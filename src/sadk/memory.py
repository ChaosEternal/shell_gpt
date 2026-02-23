import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
import json
import pathlib
import os
import sqlite3
import uuid

from typing import Any, List, Dict, override
from urllib.parse import quote

import numpy as np

import agents
#from agents import Agent, Runner, Session, SQLiteSession, RunConfig, RunContextWrapper, function_tool
from agents import (
    Agent,
    Runner,
    RunConfig,
    RunContextWrapper,    
    Session,
    SQLiteSession,
    function_tool,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)
from agents.run import ModelInputData, CallModelData
from agents.items import TResponseInputItem
from agents.model_settings import ModelSettings
from mcp.types import CallToolResult, GetPromptResult, InitializeResult, ListPromptsResult, TextContent
from openai import AsyncOpenAI
from openai.types.shared import Reasoning
from .utils import _Singleton, Render, simple_handle_response_input_item_param
from .agent_hive import AgentHive
from .tutorial import read_tutorial

DOT_SADK_CHAT_ID = ".sadk_session_id"
CONTINUITY_FILE = pathlib.Path(os.getenv("XDG_CONFIG_HOME", "~/.config")).expanduser() / "sadk" / "continuity.txt"

class Memory(SQLiteSession):
    """
    SQLiteSession with embedding vectors.
    """

    def _patch_table(self, conn, sql, patch_id):
        patch_hist_table = f"all_schema_hist"
        conn.execute(f"""
        CREATE TABLE IF NOT EXISTS  {patch_hist_table} (
            hist varchar primary key
        )
        """
        )
        try:
            conn.execute(f"insert into {patch_hist_table} values ('{patch_id}')")
        except sqlite3.IntegrityError as e:
            if e.sqlite_errorname == "SQLITE_CONSTRAINT_PRIMARYKEY":
                return
            raise e
        conn.execute(
            sql
        )
        conn.commit()


    def _patch_table_01_add_embedding(self, conn, messages_table):
        patch_id = "patch_20251122_add_embedding"
        sql = f"""
             ALTER TABLE {messages_table}
                 ADD COLUMN embedding TEXT
        """
        self._patch_table(conn, sql, patch_id)


    def _patch_table_02_add_chat_id(self, conn, agent_session):
        patch_id = "patch_20251127_add_chat_id"
        sql = f"""
        ALTER TABLE {agent_session}
        ADD COLUMN chat_id TEXT
        """
        self._patch_table(conn, sql, patch_id)

    def _patch_table_03_add_rank(self, conn, agent_session):
        patch_id = "patch_20260125_add_rank"
        sql = f"""
        ALTER TABLE {agent_session}
        ADD COLUMN rank INT
        """
        self._patch_table(conn, sql, patch_id)

    @override
    def _init_db_for_connection(self, conn: sqlite3.Connection):
        pass

    def init_database(self):
        init_conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        init_conn.execute("PRAGMA journal_mode=WAL")
        SQLiteSession._init_db_for_connection(self, init_conn)
        self._patch_table_01_add_embedding(init_conn, self.messages_table)
        self._patch_table_02_add_chat_id(init_conn, self.sessions_table)
        self._patch_table_03_add_rank(init_conn, self.sessions_table)
        init_conn.close()

    async def save_from(
        self,
        session: Session
    ):
        self.session_id = str(uuid.uuid4())
        items = await session.get_items()
        await self.add_items(items)

    @override
    def _get_connection(self):
        conn = SQLiteSession._get_connection(self)

        def hamming_dist(x1, x2):
            i1 = int(x1, 16)
            i2 = int(x2, 16)
            return (i1 ^ i2).bit_count()

        conn.create_function("hamming_dist", 2, hamming_dist)
        return conn

    async def get_items_by_session_id(self, session_id):
        conn = self._get_connection()
        cursor = conn.execute(
            f"""
            SELECT message_data, created_at FROM {self.messages_table}
            WHERE session_id = ?
            ORDER BY created_at ASC
            """,
            (session_id,),
        )
        res = []
        for message_data, created_at in cursor.fetchall():
            try:
                res.append((json.loads(message_data), created_at))
            except json.JSONDecodeError:
                continue
        return res

    async def get_recent_sessions(self, n: int) -> List[str]:
        """Fetches the n most recent session IDs from the memory database."""
        conn = self._get_connection()
        cursor = conn.execute(
            f"""
                SELECT session_id FROM {self.sessions_table}
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (n,),
        )
        return [row[0] for row in cursor.fetchall()]

    async def update_session_rank(self, session_id: str, delta: int):
        conn = self._get_connection()
        conn.execute(
            f"UPDATE {self.sessions_table} SET rank = IFNULL(rank, 1) + ? WHERE session_id = ?",
            (delta, session_id)
        )
        conn.commit()

    async def get_max_rank(self, session_ids: List[str]) -> int:
        if not session_ids:
            return 1
        conn = self._get_connection()
        placeholders = ", ".join(["?"] * len(session_ids))
        cursor = conn.execute(
            f"SELECT MAX(IFNULL(rank, 1)) FROM {self.sessions_table} WHERE session_id IN ({placeholders})",
            session_ids
        )
        row = cursor.fetchone()
        return row[0] if row and row[0] is not None else 1

    async def search_similar(
            self,
            vec: List[str],
            topk: int,
            no_payload: bool = True,
            exclude: list[int]=None):
        args = vec[0:3]
        if exclude:
            where = "AND id not in ("
            where += ", ".join(["?"] * len(exclude))
            where +=")"
            args.extend(exclude)
        else:
            where = ""
        args += [topk]
        message_column = "m.message_data," if not no_payload else ""
        SQL = f"""
        SELECT
        m.id,
        m.session_id,
        {message_column}
        ( hamming_dist(m.embedding, ?) +
          hamming_dist(m.embedding, ?) +
          hamming_dist(m.embedding, ?) )/(3.0*768) as distance,
        ifnull(s.rank, 0)
        FROM
        {self.messages_table} m
        LEFT JOIN
            {self.sessions_table} s ON m.session_id = s.session_id
        WHERE
        length(embedding) > 48
        {where}
        ORDER BY distance
        LIMIT ?
        """

        conn = self._get_connection()
        cur = conn.execute(SQL, args)
        results = cur.fetchall()
        sorted_results = sorted(results, key=lambda x: (-x[3], x[2]))
        return sorted_results[:topk]

    async def embed_memories(self, embed_gear):
        conn = self._get_connection()
        SQL = f"""
        SELECT id, session_id, message_data, embedding
        FROM {self.messages_table}
        WHERE embedding is null
        """
        UPDATE = f"""
        UPDATE {self.messages_table}
        SET embedding=(?)
        WHERE id=(?)
        """
        cur = conn.execute(SQL)
        for message_id, session_id, message_data, embedding in cur.fetchall():
            message = json.loads(message_data)
            text = None
            if message.get("type") == "reasoning" and message.get("content"):
                content = message.get("content")
                if isinstance(content, list):
                    text = "\n".join(i.get("text", "") for i in content)
                elif isinstance(content, str):
                    text = content
            if text and len(text) < 1800:
                emb = await embed_gear.get_embedding(text)
            else:
                emb = "not embedable"
            conn.execute(UPDATE, (emb, message_id))
        conn.commit()

class EmbedGear:
    def __init__(self, embedding_client, embedding_model="nomic-embed-text:latest"):
        self.embedding_model = embedding_model
        self.embedding_client = embedding_client

    async def get_embedding(self, q):
        resp = await self.embedding_client.embeddings.create(
            input=q,
            model=self.embedding_model,
            dimensions=768
        )
        vec = np.array(resp.data[0].embedding, dtype=np.float32)
        packed = np.packbits(vec > 0)
        return packed.tobytes().hex()


def create_memory():
    
    if memory_db_file := os.getenv("MEMORY_DB_PATH"):
        session_db_file = memory_db_file
    else:
        ah = AgentHive()
        config_home = ah.config_dir
        session_db_file = str(config_home / "session.db")

    mem = Memory("all", session_db_file)
    mem.init_database()
    return mem

@dataclass
class MemoryContext:
    session: Memory
    embeder: EmbedGear
    # A temporary holding area for items fetched during this turn
    memory_items: List[str] = field(default_factory=list)
    recalled_sessions: List[str] = field(default_factory=list)
    final_rank: int = 1

@function_tool
async def leave_note_and_rate(ctx: RunContextWrapper[MemoryContext], note: str, rank: int) -> str:
    """
    Rate the utility of the recalled sessions before providing the final response
    and Leave a note for the next session.

    Rating is to optimize the memory engine for future interactions.
    In the note, save instructions, context, or reminders that should be persisted
    to the beginning of the next session (after a restart).

    Args:
        note (str): The content of the note to save.
        rank (int): An integer rating between -2 and 2.
                   2: Recalled context was extremely helpful.
                   0: Context was neutral or only slightly relevant.
                  -2: Context was irrelevant or distracting.
    """
    CONTINUITY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONTINUITY_FILE, "w") as f:
        f.write(note)

    await rate_before_final(ctx, rank)
    return "Note and rank are saved."

async def search_in_memory_raw(
    ctx: RunContextWrapper[MemoryContext],
    description: str,
    summary: str,
    hypothesis: str
) -> str:
    """
    Searches memory using semantic vector similarity. Accepts multiple search angles to improve accuracy.

    CRITICAL USAGE INSTRUCTIONS:p
    - This tool uses embedding distance, NOT keyword matching.
    - Do NOT use short keywords, boolean operators, or vague queries (e.g., "python error").
    - You MUST generate full, hypothetical sentences or paragraphs that represent the *content* you expect to find.
    - The search engine matches the semantic meaning of your input against the stored memories.

    The found items will NOT be returned directly. Instead, they will be injected into
      the conversation history and will appear in chronological order relative to the current interaction.

    Args:
        queries (List[str]): A list of exactly 3 distinct hypothetical search variations to maximize recall:
        description (str): A direct, technical description of the information.
        summary (str): A conversational or colloquial summary of the event.
        hypothesis (str). A hypothetical snippet of the specific log, code, or text you are looking for.

    Returns:
        str: A string indicating whether or not some items were found.
    """
    # Access our "Session DB" from the context

    queries = [description, summary, hypothesis]
    session = ctx.context.session
    embeder = ctx.context.embeder
    memory_items = ctx.context.memory_items
#    memory_items.clear()

    v = [await embeder.get_embedding(q) for q in queries]
    try:
        l = await session.search_similar(v, 1000)
    except Exception as e:
        print(e)
        raise e
    for i, s, *_ in l:
        memory_items.append(s)
    print("Found memory, items:", len(memory_items))
    if memory_items:
        # Found it! Stage it for injection.
        return "Memory found and staged, look for RECENT MEMORY or RECALLED MEMORY in conversation."

    return f"No item found."

search_in_memory = function_tool(func=search_in_memory_raw, name_override="search_in_memory")


async def rate_before_final(ctx: RunContextWrapper[MemoryContext], rank: int) -> str:
    """
    Rate the utility of the recalled sessions before providing the final response.
    Use this to optimize the memory engine for future interactions.

    Args:
        rank (int): An integer rating between -2 and 2.
                   2: Recalled context was extremely helpful.
                   0: Context was neutral or only slightly relevant.
                  -2: Context was irrelevant or distracting.
    """
    memory = ctx.context.session
    recalled = list(set(ctx.context.recalled_sessions))

    if rank > 0:
        # Success path: Current session inherits the authority of the best recalled memory
        max_recalled_rank = await memory.get_max_rank(recalled)
        ctx.context.final_rank = max_recalled_rank + rank
    else:
        # Baseline/Failure path
        ctx.context.final_rank = rank

    # Reward/Penalize all recalled memories based on the model's judgment
    for s_id in recalled:
        await memory.update_session_rank(s_id, delta=rank)

    return "Rating accepted."

def estimate_tokens(items: List[TResponseInputItem]) -> int:
    """Roughly estimate tokens based on character count (approx 4 chars/token)."""
    total_chars = 0
    for item in items:
        content = item.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    total_chars += len(part.get("text", ""))
    return total_chars // 4

async def inject_staged_items_filter(data: CallModelData[MemoryContext]) -> ModelInputData:
    """
    An input filter that runs right before the model call.
    It fetches content for staged memories and recent sessions, removes tool calls,
    and respects a 96k cumulative token limit.
    """
    context = data.context
    if not context:
        return data.model_data

    session = context.session
    injected_messages: List[TResponseInputItem] = []

    # Starting count with current session history
    current_history = list(data.model_data.input)
    current_token_count = estimate_tokens(current_history)
    #    context.recalled_sessions.clear()
    
    TOKEN_LIMIT = 96000

    # 1. Fetch Recent Sessions (up to 20)
    recent_session_ids = await session.get_recent_sessions(20)

    # Use a set to avoid duplicating sessions if a recent one was also recalled
    seen_sessions = set()

    print("before inject, items:", len(context.memory_items))
    
    for mem_sets_name, mem_sets in [("RECALLED MEMORY", context.memory_items),
                                    ("RECENT MEMORY", recent_session_ids)]:
        if mem_sets:
            header = {"role": "system", "content": f"--- {mem_sets_name} ---"}
            print(header)
            injected_messages.append(header)
            
        print("hist_size_a", current_token_count)
        
        for session_id in mem_sets:

            if session_id in seen_sessions:
                continue
            seen_sessions.add(session_id)
            past_items = await session.get_items_by_session_id(session_id)

            if not past_items:
                continue
            
            print("hist_size_b", current_token_count)
            
            # Rule: Remove tool call results
            filtered_items = [item for item, created_at in past_items if item.get("role") != "tool"]

            session_tokens = estimate_tokens(filtered_items)

            # Rule: Only insert if we stay within the 96k total token limit
            if current_token_count + session_tokens > TOKEN_LIMIT:
                break

            current_token_count += session_tokens

            if mem_sets_name == "RECALLED MEMORY":
                print("insert")
                context.recalled_sessions.append(session_id)

            injected_messages.extend(filtered_items)

    if context.memory_items or recent_session_ids:
        footer = {"role": "system", "content": f"--- END OF MEMORY ---"}
        injected_messages.append(footer)
        
    # Combine: Recalled/Recent + Current Conversation History
    new_input = injected_messages + current_history

    # Clear the stage for the next loop
    #context.memory_items.clear()

    print("after injection", len(context.recalled_sessions))

    return ModelInputData(
        input=new_input,
        instructions=data.model_data.instructions
    )

def _get_mcp_by_typename(tname: str, name: str, params: str):
    mcp_class = getattr(agents.mcp, tname)
    class wrapped_mcp(mcp_class):
        @override
        async def call_tool(
                self,
                tool_name: str,
                arguments: dict[str, Any] | None,
                meta: dict[str, Any] | None = None,
        ) -> CallToolResult:
             res = await super().call_tool(
                 tool_name,
                 arguments,
                # meta
             )
             if len(res.model_dump_json()) > 32768:
                 return CallToolResult(isError=True, content=[TextContent(type="text", text="The function result exceeds size limit.")])
             return res
    return wrapped_mcp(name=name,
                     params=params,
                     cache_tools_list=True,
                     client_session_timeout_seconds=60)

async def main(prompt, model_name="gpt-oss:latest", agent_name="default"):
    ah = AgentHive()
    agent_cfg = ah.get_agent(agent_name)
    params = {
        key: agent_cfg[key]
        for key in ["name", "instructions"]
        if key in agent_cfg
    }
    client = AsyncOpenAI(
        base_url="http://127.0.0.1:11434/v1",
        api_key="dummy",
    )
    set_default_openai_client(client=client, use_for_tracing=False)
    set_default_openai_api("chat_completions")
    set_tracing_disabled(disabled=True)
    params["model"] = model_name
    params["model_settings"] = ModelSettings(
        tool_choice="auto",
        reasoning=Reasoning(effort="high")
    )
    # params["tools"] = [get_current_time]
    mcps = [
        _get_mcp_by_typename(x["type"], x["name"], x["params"])
        for x in agent_cfg.get("mcp", [])
    ]
    params["tools"] = [
        search_in_memory,
        leave_note_and_rate,
        read_tutorial
    ]
    
    if CONTINUITY_FILE.exists():
        with open(CONTINUITY_FILE, "r") as f:
            note = f.read()
        if note.strip():
            # Inject the note into the prompt
            prompt = f"{prompt}\n\n[System Note from previous session]:\n{note}"
    memory = create_memory()
    c = AsyncOpenAI(base_url="http://127.0.0.1:11435/v1", api_key="no")
    embeder = EmbedGear(c, "nomic-embed-text:latest")

    ctx = MemoryContext(session=memory, embeder=embeder)
    run_config = RunConfig(
        call_model_input_filter=inject_staged_items_filter,
    )
    await memory.embed_memories(embeder)
    async with AsyncExitStack() as stack:
        mcp_servers = [await stack.enter_async_context(m) for m in mcps]
        if mcps:
            params["mcp_servers"] = mcp_servers
        agent = Agent(
            **params
        )
        session = SQLiteSession("temp")
        try:
            result = Runner.run_streamed(
                agent,
                prompt,
                context=ctx,
                run_config=run_config,
                session=session)
        except agents.exceptions.MaxTurnsExceeded as e:
            print(e)
            pass

        async for event in result.stream_events():
            match event.type:
                case "raw_response_event":
                    continue
                case "agent_updated_stream_event":
                    Render().output(f"Handoff to Agent: {event.new_agent.name}")
                case "run_item_stream_event":
                    ii = event.item.to_input_item()
                    Render().output(simple_handle_response_input_item_param(ii))
                case _:
                    print("other event")
                    print(event)

        await memory.save_from(session)

    Render().output("\n---\n" + result.final_output)

def test_main():
    import asyncio
    import sys

    from openai import AsyncOpenAI

    sess = MemSession(sys.argv[1])
    search_phrase = sys.argv[2:5]
    em = sys.argv[5]
    sess.init_database()

    c = AsyncOpenAI(base_url="http://127.0.0.1:11435/v1", api_key="no")
#    m = EmbedGear(c, "embeddinggemma:300m")
#    m = EmbedGear(c, "nomic-embed-text:latest")
    m = EmbedGear(c, em)

    async def embd_and_search():
        await sess.embed_memories(m)

        v = [ await m.get_embedding(x) for x in search_phrase]
        l = await sess.search_similar(v, 10)
        for i in l:
            print(i)
    asyncio.run(embd_and_search())
    return
    #    cur = db.execute("select vec_length(vec_bit(?))", [v])
    cur = db.execute("select vec_length(?)", [v.astype(np.int8)])
    print(cur.fetchall())
    cur = db.execute("select vec_distance_hamming(vec_bit(?), vec_bit(?))", [v, v])
    print(cur.fetchall())
    db.execute("create virtual table vec_e using vec0(v bit[768])")
    db.execute("insert into vec_e(v) values (vec_bit(?))", [v])

if __name__ == "__main__":
    import sys
    asyncio.run(main(sys.argv[1]))
