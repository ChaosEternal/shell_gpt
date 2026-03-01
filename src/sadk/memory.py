import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
import json
import pathlib
from pathlib import Path
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
        session: Session,
        chat_id: str,
    ):
        self.session_id = str(uuid.uuid4())
        items = await session.get_items()
        await self.add_items(items)
        # Tag the newly saved session with its chat_id
        conn = self._get_connection()
        conn.execute(
            f"UPDATE {self.sessions_table} SET chat_id = ? WHERE session_id = ?",
            (chat_id, self.session_id),
        )
        conn.commit()

    async def save_items(
        self,
        items: list,
        chat_id: str,
    ) -> str:
        """Save raw conversation items and tag them with chat_id. Returns the new session_id."""
        self.session_id = str(uuid.uuid4())
        await self.add_items(items)
        conn = self._get_connection()
        conn.execute(
            f"UPDATE {self.sessions_table} SET chat_id = ? WHERE session_id = ?",
            (chat_id, self.session_id),
        )
        conn.commit()
        return self.session_id

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
            exclude: list[int]=None,
            chat_id: str | None = None):
        args = vec[0:3]
        where_clauses = []
        if exclude:
            placeholders = ", ".join(["?"] * len(exclude))
            where_clauses.append(f"id not in ({placeholders})")
            args.extend(exclude)
        if chat_id is not None:
            where_clauses.append("s.chat_id = ?")
            args.append(chat_id)
        where = ("AND " + " AND ".join(where_clauses)) if where_clauses else ""
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
            if text and len(text) < 4000:
                emb = await embed_gear.get_embedding(text)
            else:
                emb = "not embedable"
            conn.execute(UPDATE, (emb, message_id))
        conn.commit()

    @classmethod
    def create(cls) -> "Memory":
        """Create and initialise the Memory instance from env/config."""
        if memory_db_file := os.getenv("MEMORY_DB_PATH"):
            session_db_file = memory_db_file
        else:
            ah = AgentHive()
            session_db_file = str(ah.config_dir / "session.db")
        mem = cls("all", session_db_file)
        mem.init_database()
        return mem

    async def search_and_format(
        self,
        embeder: "EmbedGear",
        description: str,
        summary: str,
        hypothesis: str,
        chat_id: str,
        token_limit: int = 96000,
    ) -> tuple[str, List[str]]:
        """
        Vector-search past sessions belonging to chat_id and return their contents as flattened text.

        Returns:
            (text, recalled_session_ids)
            text is human-readable; tool results are replaced by RESULT_OMITTED.
        """
        v = [await embeder.get_embedding(q) for q in [description, summary, hypothesis]]
        try:
            similar = await self.search_similar(v, 1000, chat_id=chat_id)
        except Exception as e:
            print(e)
            raise

        # Deduplicate, preserving rank order
        seen: set = set()
        ranked_ids: List[str] = []
        for _msg_id, session_id, *_ in similar:
            if session_id not in seen:
                seen.add(session_id)
                ranked_ids.append(session_id)

        print(f"[Memory.search_and_format] unique sessions: {len(ranked_ids)}")

        tokens_used = 0
        output_parts: List[str] = []
        recalled: List[str] = []

        for session_id in ranked_ids:
            past_items = await self.get_items_by_session_id(session_id)
            if not past_items:
                continue

            non_tool = [item for item, _ in past_items if item.get("role") != "tool"]
            session_tokens = estimate_tokens(non_tool)
            if tokens_used + session_tokens > token_limit:
                break

            tokens_used += session_tokens
            recalled.append(session_id)

            lines = [
                line for item, _ in past_items
                if (line := _format_item_as_text(item)) is not None
            ]
            if lines:
                output_parts.append(f"=== Session {session_id} ===\n" + "\n".join(lines))

        text = "\n\n".join(output_parts) if output_parts else "No matching memory found."
        print(f"[Memory.search_and_format] sessions in result: {len(output_parts)}")
        return text, recalled


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



@dataclass
class MemoryContext:
    session: Memory
    embeder: EmbedGear
    chat_id: str
    # A temporary holding area for items fetched during this turn
    memory_items: List[str] = field(default_factory=list)
    recalled_sessions: List[str] = field(default_factory=list)
    final_rank: int = 1
    # Set to True once search_in_memory has been called; gates recent-session injection
    search_done: bool = False

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

def _format_item_as_text(item: dict) -> str | None:
    """
    Render a single conversation item as a human-readable line.
    Every branch uses the uniform format:  [role]: content
      - tool results        ->  [tool]: RESULT_OMITTED
      - assistant tool call ->  [assistant]: fn_name(args)
      - text content        ->  [role]: text
    """
    role = item.get("role", "unknown")

    # Tool results: always omit the payload
    if role == "tool":
        return "[tool]: RESULT_OMITTED"

    # Assistant tool-use calls (OpenAI chat format)
    tool_calls = item.get("tool_calls")
    if tool_calls and role == "assistant":
        parts = []
        for tc in tool_calls:
            fn = tc.get("function") or {}
            name = fn.get("name", "?")
            args = fn.get("arguments", "{}")
            parts.append(f"[tool call: {name}({args})]")
        return "\n".join(parts) if parts else None

    # Regular text content
    content = item.get("content", "")
    if isinstance(content, str):
        text = content.strip()
        return f"[{role}]: {text}" if text else None
    if isinstance(content, list):
        texts = [
            part.get("text", "").strip()
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        texts = [t for t in texts if t]
        return f"[{role}]: {' '.join(texts)}" if texts else None
    return None


async def search_in_memory_raw(
    ctx: RunContextWrapper[MemoryContext],
    description: str,
    summary: str,
    hypothesis: str
) -> str:
    """
    Search past sessions using semantic vector similarity and return their contents as text.

    CRITICAL USAGE INSTRUCTIONS:
    - This tool uses embedding distance, NOT keyword matching.
    - Do NOT use short keywords or vague queries (e.g., "python error").
    - Generate full, hypothetical sentences/paragraphs representing the *content* you expect to find.

    The returned text contains the matched sessions flattened into readable lines.
    Tool call results from those sessions are replaced with RESULT_OMITTED.
    Tool calls themselves are shown as function-call lines.

    Args:
        description (str): A direct, technical description of the information sought.
        summary (str): A conversational or colloquial summary of the event.
        hypothesis (str): A hypothetical snippet of the specific log, code, or text you are looking for.

    Returns:
        str: Matched session contents as human-readable text, or a not-found message.
    """
    mem_ctx = ctx.context
    text, recalled = await mem_ctx.session.search_and_format(
        mem_ctx.embeder, description, summary, hypothesis,
        chat_id=mem_ctx.chat_id,
    )
    mem_ctx.search_done = True
    mem_ctx.recalled_sessions.extend(recalled)
    return text


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


def _is_search_in_memory_call(item: dict) -> bool:
    """Return True if this is an assistant message that only calls search_in_memory."""
    if item.get("role") != "assistant":
        return False
    tool_calls = item.get("tool_calls") or []
    return bool(tool_calls) and all(
        (tc.get("function") or {}).get("name") == "search_in_memory"
        for tc in tool_calls
    )


def _strip_tool_outputs(items: list) -> list:
    """Remove tool outputs (role='tool') and search_in_memory calls from history items."""
    return [
        item for item in items
        if item.get("role") != "tool" and not _is_search_in_memory_call(item)
    ]


async def _build_injected_sessions(
    session: "Memory",
    session_ids: List[str],
    label: str,
    token_budget: int,
    seen_sessions: set,
    recalled_sessions: List[str] | None = None,
) -> tuple:
    """
    Fetch and filter sessions by ID, stripping tool outputs and honouring the token budget.
    Returns (injected_messages, tokens_used).
    """
    injected: List[TResponseInputItem] = []
    tokens_used = 0

    if not session_ids:
        return injected, tokens_used

    injected.append({"role": "system", "content": f"--- {label} ---"})

    for session_id in session_ids:
        if session_id in seen_sessions:
            continue
        seen_sessions.add(session_id)

        past_items = await session.get_items_by_session_id(session_id)
        if not past_items:
            continue

        filtered = _strip_tool_outputs([item for item, _ in past_items])
        session_tokens = estimate_tokens(filtered)

        if tokens_used + session_tokens > token_budget:
            break

        tokens_used += session_tokens

        if recalled_sessions is not None:
            recalled_sessions.append(session_id)

        injected.extend(filtered)

    injected.append({"role": "system", "content": f"--- END OF {label} ---"})
    return injected, tokens_used


async def inject_recent_sessions_filter(data: CallModelData[MemoryContext]) -> ModelInputData:
    """
    Input filter: injects the 20 most recent sessions (tool outputs stripped) but ONLY
    before search_in_memory has been called.  Once search_done is True the filter is a no-op
    so the model isn't re-flooded with recent history on every subsequent model call.
    Also injects a mandatory forcing message on the first turn so the model cannot skip
    the search_in_memory call even with reasoning models.
    """
    context = data.context
    if not context or context.search_done:
        # After search is done, stop injecting recent sessions
        return data.model_data

    current_history = list(data.model_data.input)
    current_token_count = estimate_tokens(current_history)
    TOKEN_LIMIT = 96000

    recent_session_ids = await context.session.get_recent_sessions(20)
    seen: set = set()

    injected, _ = await _build_injected_sessions(
        context.session,
        recent_session_ids,
        "RECENT MEMORY",
        TOKEN_LIMIT - current_token_count,
        seen,
    )

    # Hard forcing message: shown only before search_in_memory fires
    force_msg = {
        "role": "system",
        "content": (
            "[MANDATORY] Your FIRST and ONLY action right now is to call `search_in_memory`. "
            "Do NOT write any text. Do NOT think out loud. Call the tool immediately."
        ),
    }

    return ModelInputData(
        input=injected + current_history + [force_msg],
        instructions=data.model_data.instructions,
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
    client = AsyncOpenAI(
        base_url="http://127.0.0.1:11434/v1",
        api_key="dummy",
    )
    set_default_openai_client(client=client, use_for_tracing=False)
    set_default_openai_api("chat_completions")
    set_tracing_disabled(disabled=True)

    base_model_settings = ModelSettings(tool_choice="auto")

    mcps = [
        _get_mcp_by_typename(x["type"], x["name"], x["params"])
        for x in agent_cfg.get("mcp", [])
    ]

    if CONTINUITY_FILE.exists():
        with open(CONTINUITY_FILE, "r") as f:
            note = f.read()
        if note.strip():
            prompt = f"{prompt}\n\n[System Note from previous session]:\n{note}"

    memory = Memory.create()
    c = AsyncOpenAI(base_url="http://127.0.0.1:11435/v1", api_key="no")
    embeder = EmbedGear(c, "nomic-embed-text:latest")

    # Resolve chat_id: use the per-directory session file if present
    chat_id_file = Path(".") / ".sadk_session_id"
    if chat_id_file.exists():
        chat_id = chat_id_file.read_text().strip()
    else:
        chat_id = str(uuid.uuid4())
        chat_id_file.write_text(chat_id)

    ctx = MemoryContext(session=memory, embeder=embeder, chat_id=chat_id)
    await memory.embed_memories(embeder)



    async with AsyncExitStack() as stack:
        mcp_servers = [await stack.enter_async_context(m) for m in mcps]

        # ── Agent 2: only leaves a note and rating ──────────────────────────
        agent2_kwargs: dict = {
            "name": "note_taker",
            "model": model_name,
            "model_settings": base_model_settings,
            "instructions": (
                "You are a memory note-taker. Your sole responsibility is to call "
                "`leave_note_and_rate` with a concise note summarising what happened in "
                "the conversation and a ranking of how useful the recalled memories were "
                "(-2 to 2). Do nothing else."
            ),
            "tools": [leave_note_and_rate],
        }
        if mcp_servers:
            agent2_kwargs["mcp_servers"] = mcp_servers
        agent2 = Agent(**agent2_kwargs)

        # ── Agent 1: main agent – searches memory then handles the user request ──
        # search_in_memory MUST be the first tool call; once done, inject_recent_sessions_filter
        # becomes a no-op (guarded by search_done flag) so recent sessions are not re-injected.
        agent1_instructions = (
            (agent_cfg.get("instructions") or "") +
            "\n\n## TOOL PROTOCOL ADDENDUM\n"
            "**`search_in_memory` (STEP 1 — REQUIRED FIRST CALL)**\n"
            "- Derive three full-sentence queries from the user's prompt:\n"
            "  - `description`: precise technical phrasing of what is sought\n"
            "  - `summary`: colloquial/narrative restatement of the same topic\n"
            "  - `hypothesis`: a hypothetical excerpt (log line, code snippet, note) you expect to find\n"
            "- Do not use keywords. Write complete sentences.\n\n"
            "**`leave_note_and_rate` — NOT AVAILABLE TO YOU**\n"
            "- This tool is handled by the `note_taker` agent.\n"
            "- When your work is complete, hand off to `note_taker`. Do not attempt to call "
            "`leave_note_and_rate` yourself."
        )
        agent1_kwargs: dict = {
            "name": agent_cfg.get("name", "main_agent"),
            "model": model_name,
            # tool_choice="required" forces a tool call on every turn,
            # preventing the model from skipping search_in_memory on the first turn.
            "model_settings": ModelSettings(tool_choice="required"),
            "instructions": agent1_instructions,
            "tools": [search_in_memory, read_tutorial],
            "handoffs": [agent2],
        }
        if mcp_servers:
            agent1_kwargs["mcp_servers"] = mcp_servers
        agent1 = Agent(**agent1_kwargs)

        # inject_recent_sessions_filter fires only before search_in_memory is called
        run_config = RunConfig(call_model_input_filter=inject_recent_sessions_filter)
        session = SQLiteSession("temp")

        try:
            result = Runner.run_streamed(
                agent1,
                prompt,
                context=ctx,
                run_config=run_config,
                session=session,
            )
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
        except agents.exceptions.MaxTurnsExceeded as e:
            print(e)

        await memory.save_from(session, chat_id)

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
