from dataclasses import dataclass, field
import json
import pathlib
import os
import sqlite3
import uuid
import asyncio

from typing import Any, List, Dict, override, Optional
from urllib.parse import quote
from contextlib import AsyncExitStack

import numpy as np

import agents
from agents import Session, SQLiteSession, RunConfig, RunContextWrapper, function_tool, Runner, Agent
from agents.model_settings import ModelSettings
from agents.run import ModelInputData, CallModelData
from agents.items import TResponseInputItem
from openai import AsyncOpenAI

from .utils import _Singleton, Render, simple_handle_response_input_item_param
from .agent_hive import AgentHive

DOT_SADK_CHAT_ID = ".sadk_session_id"

class Memory(SQLiteSession):
    """
    SQLiteSession with embedding vectors and cross-session retrieval capabilities.
    """

    def _patch_table(self, conn, sql, patch_id):
        patch_hist_table = "all_schema_hist"
        conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {patch_hist_table} (
            hist varchar primary key
        )
        """)
        try:
            conn.execute(f"insert into {patch_hist_table} values ('{patch_id}')")
        except sqlite3.IntegrityError as e:
            if e.sqlite_errorname == "SQLITE_CONSTRAINT_PRIMARYKEY":
                return
            raise e
        conn.execute(sql)
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

    @override
    def _init_db_for_connection(self, conn: sqlite3.Connection):
        # We don't call super here to keep it clean for custom initialization
        pass

    def init_database(self):
        init_conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        init_conn.execute("PRAGMA journal_mode=WAL")
        # Ensure the base tables exist
        super()._init_db_for_connection(init_conn)
        self._patch_table_01_add_embedding(init_conn, self.messages_table)
        self._patch_table_02_add_chat_id(init_conn, self.sessions_table)
        init_conn.close()

    async def save_from(self, session: Session):
        """Copies items from a temporary session into the main memory database."""
        items = await session.get_items()
        await self.add_items(items)

    @override
    def _get_connection(self):
        conn = super()._get_connection()
        def hamming_dist(x1, x2):
            if not x1 or not x2: return 768
            try:
                i1 = int(x1, 16)
                i2 = int(x2, 16)
                return (i1 ^ i2).bit_count()
            except:
                return 768
        conn.create_function("hamming_dist", 2, hamming_dist)
        return conn

    async def get_items_by_session_id(self, session_id: str) -> List[TResponseInputItem]:
        """Retrieves all messages belonging to a specific session ID."""
        conn = self._get_connection()
        cursor = conn.execute(
            f"""
            SELECT message_data FROM {self.messages_table}
            WHERE session_id = ?
            ORDER BY created_at ASC
            """,
            (session_id,),
        )
        res = []
        for (message_data_str,) in cursor.fetchall():
            try:
                res.append(json.loads(message_data_str))
            except json.JSONDecodeError:
                continue
        return res

    async def search_similar(
            self,
            vecs: List[str],
            topk: int,
            exclude: list[int] = None):
        """Performs a multi-vector hamming distance search."""
        if not vecs: return []
        
        # Ensure we have at least 3 vectors for the query logic defined in search_in_memory
        args = list(vecs[:3])
        while len(args) < 3:
            args.append(args[0])

        where_clause = ""
        if exclude:
            placeholders = ", ".join(["?"] * len(exclude))
            where_clause = f"AND id NOT IN ({placeholders})"
            args.extend(exclude)
            
        args.append(topk)

        SQL = f"""
        SELECT
            id,
            session_id,
            ( hamming_dist(embedding, ?) +
              hamming_dist(embedding, ?) +
              hamming_dist(embedding, ?) )/(3.0*768) as distance
        FROM
            {self.messages_table}
        WHERE
            embedding IS NOT NULL AND length(embedding) > 48
            {where_clause}
        ORDER BY distance
        LIMIT ?
        """

        conn = self._get_connection()
        cur = conn.execute(SQL, args)
        return cur.fetchall()

    async def embed_memories(self, embed_gear):
        """Processes any messages that haven't been embedded yet."""
        conn = self._get_connection()
        SQL = f"SELECT id, message_data FROM {self.messages_table} WHERE embedding IS NULL"
        UPDATE = f"UPDATE {self.messages_table} SET embedding=? WHERE id=?"
        
        cur = conn.execute(SQL)
        rows = cur.fetchall()
        for message_id, message_data_str in rows:
            message = json.loads(message_data_str)
            text = None
            content = message.get("content")
            if isinstance(content, list):
                text = "\n".join(i.get("text", "") for i in content if i.get("type") == "text")
            elif isinstance(content, str):
                text = content
            
            if text and 10 < len(text) < 4000:
                emb = await embed_gear.get_embedding(text)
            else:
                emb = "not embeddable"
            
            conn.execute(UPDATE, (emb, message_id))
        conn.commit()

class EmbedGear:
    def __init__(self, embedding_client, embedding_model="nomic-embed-text:latest"):
        self.embedding_model = embedding_model
        self.embedding_client = embedding_client

    async def get_embedding(self, q: str) -> str:
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
    # Staged session IDs to be injected into the next LLM call
    memory_items: List[str] = field(default_factory=list)

@function_tool
async def search_in_memory(
    ctx: RunContextWrapper[MemoryContext],
    queries: List[str]
) -> str:
    """
    Searches memory using semantic vector similarity. Accepts multiple search angles.
    Matches are staged and injected into the prompt history automatically.
    """
    session = ctx.context.session
    embeder = ctx.context.embeder
    
    ctx.context.memory_items.clear()

    # Generate embeddings for the 3 queries
    embeddings = [await embeder.get_embedding(q) for q in queries]
    
    # Search for similar items (returning session IDs)
    results = await session.search_similar(embeddings, topk=10)
    
    unique_sessions = []
    for _, session_id, distance in results:
        if session_id not in unique_sessions:
            unique_sessions.append(session_id)

    if unique_sessions:
        ctx.context.memory_items.extend(unique_sessions)
        return f"Found {len(unique_sessions)} relevant past sessions. They will be injected into context for the next step."

    return "No relevant items found in memory."

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
    It fetches content for staged memories, removes tool calls, and respects a 32k token limit.
    """
    context = data.context
    if not context or not context.memory_items:
        return data.model_data

    session = context.session
    injected_messages: List[TResponseInputItem] = []
    
    current_token_count = 0
    TOKEN_LIMIT = 32000

    # Fetch full conversations for each staged session ID
    for session_id in context.memory_items:
        past_items = await session.get_items_by_session_id(session_id)
        if not past_items:
            continue

        # Rule: Remove tool call results as they are often spacious and less helpful for context
        filtered_items = [item for item in past_items if item.get("role") != "tool"]
        
        session_tokens = estimate_tokens(filtered_items)
        
        # Rule: Only insert if we stay within the 32k token limit
        if current_token_count + session_tokens > TOKEN_LIMIT:
            break
            
        current_token_count += session_tokens
        
        header = {"role": "system", "content": f"--- RECALLED MEMORY: SESSION {session_id} ---"}
        injected_messages.append(header)
        injected_messages.extend(filtered_items)

    # Combine: Recalled Memories + Current Conversation History
    current_history = data.model_data.input
    new_input = injected_messages + list(current_history)

    # Clear the stage so we don't re-inject the same memories in a tool-call loop
    context.memory_items.clear()

    return ModelInputData(
        input=new_input,
        instructions=data.model_data.instructions
    )

def _get_mcp_by_typename(tname: str, name: str, params: str):
    mcp_class = getattr(agents.mcp, tname)
    return mcp_class(name=name,
                     params=params,
                     cache_tools_list=True,
                     client_session_timeout_seconds=60)

async def main(prompt, model_name="gpt-4o", agent_name="default"):
    # Setup
    agent_hive = AgentHive()
    agent_cfg = agent_hive.get_agent(agent_name)
    
    # Initialize Memory
    config_home = pathlib.Path(os.getenv("XDG_CONFIG_HOME", str(pathlib.Path.home() / ".config"))) / "sadk"
    config_home.mkdir(parents=True, exist_ok=True)
    db_path = config_home / "session.db"
    
    memory = Memory("all", str(db_path))
    memory.init_database()

    # Setup Embedding Gear
    client = AsyncOpenAI(base_url="http://127.0.0.1:11435/v1", api_key="ollama")
    embeder = EmbedGear(client)

    # Context and Config
    ctx = MemoryContext(session=memory, embeder=embeder)
    run_config = RunConfig(
        call_model_input_filter=inject_staged_items_filter
    )

    params = {
        "name": agent_cfg.get("name", "MemoryAgent"),
        "instructions": agent_cfg.get("instructions", "Use search_in_memory to find past context."),
        "model": model_name,
        "tools": [search_in_memory],
        "model_settings": ModelSettings(tool_choice="auto")
    }

    async with AsyncExitStack() as stack:
        if "mcp" in agent_cfg:
            mcps = [_get_mcp_by_typename(x["type"], x["name"], x["params"]) for x in agent_cfg["mcp"]]
            params["mcp_servers"] = [await stack.enter_async_context(m) for m in mcps]

        agent = Agent(**params)
        temp_session = SQLiteSession("temp", str(db_path))

        result = await Runner.run(
            agent,
            prompt,
            context=ctx,
            run_config=run_config,
            session=temp_session
        )

        Render().output(result.final_output)

        # Persistence
        await memory.save_from(temp_session)
        await memory.embed_memories(embeder)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        asyncio.run(main(sys.argv[1]))
    else:
        print("Usage: python -m sadk.memory 'your prompt'")
