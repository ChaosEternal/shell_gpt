import asyncio
import json
import os
import pathlib
import sqlite3
import uuid
from pathlib import Path
from typing import Any, List, Dict, override

import numpy as np

from agents import SQLiteSession, Session
from agents.items import TResponseInputItem
from openai.types.shared import Reasoning

from .agent_hive import AgentHive

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

    def _patch_table_04_chat_info(self, conn):
        patch_id = "patch_20260302_chat_info"
        sql = """
        CREATE TABLE IF NOT EXISTS chat_info (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id     TEXT    NOT NULL,
            note        TEXT,
            rank        INT     DEFAULT 0,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
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
        self._patch_table_04_chat_info(init_conn)
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

    async def save_note(
        self,
        chat_id: str,
        note: str,
        rank: int,
        recalled_sessions: List[str],
    ) -> None:
        """Persist a session note and update ranks on all recalled sessions."""
        conn = self._get_connection()
        conn.execute(
            "INSERT INTO chat_info (chat_id, note, rank) VALUES (?, ?, ?)",
            (chat_id, note, rank),
        )
        conn.commit()
        for session_id in set(recalled_sessions):
            await self.update_session_rank(session_id, delta=rank)

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

    def get_note(self, chat_id: str) -> str | None:
        """Return the most recent note text for chat_id, or None if no note exists."""
        conn = self._get_connection()
        cur = conn.execute(
            "SELECT note FROM chat_info WHERE chat_id = ? ORDER BY created_at DESC LIMIT 1",
            (chat_id,),
        )
        row = cur.fetchone()
        return row[0] if row else None


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








