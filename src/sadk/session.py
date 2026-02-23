import pathlib
import os
import sqlite3
import sqlite_vec

from typing import override
from urllib.parse import quote

import numpy as np

from agents import  SQLiteSession
from .utils import _Singleton, Render, simple_handle_response_input_item_param

DOT_SADK_CHAT_ID = ".sadk_session_id"

class EmbSession(SQLiteSession):
    """
    SQLiteSession with embedding vectors.
    """
    def _patch_table(self, conn, messages_table):
        patch_id = "patch_20251122_add_embedding"
        patch_hist_table = f"{messages_table}_schema_hist"
        conn.execute(f"""
        CREATE TABLE IF NOT EXISTS  {patch_hist_table}(
            hist varchar primary key
        )
        """
        )
        try:
            conn.execute(f"insert into {messages_table}_schema_hist values ('{patch_id}')")
        except sqlite3.IntegrityError as e:
            if e.sqlite_errorname == "SQLITE_CONSTRAINT_PRIMARYKEY":
                return
            raise e
        conn.execute(
            f"""
             ALTER TABLE {messages_table}
                 ADD COLUMN embedding TEXT
            """
        )
        conn.commit()

    @override
    def _init_db_for_connection(self, conn: sqlite3.Connection):
        SQLiteSession._init_db_for_connection(self, conn)
        self._patch_table(conn, self.messages_table)

    def search_similar(self, q):
        pass


class MemoryGear:
    def __init__(self, session: EmbSession, embedding_client, embedding_model):
        self.session = EmbSession
        self.embedding_model = embedding_model
        self.embedding_client = embedding_client

    async def get_embedding(self, q):
        resp = await self.embedding_client.embeddings.create(
            input=q,
            model=self.embedding_model,
            dimensions=768
        )
        vec = np.array(resp.data[0].embedding, dtype=np.float32)
        print(len(vec), vec.shape)
        packed = np.packbits(vec > 0)
        print(len(packed), packed.shape)
        return packed #.tobytes().hex()
        
        
class SessionHive(_Singleton):
    session_db_file = None

    def __init__(self):
        if self.session_db_file is not None:
            return

        config_home = pathlib.Path(os.getenv("XDG_CONFIG_HOME", "/")) / "sadk"
        if config_home.exists():
            self.session_db_file = str(config_home / "session.db")
        else:
            self.session_db_file = str("/tmp/session.db")

    def get_session(self, session_id, session_db=None):
        if session_id == "auto":
            session_id = self._auto_chat_id()
        if session_db:
            return SQLiteSession(session_id, session_db)
        else:
            return SQLiteSession(session_id, self.session_db_file)

    async def print_history(self, session_id, items, session_db=None):
        session = self.get_session(session_id, session_db)

        for pos, item in enumerate(await session.get_items(items)):
            m = "\n"

            m += simple_handle_response_input_item_param(item)

            if pos < items - 1:
                m += "\n\n" + "-"*10 + "\n"
            Render().output(m)

    def _auto_chat_id(self) -> str:
        current_dir = pathlib.Path.cwd()

        while True:
            if (dot_sgpt_file := current_dir / DOT_SADK_CHAT_ID).is_file():
                return dot_sgpt_file.read_text().strip()
            if (dot_git := current_dir / ".git").exists():
                return "GIT_" + quote(dot_git.as_posix(), safe="")
            if current_dir == current_dir.parent:
                break
            current_dir = current_dir.parent

        if sh:=os.getenv("SHELL"):
            if os.path.basename(sh) in ["bash", "sh", "zsh"]:
                sid = os.getsid(0)
                return f"SHELL_{sid}"

        raise Exception("Can't determine chat_id for auto: no .sgpt_chat_id found, not in git repo and not in a shell")


def test_main():
    import sqlite_vec
    import asyncio
    from openai import AsyncOpenAI
    db = sqlite3.connect(":memory:")
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    c = AsyncOpenAI(base_url="http://127.0.0.1:11435/v1", api_key="no")
    m = MemoryGear("a", c, "embeddinggemma:300m")    
    v = asyncio.run(m.get_embedding("aasdasd")) #.encode("ascii")
    #    cur = db.execute("select vec_length(vec_bit(?))", [v])
    cur = db.execute("select vec_length(?)", [v.astype(np.int8)])
    print(cur.fetchall())
    cur = db.execute("select vec_distance_hamming(vec_bit(?), vec_bit(?))", [v, v])
    print(cur.fetchall())
    db.execute("create virtual table vec_e using vec0(v bit[768])")
    db.execute("insert into vec_e(v) values (vec_bit(?))", [v])

if __name__ == "__main__":
    test_main()
    


