import pathlib
import os
from urllib.parse import quote

from agents import  SQLiteSession
from .utils import _Singleton, Render, simple_handle_response_input_item_param

DOT_SADK_CHAT_ID = ".sadk_session_id"

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
