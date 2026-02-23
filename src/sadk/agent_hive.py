import pathlib
import os
import platform
import toml

import distro
import frontmatter

from .utils import _Singleton

class AgentHive(_Singleton):
    agent_hive_dir = None
    config_dir = None

    def __init__(self):
        if self.agent_hive_dir is not None:
            return
        self.config_dir = pathlib.Path(os.getenv("XDG_CONFIG_HOME", "/")) / "sadk"
        agent_hive_dir = self.config_dir / "agents"

        self.agent_hive_dir = agent_hive_dir
        if not agent_hive_dir.exists():
            self.agent_hive_dir.mkdir(parents=True)

    def _os_name(self) -> str:

        current_platform = platform.system()
        match current_platform:
            case "Linux":
                return "Linux/" + distro.name(pretty=True)
            case "Windows" | "nt":
                return "Windows " + platform.release()
            case "Darwin":
                return "Darwin/MacOS " + platform.mac_ver()[0]
            case _:
                return current_platform

    def _shell_name(self) -> str:
        current_platform = platform.system()
        if current_platform in ("Windows", "nt"):
            is_powershell = len(getenv("PSModulePath", "").split(pathsep)) >= 3
            return "powershell.exe" if is_powershell else "cmd.exe"
        return pathlib.Path(os.getenv("SHELL")).name

    def _default_agent(self):
        return dict(
            instructions=f"""Your name is Shell_ADK.
You are programming and system administration assistant.
You are managing {self._os_name()} operating system with {self._shell_name()} shell.
Provide short responses in about 100 words, unless you are specifically asked for more details.
If you need to store any data, assume it will be stored in the conversation.
When deal with time, always use tools to get current date time.
APPLY MARKDOWN formatting when possible.
""",
            name="Shell Agent")

    def _get_agent(self, name):
        agent_file = self.agent_hive_dir / f"{name}.md"
        if agent_file.is_file():
            post =  frontmatter.load(agent_file)
            agent = post.metadata
            agent["instructions"] = post.content
            return agent
        return None

    def _save_agent(self, name, config):
        agent_file = self.agent_hive_dir / f"{name}.md"
        with open(agent_file, "wb") as of:
            content = config.pop("instructions")
            if not "name" in config:
                config["name"] = name
            post = frontmatter.Post(
                content,
                **config
            )
            frontmatter.dump(post, of, handler=frontmatter.TOMLHandler())

    def get_agent(self, name, fallback=True):
        agent_cfg = self._get_agent(name)
        if agent_cfg:
            return agent_cfg
        if fallback:
            return self._default_agent()
        raise Exception(f"Agent {name} doesn't exist")

    def list_agents(self):
        if self.agent_hive_dir:
            for a in self.agent_hive_dir.glob("*.md"):
                print(a)

    def show_agent(self, name):
        agent_cfg = self.get_agent(name, fallback=False)
        if agent_cfg:
            agent_cfg["instructions"] = agent_cfg["instructions"][0:20] + "..."
            print(toml.dumps(agent_cfg))
        
    def update_agent(self, agent, instructions):
        if not instructions:
            instructions = self._default_agent()["instructions"]
        current_cfg = self._get_agent(agent)
        if current_cfg:
            current_cfg["instructions"] = instructions
        else:
            current_cfg=dict(name=agent, instructions=instructions)
        self._save_agent(agent, current_cfg)
