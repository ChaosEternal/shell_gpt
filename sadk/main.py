import asyncio
from contextlib import AsyncExitStack
import os
from pathlib import Path
import time
import sys
from typing import Annotated, Any, Dict, Optional

from agents import (
    Agent,
    Runner,
    function_tool,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)
from agents.model_settings import ModelSettings
from agents.mcp import MCPServerStreamableHttp, MCPServerSse, MCPServerStdio
import cyclopts
from cyclopts import App, Parameter, ValidationError
import distro
from openai import AsyncOpenAI
from rich.console import Console
from rich.markdown import Markdown

from .agent_hive import AgentHive
from .session import SessionHive
from .utils import Render, simple_handle_response_input_item_param

app = App(name="sadk", version="0.1.0")

@function_tool
def get_weather(city: str):
    return f"The weather in {city} is sunny."

@function_tool
def get_current_time() -> str:
    "Get current date time."
    return f"Now is {time.strftime("%a, %d %b %Y %H:%M:%S %Z")}.\n"

def get_mcp_by_typename(tname: str, name: str, params: str):
    d = {
        "MCPServerStreamableHttp": MCPServerStreamableHttp,
        "MCPServerSse":            MCPServerSse,
        "MCPServerStdio":          MCPServerStdio
    }
    return d[tname](name=name,
                    params=params,
                    cache_tools_list=True,
                    client_session_timeout_seconds=60)

async def main(model_name, agent_name, prompt, session_id):
    agent_hive = AgentHive()
    agent_cfg = agent_hive.get_agent(agent_name)
    params = {
        key: agent_cfg[key]
        for key in ["name", "instructions"]
        if key in agent_cfg
    }
    params["model"] = model_name
    params["model_settings"] = ModelSettings(tool_choice="auto")
#    params["tools"] = [get_current_time]
    mcps = [
        get_mcp_by_typename(x["type"], x["name"], x["params"])
        for x in agent_cfg.get("mcp", [])
    ]
    async with AsyncExitStack() as stack:
        mcp_servers = [await stack.enter_async_context(m) for m in mcps]
        if mcps:
            params["mcp_servers"] = mcp_servers
        agent = Agent(
            **params
        )
        if session_id == "temp":
            result = Runner.run_streamed(agent, prompt)
        else:
            session_hive = SessionHive()
            result = Runner.run_streamed(agent, prompt, session=session_hive.get_session(session_id))

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

    Render().output("\n---\n" + result.final_output)

    
@app.meta.default
def meta(
    *tokens: Annotated[str, Parameter(show=False, allow_leading_hyphen=True)],
    config_file: Annotated[Optional[Path], Parameter(help="Config file")] = None,
):
    if not config_file:
        CONFIG_FILE = Path(os.getenv("XDG_CONFIG_HOME", ".")) / "sadk" / "config.toml"

    app.config = [
        cyclopts.config.Toml(
            CONFIG_FILE,
        ),
        cyclopts.config.Env(
            "SADK_",  # Every environment variable will begin with this.
        )
    ]
    app(tokens)

def null_prompt(type_, value):
    if not value and sys.stdin.isatty():
        raise TypeError("PROMPT needed.")

@app.default
def run(
    prompt: Annotated[Optional[str], Parameter(validator=null_prompt)] = "",
    *,
    base_url: str,
    api_key: str,
    model: str,
    agent: str = "default",
    md: bool = True,
    code_theme: str = "dracula",
    session: str = "auto"
):
    """
    Invoke the agent.
    
    Parameters
    ----------
    prompt: str
        The prompt.
    base_url: str
        The base_url of your OpenAI API.
    api_key: str
        The OpenAI API Key.
    model: str
        The LLM Model to use.
    session: str
        The conversation session id.
    """
    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    set_default_openai_client(client=client, use_for_tracing=False)
    set_default_openai_api("chat_completions")
    set_tracing_disabled(disabled=True)

    if not sys.stdin.isatty():
        prompt_in = sys.stdin.read()
        prompt = f"{prompt}\n{prompt_in}"
        
    asyncio.run(main(model, agent, prompt, session))


chat_mgr = App(name="--chat", help="Chat sessions inspections.")
app.command(chat_mgr)

@chat_mgr.command(name="--show-history")
def chat_hist(
        session: str,
        start: int = 0,
        md: bool = True
):
    Render().set_no_md(md)
    asyncio.run(
        SessionHive().print_history(session, start)
    )

@chat_mgr.command(name="--show-last-chat")
def last_chat_hist(
        session: str,
        md: bool = True
):
    Render().set_no_md(md)
    asyncio.run(
        SessionHive().print_history(session, 1)
    )

@app.command(name="--list-agents")
def listagents():
    "List Agents"
    AgentHive().list_agents()

@app.command(name="--install-default")
def install_default_agent():
    "Create the default agent"
    AgentHive().update_agent("default", None)

@app.command(name="--show-agent")
def show_agent(agent: str):
    "Show agent details"
    AgentHive().show_agent(agent)

@app.command(name="--add-mcp")
def add_mcp(agent: int,
            mcp: str):
    "Add MCP to agent"
    AgentHive().add_agent_mcp(agent, mcp)

def cli():
    app.meta()
    
if __name__ == "__main__":
    cli()

