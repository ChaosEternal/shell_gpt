"""
memoryagent.py — Memory-augmented agent orchestration.

Reads agent config (instructions, MCP servers) from AgentHive, then runs a
two-agent pipeline:

  agent1 (main)   – search_in_memory → do work → hand off to note_taker
  agent2 (hidden) – leave_note_and_rate (note + rank → memory service)

Run:
    python -m sadk.memoryagent "your prompt here"
"""

from __future__ import annotations

import asyncio
import os
import uuid
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, override

import agents
from agents import (
    Agent,
    Runner,
    RunConfig,
    RunContextWrapper,
    SQLiteSession,
    function_tool,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)
from agents.model_settings import ModelSettings
from mcp.types import CallToolResult, TextContent
from openai import AsyncOpenAI

from sadk.memory_client import MemoryClient
from .agent_hive import AgentHive
from .tutorial import read_tutorial
from .utils import Render, simple_handle_response_input_item_param


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

@dataclass
class MemoryContext:
    chat_id: str
    client: MemoryClient
    recalled_sessions: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent tools
# ---------------------------------------------------------------------------

@function_tool
async def search_in_memory(
    ctx: RunContextWrapper[MemoryContext],
    description: str,
    summary: str,
    hypothesis: str,
) -> str:
    """
    Search past sessions using semantic vector similarity and return their contents as text.

    CRITICAL USAGE INSTRUCTIONS:
    - This tool uses embedding distance, NOT keyword matching.
    - Generate full hypothetical sentences, not keywords.

    Args:
        description (str): A direct, technical description of the information sought.
        summary (str): A conversational or colloquial summary of the event.
        hypothesis (str): A hypothetical snippet (log line, code, note) you expect to find.

    Returns:
        str: Matched session contents as human-readable text, or a not-found message.
    """
    mem_ctx = ctx.context
    result = await mem_ctx.client.search_in_memory(
        chat_id=mem_ctx.chat_id,
        description=description,
        summary=summary,
        hypothesis=hypothesis,
    )
    mem_ctx.recalled_sessions.extend(result.recalled_sessions)
    return result.text


@function_tool
async def leave_note_and_rate(
    ctx: RunContextWrapper[MemoryContext],
    note: str,
    rank: int,
) -> str:
    """
    Save a session note and rate the utility of recalled memories.

    Saves the note to persistent storage and updates session ranks via the memory service.
    Called by the hidden note_taker agent at the end of every run.

    Args:
        note (str): Detailed note: state, variables, decisions, next steps.
        rank (int): Memory utility rating (-2 to 2). 2 = very helpful, -2 = irrelevant.
    """
    mem_ctx = ctx.context
    # Persist note + update session ranks via the memory service
    await mem_ctx.client.leave_note_and_rate(
        chat_id=mem_ctx.chat_id,
        note=note,
        rank=rank,
        recalled_sessions=list(set(mem_ctx.recalled_sessions)),
    )
    return "Note and rank saved."



# ---------------------------------------------------------------------------
# MCP helper
# ---------------------------------------------------------------------------

def _get_mcp_by_typename(tname: str, name: str, params: dict):
    mcp_class = getattr(agents.mcp, tname)

    class _WrappedMCP(mcp_class):
        @override
        async def call_tool(
            self,
            tool_name: str,
            arguments: dict[str, Any] | None,
            meta: dict[str, Any] | None = None,
        ) -> CallToolResult:
            res = await super().call_tool(tool_name, arguments)
            if len(res.model_dump_json()) > 32768:
                return CallToolResult(
                    isError=True,
                    content=[TextContent(type="text", text="Result exceeds size limit.")],
                )
            return res

    return _WrappedMCP(
        name=name,
        params=params,
        cache_tools_list=True,
        client_session_timeout_seconds=60,
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

async def main(prompt: str, model_name: str = "gpt-oss:latest", agent_name: str = "default"):
    # ── Config ──────────────────────────────────────────────────────────────
    ah = AgentHive()
    agent_cfg = ah.get_agent(agent_name)

    oai = AsyncOpenAI(base_url="http://127.0.0.1:11434/v1", api_key="dummy")
    set_default_openai_client(client=oai, use_for_tracing=False)
    set_default_openai_api("chat_completions")
    set_tracing_disabled(disabled=True)

    base_settings = ModelSettings(tool_choice="auto")

    mcps = [
        _get_mcp_by_typename(x["type"], x["name"], x["params"])
        for x in agent_cfg.get("mcp", [])
    ]


    # ── chat_id ─────────────────────────────────────────────────────────────
    chat_id_file = Path(".") / ".sadk_session_id"
    if chat_id_file.exists():
        chat_id = chat_id_file.read_text().strip()
    else:
        chat_id = str(uuid.uuid4())
        chat_id_file.write_text(chat_id)

    async with AsyncExitStack() as stack:
        mcp_servers = [await stack.enter_async_context(m) for m in mcps]

        mem_service_url = os.getenv("MEM_SERVICE_URL", "http://localhost:8765")
        mem_client: MemoryClient = await stack.enter_async_context(
            MemoryClient(mem_service_url)
        )

        ctx = MemoryContext(
            chat_id=chat_id,
            client=mem_client,
        )

        # ── Startup note (from memory service) ───────────────────────────────
        startup_note = await mem_client.get_note(chat_id)
        if startup_note:
            prompt = f"{prompt}\n\n[System Note from previous session]:\n{startup_note}"

        # ── Agent 2: hidden note-taker ──────────────────────────────────────
        # Receives handoff from agent1; calls leave_note_and_rate then exits.
        # Its output is suppressed in the event loop below — invisible to the user.
        note_taker = Agent(
            name="note_taker",
            model=model_name,
            model_settings=base_settings,
            instructions=(
                "You are a silent memory recorder. "
                "Call `leave_note_and_rate` EXACTLY ONCE with:\n"
                "  - note: a comprehensive summary of this session (state, decisions, next steps)\n"
                "  - rank: how useful the recalled memories were (-2 to 2)\n"
                "After the tool returns, reply with a single word: 'Done.' "
                "Do not call any other tool. Do not call `leave_note_and_rate` more than once."
            ),
            tools=[leave_note_and_rate],
        )

        # ── Agent 1: main agent ─────────────────────────────────────────────
        main_instructions = (
            (agent_cfg.get("instructions") or "") +
            "\n\n## TOOL PROTOCOL ADDENDUM\n"
            "**`search_in_memory` (STEP 1 — REQUIRED FIRST CALL)**\n"
            "- Derive three full-sentence queries from the user's prompt:\n"
            "  - `description`: precise technical phrasing of what is sought\n"
            "  - `summary`: colloquial/narrative restatement of the same topic\n"
            "  - `hypothesis`: a hypothetical excerpt (log line, code snippet, note) you expect to find\n"
            "- Do not use keywords. Write complete sentences.\n\n"
            "When your work is complete, hand off to `note_taker`."
        )
        main_agent_kwargs: dict = {
            "name": agent_cfg.get("name", "main_agent"),
            "model": model_name,
            "model_settings": base_settings,
            "instructions": main_instructions,
            "tools": [search_in_memory, read_tutorial],
            "handoffs": [note_taker],
        }
        if mcp_servers:
            main_agent_kwargs["mcp_servers"] = mcp_servers
        main_agent = Agent(**main_agent_kwargs)

        # ── Run ─────────────────────────────────────────────────────────────
        run_config = RunConfig()
        session = SQLiteSession("temp")

        try:
            result = Runner.run_streamed(
                main_agent,
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
                        # Suppress the handoff announcement for the hidden note_taker
                        if event.new_agent.name != "note_taker":
                            Render().output(f"Handoff to: {event.new_agent.name}")
                    case "run_item_stream_event":
                        # Only stream output from the main agent, not from note_taker
                        agent_name_cur = getattr(event, "agent", None)
                        if agent_name_cur is None or getattr(agent_name_cur, "name", "") != "note_taker":
                            ii = event.item.to_input_item()
                            Render().output(simple_handle_response_input_item_param(ii))
                    case _:
                        pass
        except agents.exceptions.MaxTurnsExceeded as e:
            print(e)

        # ── Save session via memory service ─────────────────────────────────
        items = await session.get_items()
        if items:
            await mem_client.save_sessions(chat_id=chat_id, items=items)

    Render().output("\n---\n" + result.final_output)


if __name__ == "__main__":
    import sys
    asyncio.run(main(sys.argv[1]))
