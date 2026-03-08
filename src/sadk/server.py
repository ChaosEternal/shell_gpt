"""
Memory REST server.

Endpoints:
    POST /search_in_memory    – vector-search past sessions by chat_id
    POST /save_sessions       – persist OpenAI-format items tagged with chat_id
    POST /save_sessions_from  – persist Google ADK events (auto-converted to OpenAI format)
    POST /leave_note_and_rate – save a session note and update recalled session ranks
    GET  /get_note            – retrieve the most recent note for a chat_id

Run:
    uvicorn sadk.server:app --host 0.0.0.0 --port 8765
"""

from __future__ import annotations

import json
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from openai import AsyncOpenAI

from sadk.memory import Memory, EmbedGear


# ---------------------------------------------------------------------------
# Shared state (initialised once at startup)
# ---------------------------------------------------------------------------

_memory: Memory
_embeder: EmbedGear


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _memory, _embeder

    embed_base = os.getenv("EMBED_BASE_URL", "http://127.0.0.1:11435/v1")
    embed_model = os.getenv("EMBED_MODEL", "nomic-embed-text:latest")

    _memory = Memory.create()
    embed_client = AsyncOpenAI(base_url=embed_base, api_key="no")
    _embeder = EmbedGear(embed_client, embed_model)

    await _memory.embed_memories(_embeder)

    yield


app = FastAPI(title="Memory API", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    chat_id: str
    description: str
    summary: str
    hypothesis: str
    token_limit: int = 96000


class SearchResponse(BaseModel):
    text: str
    recalled_sessions: list[str]


class SaveRequest(BaseModel):
    chat_id: str
    items: list[dict[str, Any]]


class SaveResponse(BaseModel):
    session_id: str


class LeaveNoteRequest(BaseModel):
    chat_id: str
    note: str
    rank: int
    recalled_sessions: list[str] = []


class StatusResponse(BaseModel):
    status: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/search_in_memory", response_model=SearchResponse)
async def search_in_memory(req: SearchRequest) -> SearchResponse:
    """Vector-search past sessions belonging to chat_id and return flattened text."""
    try:
        text, recalled = await _memory.search_and_format(
            _embeder,
            req.description,
            req.summary,
            req.hypothesis,
            chat_id=req.chat_id,
            token_limit=req.token_limit,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return SearchResponse(text=text, recalled_sessions=recalled)


@app.post("/save_sessions", response_model=SaveResponse)
async def save_sessions(req: SaveRequest) -> SaveResponse:
    """Persist conversation items and tag them with chat_id."""
    if not req.items:
        raise HTTPException(status_code=400, detail="items must not be empty")
    try:
        session_id = await _memory.save_items(req.items, req.chat_id)
        await _memory.embed_memories(_embeder)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return SaveResponse(session_id=session_id)


@app.post("/leave_note_and_rate", response_model=StatusResponse)
async def leave_note_and_rate(req: LeaveNoteRequest) -> StatusResponse:
    """Save a session note to chat_info and update ranks on recalled sessions."""
    try:
        await _memory.save_note(req.chat_id, req.note, req.rank, req.recalled_sessions)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return StatusResponse(status="ok")


class NoteResponse(BaseModel):
    note: str | None


@app.get("/get_note", response_model=NoteResponse)
async def get_note(chat_id: str) -> NoteResponse:
    """Return the most recent note saved for chat_id, or null if none exists."""
    try:
        note = _memory.get_note(chat_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return NoteResponse(note=note)


# ---------------------------------------------------------------------------
# Google ADK → OpenAI format converter
# ---------------------------------------------------------------------------

def adk_events_to_openai(events: list[dict]) -> list[dict]:
    """
    Convert a list of Google ADK session events to OpenAI chat message format.

    Google ADK event structure:
        {
          "author": "user" | "model" | "<tool_name>",
          "content": {
            "role": "user" | "model",
            "parts": [
              {"text": "..."},
              {"function_call": {"name": "...", "args": {...}}},
              {"function_response": {"name": "...", "response": {...}}}
            ]
          }
        }

    OpenAI output structure:
        {"role": "user",      "content": "..."}
        {"role": "assistant", "content": "..." | null, "tool_calls": [...]}
        {"role": "tool",      "tool_call_id": "...", "name": "...", "content": "..."}
    """
    messages: list[dict] = []
    # Track generated tool_call_ids so responses can reference them
    call_id_map: dict[str, str] = {}

    for event in events:
        content = event.get("content") or {}
        parts = content.get("parts") or []
        author = event.get("author", "")
        role = content.get("role", "user")

        text_parts = [p["text"] for p in parts if "text" in p]
        call_parts = [p["function_call"] for p in parts if "function_call" in p]
        resp_parts = [p["function_response"] for p in parts if "function_response" in p]

        # ── function_response → tool message ──────────────────────────────
        for resp in resp_parts:
            fn_name = resp.get("name", "")
            call_id = call_id_map.get(fn_name, f"call_{uuid.uuid4().hex[:8]}")
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": fn_name,
                "content": json.dumps(resp.get("response", {})),
            })

        # ── function_call → assistant message with tool_calls ─────────────
        if call_parts:
            tool_calls = []
            for call in call_parts:
                fn_name = call.get("name", "")
                call_id = f"call_{uuid.uuid4().hex[:8]}"
                call_id_map[fn_name] = call_id
                tool_calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": fn_name,
                        "arguments": json.dumps(call.get("args", {})),
                    },
                })
            messages.append({
                "role": "assistant",
                "content": " ".join(text_parts) if text_parts else None,
                "tool_calls": tool_calls,
            })
            continue  # text already merged above

        # ── plain text ─────────────────────────────────────────────────────
        if text_parts:
            oai_role = "user" if (author == "user" or role == "user") else "assistant"
            messages.append({
                "role": oai_role,
                "content": " ".join(text_parts),
            })

    return messages


class SaveFromRequest(BaseModel):
    chat_id: str
    events: list[dict[str, Any]]


@app.post("/save_sessions_from", response_model=SaveResponse)
async def save_sessions_from(req: SaveFromRequest) -> SaveResponse:
    """
    Accept Google ADK session events, convert them to OpenAI format,
    and persist them tagged with chat_id.
    """
    if not req.events:
        raise HTTPException(status_code=400, detail="events must not be empty")
    try:
        items = adk_events_to_openai(req.events)
        if not items:
            raise ValueError("No convertible messages found in events")
        session_id = await _memory.save_items(items, req.chat_id)
        await _memory.embed_memories(_embeder)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return SaveResponse(session_id=session_id)


