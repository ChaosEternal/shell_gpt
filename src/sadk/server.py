"""
Memory REST server.

Endpoints:
    POST /search_in_memory    – vector-search past sessions by chat_id
    POST /save_sessions       – persist conversation items tagged with chat_id
    POST /leave_note_and_rate – save a session note and update recalled session ranks

Run:
    uvicorn sadk.server:app --host 0.0.0.0 --port 8765
"""

from __future__ import annotations

import os
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

