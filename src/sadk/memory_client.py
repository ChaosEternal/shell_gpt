"""
Client for the Memory REST server (sadk/server.py).

Usage:
    async with MemoryClient("http://localhost:8765") as client:
        result = await client.search_in_memory(
            chat_id="my-project",
            description="...",
            summary="...",
            hypothesis="...",
        )
        await client.save_sessions(chat_id="my-project", items=[...])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class SearchResult:
    text: str
    recalled_sessions: list[str] = field(default_factory=list)


class MemoryClient:
    """Async HTTP client for the Memory REST API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8765",
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "MemoryClient":
        self._client = httpx.AsyncClient(
            base_url=self._base_url, timeout=self._timeout
        )
        return self

    async def __aexit__(self, *_) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @property
    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError(
                "MemoryClient must be used as an async context manager."
            )
        return self._client

    def _raise_for(self, resp: httpx.Response) -> None:
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = resp.json().get("detail", resp.text)
            raise RuntimeError(f"Memory server error {resp.status_code}: {detail}") from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def search_in_memory(
        self,
        chat_id: str,
        description: str,
        summary: str,
        hypothesis: str,
        token_limit: int = 96000,
    ) -> SearchResult:
        """Search past sessions by chat_id using semantic similarity."""
        resp = await self._http.post(
            "/search_in_memory",
            json={
                "chat_id": chat_id,
                "description": description,
                "summary": summary,
                "hypothesis": hypothesis,
                "token_limit": token_limit,
            },
        )
        self._raise_for(resp)
        data = resp.json()
        return SearchResult(
            text=data["text"],
            recalled_sessions=data.get("recalled_sessions", []),
        )

    async def save_sessions(
        self,
        chat_id: str,
        items: list[dict[str, Any]],
    ) -> str:
        """Persist conversation items tagged with chat_id. Returns the new session_id."""
        resp = await self._http.post(
            "/save_sessions",
            json={"chat_id": chat_id, "items": items},
        )
        self._raise_for(resp)
        return resp.json()["session_id"]

    async def leave_note_and_rate(
        self,
        chat_id: str,
        note: str,
        rank: int,
        recalled_sessions: list[str] | None = None,
    ) -> None:
        """Save a session note to chat_info and update ranks on recalled sessions."""
        resp = await self._http.post(
            "/leave_note_and_rate",
            json={
                "chat_id": chat_id,
                "note": note,
                "rank": rank,
                "recalled_sessions": recalled_sessions or [],
            },
        )
        self._raise_for(resp)

    async def get_note(self, chat_id: str) -> str | None:
        """Return the most recent note for chat_id, or None if no note exists."""
        resp = await self._http.get("/get_note", params={"chat_id": chat_id})
        self._raise_for(resp)
        return resp.json().get("note")

    async def save_sessions_from(
        self,
        chat_id: str,
        events: list[dict[str, Any]],
    ) -> str:
        """
        Convert Google ADK session events to OpenAI format and persist them.

        Each event should follow the Google ADK structure::

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

        Returns the new session_id.
        """
        resp = await self._http.post(
            "/save_sessions_from",
            json={"chat_id": chat_id, "events": events},
        )
        self._raise_for(resp)
        return resp.json()["session_id"]

