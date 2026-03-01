"""
Tests for sadk/server.py

Run:
    pytest test_server.py -v
"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from sadk.server import app


# ---------------------------------------------------------------------------
# Helpers — fake Memory and EmbedGear injected via module-level patches
# ---------------------------------------------------------------------------

def _make_memory(
    search_text: str = "=== Session s1 ===\n[user]: hello",
    recalled: list[str] | None = None,
    session_id: str = "new-session-id",
):
    mem = MagicMock()
    mem.search_and_format = AsyncMock(return_value=(search_text, recalled or ["s1"]))
    mem.save_items = AsyncMock(return_value=session_id)
    mem.embed_memories = AsyncMock()
    return mem


def _make_embeder():
    emb = MagicMock()
    emb.get_embedding = AsyncMock(return_value="deadbeef" * 8)
    return emb


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def client():
    """AsyncClient backed by the FastAPI ASGI app with mocked Memory/EmbedGear."""
    mem = _make_memory()
    emb = _make_embeder()

    # Patch at the module level so server.py uses our mocks
    with (
        patch("sadk.server._memory", mem, create=True),
        patch("sadk.server._embeder", emb, create=True),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            yield ac, mem, emb


# ---------------------------------------------------------------------------
# /search_in_memory
# ---------------------------------------------------------------------------

class TestSearchInMemory(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.mem = _make_memory()
        self.emb = _make_embeder()
        self._patches = [
            patch("sadk.server._memory", self.mem, create=True),
            patch("sadk.server._embeder", self.emb, create=True),
        ]
        for p in self._patches:
            p.start()
        self.client = AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        )
        await self.client.__aenter__()

    async def asyncTearDown(self):
        await self.client.__aexit__(None, None, None)
        for p in self._patches:
            p.stop()

    async def test_search_returns_text_and_sessions(self):
        resp = await self.client.post("/search_in_memory", json={
            "chat_id": "chat-abc",
            "description": "We discussed implementing a vector search feature for the memory system",
            "summary": "Vector search implementation discussion",
            "hypothesis": "The memory search uses hamming distance on binary embeddings",
        })
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("text", body)
        self.assertIn("recalled_sessions", body)
        self.assertIsInstance(body["recalled_sessions"], list)

    async def test_search_passes_chat_id(self):
        await self.client.post("/search_in_memory", json={
            "chat_id": "my-project",
            "description": "Looking for refactoring notes",
            "summary": "Refactoring discussion",
            "hypothesis": "We refactored the Memory class to use classmethods",
        })
        call_kwargs = self.mem.search_and_format.call_args.kwargs
        self.assertEqual(call_kwargs["chat_id"], "my-project")

    async def test_search_respects_custom_token_limit(self):
        await self.client.post("/search_in_memory", json={
            "chat_id": "c",
            "description": "d", "summary": "s", "hypothesis": "h",
            "token_limit": 1024,
        })
        call_kwargs = self.mem.search_and_format.call_args.kwargs
        self.assertEqual(call_kwargs["token_limit"], 1024)

    async def test_search_no_results(self):
        self.mem.search_and_format = AsyncMock(return_value=("No matching memory found.", []))
        resp = await self.client.post("/search_in_memory", json={
            "chat_id": "c",
            "description": "d", "summary": "s", "hypothesis": "h",
        })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["recalled_sessions"], [])

    async def test_search_propagates_error(self):
        self.mem.search_and_format = AsyncMock(side_effect=RuntimeError("db gone"))
        resp = await self.client.post("/search_in_memory", json={
            "chat_id": "c",
            "description": "d", "summary": "s", "hypothesis": "h",
        })
        self.assertEqual(resp.status_code, 500)
        self.assertIn("db gone", resp.json()["detail"])


# ---------------------------------------------------------------------------
# /save_sessions
# ---------------------------------------------------------------------------

_SAMPLE_ITEMS = [
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "hi"},
]


class TestSaveSessions(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.mem = _make_memory()
        self.emb = _make_embeder()
        self._patches = [
            patch("sadk.server._memory", self.mem, create=True),
            patch("sadk.server._embeder", self.emb, create=True),
        ]
        for p in self._patches:
            p.start()
        self.client = AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        )
        await self.client.__aenter__()

    async def asyncTearDown(self):
        await self.client.__aexit__(None, None, None)
        for p in self._patches:
            p.stop()

    async def test_save_returns_session_id(self):
        resp = await self.client.post("/save_sessions", json={
            "chat_id": "chat-abc",
            "items": _SAMPLE_ITEMS,
        })
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("session_id", body)
        self.assertEqual(body["session_id"], "new-session-id")

    async def test_save_passes_chat_id_and_items(self):
        await self.client.post("/save_sessions", json={
            "chat_id": "proj-x",
            "items": _SAMPLE_ITEMS,
        })
        self.mem.save_items.assert_awaited_once()
        args = self.mem.save_items.call_args
        self.assertEqual(args.args[1], "proj-x")   # chat_id positional arg

    async def test_save_triggers_embed(self):
        await self.client.post("/save_sessions", json={
            "chat_id": "c",
            "items": _SAMPLE_ITEMS,
        })
        self.mem.embed_memories.assert_awaited()

    async def test_save_empty_items_rejected(self):
        resp = await self.client.post("/save_sessions", json={
            "chat_id": "c",
            "items": [],
        })
        self.assertEqual(resp.status_code, 400)

    async def test_save_propagates_error(self):
        self.mem.save_items = AsyncMock(side_effect=RuntimeError("disk full"))
        resp = await self.client.post("/save_sessions", json={
            "chat_id": "c",
            "items": _SAMPLE_ITEMS,
        })
        self.assertEqual(resp.status_code, 500)
        self.assertIn("disk full", resp.json()["detail"])


if __name__ == "__main__":
    unittest.main()
