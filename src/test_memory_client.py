"""
Tests for sadk/memory_client.py

Run:
    .venv/bin/pytest test_memory_client.py -v
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from sadk.memory_client import MemoryClient, SearchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(data: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = data
    resp.text = json.dumps(data)
    if status_code >= 400:
        from httpx import HTTPStatusError, Request, Response
        resp.raise_for_status.side_effect = HTTPStatusError(
            "error", request=MagicMock(), response=MagicMock(status_code=status_code)
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


# ---------------------------------------------------------------------------
# Tests for search_in_memory
# ---------------------------------------------------------------------------

class TestClientSearchInMemory(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.client = MemoryClient("http://localhost:8765")
        self.mock_post = AsyncMock()
        self.client._client = MagicMock()
        self.client._client.post = self.mock_post

    async def test_returns_search_result(self):
        self.mock_post.return_value = _make_response({
            "text": "=== Session s1 ===\n[user]: hello",
            "recalled_sessions": ["s1"],
        })
        result = await self.client.search_in_memory(
            chat_id="proj",
            description="desc",
            summary="summ",
            hypothesis="hypo",
        )
        self.assertIsInstance(result, SearchResult)
        self.assertEqual(result.text, "=== Session s1 ===\n[user]: hello")
        self.assertEqual(result.recalled_sessions, ["s1"])

    async def test_posts_to_correct_endpoint(self):
        self.mock_post.return_value = _make_response(
            {"text": "No matching memory found.", "recalled_sessions": []}
        )
        await self.client.search_in_memory(
            chat_id="x", description="d", summary="s", hypothesis="h"
        )
        self.mock_post.assert_awaited_once()
        call_args = self.mock_post.call_args
        self.assertEqual(call_args.args[0], "/search_in_memory")

    async def test_sends_all_fields(self):
        self.mock_post.return_value = _make_response(
            {"text": "t", "recalled_sessions": []}
        )
        await self.client.search_in_memory(
            chat_id="chat-1",
            description="desc",
            summary="summ",
            hypothesis="hypo",
            token_limit=4096,
        )
        payload = self.mock_post.call_args.kwargs["json"]
        self.assertEqual(payload["chat_id"], "chat-1")
        self.assertEqual(payload["description"], "desc")
        self.assertEqual(payload["summary"], "summ")
        self.assertEqual(payload["hypothesis"], "hypo")
        self.assertEqual(payload["token_limit"], 4096)

    async def test_no_results_still_ok(self):
        self.mock_post.return_value = _make_response(
            {"text": "No matching memory found.", "recalled_sessions": []}
        )
        result = await self.client.search_in_memory(
            chat_id="c", description="d", summary="s", hypothesis="h"
        )
        self.assertEqual(result.recalled_sessions, [])

    async def test_raises_on_server_error(self):
        self.mock_post.return_value = _make_response(
            {"detail": "db gone"}, status_code=500
        )
        with self.assertRaises(RuntimeError) as ctx:
            await self.client.search_in_memory(
                chat_id="c", description="d", summary="s", hypothesis="h"
            )
        self.assertIn("db gone", str(ctx.exception))


# ---------------------------------------------------------------------------
# Tests for save_sessions
# ---------------------------------------------------------------------------

_ITEMS = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]


class TestClientSaveSessions(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.client = MemoryClient("http://localhost:9999")
        self.mock_post = AsyncMock()
        self.client._client = MagicMock()
        self.client._client.post = self.mock_post

    async def test_returns_session_id(self):
        self.mock_post.return_value = _make_response(
            {"session_id": "abc-123"}
        )
        sid = await self.client.save_sessions(chat_id="proj", items=_ITEMS)
        self.assertEqual(sid, "abc-123")

    async def test_posts_to_correct_endpoint(self):
        self.mock_post.return_value = _make_response({"session_id": "x"})
        await self.client.save_sessions(chat_id="c", items=_ITEMS)
        self.assertEqual(self.mock_post.call_args.args[0], "/save_sessions")

    async def test_sends_chat_id_and_items(self):
        self.mock_post.return_value = _make_response({"session_id": "y"})
        await self.client.save_sessions(chat_id="my-chat", items=_ITEMS)
        payload = self.mock_post.call_args.kwargs["json"]
        self.assertEqual(payload["chat_id"], "my-chat")
        self.assertEqual(payload["items"], _ITEMS)

    async def test_raises_on_server_error(self):
        self.mock_post.return_value = _make_response(
            {"detail": "disk full"}, status_code=500
        )
        with self.assertRaises(RuntimeError) as ctx:
            await self.client.save_sessions(chat_id="c", items=_ITEMS)
        self.assertIn("disk full", str(ctx.exception))


# ---------------------------------------------------------------------------
# Context manager lifecycle
# ---------------------------------------------------------------------------

class TestClientLifecycle(unittest.IsolatedAsyncioTestCase):

    async def test_raises_outside_context(self):
        client = MemoryClient()
        with self.assertRaises(RuntimeError):
            await client.search_in_memory(
                chat_id="c", description="d", summary="s", hypothesis="h"
            )

    async def test_context_manager_opens_and_closes(self):
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            MockClient.return_value = instance
            async with MemoryClient("http://x") as client:
                self.assertIsNotNone(client._client)
            instance.aclose.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
