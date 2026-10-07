import unittest
from unittest.mock import patch
import asyncio
import tempfile

import httpx
from fastapi.testclient import TestClient

from heterokv.adapters import VLLMAdapter, sse_data
from heterokv.api import create_app
from heterokv.runtime import build_controller
from heterokv.runtime import benchmark
from tests.unit.test_core import config, request


class APITests(unittest.TestCase):
    def test_latest_saved_result_is_labeled(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = config() | {"duration_s": .1, "rate": 50,
                                   "runs_dir": tmp + "/runs", "ledger": tmp + "/ledger.jsonl"}
            path, _ = asyncio.run(benchmark(settings))
            with TestClient(create_app(settings)) as client:
                response = client.get("/results/latest")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["run_id"], path.name)
                self.assertEqual(response.json()["evidence_kind"], "mock")
                self.assertEqual(client.get("/metrics").status_code, 200)

    def test_browser_entrypoint(self):
        with TestClient(create_app(config())) as client:
            response = client.get("/", follow_redirects=False)
            self.assertEqual(response.status_code, 307)
            self.assertEqual(response.headers["location"], "/docs")
            self.assertEqual(client.get("/favicon.ico").status_code, 204)

    def test_stream_and_duplicate(self):
        app = create_app(config())
        with TestClient(app) as client:
            response = client.post("/generate", json={"request": request().model_dump()})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.text.count('"done":true'), 1)
            self.assertFalse(app.state.controller.reservations.owners)
            duplicate = client.post("/generate", json={"request": request().model_dump()})
            self.assertEqual(duplicate.status_code, 429)
            self.assertIn("heterokv_terminal_total", client.get("/metrics").text)

    def test_validation(self):
        with TestClient(create_app(config())) as client:
            response = client.post("/generate", json={"request": {"request_id": "bad"}})
            self.assertEqual(response.status_code, 422)


class SSETests(unittest.IsolatedAsyncioTestCase):
    async def test_multiline_comments(self):
        response = httpx.Response(200, content=b': heartbeat\ndata: {"value":\ndata: 2}\n\ndata: [DONE]\n\n')
        self.assertEqual([data async for data in sse_data(response)], ['{"value":\n2}', '[DONE]'])

    async def test_vllm_usage_and_chunk_label(self):
        def respond(req):
            return httpx.Response(200, text='data: {"choices":[{"text":"ab"}]}\n\n'
                                  'data: {"choices":[],"usage":{"completion_tokens":4}}\n\n'
                                  'data: [DONE]\n\n')
        client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        adapter = VLLMAdapter({"mock-a": "http://fixture", "mock-b": "http://fixture"}, "fixture")
        c = build_controller(config())
        plan = await c.admit(request())
        with patch("heterokv.adapters.httpx.AsyncClient", return_value=client):
            events = [event async for event in adapter.stream(request(), plan, [10] * 16)]
        self.assertFalse(events[0].exact_token_events)
        self.assertEqual(events[-1].completion_tokens, 4)
        self.assertTrue(events[-1].done)

    async def test_truncated_vllm_stream_fails(self):
        client = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(200, text='data: {"choices":[{"text":"a"}]}\n\n')))
        adapter = VLLMAdapter({"mock-a": "http://fixture", "mock-b": "http://fixture"}, "fixture")
        c = build_controller(config())
        plan = await c.admit(request())
        with patch("heterokv.adapters.httpx.AsyncClient", return_value=client):
            with self.assertRaises(RuntimeError):
                _ = [event async for event in adapter.stream(request(), plan, [10] * 16)]
