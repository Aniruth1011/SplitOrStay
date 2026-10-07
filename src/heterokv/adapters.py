"""Serving boundaries. The mock exercises orchestration, never cache correctness."""

import asyncio
import json
from collections.abc import AsyncIterator

import httpx

from heterokv.contracts import Plan, Request, TokenEvent


class MockAdapter:
    evidence_kind = "mock"
    supports_pd = True

    def __init__(self, prefill_s=0.005, token_s=0.001, transfer_s=0.002,
                 fail_at: int | None = None):
        self.prefill_s, self.token_s, self.transfer_s = prefill_s, token_s, transfer_s
        self.fail_at = fail_at

    async def stream(self, request: Request, plan: Plan, prompt=None) -> AsyncIterator[TokenEvent]:
        await asyncio.sleep(self.prefill_s)
        if plan.mode == "pd":
            await asyncio.sleep(self.transfer_s)
        for index in range(request.output_cap):
            if self.fail_at == index:
                raise RuntimeError("injected mock failure")
            if index:
                await asyncio.sleep(self.token_s)
            yield TokenEvent(text=str(index), token_count=1, exact_token_events=True)
        yield TokenEvent(done=True, completion_tokens=request.output_cap)


async def sse_data(response: httpx.Response):
    """Parse multiline SSE, including comments and events split across network reads."""
    data = []
    async for line in response.aiter_lines():
        if not line:
            if data:
                yield "\n".join(data)
                data = []
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    if data:
        yield "\n".join(data)


class VLLMAdapter:
    evidence_kind = "real-llm-unvalidated"
    supports_pd = False

    def __init__(self, endpoints: dict[str, str], model: str, controlled_length=True):
        self.endpoints, self.model = endpoints, model
        self.controlled_length = controlled_length

    async def stream(self, request: Request, plan: Plan, prompt=None):
        if plan.mode != "colocated":
            raise RuntimeError("BLOCKED: NIXL handoff adapter requires release-specific validation")
        if not isinstance(prompt, list) or len(prompt) != request.prompt_tokens:
            raise ValueError("vLLM replay requires exact tokenizer-validated prompt token IDs")
        body = {"model": self.model, "prompt": prompt, "max_tokens": request.output_cap,
                "temperature": 0, "stream": True,
                "stream_options": {"include_usage": True},
                "ignore_eos": self.controlled_length, "seed": request.seed}
        usage = None
        finished = False
        async with httpx.AsyncClient(timeout=request.timeout_s) as client:
            async with client.stream("POST", self.endpoints[plan.decode_worker].rstrip("/")
                                     + "/v1/completions", json=body,
                                     headers={"X-Request-Id": request.request_id}) as response:
                response.raise_for_status()
                async for data in sse_data(response):
                    if data == "[DONE]":
                        finished = True
                        break
                    item = json.loads(data)
                    if item.get("error"):
                        raise RuntimeError("engine returned a streaming error")
                    if item.get("usage"):
                        usage = int(item["usage"]["completion_tokens"])
                    for choice in item.get("choices", []):
                        if choice.get("text"):
                            yield TokenEvent(text=choice["text"])
        if not finished or usage is None:
            raise RuntimeError("incomplete stream or missing authoritative token usage")
        if self.controlled_length and usage != request.output_cap:
            raise RuntimeError("controlled output length mismatch")
        yield TokenEvent(done=True, completion_tokens=usage)

