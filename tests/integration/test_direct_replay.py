import asyncio
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from heterokv.artifacts import read_outcomes, validate_run
from heterokv.contracts import TokenEvent
from heterokv.runtime import benchmark


class FakeTokenizer:
    all_special_ids = []

    def get_vocab(self):
        return {str(i): i for i in range(100)}


class FakeAutoTokenizer:
    @staticmethod
    def from_pretrained(model, revision):
        assert len(revision) == 40
        return FakeTokenizer()


class FakeVLLMAdapter:
    evidence_kind = "real-llm-unvalidated"

    def __init__(self, endpoints, model, controlled_length):
        self.endpoints = endpoints

    async def stream(self, request, plan, prompt):
        assert len(prompt) == request.prompt_tokens
        for _ in range(request.output_cap):
            await asyncio.sleep(.001)
            yield TokenEvent(text="x")
        yield TokenEvent(done=True, completion_tokens=request.output_cap)


class DirectReplayTests(unittest.IsolatedAsyncioTestCase):
    async def test_replay_without_scheduler_profiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {"backend": "vllm-direct", "policy": "direct",
                      "model": "fixture", "model_revision": "a" * 40,
                      "tokenizer_revision": "a" * 40, "endpoint": "http://fixture",
                      "duration_s": .1, "drain_s": .1, "rate": 50, "workload": "W0",
                      "seed": 101, "tiny_mock": True, "allocated_gpus": 1,
                      "gpu_inventory": ["test-gpu-not-real"],
                      "runs_dir": str(Path(tmp) / "runs"),
                      "ledger": str(Path(tmp) / "ledger.jsonl")}
            module = types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer)
            with patch.dict(sys.modules, {"transformers": module}), \
                 patch.dict("os.environ", {"SLURM_JOB_ID": "fixture"}), \
                 patch("heterokv.runtime.platform.system", return_value="Linux"), \
                 patch("heterokv.runtime.VLLMAdapter", FakeVLLMAdapter):
                path, summary = await benchmark(config)
            self.assertEqual(summary["validation_status"], "COMPLETED")
            self.assertEqual(summary["counts"]["COMPLETE"], summary["offered"])
            self.assertTrue(validate_run(path)["valid"])
            self.assertTrue(all(o.timing_kind == "chunk-observed" for o in read_outcomes(path)))
