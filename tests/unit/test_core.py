import asyncio
import json
import tempfile
import time
import unittest
from contextlib import aclosing
from pathlib import Path

from pydantic import ValidationError

from heterokv.artifacts import read_outcomes, validate_run
from heterokv.contracts import ModelGeometry, Outcome, Request, Status
from heterokv.controller import Rejected
from heterokv.kv_manager import Reservations
from heterokv.metrics import paired_interval, request_metrics, summarize
from heterokv.predictor import Profile
from heterokv.runtime import benchmark, build_controller, load_config
from heterokv.workloads import generate, prepare_dataset, read_trace, write_trace


def config():
    return load_config(Path("configs/experiments/local.yaml"))


def request(name="r", **kwargs):
    return Request(**({"request_id": name, "arrival_offset": 0, "prompt_tokens": 16,
                       "output_cap": 4} | kwargs))


class MetricsTests(unittest.TestCase):
    def outcome(self, **changes):
        values = dict(run_id="test", request=request(), status=Status.COMPLETE,
                      scheduled_arrival=0, dispatch_time=.1, first_token_time=.2,
                      last_token_time=.26, stream_end_time=.3, actual_output_tokens=4)
        return Outcome(**(values | changes))

    def test_arithmetic(self):
        result = request_metrics(self.outcome())
        self.assertAlmostEqual(result["ttft_ms"], 200)
        self.assertAlmostEqual(result["tpot_ms"], 20)
        self.assertEqual(result["dispatch_delay_ms"], 100)
        self.assertTrue(result["compliant"])

    def test_one_token(self):
        result = request_metrics(self.outcome(actual_output_tokens=1, last_token_time=.2))
        self.assertIsNone(result["tpot_ms"])
        self.assertTrue(result["compliant"])

    def test_multiple_tokens_in_one_chunk_has_unknown_tpot(self):
        result = request_metrics(self.outcome(actual_output_tokens=4, last_token_time=.2))
        self.assertIsNone(result["tpot_ms"])
        self.assertFalse(result["compliant"])

    def test_empty_complete_is_not_compliant(self):
        self.assertFalse(request_metrics(self.outcome(actual_output_tokens=0))["compliant"])

    def test_failures_remain_in_denominator(self):
        outcomes = [self.outcome()]
        for index, status in enumerate((Status.REJECTED, Status.FAILED, Status.TIMED_OUT,
                                        Status.CANCELLED, Status.INCOMPLETE)):
            outcomes.append(self.outcome(request=request(str(index)), status=status))
        result = summarize(outcomes, 2, 2)
        self.assertEqual(result["offered"], 6)
        self.assertEqual(result["offered_slo_attainment"], 1 / 6)
        self.assertEqual(result["token_goodput_per_gpu"], 1)

    def test_mock_no_fake_gpu_denominator(self):
        self.assertIsNone(summarize([self.outcome()], 2, 0)["token_goodput_per_gpu"])

    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            summarize([self.outcome(), self.outcome()], 2, 1)

    def test_outside_cohort(self):
        with self.assertRaises(ValueError):
            summarize([self.outcome(scheduled_arrival=.05)], .01, 1)

    def test_nonmonotonic_rejected(self):
        with self.assertRaises(ValidationError):
            self.outcome(last_token_time=.1)

    def test_bootstrap(self):
        interval = paired_interval([2, 4, 6, 8, 10], [1, 2, 3, 4, 5])
        self.assertAlmostEqual(interval["lower95"], 2)
        with self.assertRaises(ValueError):
            paired_interval([1, 2], [0, 1])


class ContractTests(unittest.TestCase):
    def test_no_future_feature(self):
        with self.assertRaises(ValidationError):
            request(actual_output_tokens=1)

    def test_invalid_numbers(self):
        for field, value in [("prompt_tokens", 0), ("timeout_s", -1),
                             ("arrival_offset", float("nan"))]:
            with self.assertRaises(ValidationError):
                request(**{field: value})

    def test_geometry_block_rounding_gqa(self):
        geometry = ModelGeometry(layers=4, kv_heads=2, head_dim=8, max_context=128)
        self.assertEqual(geometry.kv_bytes(17), 2 * 4 * 2 * 8 * 32 * 2)

    def test_deterministic_workloads(self):
        for kind in (f"W{i}" for i in range(6)):
            c = config() | {"workload": kind, "duration_s": 200}
            self.assertEqual(generate(c, 101), generate(c, 101))
            self.assertNotEqual(generate(c, 101), generate(c, 102))

    def test_trace_roundtrip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            records = generate(config(), 101)
            write_trace(path, records)
            self.assertEqual(read_trace(path), records)
            with self.assertRaises(FileExistsError):
                write_trace(path, records)

    def test_profile_coverage(self):
        p = Profile("a", (1, 100), (1., 100.), (1, 100), (1., 2.), .1)
        self.assertAlmostEqual(p.predict(50, 50, 10, 0)[0], 50)
        with self.assertRaises(ValueError):
            p.predict(101, 50, 0, 0)


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.c = build_controller(config())

    def candidates(self, req=None, pd=True):
        return self.c.scheduler.candidates(req or request(), self.c.workers,
                                            {w.worker_id: 4096 for w in self.c.workers},
                                            time.monotonic(), pd)

    def test_real_adapter_capability_filters_pd(self):
        self.assertTrue(all(p.mode == "colocated" for p in self.candidates(pd=False)))

    def test_stale_telemetry(self):
        for w in self.c.workers:
            w.telemetry_time = 0
        self.assertEqual(self.candidates(), [])

    def test_fixed_roles(self):
        self.c.workers[0].roles = {"prefill"}
        self.c.workers[1].roles = {"decode"}
        plans = self.candidates()
        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0].mode, "pd")

    def test_compatibility(self):
        self.c.workers[1].cache_signature = "different"
        self.assertTrue(all(p.mode == "colocated" for p in self.candidates()))

    def test_context_limit(self):
        self.assertEqual(self.candidates(request(prompt_tokens=32768)), [])

    def test_missing_profile_rejects(self):
        self.c.scheduler.profiles.clear()
        self.assertEqual(self.candidates(), [])

    def test_full_cap_reserved_even_with_short_prediction(self):
        self.c.scheduler.output_estimates = {"chat": 1}
        for plan in self.candidates(request(output_cap=1024)):
            self.assertEqual(plan.reserved_blocks[plan.decode_worker], 65)

    def test_transfer_guard_prefers_local(self):
        selected = self.c.scheduler.choose(self.candidates(), self.c.workers)
        self.assertEqual(selected.mode, "colocated")

    def test_prefill_first_token_excludes_decode_queue_from_ttft(self):
        from dataclasses import replace
        self.c.workers[0].roles = {"prefill"}
        self.c.workers[1].roles = {"decode"}
        self.c.workers[1].queued_tokens = 1000
        pair = ("mock-a", "mock-b")
        self.c.scheduler.links[pair] = replace(self.c.scheduler.links[pair], first_token="prefill")
        plan = self.candidates()[0]
        self.assertEqual(plan.ttft_ms, 5)
        self.assertGreater(plan.tpot_ms, 1)


class AsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_atomic_reservations(self):
        r = Reservations({"a": 10, "b": 10})
        results = await asyncio.gather(*(r.reserve(str(i), {"a": 6, "b": 6}) for i in range(50)))
        self.assertEqual(sum(results), 1)
        self.assertEqual(r.available("a"), 4)
        winner = next(iter(r.owners))
        self.assertTrue(await r.release(winner))
        self.assertFalse(await r.release(winner))

    async def test_completion_cleanup(self):
        c = build_controller(config())
        tokens = [e async for e in c.stream(request())]
        self.assertTrue(tokens[-1].done)
        self.assertEqual(c.states["r"], Status.COMPLETE)
        self.assertFalse(c.reservations.owners)

    async def test_failure_cleanup(self):
        c = build_controller(config() | {"mock": {"fail_at": 1}})
        with self.assertRaises(RuntimeError):
            _ = [e async for e in c.stream(request())]
        self.assertEqual(c.states["r"], Status.FAILED)
        self.assertFalse(c.reservations.owners)

    async def test_pd_failure_releases_both_workers(self):
        c = build_controller(config() | {"mock": {"fail_at": 0}})
        c.workers[0].roles = {"prefill"}
        c.workers[1].roles = {"decode"}
        plan = await c.admit(request())
        self.assertEqual(len(c.reservations.owners["r"]), 2)
        with self.assertRaises(RuntimeError):
            _ = [e async for e in c.stream(request(), admitted=plan)]
        self.assertFalse(c.reservations.owners)

    async def test_backpressure_does_not_eagerly_consume(self):
        c = build_controller(config())
        emitted = []
        original = c.adapter.stream

        async def tracked(*args):
            async for event in original(*args):
                emitted.append(event)
                yield event

        c.adapter.stream = tracked
        async with aclosing(c.stream(request())) as stream:
            await anext(stream)
            await asyncio.sleep(.01)
            self.assertEqual(len(emitted), 1)
        self.assertFalse(c.reservations.owners)

    async def test_cancel_before_first_token(self):
        c = build_controller(config() | {"mock": {"prefill_s": 10}})

        async def consume():
            return [e async for e in c.stream(request())]

        task = asyncio.create_task(consume())
        await asyncio.sleep(.01)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(c.reservations.owners)

    async def test_timeout_cleanup(self):
        c = build_controller(config())
        with self.assertRaises(TimeoutError):
            _ = [e async for e in c.stream(request(timeout_s=.0001))]
        self.assertEqual(c.states["r"], Status.TIMED_OUT)
        self.assertFalse(c.reservations.owners)

    async def test_disconnect_cleanup(self):
        c = build_controller(config())
        async with aclosing(c.stream(request())) as stream:
            await anext(stream)
        self.assertEqual(c.states["r"], Status.CANCELLED)
        self.assertFalse(c.reservations.owners)

    async def test_duplicate_terminal(self):
        c = build_controller(config())
        _ = [e async for e in c.stream(request())]
        with self.assertRaises(Rejected):
            await c.admit(request())
        self.assertEqual(c.states["r"], Status.COMPLETE)

    async def test_bounded_admission(self):
        c = build_controller(config() | {"max_inflight": 1})
        await c.admit(request("one"))
        with self.assertRaises(Rejected):
            await c.admit(request("two"))
        await c.finish(request("one"), Status.CANCELLED)
        self.assertFalse(c.reservations.owners)

    async def test_run_reconciliation_and_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = config() | {"duration_s": .15, "rate": 100, "runs_dir": tmp + "/runs",
                            "ledger": tmp + "/ledger.jsonl"}
            path, summary = await benchmark(c)
            self.assertEqual(summary["validation_status"], "COMPLETED")
            self.assertTrue(validate_run(path)["valid"], validate_run(path))
            self.assertEqual(len(read_outcomes(path)), len(generate(c, 101)))
            (path / "stdout.log").write_text("tampered")
            self.assertFalse(validate_run(path)["valid"])

    async def test_bounded_drain_keeps_cancelled_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = config() | {"duration_s": .05, "rate": 100, "drain_s": 0,
                            "runs_dir": tmp + "/runs", "ledger": tmp + "/ledger.jsonl",
                            "mock": {"prefill_s": 1}}
            _, summary = await benchmark(c)
            self.assertEqual(summary["offered"], len(generate(c, 101)))
            self.assertGreater(summary["counts"].get("CANCELLED", 0), 0)


class DatasetTests(unittest.TestCase):
    def test_eligibility_and_future_output_hidden(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.csv"
            source.write_text("t,p,o\n0,16,10\n1,16,0\n2,10000,10\n")
            cfg = {"source": str(source), "out": tmp + "/out", "dataset_id": "fixture",
                   "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                   "mode": "natural", "output_cap": 128, "timestamp_seconds_per_unit": 1,
                   "speedup": 2, "max_context": 1024, "ttft_slo_ms": 500,
                   "tpot_slo_ms": 50, "seed": 101,
                   "windows": [{"start": 0, "end": 10, "id": "dev", "split": "dev"}],
                   "columns": {"timestamp": "t", "prompt": "p", "output": "o"}}
            result = prepare_dataset(cfg)
            self.assertEqual(result["excluded"], 2)
            requests = read_trace(Path(tmp) / "out/dev.jsonl")
            self.assertEqual(requests[0].output_cap, 128)
            self.assertNotIn("prescribed_output_tokens", requests[0].model_dump())
            normalized = json.loads((Path(tmp) / "out/normalized.json").read_text())
            self.assertEqual(normalized[0]["prescribed_output_tokens"], 10)
