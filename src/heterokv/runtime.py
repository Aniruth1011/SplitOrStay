import asyncio
import hashlib
import json
import os
import platform
import time
import traceback
from contextlib import aclosing
from pathlib import Path

import yaml

from heterokv.adapters import MockAdapter, VLLMAdapter
from heterokv.artifacts import RunBundle
from heterokv.contracts import ModelGeometry, Outcome, Plan, Status, Worker
from heterokv.controller import Controller, Rejected
from heterokv.kv_manager import Reservations
from heterokv.predictor import Link, Profile
from heterokv.scheduler import Scheduler
from heterokv.workloads import generate, prompt_ids, read_trace


def load_config(path):
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("configuration must be a mapping")
    return value


def require_gpu_allocation():
    if platform.system() != "Linux" or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("real vLLM work requires a Linux compute node inside a Slurm allocation")


def build_controller(config, event_sink=None):
    backend = config.get("backend", "mock")
    workers = [Worker(**(w | {"telemetry_time": time.monotonic()})) for w in config["workers"]]
    profiles = {p["worker_id"]: Profile(**p) for p in config.get("profiles", [])}
    links = {(p["source"], p["destination"]): Link(**p["profile"])
             for p in config.get("links", [])}
    if backend == "mock":
        if config.get("allocated_gpus", 0) != 0:
            raise ValueError("mock runs must report zero physically allocated GPUs")
        adapter = MockAdapter(**config.get("mock", {}))
    elif backend == "vllm":
        require_gpu_allocation()
        if not config.get("model_revision") or not config.get("tokenizer_revision"):
            raise ValueError("pin model and tokenizer revisions")
        if config.get("profile_evidence_kind") != "real-llm" or not config.get("profile_run_ids"):
            raise ValueError("real routing requires measured profiles linked to real-LLM run IDs")
        if any(w.gpu_type == "mock" or w.gpu_uuid.startswith("none") for w in workers):
            raise ValueError("mock worker identity cannot be used for real serving")
        # Configured snapshots expire; only an external telemetry feed may refresh real workers.
        for worker in workers:
            worker.telemetry_time = 0
        adapter = VLLMAdapter({w.worker_id: w.endpoint for w in workers}, config["model"],
                              config.get("mode", "controlled") == "controlled")
    else:
        raise ValueError(f"unknown backend: {backend}")
    scheduler = Scheduler(ModelGeometry(**config["geometry"]), profiles, links,
                          policy=config.get("policy", "selective"),
                          **config.get("scheduler", {}))
    return Controller(workers, scheduler, adapter, config.get("max_inflight", 128), event_sink)


class DirectVLLMController:
    """Pilot-only single endpoint replay before phase profiles exist.

    Engine admission and memory management remain inside vLLM. These runs cannot
    certify controller memory safety or serve as a tuned equal-admission baseline.
    """

    def __init__(self, config):
        require_gpu_allocation()
        if config.get("policy") != "direct" or not config.get("model_revision") \
                or config.get("tokenizer_revision") != config.get("model_revision"):
            raise ValueError("direct pilot requires policy direct and matching pinned revisions")
        self.adapter = VLLMAdapter({"direct": config["endpoint"]}, config["model"],
                                   config.get("mode", "controlled") == "controlled")
        self.reservations = Reservations({})
        self.event_sink = lambda *args: None

    async def admit(self, request):
        plan = Plan(mode="colocated", prefill_worker="direct", decode_worker="direct",
                    queue_ms=0, prefill_ms=0, transfer_ms=0, tpot_ms=0, ttft_ms=0,
                    estimated_output_tokens=request.output_cap, reserved_blocks={}, risk=0,
                    confidence="no-prediction", reason="direct vLLM pilot; predictions unavailable")
        self.event_sink("selected_plan", request.request_id, plan.model_dump())
        return plan

    async def stream(self, request, prompt=None, admitted=None):
        async for event in self.adapter.stream(request, admitted, prompt):
            yield event


async def refresh_workers(controller, config):
    """Mock heartbeats are labeled; real snapshots are local monotonic, same-host only."""
    while True:
        if config.get("backend", "mock") == "mock":
            for worker in controller.workers:
                worker.telemetry_time = time.monotonic()
        elif config.get("telemetry_file"):
            try:
                data = json.loads(Path(config["telemetry_file"]).read_text())
                if data["hostname"] != platform.node():
                    raise ValueError("cross-host monotonic timestamps are not comparable")
                samples = {w["worker_id"]: w for w in data["workers"]}
                for worker in controller.workers:
                    sample = samples[worker.worker_id]
                    worker.telemetry_time = sample["monotonic_time"]
                    worker.healthy = sample["healthy"]
            except (OSError, ValueError, KeyError):
                pass  # Existing snapshot expires; never invent a heartbeat.
        await asyncio.sleep(0.2)


async def benchmark(config):
    requests = read_trace(Path(config["trace"])) if config.get("trace") else generate(
        config, config.get("seed", 101))
    if any(r.arrival_offset >= config["duration_s"] for r in requests):
        raise ValueError("trace contains arrivals outside observation window")
    direct = config.get("backend") == "vllm-direct"
    controller = DirectVLLMController(config) if direct else build_controller(config)
    bundle = RunBundle(Path(config.get("runs_dir", "results/runs")), config, requests,
                       controller.adapter.evidence_kind,
                       Path(config.get("ledger", "experiments/ledger.jsonl")))
    controller.event_sink = bundle.event
    bundle.state("RUNNING")
    outcomes = {r.request_id: Outcome(run_id=bundle.run_id, request=r,
                                    scheduled_arrival=r.arrival_offset,
                                    policy=config.get("policy", "selective")) for r in requests}
    tasks = set()
    heartbeat = None
    state, reason = "COMPLETED", None
    try:
        prompts = {}
        if config.get("backend") in {"vllm", "vllm-direct"}:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(config["model"],
                                                       revision=config["tokenizer_revision"])
            for request in requests:
                seed = int(hashlib.sha256(request.request_id.encode()).hexdigest()[:16], 16)
                prompts[request.request_id] = prompt_ids(tokenizer, request.prompt_tokens, seed)
        if not direct:
            heartbeat = asyncio.create_task(refresh_workers(controller, config))
        await asyncio.sleep(0)
        start = time.monotonic()

        async def dispatch(request):
            outcome = outcomes[request.request_id]
            outcome.dispatch_time = time.monotonic() - start
            try:
                async with asyncio.timeout(max(0.000001,
                        request.timeout_s - (outcome.dispatch_time - request.arrival_offset))):
                    plan = await controller.admit(request)
                    outcome.plan = plan
                    if not direct:
                        outcome.gateway_accept_time = time.monotonic() - start
                    async with aclosing(controller.stream(
                            request, prompts.get(request.request_id), admitted=plan)) as stream:
                        async for event in stream:
                            now = time.monotonic() - start
                            if event.text or event.token_count:
                                if outcome.first_token_time is None:
                                    outcome.first_token_time = now
                                outcome.last_token_time = now
                                outcome.chunk_times.append(now)
                                outcome.actual_output_tokens += event.token_count
                                if event.exact_token_events:
                                    outcome.timing_kind = "exact-token-events"
                                    outcome.token_times.append(now)
                            if event.done:
                                if event.completion_tokens is not None:
                                    outcome.actual_output_tokens = event.completion_tokens
                                outcome.status = Status.COMPLETE
            except Rejected as exc:
                outcome.status, outcome.rejection_reason = Status.REJECTED, str(exc)
            except TimeoutError:
                outcome.status = Status.TIMED_OUT
            except asyncio.CancelledError:
                outcome.status = Status.CANCELLED
            except Exception as exc:
                outcome.status, outcome.exception_type = Status.FAILED, type(exc).__name__
                with (bundle.path / "stderr.log").open("a") as handle:
                    handle.write(traceback.format_exc())
            finally:
                outcome.stream_end_time = time.monotonic() - start
                bundle.outcome(outcome)
                bundle.event("outcome", request.request_id,
                             {"status": outcome.status, "exception_type": outcome.exception_type})

        for request in requests:
            await asyncio.sleep(max(0, start + request.arrival_offset - time.monotonic()))
            tasks = {task for task in tasks if not task.done()}
            if len(tasks) >= config.get("client_concurrency", 256):
                outcome = outcomes[request.request_id]
                outcome.status = Status.REJECTED
                outcome.rejection_reason = "client-concurrency-limit"
                outcome.stream_end_time = time.monotonic() - start
                bundle.outcome(outcome)
                state, reason = "INVALID", "load generator client concurrency limit reached"
                continue
            tasks.add(asyncio.create_task(dispatch(request)))
        await asyncio.sleep(max(0, start + config["duration_s"] - time.monotonic()))
        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=config.get("drain_s", 120))
            for task in pending:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.CancelledError:
        state, reason = "ABORTED", "benchmark cancelled"
    except Exception as exc:
        state, reason = "FAILED", type(exc).__name__
        with (bundle.path / "stderr.log").open("a") as handle:
            handle.write(traceback.format_exc())
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if heartbeat:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
        if controller.reservations.owners:
            state, reason = "INVALID", "leaked controller reservations"
        summary = bundle.finalize(list(outcomes.values()), state, reason)
    return bundle.path, summary
