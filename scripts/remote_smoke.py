"""One allocated-GPU streamed request, independent of scheduler profiles.

This is an unvalidated real-LLM smoke test, not a phase profile or correctness campaign.
It preserves artifacts on connection failures and does not claim cache handoff support.
"""
import argparse
import asyncio
import json
import os
import platform
import re
import subprocess
import time
import traceback
from pathlib import Path

from heterokv.adapters import VLLMAdapter
from heterokv.artifacts import RunBundle
from heterokv.contracts import Outcome, Plan, Request, Status
from heterokv.workloads import prompt_ids


async def smoke(args):
    if platform.system() != "Linux" or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("run only inside an allocated Linux/Slurm GPU job")
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise ValueError("provide the immutable 40-character model/tokenizer commit")
    if args.allocated_gpus < 1:
        raise ValueError("allocated GPU count must include all reserved GPUs and be positive")
    inventory = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,name,memory.total,driver_version", "--format=csv,noheader"],
        text=True, capture_output=True, check=True).stdout.strip().splitlines()
    if not inventory:
        raise RuntimeError("no visible GPUs")
    request = Request(request_id="single-gpu-smoke", arrival_offset=0,
                      prompt_tokens=args.prompt_tokens, output_cap=args.output_tokens,
                      timeout_s=args.timeout)
    config = {"experiment_id": "E0-single-gpu-smoke", "backend": "vllm", "policy": "direct",
              "model": args.model, "model_revision": args.revision,
              "tokenizer_revision": args.revision, "duration_s": args.timeout,
              "drain_s": 0, "allocated_gpus": args.allocated_gpus, "gpu_inventory": inventory,
              "endpoint": args.endpoint, "seed": request.seed,
              "note": "single request smoke, not a capacity or phase-profile run"}
    bundle = RunBundle(Path("results/runs"), config, [request], "real-llm-unvalidated")
    bundle.state("RUNNING")
    outcome = Outcome(run_id=bundle.run_id, request=request, scheduled_arrival=0, policy="direct")
    status, reason = "COMPLETED", None
    start = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
        tokens = prompt_ids(tokenizer, args.prompt_tokens, request.seed)
        # Dummy zero costs are metadata for a direct request, never predictor observations.
        plan = Plan(mode="colocated", prefill_worker="direct", decode_worker="direct",
                    queue_ms=0, prefill_ms=0, transfer_ms=0, tpot_ms=0, ttft_ms=0,
                    estimated_output_tokens=args.output_tokens, reserved_blocks={}, risk=0,
                    reason="direct isolated smoke; no controller cost model")
        adapter = VLLMAdapter({"direct": args.endpoint}, args.model)
        start = time.monotonic()
        outcome.dispatch_time = 0
        async with asyncio.timeout(args.timeout):
            async for event in adapter.stream(request, plan, tokens):
                offset = time.monotonic() - start
                if event.text:
                    if outcome.first_token_time is None:
                        outcome.first_token_time = offset
                    outcome.last_token_time = offset
                    outcome.chunk_times.append(offset)
                if event.done:
                    outcome.actual_output_tokens = event.completion_tokens
                    outcome.status = Status.COMPLETE
                    outcome.stream_end_time = offset
        # Preserve the predeclared observation window even for a short smoke request.
        await asyncio.sleep(max(0, args.timeout - (time.monotonic() - start)))
    except TimeoutError:
        outcome.status = Status.TIMED_OUT
        status, reason = "FAILED", "request deadline"
    except asyncio.CancelledError:
        outcome.status = Status.CANCELLED
        status, reason = "ABORTED", "cancelled"
    except Exception as exc:
        outcome.status = Status.FAILED
        outcome.exception_type = type(exc).__name__
        status, reason = "FAILED", type(exc).__name__
        with (bundle.path / "stderr.log").open("a") as handle:
            handle.write(traceback.format_exc())
    finally:
        if outcome.stream_end_time is None:
            outcome.stream_end_time = 0 if start is None else time.monotonic() - start
        bundle.outcome(outcome)
        result = bundle.finalize([outcome], status, reason)
        print(json.dumps({"path": str(bundle.path), "summary": result}, indent=2))
    return 0 if status == "COMPLETED" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--allocated-gpus", required=True, type=int)
    parser.add_argument("--prompt-tokens", default=128, type=int)
    parser.add_argument("--output-tokens", default=16, type=int)
    parser.add_argument("--timeout", default=30, type=float)
    raise SystemExit(asyncio.run(smoke(parser.parse_args())))
