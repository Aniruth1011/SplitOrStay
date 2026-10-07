"""Unique evidence bundles and append-only execution state transitions."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
import uuid
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from heterokv.contracts import Outcome
from heterokv.metrics import summarize


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def append_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_EX)
        view = memoryview(data)
        while view:
            count = os.write(fd, view)
            view = view[count:]
        os.fsync(fd)
    finally:
        os.close(fd)


def git_identity():
    def git(*args):
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"commit": git("rev-parse", "HEAD"), "dirty_diff": git("diff", "HEAD"),
            "untracked": git("ls-files", "--others", "--exclude-standard")}


class RunBundle:
    def __init__(self, root: Path, config: dict, requests, evidence_kind: str,
                 ledger=Path("experiments/ledger.jsonl")):
        self.run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:12]
        self.path = root / self.run_id
        self.path.mkdir(parents=True, exist_ok=False)
        self.ledger, self.started, self.sequence = ledger, time.monotonic(), 0
        self.config = config
        with zipfile.ZipFile(self.path / "source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
            for source in sorted(Path("src").rglob("*.py")):
                archive.write(source, str(source))
            for source in (Path("pyproject.toml"), Path("AGENTS.md"), Path("SKILLS.md")):
                if source.exists():
                    archive.write(source, str(source))
        self.manifest = {
            "run_id": self.run_id, "schema_version": 1, "config": config,
            "experiment_id": config.get("experiment_id", "local-smoke"),
            "evidence_kind": evidence_kind, "platform": platform.platform(),
            "python": platform.python_version(), "git": git_identity(),
            "source_sha256": {str(p): sha256(p) for p in sorted(Path("src").rglob("*.py"))},
            "source_archive_sha256": sha256(self.path / "source.zip"),
            "dependencies": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
            "lock_sha256": sha256(Path("uv.lock")) if Path("uv.lock").exists() else None,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "gpu_inventory": config.get("gpu_inventory", []),
            "allocated_gpus": config.get("allocated_gpus", 0),
            "model_revision": config.get("model_revision"),
            "tokenizer_revision": config.get("tokenizer_revision"),
            "trace_sha256": hashlib.sha256("\n".join(
                r.model_dump_json() for r in requests).encode()).hexdigest(),
            "offered_ids": [r.request_id for r in requests],
            "timing_convention": "single client monotonic offsets; arrival cohort; bounded drain",
            "unavailable_fields": "engine phase/cache/NVML measurements absent unless explicitly collected",
            "retry_of": config.get("retry_of"),
        }
        write_json(self.path / "manifest.json", self.manifest)
        for name in ("events.jsonl", "outcomes.jsonl", "stdout.log", "stderr.log"):
            (self.path / name).touch(exist_ok=False)
        self.state("PLANNED")

    def state(self, status, reason=None):
        event = {"run_id": self.run_id, "state": status, "reason": reason,
                 "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        append_json(self.ledger, event)
        self.event("run_state", None, event)

    def event(self, kind, request_id, payload):
        self.sequence += 1
        # Buffered per-process append; detailed outcomes also have a crash-recovery journal.
        with (self.path / "events.jsonl").open("a") as handle:
            handle.write(json.dumps({"seq": self.sequence, "run_id": self.run_id,
                                     "offset": time.monotonic() - self.started,
                                     "request_id": request_id, "type": kind,
                                     "payload": payload}) + "\n")

    def outcome(self, outcome: Outcome):
        with (self.path / "outcomes.jsonl").open("a") as handle:
            handle.write(outcome.model_dump_json() + "\n")

    def finalize(self, outcomes, state="COMPLETED", reason=None):
        if {o.request.request_id for o in outcomes} != set(self.manifest["offered_ids"]):
            state, reason = "INVALID", "offered outcome mismatch"
        rows = []
        for o in outcomes:
            row = o.model_dump(mode="json")
            row["request_json"] = json.dumps(row.pop("request"))
            row["plan_json"] = json.dumps(row.pop("plan"))
            row.update(request_id=o.request.request_id, prompt_tokens=o.request.prompt_tokens,
                       requested_output_cap=o.request.output_cap, seed=o.request.seed,
                       workload_class=o.request.workload_class,
                       ttft_slo_ms=o.request.ttft_slo_ms, tpot_slo_ms=o.request.tpot_slo_ms)
            row.update(predicted_output_tokens=o.plan.estimated_output_tokens if o.plan else None,
                       plan_mode=o.plan.mode if o.plan else None,
                       prefill_worker=o.plan.prefill_worker if o.plan else None,
                       decode_worker=o.plan.decode_worker if o.plan else None,
                       predicted_costs=json.dumps(o.plan.model_dump()) if o.plan else None)
            for field in ("queue_ms", "prefill_ms", "transfer_visible_stall_ms",
                          "transfer_elapsed_ms", "transfer_bytes", "cache_reserved_bytes"):
                row[field] = None  # Engine measurements unavailable; never copy predictions here.
            rows.append(row)
        pq.write_table(pa.Table.from_pylist(rows) if rows else pa.table({"request_id": []}),
                       self.path / "requests.parquet")
        # Empty, explicitly typed telemetry does not imply hardware was sampled.
        pq.write_table(pa.table({"worker_id": pa.array([], type=pa.string()),
                                 "offset": pa.array([], type=pa.float64()),
                                 "memory_used_bytes": pa.array([], type=pa.int64())}),
                       self.path / "telemetry.parquet")
        summary = summarize(outcomes, self.config["duration_s"], self.config.get("allocated_gpus", 0))
        summary.update(run_id=self.run_id, evidence_kind=self.manifest["evidence_kind"],
                       validation_status=state, reason=reason)
        write_json(self.path / "summary.json", summary)
        self.state(state, reason)
        with (self.path / "SHA256SUMS").open("x") as handle:
            for path in sorted(self.path.iterdir()):
                if path.name != "SHA256SUMS":
                    handle.write(f"{sha256(path)}  {path.name}\n")
        return summary


def read_outcomes(path):
    result = []
    for row in pq.read_table(path / "requests.parquet").to_pylist():
        request = json.loads(row.pop("request_json"))
        plan = json.loads(row.pop("plan_json"))
        fields = {k: v for k, v in row.items() if k in Outcome.model_fields}
        result.append(Outcome(**fields, request=request, plan=plan))
    return result


def validate_run(path: Path):
    errors = []
    required = {"manifest.json", "requests.parquet", "telemetry.parquet", "events.jsonl",
                "outcomes.jsonl", "stdout.log", "stderr.log", "summary.json"}
    try:
        entries = {}
        for line in (path / "SHA256SUMS").read_text().splitlines():
            expected, name = line.split("  ", 1)
            if Path(name).name != name or name in entries:
                raise ValueError("unsafe or duplicate hash entry")
            entries[name] = expected
            if sha256(path / name) != expected:
                errors.append(f"hash mismatch: {name}")
        if not required <= entries.keys():
            errors.append("missing required hash entries")
        manifest = json.loads((path / "manifest.json").read_text())
        outcomes = read_outcomes(path)
        if sorted(o.request.request_id for o in outcomes) != sorted(manifest["offered_ids"]):
            errors.append("offered outcomes do not reconcile")
        recomputed = summarize(outcomes, manifest["config"]["duration_s"], manifest["allocated_gpus"])
        saved = json.loads((path / "summary.json").read_text())
        for key, value in recomputed.items():
            if json.loads(json.dumps(value)) != saved.get(key):
                errors.append(f"summary mismatch: {key}")
    except (OSError, ValueError, KeyError) as exc:
        errors.append(str(exc))
    return {"run_id": path.name, "valid": not errors, "errors": errors}
