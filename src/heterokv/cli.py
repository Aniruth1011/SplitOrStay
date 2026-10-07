import argparse
import asyncio
import importlib.util
import json
import os
import platform
import re
import sys
from pathlib import Path

from heterokv.artifacts import read_outcomes, sha256, validate_run, write_json
from heterokv.metrics import summarize
from heterokv.runtime import benchmark, load_config
from heterokv.workloads import generate, prepare_dataset, write_trace


def doctor(config):
    backend = config.get("backend", "mock")
    gpu_requested = backend in {"vllm", "vllm-direct"}
    blockers = []
    if gpu_requested:
        if platform.system() != "Linux":
            blockers.append("NVIDIA engine requires remote Linux; use mock on this Mac")
        if not os.environ.get("SLURM_JOB_ID"):
            blockers.append("GPU execution requires an allocated Slurm compute job")
        for package in ("transformers",):
            if importlib.util.find_spec(package) is None:
                blockers.append(f"missing client dependency: {package}")
        keys = ["model_revision", "tokenizer_revision", "gpu_inventory"]
        if backend == "vllm":
            keys.extend(["profiles", "profile_run_ids"])
        else:
            keys.extend(["model", "endpoint", "engine_settings"])
            if config.get("policy") != "direct":
                blockers.append("direct pilot must declare policy: direct")
            if config.get("allocated_gpus", 0) < 1:
                blockers.append("record the total allocated GPU count")
        for key in keys:
            if not config.get(key):
                blockers.append(f"missing frozen/measured {key}")
        if config.get("model_revision") != config.get("tokenizer_revision") or not re.fullmatch(
                r"[0-9a-f]{40}", config.get("model_revision") or ""):
            blockers.append("model and tokenizer must use the same immutable 40-character commit")
        if backend == "vllm" and config.get("profile_evidence_kind") != "real-llm":
            blockers.append("routing profiles must be real-LLM measurements")
    return {"platform": platform.platform(), "python": platform.python_version(),
            "backend": config.get("backend", "mock"), "blockers": blockers,
            "ready": not blockers, "gpu_acceptance": "BLOCKED: no validated hardware evidence"}


def summarize_runs(root, out):
    out.mkdir(parents=True, exist_ok=False)
    summaries = []
    for run in sorted(root.iterdir()):
        if not run.is_dir():
            continue
        validation = validate_run(run)
        if not validation["valid"]:
            summaries.append(validation)
            continue
        manifest = json.loads((run / "manifest.json").read_text())
        summary = summarize(read_outcomes(run), manifest["config"]["duration_s"],
                            manifest["allocated_gpus"])
        summary.update(run_id=run.name, evidence_kind=manifest["evidence_kind"],
                       source_manifest_sha256=sha256(run / "manifest.json"),
                       model_revision=manifest["model_revision"],
                       policy=manifest["config"].get("policy"), valid=True)
        summaries.append(summary)
    write_json(out / "summaries.json", summaries)
    return summaries


def plot_summaries(source, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = json.loads((source / "summaries.json").read_text())
    out.mkdir(parents=True, exist_ok=False)
    # Separate evidence classes; no fabricated performance curve or aggregated claims.
    for kind in sorted({r["evidence_kind"] for r in rows if r.get("valid")}):
        data = [r for r in rows if r.get("valid") and r["evidence_kind"] == kind]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(range(len(data)), [r["request_goodput"] for r in data])
        ax.set_xticks(range(len(data)), [r["run_id"] for r in data], rotation=30, ha="right",
                      fontsize=7)
        ax.set_ylabel("Compliant requests / observation second")
        ax.set_title(f"{kind} — individual runs; no comparative improvement claim")
        fig.tight_layout()
        fig.savefig(out / f"{kind}.png", dpi=160)
        plt.close(fig)
        write_json(out / f"{kind}-caption.json", data)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="heterokv")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("doctor", "serve", "benchmark", "profile", "sweep"):
        child = sub.add_parser(name)
        child.add_argument("--config", required=True, type=Path)
    for name in ("replay", "evaluate"):
        sub.add_parser(name).add_argument("--manifest", required=True, type=Path)
    trace = sub.add_parser("trace").add_subparsers(dest="action", required=True)
    gen = trace.add_parser("generate")
    gen.add_argument("--config", required=True, type=Path)
    gen.add_argument("--seed", required=True, type=int)
    gen.add_argument("--out", type=Path)
    runs = sub.add_parser("runs").add_subparsers(dest="action", required=True)
    runs.add_parser("validate").add_argument("--runs", type=Path, required=True)
    summary = sub.add_parser("summarize")
    summary.add_argument("--runs", type=Path, required=True)
    summary.add_argument("--out", type=Path, required=True)
    plot = sub.add_parser("plot")
    plot.add_argument("--input", type=Path, required=True)
    plot.add_argument("--out", type=Path, required=True)
    dataset = sub.add_parser("dataset").add_subparsers(dest="action", required=True)
    dataset.add_parser("prepare").add_argument("--config", required=True, type=Path)
    dataset.add_parser("validate").add_argument("--registry", required=True, type=Path)
    report = sub.add_parser("report")
    report.add_argument("--registry", required=True, type=Path)
    report.add_argument("--ledger", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = execute(args)
        if result is not None:
            print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"heterokv: {exc}", file=sys.stderr)
        return 2


def execute(args):
    if args.command == "doctor":
        result = doctor(load_config(args.config))
        if not result["ready"]:
            print(json.dumps(result, indent=2))
            raise RuntimeError("remote acceptance prerequisites are blocked")
        return result
    if args.command == "trace":
        config = load_config(args.config)
        path = args.out or Path(f"datasets/prepared/{config.get('workload', 'W1')}-{args.seed}.jsonl")
        requests = generate(config, args.seed)
        write_trace(path, requests)
        return {"path": str(path), "requests": len(requests), "sha256": sha256(path)}
    if args.command in {"benchmark", "replay"}:
        config = load_config(args.config if args.command == "benchmark" else args.manifest)
        if config.get("backend") in {"vllm", "vllm-direct"}:
            report = doctor(config)
            if not report["ready"]:
                raise RuntimeError("; ".join(report["blockers"]))
        path, summary = asyncio.run(benchmark(config))
        if summary["validation_status"] not in {"COMPLETED"}:
            print(json.dumps({"path": str(path), "summary": summary}, indent=2))
            raise RuntimeError("run failed or invalid; evidence retained")
        return {"path": str(path), "summary": summary}
    if args.command == "serve":
        import uvicorn
        from heterokv.api import create_app
        config = load_config(args.config)
        if config.get("backend") in {"vllm", "vllm-direct"}:
            report = doctor(config)
            if not report["ready"]:
                raise RuntimeError("; ".join(report["blockers"]))
        uvicorn.run(create_app(config), host=config.get("host", "127.0.0.1"),
                    port=config.get("port", 8080))
    elif args.command == "runs":
        result = [validate_run(p) for p in sorted(args.runs.iterdir()) if p.is_dir()]
        if any(not r["valid"] for r in result):
            print(json.dumps(result, indent=2))
            raise RuntimeError("invalid or incomplete evidence bundles")
        return result
    elif args.command == "summarize":
        return summarize_runs(args.runs, args.out)
    elif args.command == "plot":
        plot_summaries(args.input, args.out)
    elif args.command == "dataset":
        if args.action == "prepare":
            return prepare_dataset(load_config(args.config))
        records = json.loads(args.registry.read_text())
        for record in records["datasets"]:
            if record.get("path") and sha256(Path(record["path"])) != record["sha256"]:
                raise ValueError("dataset hash mismatch")
        return {"registry_valid": True, "datasets": records["datasets"],
                "note": "PLANNED entries are not downloaded or replay-validated"}
    elif args.command == "report":
        registry = load_config(args.registry)
        events = [json.loads(line) for line in args.ledger.read_text().splitlines() if line]
        return {"planned": registry, "execution_events": events,
                "selection": "BLOCKED pending measured validation and frozen held-out campaign"}
    elif args.command == "sweep":
        from heterokv.campaign import sweep
        return asyncio.run(sweep(load_config(args.config)))
    elif args.command == "evaluate":
        from heterokv.campaign import evaluate
        return asyncio.run(evaluate(load_config(args.manifest)))
    elif args.command == "profile":
        raise RuntimeError(f"BLOCKED: {args.command} is not implemented for hardware campaigns; "
                           "see docs/status.md. No experiment has been executed.")


if __name__ == "__main__":
    sys.exit(main())
