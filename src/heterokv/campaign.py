"""Small bounded comparison campaigns; design selection remains evidence-dependent."""

import asyncio
import copy
import random
from pathlib import Path

from heterokv.artifacts import sha256, write_json
from heterokv.runtime import benchmark, load_config


async def sweep(config):
    if config.get("stage") != "validation":
        raise ValueError("sweeps are validation-only")
    if config.get("dataset_id") == "azure2024":
        raise ValueError("Azure 2024 is reserved for frozen held-out evaluation")
    base = load_config(config["base_config"])
    if base.get("backend") != "mock":
        raise RuntimeError("BLOCKED: real campaign needs hardware/connector acceptance gates")
    candidates = config["candidates"]
    if not 1 <= len(candidates) <= 24:
        raise ValueError("screening requires between 1 and 24 explicit candidates")
    seeds = config["seeds"]
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate seeds are not independent paired repetitions")
    out = Path(config["out"])
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "declaration.json", config)
    results = []
    rng = random.Random(config.get("order_seed", 101))
    for seed in seeds:
        order = list(candidates)
        rng.shuffle(order)
        for candidate in order:
            run = copy.deepcopy(base)
            allowed = {"policy", "scheduler", "max_inflight"}
            if not set(candidate) <= allowed:
                raise ValueError("candidate modifies a controlled workload/resource variable")
            run.update(candidate)
            run["seed"] = seed
            run["experiment_id"] = config["experiment_id"]
            path, summary = await benchmark(run)
            results.append({"seed": seed, "candidate": candidate, "path": str(path),
                            "manifest_sha256": sha256(path / "manifest.json"), "summary": summary})
            write_json(out / f"attempt-{len(results):04d}.json", results[-1])
    write_json(out / "results.json", results)
    return {"attempts": len(results), "path": str(out), "evidence_kind": "mock",
            "selection": "none; mock outcomes cannot choose the hardware winner"}


def verify_frozen(manifest):
    if not manifest.get("frozen"):
        raise ValueError("manifest is not frozen")
    if set(manifest["validation_seeds"]) & set(manifest["test_seeds"]):
        raise ValueError("held-out seeds overlap validation")
    if len(set(manifest["test_seeds"])) < 5:
        raise ValueError("at least five distinct held-out paired seeds required")
    for file, expected in manifest["input_hashes"].items():
        if sha256(Path(file)) != expected:
            raise ValueError(f"frozen input changed: {file}")
    if manifest["config"] not in manifest["input_hashes"]:
        raise ValueError("configuration must be included in frozen hashes")
    return load_config(manifest["config"])


async def evaluate(manifest):
    config = verify_frozen(manifest)
    if config.get("backend") != "mock":
        raise RuntimeError("BLOCKED: real held-out campaign requires accepted hardware evidence")
    results = []
    for seed in manifest["test_seeds"]:
        path, summary = await benchmark(config | {"seed": seed})
        results.append({"path": str(path), "summary": summary})
    return {"evidence_kind": "mock", "results": results,
            "note": "Replay mechanics only; this does not satisfy real held-out acceptance"}
