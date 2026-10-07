import csv
import hashlib
import json
import math
import random
from pathlib import Path

from heterokv.contracts import Request

CLASSES = {"chat": (512, 128, 500), "generation": (512, 1024, 500),
           "rag": (8192, 256, 1500), "document_qa": (16384, 128, 3000)}


def generate(config: dict, seed: int) -> list[Request]:
    rng = random.Random(seed)
    duration, rate = float(config["duration_s"]), float(config["rate"])
    kind = config.get("workload", "W1")
    if (not math.isfinite(duration) or not math.isfinite(rate) or duration <= 0 or rate <= 0
            or kind not in {f"W{i}" for i in range(6)}):
        raise ValueError("invalid duration, rate or workload")
    at, result = 0.0, []
    # Thinning preserves piecewise burst intensity without stepping over boundaries.
    peak_rate = rate * (1.5 if kind == "W2" else 1)
    while True:
        at += 1 / rate if kind == "W0" else rng.expovariate(peak_rate)
        if at >= duration:
            break
        if kind == "W2":
            multiplier = 0.5 if at % 120 < 90 else 1.5
            if rng.random() > multiplier / 1.5:
                continue
        mix = {"W0": {"chat": .7, "generation": .3},
               "W1": dict.fromkeys(CLASSES, .25), "W2": {"chat": .5, "rag": .5},
               "W4": {"document_qa": .7, "rag": .3},
               "W5": {"generation": .8, "chat": .2}}.get(kind)
        if kind == "W3":
            mix = [{"chat": .8, "rag": .2}, {"chat": .2, "rag": .8},
                   {"generation": .6, "document_qa": .4}][int(at // 60) % 3]
        name = rng.choices(list(mix), weights=list(mix.values()))[0]
        prompt, output, ttft = CLASSES[name]
        if config.get("tiny_mock"):
            prompt, output = 16, 4
        result.append(Request(request_id=f"{seed}-{len(result):08d}", arrival_offset=at,
                              prompt_tokens=prompt, output_cap=output, workload_class=name,
                              ttft_slo_ms=ttft, tpot_slo_ms=50, seed=seed,
                              timeout_s=config.get("timeout_s", 120)))
    return result


def write_trace(path: Path, requests: list[Request]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        for request in requests:
            handle.write(request.model_dump_json() + "\n")


def read_trace(path: Path):
    requests = [Request.model_validate_json(line) for line in path.read_text().splitlines() if line]
    if len({r.request_id for r in requests}) != len(requests):
        raise ValueError("duplicate request IDs")
    if any(b.arrival_offset < a.arrival_offset for a, b in zip(requests, requests[1:])):
        raise ValueError("trace must be ordered")
    return requests


def prompt_ids(tokenizer, count: int, seed: int):
    """Varied deterministic vocabulary IDs; no shared padding prefix or text quality claim."""
    excluded = set(tokenizer.all_special_ids)
    valid = sorted(set(tokenizer.get_vocab().values()) - excluded)
    if not valid:
        raise ValueError("empty non-special vocabulary")
    rng = random.Random(seed)
    result = rng.choices(valid, k=count)
    if len(result) != count:
        raise ValueError("prompt length mismatch")
    return result


def prepare_dataset(config: dict):
    """Explicit column mapping avoids guessing dataset release schemas or timestamp units."""
    source, output = Path(config["source"]), Path(config["out"])
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != config["sha256"]:
        raise ValueError("source hash does not match frozen dataset identity")
    if config["mode"] not in {"controlled", "natural"}:
        raise ValueError("unknown replay mode")
    scale = float(config["timestamp_seconds_per_unit"])
    speedup = float(config["speedup"])
    if scale <= 0 or speedup <= 0:
        raise ValueError("positive timestamp scale and speedup required")
    windows = config["windows"]
    for i, a in enumerate(windows):
        if a["end"] <= a["start"]:
            raise ValueError("invalid window")
        for b in windows[i + 1:]:
            if max(a["start"], b["start"]) < min(a["end"], b["end"]):
                raise ValueError("overlapping split windows")
    output.mkdir(parents=True, exist_ok=False)
    reports, normalized, sessions = [], [], {}
    columns = config["columns"]
    with source.open(newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            reason, window = None, None
            try:
                timestamp = float(row[columns["timestamp"]]) * scale
                prompt, prescribed = int(row[columns["prompt"]]), int(row[columns["output"]])
                if not math.isfinite(timestamp) or prompt <= 0 or prescribed <= 0:
                    raise ValueError("invalid timestamp or historical zero/negative token count")
                window = next((w for w in windows if w["start"] <= timestamp < w["end"]), None)
                if window is None:
                    reason = "outside-predeclared-windows"
                cap = prescribed if config["mode"] == "controlled" else int(config["output_cap"])
                if prompt + cap > config["max_context"]:
                    reason = "context-limit"
                session = row.get(columns.get("session", ""))
                if window and session:
                    previous = sessions.setdefault(session, window["split"])
                    if previous != window["split"]:
                        raise ValueError("session crosses splits")
                if reason is None:
                    request = Request(request_id=f"{config['dataset_id']}-{index}",
                                      arrival_offset=(timestamp - window["start"]) / speedup,
                                      prompt_tokens=prompt, output_cap=cap,
                                      ttft_slo_ms=config["ttft_slo_ms"],
                                      tpot_slo_ms=config["tpot_slo_ms"], seed=config["seed"])
                    normalized.append({"request": request.model_dump(), "source_row": index,
                                       "source_session": session, "window": window["id"],
                                       "split": window["split"],
                                       "prescribed_output_tokens": prescribed})
            except (ValueError, KeyError) as exc:
                if "session crosses" in str(exc):
                    raise
                reason = str(exc)
            reports.append({"source_row": index, "eligible": reason is None, "reason": reason})
    normalized.sort(key=lambda x: (x["window"], x["request"]["arrival_offset"]))
    (output / "normalized.json").write_text(json.dumps(normalized, indent=2))
    (output / "eligibility.json").write_text(json.dumps(reports, indent=2))
    for window in windows:
        write_trace(output / f"{window['id']}.jsonl", [Request(**r["request"]) for r in normalized
                                                     if r["window"] == window["id"]])
    identity = {"config": config, "source_sha256": digest, "rows": len(reports),
                "eligible": len(normalized), "excluded": len(reports) - len(normalized)}
    (output / "manifest.json").write_text(json.dumps(identity, indent=2))
    return identity
