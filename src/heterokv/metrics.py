from collections import Counter

import numpy as np

from heterokv.contracts import Outcome, Status


def request_metrics(o: Outcome):
    ttft = None if o.first_token_time is None else 1000 * (
        o.first_token_time - o.scheduled_arrival)
    tpot = None
    if (o.actual_output_tokens > 1 and o.first_token_time is not None
            and o.last_token_time is not None and o.last_token_time > o.first_token_time):
        tpot = 1000 * (o.last_token_time - o.first_token_time) / (o.actual_output_tokens - 1)
    compliant = (o.status == Status.COMPLETE and o.actual_output_tokens > 0 and ttft is not None
                 and ttft <= o.request.ttft_slo_ms
                 and (o.actual_output_tokens == 1 or
                      (tpot is not None and tpot <= o.request.tpot_slo_ms)))
    return {"ttft_ms": ttft, "tpot_ms": tpot, "compliant": compliant,
            "e2e_ms": None if o.stream_end_time is None else
            1000 * (o.stream_end_time - o.scheduled_arrival),
            "dispatch_delay_ms": None if o.dispatch_time is None else
            1000 * (o.dispatch_time - o.scheduled_arrival),
            "gateway_ttft_ms": None if o.gateway_accept_time is None or ttft is None else
            1000 * (o.first_token_time - o.gateway_accept_time)}


def summarize(outcomes: list[Outcome], window_s: float, allocated_gpus: int):
    if window_s <= 0 or allocated_gpus < 0:
        raise ValueError("positive observation window and nonnegative GPU count required")
    ids = [o.request.request_id for o in outcomes]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate logical outcomes")
    if any(not 0 <= o.scheduled_arrival < window_s for o in outcomes):
        raise ValueError("outcome is outside the declared arrival cohort")
    metrics = [request_metrics(o) for o in outcomes]
    successful = [o for o, m in zip(outcomes, metrics, strict=True) if m["compliant"]]
    counts = Counter(o.status.value for o in outcomes)
    offered = len(outcomes)
    result = {"offered": offered, "counts": dict(counts), "compliant": len(successful),
              "offered_slo_attainment": len(successful) / offered if offered else None,
              "rejection_fraction": counts["REJECTED"] / offered if offered else None,
              "request_goodput": len(successful) / window_s,
              "token_goodput_per_gpu": sum(o.actual_output_tokens for o in successful)
              / window_s / allocated_gpus if allocated_gpus else None,
              "steady_window_completion_throughput": sum(
                  o.status == Status.COMPLETE and o.stream_end_time is not None
                  and o.stream_end_time < window_s for o in outcomes) / window_s,
              "allocated_gpus": allocated_gpus, "window_s": window_s,
              "timing_convention": "arrival cohort; fixed window denominator; bounded drain",
              "timing_kinds": sorted({o.timing_kind for o in outcomes})}
    accepted = offered - counts["REJECTED"]
    result["accepted_slo_attainment"] = len(successful) / accepted if accepted else None
    result["runtime_failure_fraction"] = sum(counts[s] for s in
                                             ("FAILED", "TIMED_OUT", "CANCELLED", "INCOMPLETE")) \
        / offered if offered else None
    for name in ("ttft_ms", "tpot_ms", "dispatch_delay_ms", "e2e_ms"):
        values = [m[name] for m in metrics if m[name] is not None]
        result[name] = {f"p{p}": float(np.percentile(values, p)) if values else None
                        for p in (50, 95, 99)}
    result["p99_exploratory"] = counts["COMPLETE"] < 10000
    result["per_class"] = {}
    for name in sorted({o.request.workload_class for o in outcomes}):
        indexes = [i for i, o in enumerate(outcomes) if o.request.workload_class == name]
        result["per_class"][name] = {"offered": len(indexes), "offered_slo_attainment":
                                     sum(metrics[i]["compliant"] for i in indexes) / len(indexes)}
    # A drain must not conceal backlog: report a time series and a late-window slope.
    if offered:
        grid = np.linspace(0, window_s, 21)
        backlog = [int(sum(o.scheduled_arrival <= t and
                       (o.stream_end_time is None or o.stream_end_time > t) for o in outcomes))
                   for t in grid]
        result["backlog"] = list(zip(grid.tolist(), backlog, strict=True))
        result["late_backlog_slope_per_s"] = float(np.polyfit(grid[10:], backlog[10:], 1)[0])
    result["capacity_qualified"] = False  # Requires a repeated rate campaign and stability review.
    return result


def paired_interval(candidate, baseline, seed=101, repetitions=10000, ratio=True):
    """Run-level paired bootstrap; the caller must align session/trace pairs."""
    a, b = np.asarray(candidate, dtype=float), np.asarray(baseline, dtype=float)
    if a.shape != b.shape or len(a) < 2 or not np.all(np.isfinite(a + b)):
        raise ValueError("at least two finite aligned run pairs required")
    if ratio and (np.any(b <= 0) or np.any(a <= 0)):
        raise ValueError("nonpositive goodput: report absolute differences, no epsilon")
    samples = np.log(a / b) if ratio else a - b
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(samples, size=(repetitions, len(samples))), axis=1)
    values = [float(np.mean(samples)), *np.percentile(means, [2.5, 97.5]).tolist()]
    if ratio:
        values = np.exp(values).tolist()
    return dict(zip(("estimate", "lower95", "upper95"), values, strict=True))
