"""Role-constrained point-estimate scheduling with conservative full-growth reservations."""

import math

from heterokv.contracts import ModelGeometry, Plan, Request, Worker
from heterokv.predictor import Link, Profile

POLICIES = {"round_robin", "least_queue", "token_work", "colocated", "static_pd",
            "transfer_unaware", "selective"}


class Scheduler:
    def __init__(self, geometry: ModelGeometry, profiles: dict[str, Profile],
                 links: dict[tuple[str, str], Link], policy="selective", stale_s=1.0,
                 benefit_margin=0.1, uncertainty=1.0, output_estimates=None):
        if policy not in POLICIES:
            raise ValueError(f"unknown policy: {policy}")
        if stale_s <= 0 or benefit_margin < 0 or uncertainty < 1:
            raise ValueError("invalid scheduler guard")
        self.geometry, self.profiles, self.links = geometry, profiles, links
        self.policy, self.stale_s = policy, stale_s
        self.benefit_margin, self.uncertainty = benefit_margin, uncertainty
        self.output_estimates = output_estimates or {}
        self.cursor = 0

    def candidates(self, request: Request, workers: list[Worker], free: dict[str, int],
                   now: float, supports_pd: bool) -> list[Plan]:
        if request.prompt_tokens + request.output_cap > self.geometry.max_context:
            return []
        estimate = max(1, min(request.output_cap,
                            self.output_estimates.get(request.workload_class, request.output_cap)))
        eligible = [w for w in workers if w.healthy and 0 <= now - w.telemetry_time <= self.stale_s]
        plans = []
        for p in eligible:
            for d in eligible:
                colocated = p.worker_id == d.worker_id
                if colocated and "colocated" not in p.roles:
                    continue
                if not colocated and (not supports_pd or "prefill" not in p.roles
                                      or "decode" not in d.roles):
                    continue
                if p.model_revision != d.model_revision or p.cache_signature != d.cache_signature:
                    continue
                blocks = {d.worker_id: math.ceil(
                    (request.prompt_tokens + request.output_cap) / d.block_tokens)}
                if not colocated:
                    blocks[p.worker_id] = math.ceil((request.prompt_tokens + 1) / p.block_tokens)
                if any(free.get(k, 0) < v for k, v in blocks.items()):
                    continue
                try:
                    prefill, _, pq = self.profiles[p.worker_id].predict(
                        request.prompt_tokens, request.prompt_tokens + estimate,
                        p.queued_tokens, p.active_sequences)
                    _, tpot, dq = self.profiles[d.worker_id].predict(
                        request.prompt_tokens, request.prompt_tokens + estimate,
                        d.queued_tokens, d.active_sequences)
                    link = None if colocated else self.links[p.worker_id, d.worker_id]
                except (KeyError, ValueError):
                    continue
                transfer = 0 if link is None else link.cost(
                    self.geometry.kv_bytes(request.prompt_tokens, p.block_tokens))
                modeled_transfer = 0 if self.policy == "transfer_unaware" else transfer
                queue = pq if colocated else pq + dq
                ttft = queue + prefill
                if link is not None:
                    if link.first_token == "decode":
                        ttft += modeled_transfer + tpot
                    else:
                        ttft = pq + prefill
                        if estimate > 1:
                            tpot += (dq + modeled_transfer) / (estimate - 1)
                ttft *= self.uncertainty
                tpot *= self.uncertainty
                risk = max(ttft / request.ttft_slo_ms,
                           tpot / request.tpot_slo_ms if estimate > 1 else 0)
                if risk > 1:
                    continue
                plans.append(Plan(mode="colocated" if colocated else "pd",
                                  prefill_worker=p.worker_id, decode_worker=d.worker_id,
                                  queue_ms=queue, prefill_ms=prefill, transfer_ms=transfer,
                                  tpot_ms=tpot, ttft_ms=ttft, estimated_output_tokens=estimate,
                                  reserved_blocks=blocks, risk=risk))
        return plans

    def choose(self, plans: list[Plan], workers: list[Worker]) -> Plan | None:
        colocated = [p for p in plans if p.mode == "colocated"]
        by_id = {w.worker_id: w for w in workers}
        if self.policy in {"round_robin", "least_queue", "token_work", "colocated"}:
            plans = colocated
        elif self.policy == "static_pd":
            plans = [p for p in plans if p.mode == "pd"]
        if not plans:
            return None
        if self.policy == "round_robin":
            ordered = sorted(plans, key=lambda p: p.decode_worker)
            result = ordered[self.cursor % len(ordered)]
            self.cursor += 1
            return result
        if self.policy in {"least_queue", "token_work"}:
            field = "queued_requests" if self.policy == "least_queue" else "queued_tokens"
            return min(plans, key=lambda p: (getattr(by_id[p.decode_worker], field),
                                            p.total_ms, p.decode_worker))
        if self.policy == "selective" and colocated:
            best_local = min(colocated, key=lambda p: p.total_ms)
            plans = [p for p in plans if p.mode == "colocated"
                     or p.total_ms < best_local.total_ms * (1 - self.benefit_margin)]
        return min(plans, key=lambda p: (p.risk, p.total_ms, p.prefill_worker, p.decode_worker))
