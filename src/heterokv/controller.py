import asyncio
import time
from collections.abc import Callable

from heterokv.contracts import TERMINAL, Plan, Request, Status, Worker
from heterokv.kv_manager import Reservations
from heterokv.scheduler import Scheduler

NEXT = {
    Status.ARRIVED: {Status.ADMITTED}, Status.ADMITTED: {Status.QUEUED},
    Status.QUEUED: {Status.PREFILL}, Status.PREFILL: {Status.TRANSFER, Status.DECODE},
    Status.TRANSFER: {Status.DECODE}, Status.DECODE: {Status.COMPLETE},
}


class Rejected(Exception):
    pass


class Controller:
    def __init__(self, workers: list[Worker], scheduler: Scheduler, adapter,
                 max_inflight=128, event_sink: Callable | None = None):
        if len({w.worker_id for w in workers}) != len(workers) or max_inflight < 1:
            raise ValueError("unique workers and positive inflight bound required")
        self.workers, self.scheduler, self.adapter = workers, scheduler, adapter
        self.reservations = Reservations({w.worker_id: w.cache_blocks for w in workers})
        self.max_inflight = max_inflight
        self.states: dict[str, Status] = {}
        self.plans: dict[str, Plan] = {}
        self.event_sink = event_sink or (lambda *args: None)

    def transition(self, request_id: str, status: Status):
        old = self.states[request_id]
        if old in TERMINAL or (status not in TERMINAL and status not in NEXT.get(old, set())):
            raise RuntimeError(f"invalid lifecycle transition {old} -> {status}")
        if status == Status.COMPLETE and old != Status.DECODE:
            raise RuntimeError("completion before decode")
        self.states[request_id] = status
        self.event_sink("lifecycle", request_id, {"previous": old, "status": status})

    async def admit(self, request: Request) -> Plan:
        async with self.reservations.lock:
            if request.request_id in self.states:
                raise Rejected("duplicate request ID")
            self.states[request.request_id] = Status.ARRIVED
            self.event_sink("arrival", request.request_id, {})
            free = {w.worker_id: self.reservations.available(w.worker_id) for w in self.workers}
            candidates = self.scheduler.candidates(request, self.workers, free,
                                                   time.monotonic(), self.adapter.supports_pd)
            plan = self.scheduler.choose(candidates, self.workers)
            self.event_sink("candidates", request.request_id,
                            {"plans": [p.model_dump() for p in candidates]})
            if len(self.reservations.owners) >= self.max_inflight or plan is None:
                self.transition(request.request_id, Status.REJECTED)
                raise Rejected("bounded admission or no fresh compatible SLO/memory-feasible route")
            if not self.reservations.reserve_locked(request.request_id, plan.reserved_blocks):
                self.transition(request.request_id, Status.REJECTED)
                raise Rejected("capacity unavailable")
            self.plans[request.request_id] = plan
            for w in self.workers:
                if w.worker_id in plan.reserved_blocks:
                    w.queued_requests += 1
                    w.queued_tokens += request.prompt_tokens + request.output_cap
            self.transition(request.request_id, Status.ADMITTED)
            self.transition(request.request_id, Status.QUEUED)
            self.event_sink("reserve", request.request_id, plan.model_dump())
            return plan

    async def finish(self, request: Request, status: Status):
        async with self.reservations.lock:
            if self.states.get(request.request_id) in TERMINAL:
                return
            self.transition(request.request_id, status)
            blocks = self.reservations.owners.pop(request.request_id, None)
            if blocks is not None:
                for w in self.workers:
                    if w.worker_id in blocks:
                        w.queued_requests -= 1
                        w.queued_tokens -= request.prompt_tokens + request.output_cap
                self.event_sink("release", request.request_id, {"blocks": blocks})

    async def stream(self, request: Request, prompt=None, admitted: Plan | None = None):
        plan = admitted or await self.admit(request)
        terminal = Status.FAILED
        done = False
        try:
            async with asyncio.timeout(request.timeout_s):
                self.transition(request.request_id, Status.PREFILL)
                if plan.mode == "pd":
                    self.transition(request.request_id, Status.TRANSFER)
                # Lifecycle labels describe orchestration; no engine phase durations are inferred.
                self.transition(request.request_id, Status.DECODE)
                async for event in self.adapter.stream(request, plan, prompt):
                    if event.done:
                        done = True
                        terminal = Status.COMPLETE
                        yield event
                        break
                    yield event
                if not done:
                    raise RuntimeError("adapter ended without completion acknowledgment")
        except TimeoutError:
            terminal = Status.TIMED_OUT
            raise
        except (asyncio.CancelledError, GeneratorExit):
            terminal = Status.CANCELLED
            raise
        finally:
            cleanup = asyncio.create_task(self.finish(request, terminal))
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
                raise

