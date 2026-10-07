"""Validated public contracts. Actual output lengths are deliberately absent from Request."""

import math
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, validate_assignment=True)


class Status(StrEnum):
    ARRIVED = "ARRIVED"
    ADMITTED = "ADMITTED"
    QUEUED = "QUEUED"
    PREFILL = "PREFILL"
    TRANSFER = "TRANSFER"
    DECODE = "DECODE"
    COMPLETE = "COMPLETE"
    REJECTED = "REJECTED"
    TIMED_OUT = "TIMED_OUT"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    INCOMPLETE = "INCOMPLETE"


TERMINAL = frozenset(list(Status)[7:])


class Request(Contract):
    request_id: str = Field(min_length=1, max_length=128)
    arrival_offset: float = Field(ge=0)
    prompt_tokens: int = Field(gt=0)
    output_cap: int = Field(gt=0)
    workload_class: str = "chat"
    ttft_slo_ms: float = Field(gt=0, default=500)
    tpot_slo_ms: float = Field(gt=0, default=50)
    priority: int = 0
    timeout_s: float = Field(gt=0, default=120)
    seed: int = 101


class Worker(Contract):
    worker_id: str
    allocation_id: str
    node_id: str
    gpu_uuid: str
    gpu_type: str
    model_revision: str
    cache_signature: str
    roles: set[str] = Field(default_factory=lambda: {"colocated"})
    healthy: bool = True
    telemetry_time: float = Field(ge=0)
    cache_blocks: int = Field(ge=0)
    block_tokens: int = Field(gt=0, default=16)
    queued_requests: int = Field(ge=0, default=0)
    queued_tokens: int = Field(ge=0, default=0)
    active_sequences: int = Field(ge=0, default=0)
    topology: str = "unknown"
    endpoint: str | None = None


class ModelGeometry(Contract):
    layers: int = Field(gt=0)
    kv_heads: int = Field(gt=0)
    head_dim: int = Field(gt=0)
    bytes_per_element: int = Field(gt=0, default=2)
    max_context: int = Field(gt=0)

    def kv_bytes(self, tokens: int, block_tokens: int = 16) -> int:
        """Conventional dense attention, TP=1; not a total VRAM estimate."""
        rounded = math.ceil(tokens / block_tokens) * block_tokens
        return 2 * self.layers * self.kv_heads * self.head_dim * rounded * self.bytes_per_element


class Plan(Contract):
    mode: str
    prefill_worker: str
    decode_worker: str
    queue_ms: float = Field(ge=0)
    prefill_ms: float = Field(ge=0)
    transfer_ms: float = Field(ge=0)
    tpot_ms: float = Field(ge=0)
    ttft_ms: float = Field(ge=0)
    estimated_output_tokens: int = Field(gt=0)
    reserved_blocks: dict[str, int]
    risk: float = Field(ge=0)
    confidence: str = "uncalibrated-point-estimate"
    reason: str = "feasible"
    configuration_version: str = "v1"

    @property
    def total_ms(self) -> float:
        return self.ttft_ms + self.tpot_ms * (self.estimated_output_tokens - 1)


class TokenEvent(Contract):
    text: str = ""
    token_count: int = Field(ge=0, default=0)
    exact_token_events: bool = False
    completion_tokens: int | None = Field(ge=0, default=None)
    done: bool = False


class Outcome(Contract):
    run_id: str
    request: Request
    trace_id: str = "synthetic"
    status: Status = Status.INCOMPLETE
    scheduled_arrival: float
    dispatch_time: float | None = None
    gateway_accept_time: float | None = None
    first_token_time: float | None = None
    last_token_time: float | None = None
    stream_end_time: float | None = None
    actual_output_tokens: int = Field(ge=0, default=0)
    token_times: list[float] = Field(default_factory=list)
    chunk_times: list[float] = Field(default_factory=list)
    timing_kind: str = "chunk-observed"
    rejection_reason: str | None = None
    exception_type: str | None = None
    retry_count: int = 0
    policy: str = "unknown"
    plan: Plan | None = None

    @model_validator(mode="after")
    def ordered_times(self):
        times = [self.scheduled_arrival, self.dispatch_time, self.first_token_time,
                 self.last_token_time, self.stream_end_time]
        known = [x for x in times if x is not None]
        if any(b < a for a, b in zip(known, known[1:])):
            raise ValueError("client timestamps must be monotonic")
        return self

