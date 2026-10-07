# HeteroKV — Agent Instructions

## 1. Mission and problem statement

Build and evaluate a cost-aware controller for LLM serving on heterogeneous NVIDIA GPUs. For each request, choose whether prefill and decode should run together or on separate workers, accounting for queueing, phase execution time, KV-cache transfer cost, and destination memory capacity. Maximize requests that meet both first-token and token-generation latency objectives within a fixed GPU allocation.

Research question: **When does selective prefill/decode disaggregation improve SLO-compliant goodput over tuned colocated serving and static disaggregation, and when should it be avoided?**

The contribution is the measured decision policy and its evaluation. Use an existing inference engine. Do not claim that heterogeneous placement or disaggregation is new by itself. Maintain a related-work table and identify the narrower contribution supported by experiments.

Read `SKILLS.md` before implementation; it contains the execution playbooks, experiment matrix, and target-selection procedure. These are repository instructions, not an installed ChatGPT skill package.

Final evaluation-plan revision: 2026-10-06. This revision adds public trace benchmarks, mandatory comparison campaigns, and an auditable experiment ledger. It describes work to perform; it does not imply the experiments have already run.

## 2. Scientific and engineering rules

- Treat every numerical target in these files as a goal, never as an achieved result.
- Keep simulated, synthetic-transfer, real-LLM, and real-heterogeneous results explicitly separate.
- Never invent timings, bandwidths, GPU allocations, compatibility, or benchmark improvements.
- Preserve raw request outcomes, including rejects, timeouts, disconnects, retries, and incomplete requests. A failed or rejected request is not SLO compliant.
- Use the same model revision, tokenizer, precision, workloads, GPU pool, and offered load for policy comparisons.
- Build strong baselines, including chunked-prefill colocated vLLM. Give baselines comparable validation tuning budgets.
- Use validation workloads to choose parameters; freeze the selected configuration before final held-out evaluation.
- Do not provide the scheduler with future arrivals or actual future output lengths. An oracle may be an explicitly labeled diagnostic baseline only.
- Negative results are valid. If selective disaggregation loses on a topology, publish the crossover and explanation; do not weaken the baseline or quietly change the SLO.
- Do GPU work only inside allocated compute jobs. Manage only this project's processes and resources.
- Continue independent implementation when hardware is unavailable, but mark GPU acceptance gates blocked. CPU mocks cannot satisfy them.

## 3. Scope and feasibility gates

### Core deliverable

One dense decoder-only model; two or more GPU workers; streamed responses; colocated and prefill/decode paths; measured hardware profiles; transfer-aware scheduling; destination KV capacity reservations; reproducible benchmarks and analysis.

Target two supported GPU types, for example A100 plus A40 or L40S. Begin with one GPU, then two homogeneous GPUs to validate transfer correctness, then a simultaneously allocated heterogeneous pair across nodes. The final heterogeneous claim requires simultaneous execution on different physical GPU types. Four GPUs are an extension, not a prerequisite.

### Integration constraints

1. Discovery node types have different GPU counts. Do not assume four GPUs on one node or mixed GPU types within one node. Discover actual resources and validate Slurm placement.
2. Pin a tested vLLM, PyTorch, CUDA, connector, and transport combination. Check the release-specific hardware requirements. V100/P100 must not be assumed compatible with the selected modern stack; exclude them unless a separate tested backend supports them.
3. Start with tensor parallelism = 1 on each worker, identical model/KV dtypes, and compatible attention/cache configurations. Keep connector handshake checks enabled.
4. Prefer the selected vLLM KV connector's data plane, initially NIXL if supported by the allocation. `torch.distributed`/NCCL are tools for controlled transfer microbenchmarks or a separately validated adapter. A tensor send alone does not import cache blocks into vLLM.
5. Verify that the chosen adapter can express each candidate plan. Producer/consumer engine roles may be fixed. Do not assume an existing worker can dynamically change roles or a request can freely switch between engines. Include replica/model-memory costs and role-pool overhead in resource accounting.
6. Separate prefill-to-decode handoff from migration during active decode. The former is core; the latter is an extension requiring explicit engine support and correctness tests.
7. Disaggregated prefill is experimental. Benefits to goodput or latency are hypotheses to measure, not guaranteed throughput gains.

If mixed allocations or connector support fail, finish homogeneous correctness and profiling, write `docs/limitations.md`, and keep heterogeneous claims blocked. Do not replace real experiments with an unlabeled simulator.

### Extensions after core completion

Host-memory KV offload and recomputation, prefix-aware routing, worker-failure recovery, four-GPU scaling, and slow resource reallocation. Each requires an independent experiment and adapter capability check. SSD tiering, custom CUDA kernels, mixed precision across phases, arbitrary active-decode migration, and heterogeneous tensor parallelism are outside the initial scope.

## 4. Technology choices

| Layer | Default | Responsibility |
| --- | --- | --- |
| Development | Python, uv, pytest, Ruff | Controller, contracts, packaging, checks |
| Execution | PyTorch, CUDA, Hugging Face tokenizer/config | Profiling, model metadata, tensor experiments |
| Serving | Pinned vLLM | Model execution, continuous batching, chunked prefill, engine-owned KV blocks |
| KV handoff | One tested vLLM connector; NIXL first candidate | Actual cache export/import and transfer acknowledgments |
| Controller | asyncio, typed Python | Feasible-plan enumeration, predictions, admission, reservations |
| API | FastAPI, streaming HTTP/SSE, HTTPX | Client-visible streaming and asynchronous trace replay |
| Independent benchmark client | Pinned NVIDIA AIPerf or vLLM serving benchmark | Cross-check client timing and replay public workloads where supported |
| Control RPC | HTTP initially; gRPC if required | Worker status and control metadata |
| Predictions | Lookup/interpolation, linear model, XGBoost | Compare simple and learned latency estimators |
| HPC | Slurm, Apptainer or tested environment | Allocation, launch, reproducibility |
| Telemetry | NVIDIA NVML Python bindings | Per-worker utilization and memory observations |
| Metrics | prometheus-client; Grafana optional | Operational metrics and dashboards |
| Experiment data | JSON manifests, Parquet, NumPy/Pandas | Durable raw results and analysis |
| Figures | Matplotlib | Exportable benchmark figures |

Do not add unrelated infrastructure. Dashboard completion is not a prerequisite for the first benchmark. Install dependencies on the platform where they execute; the Mac/CPU environment must support contracts and trace generation without requiring CUDA.

## 5. Planned repository layout

```text
AGENTS.md
SKILLS.md
pyproject.toml
src/heterokv/
  contracts.py
  api/
  adapters/              # Serving-engine and connector interfaces
  controller/            # Request lifecycle, reservations, admission
  profiler/              # Prefill, decode, topology, transfer
  predictor/             # Lookup, linear, XGBoost, calibration
  scheduler/             # Baselines and adaptive policy
  kv_manager/            # Ownership and reservation accounting
  workloads/             # Seeded trace generation and replay
  metrics/
configs/
  models/
  slurm/
  workloads/
  experiments/
tests/
  unit/
  integration/
  gpu/
datasets/               # Registry, immutable inputs and normalized trace metadata
experiments/            # Predeclared comparisons and append-only run ledger
scripts/
slurm/
docs/
  architecture.md
  compatibility.md
  experiment-plan.md
  metrics.md
  decisions.md
  limitations.md
results/
  runs/                 # Canonical evidence bundle per unique run ID
  raw/
  profiles/
  manifests/
  summaries/
  figures/
```

This is a planned layout. Do not report directories or functions as implemented until they exist and run.

## 6. Core contracts and scheduling policy

Request fields: unique ID, arrival offset, tokenized prompt length, requested output cap, workload class, TTFT SLO, TPOT SLO, priority, timeout, and trace seed. Output length estimates must use only arrival-time information. Record actual generated length after completion.

Worker fields: allocation/node/GPU identifiers, GPU type, model revision, engine role/capabilities, cache compatibility signature, health, running/queued token work, cache block availability/reservations, last telemetry time, and transport topology.

Plan fields: execution mode, prefill/decode worker IDs, predicted queue/prefill/transfer/decode costs, estimated output length, predicted SLO risk, reserved cache capacity, confidence, reason, and configuration version.

Request lifecycle: `ARRIVED -> ADMITTED -> QUEUED -> PREFILL -> TRANSFER(optional) -> DECODE -> COMPLETE`. Alternative terminal outcomes: `REJECTED`, `TIMED_OUT`, `FAILED`, `CANCELLED`. Terminal transitions must release all owned reservations exactly once. Transfer retries must not emit duplicate tokens or duplicate terminal responses.

Core algorithm:

1. Filter unhealthy, stale, incompatible, and memory-infeasible plans. Permit only routes supported by the worker roles.
2. Estimate phase costs and queue work using token/context-aware profiles. Account for decode GPU contention and future cache growth.
3. Include colocated routes as genuine candidates. Estimate disaggregated TTFT using the adapter's first-token protocol: if the first token comes from prefill, transfer may affect later tokens rather than TTFT. Do not hard-code one timing definition for all connectors.
4. Rank plans by estimated probability of meeting both SLOs, then expected resource cost. Implement a transparent point-estimate policy first; add calibrated uncertainty only if it helps on validation traces.
5. Disaggregate only when the expected benefit exceeds transfer/setup cost plus an uncertainty margin. Search the margin experimentally.
6. Reserve destination capacity before starting handoff. Admission and reservation must be atomic under concurrent requests. Never evict live state without supported offload/recompute semantics.
7. Bound queues and define explicit overload handling. Default: reject infeasible requests with a recorded reason. Compare policies under the same admission rule before evaluating policy-specific admission.

Memory estimate for a conventional dense-attention KV cache:

`KV bytes = 2 * layers * KV_heads * head_dimension * resident_tokens * bytes_per_element`

Derive values from the actual model, including grouped-query attention. Round to engine cache blocks, account for TP sharding/replication, reserve for generation growth, and measure overhead. Weights, runtime workspaces, CUDA graphs, allocator behavior, and transfer buffers also use VRAM. This formula is not universal for all architectures.

## 7. Mandatory metrics and targets

### Definitions

Measure client-facing timing with one client's monotonic clock. Let `a` be scheduled arrival, `f` first received output token, `l` last received output token, `c` stream completion, and `n` output tokens:

- TTFT = `f - a`; includes any load-generator dispatch delay. Also report gateway-accept-to-first-token and dispatch delay separately.
- TPOT = `(l - f)/(n - 1)` for `n > 1`. One-token TPOT is N/A, with no fictitious zero value.
- ITL = adjacent output-token arrival gaps. HTTP chunks can contain several tokens: use exact token events where available or label chunk-gap measurements separately.
- E2E latency = `c - a`.
- Request compliant = successfully completed and TTFT <= its class SLO and TPOT <= its class SLO. One-token requests evaluate the TTFT condition only.
- Offered SLO attainment = compliant requests / all offered requests. Rejected, failed, timed-out, and incomplete requests count as noncompliant.
- Request goodput = compliant requests / observation-window seconds.
- Token goodput/GPU = output tokens from compliant requests / observation-window seconds / all allocated GPUs. Include idle prefill GPUs; never count prompt tokens here.
- Max sustainable compliant arrival rate = highest tested rate with offered SLO attainment >=95%, rejection <=1%, non-rejection runtime failure <=0.1%, and no persistently growing backlog.

Use an arrival cohort during a fixed observation window and a predeclared bounded drain to classify outcomes; keep the denominator as the observation window and label this cohort convention. Also report steady-window completion throughput. At sustained-load capacity points, check arrivals versus completions and backlog slope so drain cannot conceal instability.

### Target scoreboard

These targets are proposed project goals, not literature-derived guarantees. Compare against the **strongest validation-selected feasible baseline**, on the identical GPU pool and frozen workloads.

| Metric | Core target | Stretch target / qualification |
| --- | --- | --- |
| Token goodput/GPU at matched offered load | >=20% higher geometric-mean ratio across predeclared mixed/bursty/skewed challenge cells | >=35%; paired 95% CI lower bound must exceed 1.0 for an improvement claim |
| Maximum sustainable compliant arrival rate | >=15% higher | >=25%; use the same rate-search resolution |
| Offered SLO attainment below capacity | >=95% overall and >=90% in each workload class | >=99% overall; report offered and accepted denominators |
| Rejection below capacity | <=1% | <=0.1%; all rejected requests remain in offered denominator |
| p95 TTFT at matched load | >=15% lower in the predeclared burst/long-prompt stress cells | >=25%; report per class and overall |
| p95 request TPOT | <=5% regression relative to baseline in every core class | >=10% reduction in decode-contention cells |
| Scheduler decision overhead | p95 <=2 ms and p99 <=5 ms at target request rate for <=4 workers | p95 <=1 ms; report gateway/serialization overhead separately |
| Profile-based latency predictor | Held-out median absolute percentage error <=15%, p90 <=30% per predicted component | <=10% / <=20%; use a 1 ms denominator floor and also report absolute error |
| KV transfer efficiency | >=70% of same-topology, same-size, same-transport isolated transfer bandwidth | >=85%; do not compare against advertised PCIe/network bandwidth |
| Transfer tax | Median visible transfer stall <=20% of E2E latency among disaggregated requests | <=10%; also report p95 and bytes transferred per offered request |
| Memory correctness | 0 reproducible OOMs attributable to controller over-admission; 0 leaked reservations | Maintain >=10% device-memory headroom initially; tune only on validation |
| Output/cache correctness | 0 detected cache/ownership errors; >=99.9% greedy token agreement in transfer validation | Exact match where the backend is deterministic; investigate every mismatch |
| Worker failure (extension) | Detect <=5 s; restore >=90% of reduced-pool reference goodput within 30 s after detection | Compare remaining GPUs, not the original full-pool capacity |
| Slow role adjustment (extension) | >=90% of new-mix steady-state goodput within 30 s | Include restart/loading disruption; capability gate required |

Prevent cherry-picking: designate load levels and cells before final tests; include low-load, transfer-unfavorable and overload controls. Target <=5% goodput regression versus the strongest baseline in transfer-unfavorable control cells. If not achieved, report it. No minimum GPU-utilization target: higher utilization is useful only if SLO results improve.

### Initial SLOs to attempt

Use this starting table for a small/medium dense model after a feasibility pilot:

| Workload | Prompt tokens | Output tokens | TTFT SLO | Request TPOT SLO |
| --- | ---: | ---: | ---: | ---: |
| Chat | 512 | 128 | 500 ms | 50 ms/token |
| Generation | 512 | 1024 | 500 ms | 50 ms/token |
| RAG | 8192 | 256 | 1500 ms | 50 ms/token |
| Document QA | 16384 | 128 | 3000 ms | 50 ms/token |

These are proposed service objectives, not predictions about CARC performance. Check native context limits and memory before use. If the pilot shows they are infeasible, define a separate hardware-calibrated track: per-class SLO = 1.5x the p95 isolated TTFT/TPOT of a preselected feasible reference worker. Record both the original and calibrated tables and the reason; freeze the chosen primary track before policy tuning. Never change SLOs after examining held-out policy outcomes.

## 8. Implementation phases and acceptance gates

| Phase | Required work | Completion evidence |
| --- | --- | --- |
| P0: contracts and feasibility | Architecture, version pinning, model selection, Slurm discovery, worker roles, metric definitions | Compatibility report plus one real single-GPU streamed request |
| P1: honest measurement | Open-loop traces, raw outcome logging, counters, TTFT/TPOT validation | Seeded replay; known synthetic timings reproduce expected metrics; no dropped outcomes |
| P2: hardware profiling | Single-GPU prefill/decode and same-/cross-node transfer profiling | Repeated measured profiles with topology/model metadata |
| P3: real KV handoff | Connector adapter, reservations, cleanup, homogeneous then heterogeneous transfer | Correct continuation on >=1000 transfer cases spanning prompt lengths; no duplicate first token |
| P4: baselines | Tuned colocation, static PD splits, round-robin, least-queue, token-work-aware colocation | Same trace and resource budget replayed across each policy |
| P5: adaptive controller | Lookup predictor, selective PD, transfer/memory guards, bounded admission | Feasibility tests pass; initial controlled GPU comparison |
| P6: experiments and selection | Predictor/policy/threshold sweeps, held-out tests, ablations | Frozen winner manifest, raw data, confidence intervals, regression controls |
| P7: optional extensions | Prefix/offload/failure/role-control experiments | Independent capability gates and ablations; no inherited success claims |
| P8: final report | Figures, reproducible commands, limitations, evidence-backed results | Result table regenerated from raw artifacts; achieved/failed/blocked targets explicit |

Do not skip KV correctness to produce performance charts. Do not spend the full GPU budget on a Cartesian-product sweep; follow the staged search in `SKILLS.md`.

## 9. Verification and final deliverables

Required tests: metric arithmetic and denominators; deterministic trace generation; candidate feasibility; no future-information leakage; simultaneous memory reservations; release on cancel/timeout/transfer failure; stale telemetry; connector continuation; API backpressure; outcome reconciliation; report regeneration.

Expose documented commands for environment checks, worker launch, profiling, trace generation, policy replay, summarization and plotting. The commands in `SKILLS.md` define the intended CLI contract; mark each unavailable command until implemented. Use short CPU checks after controller changes and GPU integration checks after connector changes.

Each benchmark manifest must store commit, dependency/container identities, model/tokenizer revision, GPU UUID/type/count, Slurm job ID, topology, role layout, engine settings, policy parameters, dataset/trace hashes, seeds, SLOs, timing convention, run/drain limits, exclusions, and artifact paths.

Final evidence package: raw Parquet, immutable run manifests, hardware profiles, baseline tuning records, validation selection, held-out results with intervals, ablations, figures, failure logs, related work, and a short README. Resume claims must identify tested GPUs, model, baseline, workload and measured improvement. Use placeholders until measured.

## 10. Primary references to check when pinning the environment

Reviewed for this plan on 2026-10-06 UTC; these pages evolve. Record release-specific links and commits in the compatibility report.

- [vLLM disaggregated prefill](https://docs.vllm.ai/en/latest/features/disagg_prefill/)
- [vLLM NIXL usage](https://docs.vllm.ai/en/latest/features/nixl_connector_usage/)
- [vLLM NIXL compatibility](https://docs.vllm.ai/en/latest/features/nixl_connector_compatibility/)
- [vLLM GPU requirements, versioned example](https://docs.vllm.ai/en/v0.28.0/getting_started/installation/gpu/)
- [USC Discovery resource overview](https://www.carc.usc.edu/user-guides/hpc-systems/discovery/resource-overview-discovery)
- [USC running jobs](https://www.carc.usc.edu/user-guides/hpc-systems/using-our-hpc-systems/running-jobs)

Prefer the matching version's connector guide and compatibility matrix over conflicting example snippets. Verify hardware and transport empirically before adding them to the supported set.

## 11. Required public benchmark campaign

Use production traces in addition to the synthetic W0-W5 workloads. A trace supplies arrivals and token demand; performance must be measured on our model and CARC allocation. Original service timings are not our baseline.

| Dataset | Role in this project | Source |
| --- | --- | --- |
| Azure LLM inference traces 2023 | Primary development/validation; separate code and conversation results | [Azure 2023 / Splitwise](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md) |
| BurstGPT | Validation of bursts and changing demand; reserve separate untouched windows for confirmation | [BurstGPT repository](https://github.com/HPMLL/BurstGPT) |
| Azure LLM inference traces 2024 | External held-out evaluation after design/configuration freeze | [Azure 2024 / DynamoLLM](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md) |
| Synthetic length/topology grid | Mechanism isolation, long-context and transfer-unfavorable controls | Seeded generator committed in this repository |

Azure traces provide timestamp, context-token count and generated-token count without customer prompt text. Implement reproducible prompt construction using the selected tokenizer. Public trace replay does not prove answer quality or prefix-cache realism. BurstGPT contains source-service failures in some releases; define eligibility before policy testing, log exclusions, and never count those historical failures as failures produced by HeteroKV.

Before tuning, freeze dataset versions/hashes, eligible rows, development/validation/test windows, original timestamp units, replay scaling and prompt construction. Choose windows by predeclared traffic statistics, including ordinary and peak periods; never select windows based on where the proposed policy wins. Group multi-turn sessions when identifiers exist so they cannot leak across splits.

Run two distinct modes:

- **Controlled length replay:** Reproduce input sizes and enforce output sizes where the backend supports it. Explicitly label output-length knowledge through the supplied cap. This measures the serving mechanism under controlled demand.
- **Natural generation:** Use text prompts and realistic output caps; predict actual length using arrival-time information. Keep trace-recorded future output counts out of scheduler features. Log deviations between prescribed and actual output length.

Measure near-capacity performance and max sustainable load using the same numeric rates for every compared policy. At least one supported simultaneous heterogeneous allocation must be used for the final heterogeneous claims. If unavailable, record the campaign as blocked and report the narrower verified scope.

Independent measurement candidate: [NVIDIA AIPerf BurstGPT replay](https://docs.nvidia.com/aiperf/dev/tutorials/datasets-inputs/profile-with-burst-gpt-traces). Verify the installed release's timestamp, output-length and failure semantics; a supported input format alone does not satisfy our outcome-accounting requirements.

## 12. Mandatory experiments, records and selection

**Do not stop at the first working scheduler or at one favorable benchmark.** Conduct a staged campaign covering different approaches, tune strong baselines, document all attempts and select the final method using validation evidence.

Minimum core comparisons: tuned colocation, both feasible static PD orientations, request-count versus token-work routing, transfer-unaware versus transfer-aware selective PD, lookup versus linear versus XGBoost prediction, and conservative versus unguarded prediction. Compare engine token budgets, migration-benefit thresholds, output-length estimation and safe memory margins. Use ablations to attribute benefits. Test queue ordering (FIFO/EDF/shortest predicted work) only through an adapter that can actually control it; otherwise mark it unsupported.

The campaign must include:

1. Real hardware/connector correctness and profiling.
2. Strong baseline tuning on validation workloads.
3. Controlled length/topology break-even tests.
4. Public-trace replay, at common rates across policies.
5. Focused parameter search and one-component-at-a-time ablations.
6. Frozen final design, held-out trace/session evaluation and paired repetitions.
7. A final table showing gains, losses, uncertainty, cost and failed/blocked targets.

Every experiment gets a pre-run hypothesis, comparator, controlled variables, primary metric, expected mechanism and pass/fail criteria. Keep planned experiments and executed runs separate. Do not mark an experiment completed from a configuration file or simulated output alone.

### Required logging artifacts

- `experiments/registry.yaml`: experiment definitions and predeclared comparisons.
- `experiments/ledger.jsonl`: append-only state and decision events for all attempts.
- `results/runs/<run_id>/manifest.json`: exact input/config/environment/resource identity.
- `results/runs/<run_id>/requests.parquet`: every offered logical request and terminal outcome.
- `results/runs/<run_id>/telemetry.parquet`: GPU and engine/cache/queue observations.
- `results/runs/<run_id>/events.jsonl`: decisions, transfers, reservations, retries and worker faults.
- `results/runs/<run_id>/stdout.log`, `stderr.log`: execution/debugging evidence.
- `results/runs/<run_id>/summary.json`: derived metrics, counts and validation status.
- `results/runs/<run_id>/SHA256SUMS`: evidence integrity hashes.
- `docs/experiment-log.md`: readable narrative of hypotheses, findings and next decisions.

Create a unique run ID for every execution attempt and link retries to their original attempt. Record failed, cancelled, incomplete and invalid runs with reasons. A new attempt must never overwrite prior artifacts. Redact credentials and avoid logging prompt content by default.

Record states `PLANNED`, `RUNNING`, `COMPLETED`, `FAILED`, `ABORTED`, `INVALID` or `BLOCKED` as append-only transitions. An invalid run remains visible with a predeclared exclusion reason, such as wrong hardware or broken load replay. Report excluded counts. A disappointing result is not an exclusion reason.

The final decision record must list all candidate approaches and why each was selected, rejected, tied, blocked or deferred. Use paired confidence intervals and per-workload results, not only aggregate averages. Every headline number must link to raw run IDs, the comparator configuration and the analysis command that generated it.

Do not relax the evaluation protocol to reach the proposed performance targets. If the best result is a simple lookup policy or tuned colocation, report it and explain the measured crossover. Complete the evidence package even when the performance hypothesis is disproved.
