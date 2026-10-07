# HeteroKV — Implementation and Experiment Playbooks

This file defines how a coding agent should implement, compare, and select the HeteroKV design. Read `AGENTS.md` first. Numerical targets are goals; achieved values require real artifacts. This is a project workflow document, not an installed assistant skill.

Final evaluation-plan revision: 2026-10-06. Public-trace replay and the experiment ledger below are required components of the implementation.

## 1. Working procedure for every task

1. Read the applicable contracts, configuration and previous decision record.
2. State a concrete hypothesis or intended behavior and the evidence needed to verify it.
3. Implement the smallest complete slice that produces that evidence.
4. Run relevant CPU tests; run GPU tests when the slice touches the engine, memory or transport.
5. Save exact settings and raw outcomes before analysis.
6. Compare against the appropriate baseline under equal resources.
7. Record the decision in `docs/decisions.md`: what changed, evidence, tradeoffs, unresolved constraints and next experiment.

If a capability is missing, mark it `BLOCKED` and implement independent pieces. Do not represent a stub as a working GPU integration. Keep a checklist of implemented, verified, blocked and optional work.

## 2. Playbook: hardware and software feasibility

**Trigger:** Before selecting the engine version, requesting a GPU configuration, or writing mixed-node launch scripts.

**Procedure**

- Discover resources using CARC-supported commands, for example `noderes -c -p gpu`, `sinfo`, and `scontrol show node`. Inspect actual GPUs and topology inside an allocation using `nvidia-smi` and `nvidia-smi topo -m` where supported.
- Record GPU UUIDs, VRAM, driver, compute capability, node type, CPU/RAM, Slurm GPU binding, transport interfaces and supported paths. Never infer NVLink, RDMA or measured bandwidth from a GPU name.
- Verify Python, CUDA, PyTorch, vLLM, connector and transport compatibility. Prefer a common environment and attention configuration for the selected A100/A40/L40S pair.
- Start with a small dense model that fits on the smallest selected GPU with room for cache and workspaces. Pin exact weights and tokenizer revisions. Progress to a larger model only after correctness.
- Start with tensor parallelism = 1. Check model context limits. Exclude unsupported older GPUs from the core rather than changing the backend mid-comparison.
- Define an adapter capability matrix: colocated generation, PD producer, PD consumer, cache handoff, CPU offload, prefix reuse, active-decode migration, cancellation and role change.
- Request simultaneous mixed resources through validated Slurm configuration. Homogeneous multi-node directives do not guarantee different GPU types. Separate independent jobs require a tested rendezvous, overlap and network accessibility; document those limitations.
- Keep addresses/ports and credentials out of committed public configurations. Bind services within the allocation and use the site's supported access method for interactive inspection.

**Outputs:** `docs/compatibility.md`, environment lock/container identity, supported allocation configurations, single-GPU smoke log, two-worker handoff smoke log.

**Gate:** Both workers produce correct streamed continuation using the pinned model and compatible cache configuration. Mixed GPUs are supported only after the real mixed-pair test succeeds.

## 3. Playbook: trustworthy telemetry and request logging

**Trigger:** Before any latency or goodput claim.

**Required request columns**

`run_id`, `request_id`, `trace_id`, `seed`, `class`, `scheduled_arrival`, `dispatch_time`, `gateway_accept_time`, `first_token_time`, `last_token_time`, `stream_end_time`, `prompt_tokens`, `requested_output_cap`, `predicted_output_tokens`, `actual_output_tokens`, `ttft_slo_ms`, `tpot_slo_ms`, `status`, `rejection_reason`, `retry_count`, `policy`, `plan_mode`, `prefill_worker`, `decode_worker`, `queue_ms`, `prefill_ms`, `transfer_visible_stall_ms`, `transfer_elapsed_ms`, `transfer_bytes`, `predicted_costs`, `cache_reserved_bytes`, `exception_type`.

Fields unavailable from the engine must be null with an explanation, not guessed. Distinguish measured GPU event durations from client wall time and predictor estimates. Cross-node wall timestamps do not establish a trustworthy phase duration without clock treatment; use local timers and correlation IDs.

**Telemetry columns:** worker/GPU/node ID, monotonic sample offset, utilization, physical memory used/total, engine cache used/free/reserved blocks, running/queued sequences and token work, power if available, health and telemetry age. Sample initially at 200 ms; compare against 1 s if overhead is measurable.

**Rules**

- Use open-loop arrivals: the next arrival cannot depend on the previous response completing. Use bounded client concurrency high enough to maintain intended dispatch and record dispatch lag.
- Validate the load generator can sustain at least 2x the maximum planned arrival rate against a dummy endpoint. If it cannot, fix it or label the run client-limited.
- SSE events can contain multiple tokens. Validate tokenizer accounting, buffering and first-token handling. Report request TPOT separately from engine-token ITL and client chunk gaps.
- Keep unique request IDs through retries. Count one offered logical request once and charge all wasted compute/transfer activity to the run.
- Reconcile: offered = successful + rejected + failed + timed_out + cancelled + incomplete. Do not drop incomplete requests from summaries.
- Export aggregate Prometheus counters/histograms; do not use request IDs as metric labels. Raw logs own the detailed evidence.

**Tests:** known timestamp sequences, one-token output, empty response, multi-token chunks, lost connection, bounded drain, and deliberate reject/timeout cases. Independently recompute metrics from a small fixture.

## 4. Playbook: hardware profiling and cost modeling

**Trigger:** Before adaptive scheduling.

### Profiling matrix

| Quantity | Initial axes | Required controls |
| --- | --- | --- |
| Prefill duration | Prompt 128, 512, 2048, 8192, 16384 tokens; concurrent sequences 1, 2, 4, 8 | GPU/model fixed per profile; log actual batching and engine token budget |
| Decode TPOT | Context 512, 2048, 8192, 16384; active sequences 1, 2, 4, 8; output 128/512 | Distinguish decode-only microprofile from serving under prefill interference |
| Transfer | Payload 16, 64, 256, 1024 MiB plus model-derived sizes that fit | Both directions; same-node/cross-node; actual chosen transport |
| Connector overhead | Actual cache handoff for short/long prompts | Metadata, block mapping, staging and acknowledgment included |
| Queue behavior | Several offered loads and prompt/output mixes | Log queued tokens and active decode work, not only request count |

Use the feasible subset; do not force unsafe cells. For continuous batching, concurrency is a control and realized batch size is an observation. Record max batched tokens, max sequences, chunked prefill and cache settings.

Warm up until timing stabilizes. For a first profile aim for >=30 measured repetitions per cell and >=500 real-LLM observations per GPU type, including under-load measurements. Increase only where variance or coverage warrants it. Synchronize GPU timing correctly; avoid timing only asynchronous launch overhead.

Estimate transfer as setup cost plus bytes/effective bandwidth, then measure prediction residuals. Synthetic tensors establish transport performance, not proof of cache handoff or end-to-end inference.

### Predictors to compare

| ID | Method | Purpose |
| --- | --- | --- |
| P0 | Lookup with interpolation | Transparent non-learned reference |
| P1 | Linear/piecewise regression | Low-cost structured cost model |
| P2 | XGBoost component predictors | Nonlinear phase/transfer prediction |
| P3 | Selected predictor plus empirical uncertainty margin | Conservative SLO-aware decisions |
| P4 | Oracle with actual future length/timing | Diagnostic ceiling only; never deploy or headline |

Features: GPU type, prompt tokens, arrival-time output estimate/cap, active sequences, queued token work, context lengths, available/reserved cache, engine configuration, topology and payload size. Training labels are measured prefill, decode-step/request TPOT, transfer and queue delays where observed. Do not use realized completion duration or actual output length as inference-time features.

Split by whole profiling runs and length/load regions. Reserve held-out hardware sessions and workload seeds; do not randomly split neighboring rows from the same run into train/test. Train/tune with separate validation. Report component-wise median/p90 percentage errors and absolute ms errors, especially underprediction near SLO boundaries.

Output length alternatives: configured cap, historical median by class, historical p90 by class. The trace's future realized length must remain hidden from the scheduler. Compare sensitivity to +/-25% and +/-50% length-estimation error.

**Selection:** A learned predictor is adopted only if held-out prediction and validation scheduling outcomes improve enough to justify its overhead. Lookup is an acceptable final winner.

## 5. Playbook: real KV handoff and memory correctness

**Trigger:** Implementing the disaggregated path or cache-related extension.

**Core ownership flow**

1. Validate model/cache compatibility and feasible producer/consumer roles.
2. Reserve rounded destination blocks for prompt plus a conservative generation-growth budget.
3. Run prefill and obtain the connector's cache handle plus any generated first-token state.
4. Transfer with a deadline and explicit acknowledgment; account for source/destination overlap and staging buffers.
5. Decode with the correct position IDs, token history, sampling state and first-token semantics.
6. Release the source only after the protocol permits it. Retain/pin all live destination blocks until completion/cancellation.
7. Release reservations on every terminal path and log failures. Retries must be idempotent at the logical-request layer.

Test >=1000 cases across short/long prompts and outputs, both directions, cancellation and transfer failure. Use greedy generation first. Compare against colocated outputs and a no-transfer numerical-control run across GPU types. All systematic cache-related divergence must block acceptance; the >=99.9% token-agreement goal is not permission to ignore corruption. If heterogeneous numerical behavior prevents bitwise agreement, report it and use logit tolerances defined before the test.

Do not use a shared-filesystem cache example as evidence of high-performance GPU transfer. Do not manipulate private cache blocks with an unversioned patch. If an engine patch is necessary, isolate it, pin its commit and add integration tests.

**Memory-pressure extension candidates:** route elsewhere, delay with a bound, reject, CPU offload, or recompute. A metadata eviction table alone cannot offload live vLLM cache. Compare CPU offload and recompute only if the adapter implements valid semantics. Prefix eviction must exclude pinned active state.

## 6. Playbook: baseline and adaptive policy implementation

Implement the following policy family before the final comparison:

| ID | Policy | What it isolates |
| --- | --- | --- |
| B0 | Colocated round-robin over feasible workers | Simple placement |
| B1 | Colocated least queued requests | Queue-count signal |
| B2 | Colocated predicted remaining token work | Strong workload-aware baseline |
| B3 | Best tuned colocated vLLM with chunked prefill | Strong serving baseline with interference mitigation |
| B4 | Static disaggregation, each feasible fixed role split and orientation | Benefit/cost of fixed PD placement |
| B5 | Selective PD using queues and phase profiles but zero transfer penalty | Value of explicit data-movement modeling |
| H0 | Transfer-aware selective PD with lookup cost model | Core proposed controller |
| H1 | H0 with selected learned predictor | Value of learned latency prediction |
| H2 | H1/H0 with uncertainty guard | Robustness to prediction error |

B0-B3 share the same available colocated pool. For B4-H2 use feasible deployed role layouts; account for every allocated GPU and any separate colocated pool. Compare the best feasible baseline deployment against the adaptive deployment at equal total resources. Also run matched-role-layout comparisons to isolate routing from deployment differences. On a two-GPU fixed-role implementation, acknowledge that some colocated choices are unavailable instead of inventing role switching.

Adaptive policy contract:

- Enumerate supported colocated and PD candidates.
- Predict first-token timing according to the connector's first-token protocol, output token pace, transfer cost and reservation feasibility.
- Prefer the candidate with greatest predicted probability of satisfying both SLOs; use a clear deterministic tie-breaker.
- Require a positive benefit margin before paying migration cost. If using point estimates, reject plans with predicted SLO violations plus the selected safety margin.
- Fall back to a compatible feasible colocated route if profiles are missing or telemetry is stale. Reject explicitly if no valid fallback exists.
- Use the engine's existing continuous batching. Initially tune engine batch/token budgets; do not introduce a second outer batch queue without a separate hypothesis and experiment.

Queue length, utilization and predicted remaining work should be separate ablation signals. GPU utilization alone is not a completion-time estimator.

## 7. Workload design

Generate seeded tokenized prompts with exact validated lengths and feasible context limits. Start with synthetic fixed lengths to isolate mechanisms, then use a fixed versioned public or locally provided text corpus to test natural early stopping and length uncertainty. Record hashes and licensing. Avoid meaningful content quality claims from repetitive synthetic prompts.

Use the four workload classes and initial SLOs in `AGENTS.md`. Vary lengths around the class center and add held-out lengths (for example 1024, 4096 and 12288 when supported). Run a variable-output track where the scheduler sees only a cap/estimate; fixed-output runs isolate costs and must be labeled.

| Trace ID | Traffic | Composition / purpose |
| --- | --- | --- |
| W0 | Constant low load | No-transfer-overhead regression control; 70% chat, 30% generation |
| W1 | Stationary Poisson | 25% each class; main mixed-load curve |
| W2 | Bursty open loop | 0.5x baseline capacity for 90 s, 1.5x for 30 s, repeat; 50% chat, 50% RAG |
| W3 | Composition shift | 60 s each: 80% chat/20% RAG; 20% chat/80% RAG; 60% generation/40% document QA |
| W4 | Long-prompt pressure | 70% document QA, 30% RAG; near cache capacity |
| W5 | Decode pressure | 80% generation, 20% chat; output-length uncertainty |
| W6 | Prefix-locality extension | Hot-prefix distribution and popularity shift; caching equally enabled across comparisons |
| W7 | Worker failure extension | Kill one project-owned worker at a predeclared offset; retry/recovery accounting |

Scale bursts using a capacity measured on validation for one fixed reference baseline; keep the resulting numeric rates identical for every test policy. Do not scale offered load independently by each policy's capacity.

Core load sweep: 0.25, 0.50, 0.75, 1.00, 1.25 and 1.50 times reference capacity. Add points around the 95% compliance crossover with <=5% relative rate resolution. Record actual offered and dispatched rates. The 95% attainment goal applies below capacity; overload stress intentionally measures degradation and rejects.

## 8. Experiment matrix: search before choosing the final design

### Hypotheses and required evidence

| Experiment | Question / comparison | Primary evidence | Decision rule |
| --- | --- | --- | --- |
| E0: feasibility and profiling | Which actual GPUs/topologies and engine roles are usable? | Compatibility, phase curves, transfer curves | Exclude invalid configurations; no performance claims from a failed handoff |
| E1: colocation tuning | Which chunked-prefill/token-budget configuration is strongest? | Goodput, TTFT, TPOT, memory across validation loads | Retain strongest valid colocated baseline, not defaults |
| E2: static PD crossover | When do both PD orientations/splits beat colocation? | Break-even surface over prompt/output length and topology | Identify where movement pays and where it loses |
| E3: policy ladder | B0-B5 versus H0-H2 | Goodput and SLO attainment on identical traces | Keep policies on the validation Pareto frontier |
| E4: predictor comparison | Lookup vs regression vs XGBoost | Held-out error, underprediction, decision cost, validation goodput | Choose on scheduling value; accuracy alone is insufficient |
| E5: migration guard | How much benefit should be required before transfer? | Goodput, bytes/offered request, low-load regressions | Select guard without degrading transfer-unfavorable controls >5% |
| E6: batching sensitivity | Do gains persist across strong engine tuning? | Token-budget sweep, TTFT/TPOT, realized batches | Compare equally tuned baseline and adaptive settings |
| E7: memory/admission | Which safety margin and growth estimate avoid OOM without excess rejection? | OOMs, reservations, offered compliance, rejection | Eliminate incorrect configurations first; maximize offered goodput |
| E8: uncertainty robustness | What if output or runtime estimates are wrong/stale? | Perturbed-estimate results and per-class misses | Choose uncertainty guard if useful; document instability |
| E9: ablations | Which component actually provides the gains? | Full policy vs each disabled component | Retain useful components; explain ineffective ones |
| E10: held-out final evaluation | Does the frozen design generalize? | New seeds, session, lengths and traffic mixes; paired intervals | Report all predeclared cells; no retuning on test results |
| E11: extensions | Are prefix/offload/failure/role changes independently valuable? | Separate controlled comparisons | Include only after capability and correctness gates |

### Parameter candidates

| Parameter | Candidate values / guard |
| --- | --- |
| Engine max batched tokens | 2048, 4096, 8192, 16384 where model/version support them |
| Engine max sequences | 4, 8, 16, 32 where memory permits |
| Chunked prefill | Enabled; disabled as diagnostic; retain correct context handling |
| Migration required predicted benefit | 0%, 10%, 20%, 30% beyond modeled transfer cost |
| Prediction uncertainty multiplier | 1.0, 1.1, 1.2, 1.3; apply per-component residual calibration |
| Device memory headroom | 10%, 15%, 20%; test 5% only after no-OOM validation |
| Output estimate | Request cap, class median, class p90 |
| Telemetry stale cutoff | 500 ms, 1000 ms, 2000 ms |
| Admission | Identical bounded rule for policy study; deadline-aware variant as separate experiment |
| Prefix cache | Off for core cache-movement isolation; on equally for prefix experiments |
| Slow adjustment interval (extension) | 5, 15, 30 s with minimum dwell and measurable reload cost |

Do not run the full Cartesian product. Parameter choices are candidates, not requirements to adopt every technology.

### Staged search and budget control

1. **Pilot:** Determine feasible GPUs, model size, SLOs, trace replay capacity and transfer sizes. Freeze metrics. Aim for short 60-180 s runs rather than lengthy unstable tests.
2. **Baseline calibration:** Tune B3/B4 on validation only; measure reference capacity and freeze absolute loads. Predeclare the strongest-baseline comparator and challenge/control cells.
3. **Screening:** Evaluate each feasible policy on W1-W3 at two informative load levels, two validation seeds and 180 s measurement windows. Remove clearly dominated candidates. Do not draw final statistical conclusions from screening.
4. **Focused tuning:** Use coordinate search or at most 24 seeded random configurations for the surviving adaptive policy, with an equal stated evaluation budget for the major baseline families. Tune predictor, transfer guard and memory margin in stages; avoid combining several untested changes.
5. **Validation selection:** Evaluate the top three configurations across all core W0-W5 using three validation seeds. Select one final adaptive configuration and the baseline configurations. Save the frozen manifest and hashes.
6. **Held-out confirmation:** Compare the frozen winner with strong frozen baselines at the predeclared cells, using >=5 paired seeds, 30-60 s stable warmup, >=300 s measurement windows and a fixed drain cap (initially 120 s). Longer-output traces may require a revised cap established in the pilot. Aim for >=2000 offered requests per final cell across repeats; lengthen low-rate cells if feasible. Make p99 claims only with adequate completed samples (prefer >=10000); otherwise label p99 exploratory.
7. **Ablations and stress:** Remove one component at a time from the frozen design. Run estimation error, topology, overload and extension tests without changing the primary winner.

Publish a run-budget estimate before large sweeps: `GPU-hours = sum(allocated_GPUs * allocation_seconds / 3600)`, including startup/warmup/drain. Begin with a <=20 GPU-hour screening budget and revise the confirmation budget based on measured capacity and available allocation. Batch comparisons within an allocation when possible. These are planning budgets, not requests to reserve hardware now.

## 9. Selecting the best design

The best design is the one supported by held-out results for the stated service objectives, not the most complex algorithm.

### Validation selection order

1. Disqualify cache-incorrect, memory-incorrect, incompatible or future-information-leaking configurations.
2. Enforce the below-capacity compliance/rejection targets, per-class TPOT regression limit, and <=5% goodput regression on transfer-unfavorable controls.
3. Maximize geometric-mean token-goodput/GPU ratio across the predeclared challenge cells. Report absolute values and per-cell ratios too. Cells with zero baseline goodput have undefined ratios: use an absolute comparison and list them separately; never add an arbitrary epsilon.
4. Use maximum sustainable compliant arrival rate as the secondary objective.
5. If validation gains are within 3% and uncertain, choose the simpler policy with lower decision/transfer overhead.
6. Freeze the chosen configuration before test runs. If no candidate passes, report the best Pareto tradeoffs and which gates failed; do not fabricate a winner meeting every target.

### Statistics and fairness

- Pair policies by identical trace seed and, where possible, allocation/session. Randomize policy order within each block to reduce time/thermal bias. Reset or equivalently warm caches between policies.
- Report individual run values, median and variation. Bootstrap paired run-level ratios/differences for 95% confidence intervals; use block bootstrap for temporally correlated within-run latency. Requests within a run are not independent experiment repetitions.
- No test-set hyperparameter search. A new design informed by test failures requires a newly designated held-out evaluation.
- Do not compare four adaptive GPUs with two baseline GPUs. Normalize by all allocated GPUs and also report node/CPU resource costs.
- Strong colocated vLLM may win. Static PD may win when the mix is stable. Explain the region where each strategy is appropriate.
- Required ablations: transfer cost removed, output estimate replaced by cap, predictor replaced by lookup, uncertainty margin removed, queue work replaced by request count. Keep memory correctness guards enabled; memory safety is not an optional ablation.
- Admission improvement must show offered goodput and rejected fraction. High accepted-only compliance is not enough.
- Prefix hit rate is a conditional diagnostic, not a mandatory score: define cached prefix tokens reused / eligible prefix tokens. Enable caching equally when comparing caching-aware policies.

## 10. Required figures and result tables

1. Token goodput/GPU versus offered arrival rate, with confidence intervals.
2. Offered SLO attainment and rejected fraction versus load, per class and overall.
3. p50/p95 TTFT and request TPOT versus load, per workload class.
4. Time series during W2/W3: arrivals, goodput, queue token work, cache occupancy and placement decisions. Align all on run offsets.
5. Break-even heatmap: prompt/output lengths versus colocation/PD winner for each measured topology; distinguish measured cells from interpolated predictions.
6. Actual versus predicted phase/transfer latency, with underprediction highlighted.
7. Ablation table: goodput, TTFT, TPOT, bytes/offered request, decision time and rejection.
8. Optional failure/offload table with availability, retries, recovery, useful and wasted transfer bytes.

All figures must be regenerated from raw artifacts by a committed script. Caption every figure with hardware, model/revision, workload, SLOs, sample count and comparison baseline. Never draw a performance curve with hypothetical values in the final evidence report.

## 11. Intended CLI and reproducibility contract

Implement and document these operations; the examples are the target interface, not existing commands:

```bash
uv run heterokv doctor --config configs/experiments/pilot.yaml
uv run heterokv profile --config configs/experiments/profile.yaml
uv run heterokv trace generate --config configs/workloads/mixed.yaml --seed 101
uv run heterokv serve --config configs/experiments/colocated.yaml
uv run heterokv benchmark --config configs/experiments/validation.yaml
uv run heterokv sweep --config configs/experiments/search.yaml
uv run heterokv evaluate --manifest results/manifests/frozen-test.json
uv run heterokv summarize --runs results/runs --out results/summaries
uv run heterokv plot --input results/summaries --out results/figures
uv run pytest tests/unit
uv run ruff check .
```

Use configuration files for multiline settings. Provide Slurm launch scripts with configurable account, GPU type/count, nodes, memory, CPUs, wall time, ports and model-cache location. Validate actual Slurm syntax and binding on CARC; mark untested templates.

Each run writes a unique immutable manifest, raw request/telemetry Parquet and an outcome summary. Hash input traces and frozen configs. Preserve unsuccessful runs and exclusions with reasons. A report must not silently overwrite prior evidence.

## 12. Definition of done

- [ ] A pinned real model executes on the supported GPUs inside valid allocations.
- [ ] Colocated and PD streaming paths pass continuation/ownership tests.
- [ ] Heterogeneous runs use simultaneously allocated different physical GPU types.
- [ ] Strong baselines are tuned with a recorded comparable budget.
- [ ] Hardware profiles and predictors use measured data without leakage.
- [ ] Open-loop traces cover low-load, mixed, bursty, shift, prompt-pressure and decode-pressure cases.
- [ ] Offered-request outcomes reconcile; no dropped rejects/timeouts.
- [ ] Final policy is selected on validation and frozen before >=5 paired held-out repetitions.
- [ ] Main figures, confidence intervals and ablations regenerate from raw artifacts.
- [ ] Every target is labeled achieved, missed, blocked or not attempted, with evidence.
- [ ] Final README states where the method helps and where it loses.
- [ ] Optional features are claimed only when independently implemented and evaluated.

The core project can be complete even if the proposed performance gains are not achieved. A correct implementation with a rigorous negative result and measured break-even conditions is a defensible systems project.

## 13. Playbook: public dataset preparation and replay

**Trigger:** Before claiming realism or final generalization.

### Dataset roles

Use [Azure 2023](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md) for primary development/validation, [BurstGPT](https://github.com/HPMLL/BurstGPT) for burst robustness with disjoint confirmation windows, and [Azure 2024](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md) as the external held-out campaign. Keep the synthetic grid: public traces may not sufficiently cover long contexts or unfavorable transfers.

1. Download a pinned release or immutable asset, record source URL, retrieval date, license, paper attribution, SHA-256 and original schema in `datasets/registry.json`.
2. Inspect timestamp units, ordering, token-count ranges, nulls, duplicates, session identifiers and historical zero-output failures. Generate an eligibility report before comparing policies.
3. Preserve raw input unchanged. Normalize eligible requests into a common schema with source row ID, source dataset/window/session, relative arrival seconds, prescribed input/output counts and eligibility reason. Never silently clamp long contexts; report exclusion or a separate transformed track.
4. Pin tokenizer/model revision and validate actual input tokens after templates/special tokens are applied. Record the mapping from original token counts to actual served counts.
5. Choose ordinary/peak/shift windows using predeclared arrival and length statistics. Start with contiguous 5-15 minute segments. Use disjoint time/session splits; choose held-out IDs without inspecting policy outcomes.
6. Match empirical distributions: arrivals, interarrival gaps, prompt/output lengths and their correlation. Plot raw versus normalized/replayed distributions. A row sampler that discards arrival structure is not faithful trace replay.
7. Scale arrival times using `replay_offset = (source_time - window_start) / speedup`; apply the same speedup for every policy. Record dispatch lag. Speedup changes offered load and time correlation, so label it and show the achieved rate.
8. Generate varied prompt token sequences with deterministic seeds. Identical padding can create artificial shared prefixes; disable prefix caching for the core isolation track or prevent accidental prefix overlap. Session IDs alone do not reveal actual shared prefix text.
9. In controlled-length mode, force the prescribed number of output tokens only if the backend supports it; record EOS handling and verify actual counts. In natural-generation mode, preserve ordinary stop behavior and use only cap/arrival-time estimates in scheduling.
10. For BurstGPT releases containing historical source failures, keep all source rows in the ingestion report, define the performance-replay eligible set in advance, and separately log excluded counts. All failures generated by our own replay remain in the offered-request denominator.

### Measurement cross-check

Evaluate pinned [NVIDIA AIPerf](https://docs.nvidia.com/aiperf/dev/tutorials/datasets-inputs/profile-with-burst-gpt-traces) as an independent streaming/replay client. It supports a BurstGPT input mode and fixed-schedule replay. Verify actual generated lengths, scheduling units and counters on a small fixture. vLLM's serving benchmark is another cross-check for supported prompt datasets; exact trace replay may still need the custom generator.

Use a mock streaming endpoint with known delays to compare independent-client timing against the project's metric calculator. Differences beyond 5% or 5 ms, whichever is larger, require an explained timestamp/chunking convention before performance reporting. Make each definition explicit instead of choosing whichever tool gives a larger gain.

**Outputs:** dataset registry, eligibility/split reports, normalized immutable traces, prompt-construction manifest, replay-fidelity figures and independent-client validation.

**Gate:** An identical replay manifest produces identical scheduled requests and prompt construction across policies; empirical dispatch matches the intended workload within the declared tolerance. Held-out Azure 2024 data has not been used to tune policy parameters or predictor weights.

## 14. Playbook: comprehensive experiment logging

**Trigger:** Every profile, trial, search candidate, correctness run, GPU benchmark and final evaluation.

### Before starting a run

Create an experiment record with the following fields:

```yaml
experiment_id: E5-transfer-guard
stage: validation
hypothesis: A transfer-benefit margin reduces unhelpful handoffs under bursts.
independent_variable: migration_benefit_margin
candidate_values: [0.0, 0.1, 0.2, 0.3]
comparator: strongest-frozen-feasible-baseline
controls: [model, tokenizer, GPU_pool, trace, SLOs, engine_settings]
primary_metric: compliant_output_tokens_per_second_per_allocated_GPU
secondary_metrics: [offered_SLO_attainment, p95_TTFT, p95_TPOT, rejection, transfer_bytes]
selection_rule: validation-only constrained goodput improvement
seeds: [101, 102]
status: PLANNED
```

This is an illustrative declaration, not an executed experiment. Final held-out seeds must be different from validation seeds.

Allocate a unique `run_id`, link `experiment_id`, and write a manifest before launch. Store resolved configuration, policy class/version, predictor artifact/hash, code commit plus dirty diff if applicable, command/arguments, environment lock, container digest, driver/CUDA/engine versions, model revision, allocation topology/GPU UUIDs, Slurm identity, dataset/trace hash, prompt mode, rates, SLOs, seed, warmup/measurement/drain and timeout settings.

Use the same append-only run states as `AGENTS.md`: `PLANNED`, `RUNNING`, `COMPLETED`, `FAILED`, `ABORTED`, `INVALID`, and `BLOCKED`. Emit transition events with a timestamp and reason; retain previous states as history.

### During execution

Write structured events with sequence number, run/request ID, local monotonic offset, node/worker ID, event type and payload. Required events:

- request arrival, admission/rejection and terminal outcome;
- candidate plans, selected plan, rejection/selection reason and prediction version;
- predicted prefill/queue/transfer/TPOT costs and memory feasibility for each sampled decision;
- block reservation/release, transfer start/ack/failure and cache ownership change;
- telemetry-stale fallback, worker health change, timeout, cancellation and retry;
- configuration change, overload episode, OOM and process exit.

For high-rate runs, log selected-plan/outcome events for every request and candidate-detail events for a predeclared deterministic sample. Record the sampling rate and measure logging overhead. Core request outcome logging cannot be sampled away. Buffer I/O off the scheduling path and report dropped debug-event counters; never silently drop outcome records.

Keep stdout/stderr and structured exception traces. Sample GPU telemetry at the frozen interval. Do not log tokens/prompts by default. Collect all node outputs into the run directory without relying on inter-node wall-clock subtraction.

### After execution

1. Persist terminal state and exit reason even after timeout or nonzero exit. Use cleanup/finalization handlers where possible; reconcile unfinished runs after crashes.
2. Reconcile offered outcomes and worker reservation counts. Verify cache cleanup, GPU identity, dispatch fidelity, intended engine settings and sufficient samples.
3. Mark `INVALID` only for a documented protocol/measurement violation; include the evidence and exclusion reason. OOMs or policy failures are valid negative findings unless the declared experiment was incorrectly executed.
4. Derive summaries from raw logs. Preserve original measurements and keep analysis-code version/hash beside each derived result.
5. Calculate actual GPU-hours, CPU/load-client overhead, transfer activity and wasted retry work.
6. Append findings to the ledger and readable experiment log: hypothesis supported/refuted/inconclusive; absolute metrics; comparator; intervals; mechanism; regressions; next decision.
7. Hash the final artifact bundle. Never reuse a run ID or overwrite the result of a previous attempt.

### Human-readable experiment log entry

```markdown
### E5 / <run IDs> — Transfer-benefit margin
- Hypothesis and predeclared primary metric:
- Dataset/window, load, GPU topology and model:
- Configurations and comparator:
- Raw artifacts and analysis command:
- Results, paired confidence intervals and per-class outcomes:
- Failures, excluded runs and reasons:
- Interpretation: supported / refuted / inconclusive:
- Decision: select / reject / tie / blocked / defer:
- Next experiment and remaining uncertainty:
```

Fill values only after measurement. Preserve unsuccessful approaches and tuning attempts. The log must let another engineer reproduce both the selected design and the reasons alternatives were rejected.

## 15. Final campaign and evidence checklist

Run staged search within the budget in section 8, rather than every combination. The minimum evidence campaign is:

| Campaign | Approaches / factors | Required record |
| --- | --- | --- |
| Correctness | Colocated vs real handoff, both transfer directions, cancel/failure | Token/ownership validation, run artifacts, capability gates |
| Hardware | Each feasible GPU type and actual link topology | Phase and transfer profiles with repetitions |
| Baselines | B0-B4; strong chunked-prefill and static-role tuning | All settings tried, validation score, tuning budget |
| Adaptive routing | Transfer-unaware B5 vs H0; H0 vs best baselines | Candidate decisions, transfer tax, goodput and regression controls |
| Predictors | Lookup, linear, XGBoost, uncertainty guard | Grouped splits, held-out prediction error, scheduling effect and overhead |
| Tuning | Token budget, sequences, margin, memory headroom, output estimate | Candidate configuration ledger, invalid/failed attempts, chosen settings |
| Queue policy, capability dependent | FIFO, EDF, shortest predicted remaining work at the same routing configuration | Proof the adapter changes ordering; starvation/per-class impact; otherwise blocked |
| Public traces | Azure 2023, BurstGPT, untouched Azure 2024 | Dataset/replay identity and per-source results |
| Robustness | Low load, overload, bursts, shifts, inaccurate length/cost estimates | Gains and losses; all rejects/timeouts; confidence intervals |
| Ablations | Remove transfer cost, learned predictor, uncertainty or token-work queue estimate | Change one component; preserve safety and resources |
| Extensions | Prefix reuse, CPU offload/recompute, worker fault or slow role control | Independent capability tests and separate comparisons |

Distinguish the two questions: matched-layout comparisons explain the algorithm; equal-total-GPU best-deployment comparisons measure practical service value. Record when fixed producer/consumer roles prevent a supposedly adaptive route from running. A policy name without a working execution path is not a benchmark participant.

Finalize `docs/final-selection.md` with:

- the chosen design and frozen manifest;
- every candidate considered, including simple methods and unsuccessful trials;
- validation selection evidence and equal-resource baseline details;
- held-out results with five or more paired repetitions and confidence intervals;
- per-dataset/per-class results, transfer-unfavorable controls and worst regressions;
- achieved/missed/blocked targets, limitations and hardware/compute costs;
- exact commands to regenerate metrics and figures from raw logs.

Update the README and any resume bullets only from this report. Do not claim a 20% gain because it was a target. If improvement is uncertain, say so; if colocation is the strongest design on the measured topology, publish that conclusion.

Additional intended CLI operations to implement and document:

```bash
uv run heterokv dataset prepare --config configs/datasets/azure2023.yaml
uv run heterokv dataset validate --registry datasets/registry.json
uv run heterokv replay --manifest configs/experiments/public-trace-validation.yaml
uv run heterokv runs validate --runs results/runs
uv run heterokv report --registry experiments/registry.yaml --ledger experiments/ledger.jsonl
```

These remain interface proposals until implemented and verified. Final acceptance requires an experiment registry, complete run ledger, public-trace fidelity validation, frozen selection record and regenerable results; a dashboard or a single favorable screenshot is insufficient.
