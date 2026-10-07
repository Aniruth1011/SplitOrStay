# HeteroKV

For the current implementation inventory and copyable Mac/remote commands, start with
[completed.md](completed.md).

A research implementation for transfer-aware LLM request placement, with CPU development
on macOS and a separate NVIDIA/Linux serving path. Read [AGENTS.md](AGENTS.md) and
[SKILLS.md](SKILLS.md) for the full evaluation protocol.

**Current scope:** the Mac controller, streaming API, synthetic replay, dataset preparation,
and evidence tools run locally. The vLLM colocated HTTP adapter and Slurm template are
implemented but have not been tested on GPUs. Real NIXL handoff, hardware profiling, tuned
baselines, and the heterogeneous evaluation remain unfinished. No GPU performance gain is claimed.

## Run on your Mac

Use Python 3.13 (3.11–3.14 are declared; only 3.13 has been exercised here). CUDA, PyTorch,
and vLLM are not dependencies of the local package.

```bash
uv sync --python 3.13
uv run heterokv doctor --config configs/experiments/local.yaml
uv run pytest
uv run heterokv benchmark --config configs/experiments/local.yaml
uv run heterokv runs validate --runs results/runs
uv run heterokv summarize --runs results/runs --out results/summaries/local-001
uv run heterokv plot --input results/summaries/local-001 --out results/figures/local-001
uv run heterokv serve --config configs/experiments/local.yaml
```

The server listens at `http://127.0.0.1:8080`, which opens interactive request documentation.
The documentation is also available at `/docs`,
health at `/health`, live server counters at `/metrics`, and the most recent valid saved
benchmark summary at `/results/latest`. The live counters only reflect requests sent to this
server process; a CLI benchmark writes a separate run bundle. `POST /generate` accepts a `request`
object and optional `prompt_token_ids`, and returns SSE. Example body:

```json
{"request":{"request_id":"example-1","arrival_offset":0,"prompt_tokens":16,"output_cap":4,"seed":101}}
```

Mock outputs are numeric placeholders. The Mac GPU is not used to emulate NVIDIA performance.
Every run says `evidence_kind: mock`, reports zero allocated GPUs, and leaves GPU-normalized
goodput null. Figures are labeled by evidence kind. Output directories are exclusive: use a
new summary/figure/campaign directory for each generation.

Network access was unavailable during implementation, so a portable `uv.lock` could not be
resolved and pytest/Ruff could not be installed. CPU tests were executed with `unittest` and
already cached dependencies. On this workspace the verified offline equivalent is:

```bash
python3.13 scripts/cached_python.py unittest discover -v
python3.13 scripts/cached_python.py heterokv.cli benchmark --config configs/experiments/local.yaml
```

This helper needs the existing `.uv-cache` archives and matching Python ABI; it is not a
deployment method. Run `uv sync` and commit its generated lock when network access is available.

## Workloads and evidence

```bash
uv run heterokv trace generate --config configs/workloads/mixed.yaml --seed 101
uv run heterokv dataset prepare --config configs/datasets/example.local.yaml
uv run heterokv dataset validate --registry datasets/registry.json
uv run heterokv sweep --config configs/experiments/search.yaml
uv run heterokv report --registry experiments/registry.yaml --ledger experiments/ledger.jsonl
```

Copy the dataset example and supply the pinned input file/hash, explicit column mapping,
timestamp units, context limit, and predeclared disjoint windows. Public datasets have not
been downloaded. Natural mode keeps source future output counts out of scheduler requests.
Prompt IDs are deterministic varied vocabulary samples; they do not establish answer quality
or realistic prefix sharing. Sweep currently verifies mock policy plumbing only.

Run bundles contain manifests, raw request Parquet, an outcome journal, structured events,
empty explicitly unavailable GPU telemetry, execution logs, summaries, and SHA-256 hashes.
The append-only ledger includes every benchmark attempt. Summaries reconcile all offered
requests, including failed/rejected/cancelled/incomplete outcomes.

## Move to the remote system

Follow [docs/remote.md](docs/remote.md). Keep the controller package independent from the
GPU engine environment; use the same trace and configuration files on both platforms.
The `direct-pilot.example.yaml` configuration can replay real colocated vLLM requests before
hardware phase profiles are available. It requires an allocated remote GPU and has not been
tested on one here. The adaptive real controller still needs measured profiles and connector
validation.
`doctor --config configs/experiments/pilot.yaml` intentionally reports missing remote inputs.
Never use the local mock profiles in a GPU experiment.

See [docs/status.md](docs/status.md) for command and acceptance status,
[docs/architecture.md](docs/architecture.md) for contracts, and
[docs/limitations.md](docs/limitations.md) for unfinished work and measurement restrictions.
