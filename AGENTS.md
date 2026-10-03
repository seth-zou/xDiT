# AGENTS.md

This file provides guidance for autonomous AI agents working on the xDiT codebase.

## Project Overview

xDiT is a scalable inference engine for Diffusion Transformers (DiTs) that runs on
multiple computing devices (multi-GPU / multi-node). It parallelizes DiT inference
via a hybrid of several strategies — PipeFusion, Unified Sequence Parallelism (USP),
CFG Parallel, and Data Parallel — and accelerates single-GPU execution through
`torch.compile`, `onediff`, and caching techniques (TeaCache, First-Block-Cache,
DiTFastAttn).

The published package name is **`xfuser`** (not `xdit`). Import everything from
`xfuser` and launch examples with `torchrun`.

## Repository Layout

```
xfuser/                  # Main Python package (import name: `xfuser`)
  config/                # EngineConfig, InputConfig, xFuserArgs (CLI parsing)
  core/
    distributed/         # parallel_state, group_coordinator, runtime_state
    cache_manager/       # TeaCache / First-Block-Cache runtime caches
    fast_attention/      # DiTFastAttn acceleration
    long_ctx_attention/  # USP / ring / Ulysses attention integration
  model_executor/
    layers/              # Parallel wrappers for attn, conv, embeddings, feedforward
    models/              # Parallel wrappers for transformer backbones (PixArt, Flux, ...)
    pipelines/           # Parallel wrappers for diffusers pipelines + register
    schedulers/          # Parallel scheduler wrappers
    base_wrapper.py      # xFuserBaseWrapper — getattr pass-through base class
  parallel.py            # xDiTParallel — top-level entry point for users
  envs.py                # Runtime environment + package version checks
  logger.py
  ray/                   # Optional Ray-based execution
entrypoints/             # launch.py for the optional HTTP service
examples/                # Per-model example scripts and run.sh drivers
tests/                   # pytest test suite (needs GPU + >=2 ranks)
benchmark/               # Latency / FID benchmarking scripts
docs/                    # Method docs, performance reports, developer guide
docker/                  # Dockerfile for the dev image
setup.py                 # Package metadata + install_requires
pytest.ini               # pytest config (tee-sys capture, verbose, INFO logging)
.pre-commit-config.yaml # ruff (lint) + black (format) via pre-commit
ruff.toml                # ruff linter configuration
```

## Setup & Installation

xDiT requires **Python >= 3.10** and a CUDA-enabled PyTorch build (>= 2.2.0;
`torch>=2.1.0` is the install floor but CUDA Graph + NCCL needs 2.2.0+).

### Install from source (recommended for development)

```bash
pip install -e .
# With diffusers and flash attention (strongly recommended on GPU):
pip install -e ".[diffusers,flash-attn]"
```

### Install from pip

```bash
pip install xfuser
pip install "xfuser[diffusers,flash-attn]"
```

### Key runtime dependencies

- `torch>=2.1.0` (>= 2.2.0 for CUDA Graph with NCCL)
- `accelerate>=0.33.0`, `transformers>=4.39.1`
- `diffusers>=0.31.0` (extras: `diffusers`) — required for the pipeline wrappers.
  Newer models (CogVideoX, Flux) may need a diffusers built from `main`.
- `flash-attn>=2.6.0` (extras: `flash-attn`) — required when `ring_degree > 1`
  for best GPU performance. Without it, xDiT falls back to a PyTorch ring
  attention implementation (useful for NPU compatibility).
- `yunchang>=0.6.0` — self-maintained long-context-attention library for USP.
- `distvae` — parallel VAE support.
- `optimum-quanto` (extras) — needed for `--use_fp8_t5_encoder`.
- `flask` (extras) — needed for the HTTP service in `entrypoints/`.

### Pre-commit hooks

```bash
pip install pre-commit
pre-commit install
```

This runs `ruff` (linter, auto-fixes safe violations) and `black` (formatter,
targeting python3.10) on staged files.

## Build & Test

### Run the test suite

Tests require GPUs and a multi-rank launcher. From the repo root:

```bash
# Single test file (CPU-runnable parts will still need a GPU for full coverage)
pytest tests/

# Distributed tests are launched per-script; see tests/context_parallel/debug_tests.py
# as an example pattern:
python tests/context_parallel/debug_tests.py
```

`pytest.ini` configures verbose output with `--capture=tee-sys`, INFO-level CLI
logging, and `--durations=0`. The distributed adapter tests under
`tests/context_parallel/test_diffusers_adapters.py` pin `CUDA_VISIBLE_DEVICES`
and spawn pytest via subprocess — they require 2+ visible GPUs.

### Run CPU-only smoke tests (no GPU required)

`tests/test_cpu_smoke.py` is a CPU-runnable pytest suite that does not require
CUDA, Ray, or model weights. It guards the `import xfuser` path on CPU-only
torch, exercises the core `CacheManager` and config dataclasses on CPU, and
drives the HTTP service end-to-end against the `MockEngine`. Run it with:

```bash
pytest tests/test_cpu_smoke.py -v
```

This is the fastest way to verify the package and the HTTP service are
interactive-QA ready on a machine without GPUs.

### Run an example

Examples live in `examples/` and are driven by `torchrun`. The product of all
parallel degrees (ulysses × ring × pipefusion × cfg × data) must equal the
number of devices.

```bash
bash examples/run.sh   # edit MODEL_TYPE / N_GPUS / parallel degrees first
```

Or launch directly:

```bash
torchrun --nproc_per_node=8 examples/pixartalpha_example.py \
  --model models/PixArt-XL-2-1024-MS \
  --pipefusion_parallel_degree 2 --ulysses_degree 2 \
  --num_inference_steps 20 --warmup_steps 1 \
  --prompt "A cute dog" --use_cfg_parallel
```

### Docker

A developer image is published as `thufeifeibear/xdit-dev` (see `docker/Dockerfile`).

## Development Workflow

### Architecture: wrapper + register pattern

xDiT does **not** fork diffusers models. It wraps them. Four base wrappers under
`model_executor/` each parallelize a different component:

- `xFuserPipelineBaseWrapper` — diffusion pipelines (`pipelines/`)
- `xFuserModelBaseWrapper` — transformer/unet backbones (`models/`)
- `xFuserLayerBaseWrapper` — low-level layers: attn, conv, embeddings (`layers/`)
- `xFuserSchedulerBaseWrapper` — schedulers (`schedulers/`)

Each base wrapper passes attributes through to the wrapped module via `__getattr__`,
so wrapped code can reuse upstream diffusers logic directly. Parallel behavior is
injected in `__init__` by walking the module tree and replacing submodules with
their parallel wrappers via a **register** lookup.

Registers (`model_executor/{pipelines,models,layers,schedulers}/register.py`) map
an original diffusers/torch class to its xDiT parallel wrapper. Adding a new model
means: write a wrapper subclass of the appropriate base, decorate it with
`@xFuser...Register.register(OriginalClass)`, and the auto-wrapping in the parent
base class picks it up automatically.

### Adding a new model

Follow the step-by-step guides under `docs/developer/adding_models/`:

- `adding_model_cfg.md` — base pipeline wrapper for a new model
- `adding_model_usp.md` / `adding_model_usp.py` — adding USP support
- `adding_model_pipefusion.md` — adding PipeFusion support

See also `docs/developer/The_implement_design_of_xdit_framework.md` for the full
architecture write-up (parallel state, group coordinators, runtime state).

### Runtime state

Distributed setup lives in `core/distributed/`:

- `parallel_state.py` — initializes the process group and communication groups
  (data, cfg, sequence, pipeline, tensor parallel groups). Query ranks/world sizes
  via `get_world_group()`, `get_data_parallel_rank()`, `get_sp_group()`, etc.
- `group_coordinator.py` — wraps communication collectives for a parallel group.
- `runtime_state.py` — holds non-communication runtime metadata (patch partitioning,
  runtime config, input config). Access via `get_runtime_state()`.

All of these are global singletons. Models read metadata from them at runtime; do
not pass distributed state through arguments.

### CLI arguments

All user-facing flags are defined as dataclass fields on `xFuserArgs`
(`config/args.py`) and registered with `FlexibleArgumentParser`, which normalizes
`--kebab-case` and `--snake_case` interchangeably. When adding a flag, add it as a
dataclass field **and** an `add_argument` call in `xFuserArgs.add_cli_args`.

## Conventions

- **Linting**: `ruff` (configured in `ruff.toml`) catches code-quality issues
  (unused imports/variables, undefined names, bare excepts, mutable default
  arguments, f-strings without placeholders, etc.). Run `ruff check .` or
  `ruff check . --fix` before committing. Ruff also runs in pre-commit and CI.
- **Formatting**: `black` (python3.10 target), enforced via pre-commit. Run
  `pre-commit run --all-files` before committing.
- **Python version**: 3.10+. Do not use 3.11+-only syntax.
- **Logging**: use `from xfuser.logger import init_logger; logger = init_logger(__name__)`
  — never `print()` or bare `logging.getLogger()`.
- **Imports**: import from the package root (`from xfuser.config import ...`),
  not via deep relative paths, outside of intra-package `__init__.py` re-exports.
- **Parallel degrees**: the invariant
  `dp_degree × cfg_degree × sp_degree × tp_degree × pp_degree == dit_parallel_size`
  (== world_size by default) is enforced in `ParallelConfig.__post_init__`. New code
  must not break it.
- **Do not** modify wrapped diffusers modules in place; always go through the
  wrapper + register pattern so parallel state stays consistent.
- **`warmup_steps`** must be set when using PipeFusion (serial warmup phase).
  A value of 0 is acceptable for PixArt but may hurt other models.
- Results and `*.mp4` outputs are gitignored; place generated artifacts in
  `results/` (created by `examples/run.sh`).

## Common Tasks

- **Run an existing model with a new parallel config**: edit `examples/run.sh`
  (`MODEL_TYPE`, `N_GPUS`, `PARALLEL_ARGS`, `CFG_ARGS`) and `bash examples/run.sh`.
- **Benchmark**: see `benchmark/run.sh` and `benchmark/single_node_latency_test.py`.
- **Launch the HTTP service**: `python entrypoints/launch.py` (requires the `flask`
  extra); see `docs/developer/Http_Service.md`. For a CPU-only interactive-QA
  mode that needs no GPU, Ray, or model weights, use
  `python entrypoints/launch.py --mock` (the `MockEngine` produces a
  deterministic placeholder PNG through the same request/response contract).
- **Debug a parallel run**: reduce to `--nproc_per_node=1` with all parallel degrees
  at 1 to isolate logic errors from communication issues, then scale up.

## Further Reading

- README.md — supported models matrix, performance reports, quick start
- docs/developer/ — design docs and model integration guides
- docs/methods/ — PipeFusion, USP, hybrid parallel, CFG parallel, parallel VAE
- docs/performance/ — per-model performance reports
