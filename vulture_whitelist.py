"""Whitelist for vulture (dead code detector).

xDiT uses several patterns that vulture cannot follow statically and would
flag as unused code. This whitelist documents each such pattern so vulture
can still surface *genuinely* dead code without drowning the report in
false positives.

Run vulture with:

    vulture xfuser entrypoints vulture_whitelist.py

A finding is "real" if it is NOT explained here. When you intentionally rely
on one of the patterns below, keep the matching whitelist entry so CI stays
green.

Each block below references a name that vulture would otherwise report as
unused. Referencing a name in this whitelist tells vulture "this name is used
somewhere", silencing the false positive at its real definition/import site.
"""

import importlib

# Import through importlib so the whitelist module itself does not introduce
# new "unused import" findings; the imported modules are accessed by name below.
_envs = importlib.import_module("xfuser.envs")
_ring = importlib.import_module("xfuser.core.long_ctx_attention.ring.ring_flash_attn")
_ray_utils = importlib.import_module("xfuser.ray.pipeline.ray_utils")
_pipe_utils = importlib.import_module("xfuser.ray.pipeline.pipeline_utils")

# ---------------------------------------------------------------------------
# 1. PEP 562 lazy module attributes: xfuser/envs.py exposes environment
#    variables (MASTER_ADDR, MASTER_PORT, CUDA_HOME, CUDA_VISIBLE_DEVICES,
#    XDIT_LOGGING_LEVEL) through a module __getattr__ that lazily evaluates
#    lambdas in `environment_variables`. Callers do
#    `from xfuser.envs import MASTER_ADDR`, which vulture cannot see.
# ---------------------------------------------------------------------------
_envs.__getattr__
_envs.__dir__
_envs.MASTER_ADDR
_envs.MASTER_PORT
_envs.CUDA_HOME
_envs.CUDA_VISIBLE_DEVICES
_envs.XDIT_LOGGING_LEVEL

# ---------------------------------------------------------------------------
# 2. Optional-dependency availability checks (try/except ImportError).
#    ring_flash_attn.py imports _flash_attn_forward / pytorch_attn_forward
#    only to validate flash_attn availability and to provide a fallback
#    implementation; the names are dispatched at runtime through yunchang's
#    RingFlashAttnFunc, which vulture cannot follow.
# ---------------------------------------------------------------------------
_ring._flash_attn_forward
_ring.pytorch_attn_forward

# envs.check_long_ctx_attn() imports these from yunchang inside a try/except
# purely to validate availability; they are not referenced afterwards.
LongContextAttentionQKVPacked
ring_flash_attn_func
UlyssesAttention

# ray_utils imports PlacementGroup as a string-quoted type annotation and to
# assert ray is importable.
PlacementGroup

# ---------------------------------------------------------------------------
# 3. Ray worker classes referenced by dotted *strings*.
#    pipeline_utils.py imports DiTWorker/VAEWorker so the worker module is
#    loaded, but dispatches workers via strings passed to RayWorkerWrapper
#    (e.g. "xfuser.ray.worker.worker.DiTWorker"). Vulture cannot link the
#    string to the class, so it flags the import.
# ---------------------------------------------------------------------------
DiTWorker
VAEWorker

# RayDiffusionPipeline is the public entry point of the optional Ray backend;
# it is constructed via its from_pretrained classmethod by callers that import
# it dynamically only when Ray is available, which vulture cannot detect.
_pipe_utils.RayDiffusionPipeline
_pipe_utils.RayDiffusionPipeline.total_workers
