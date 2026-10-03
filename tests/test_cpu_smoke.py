"""CPU-runnable smoke tests for the xDiT HTTP service and core components.

These tests intentionally avoid CUDA, Ray, and multi-GB model weights so that
the interactive-QA path of the service can be exercised end-to-end on a
CPU-only machine (such as CI). They cover:

1. ``import xfuser`` succeeds on CPU-only torch (regression guard for the
   ``envs.py`` GPU-gating fix).
2. The core ``CacheManager`` (naive + sequence-parallel cache update) works
   on CPU tensors.
3. The ``xFuserArgs`` / config dataclasses construct correctly on CPU.
4. The FastAPI HTTP service (``entrypoints.launch``) is importable on CPU and
   the ``MockEngine`` produces a deterministic PNG via ``/generate``.

Run with::

    pytest tests/test_cpu_smoke.py
"""

import base64
import io
import os

import pytest
import torch
from fastapi.testclient import TestClient

# ``xfuser`` must import successfully even when torch has no CUDA support. This
# is a regression test for the envs.py fix that previously crashed on
# ``torch.cuda.get_device_name`` during import.
import xfuser  # noqa: F401


def test_xfuser_imports_on_cpu():
    """The package must import on a CPU-only torch build."""
    import xfuser as xf

    assert xf is not None
    # The envs checker should have set flash_attn availability to False on CPU.
    import xfuser.envs as envs

    info = envs.PACKAGES_CHECKER.get_packages_info()
    assert info["has_flash_attn"] is False


def test_cache_manager_naive_cache_on_cpu():
    """CacheManager naive-cache path works on CPU tensors."""
    from xfuser.core.cache_manager.cache_manager import (
        CacheManager,
        get_cache_manager,
    )

    mgr = CacheManager()

    class DummyLayer:
        pass

    layer = DummyLayer()
    mgr.register_cache_entry(layer, layer_type="attn", cache_type="naive_cache")
    kv = torch.randn(2, 8, 4)
    out = mgr.update_and_get_kv_cache(kv, layer=layer, slice_dim=1, layer_type="attn")
    assert out.shape == (2, 8, 4)
    # Naive cache (no pipeline patch mode) returns the new kv directly.
    assert torch.allclose(out, kv)
    # The singleton cache manager should also be usable.
    assert get_cache_manager() is not None


def test_xfuser_args_and_input_config_construct_on_cpu():
    """xFuserArgs and config dataclasses construct on CPU."""
    from xfuser.config import (
        DataParallelConfig,
        EngineConfig,
        InputConfig,
        ModelConfig,
        ParallelConfig,
        PipeFusionParallelConfig,
        RuntimeConfig,
        SequenceParallelConfig,
        TensorParallelConfig,
    )
    from xfuser.config.config import FastAttnConfig

    input_config = InputConfig(height=512, width=512, prompt="a test prompt")
    assert input_config.height == 512
    assert input_config.width == 512
    assert input_config.batch_size == 1

    parallel_config = ParallelConfig(
        dp_config=DataParallelConfig(dit_parallel_size=1),
        sp_config=SequenceParallelConfig(ulysses_degree=1, ring_degree=1, dit_parallel_size=1),
        pp_config=PipeFusionParallelConfig(dit_parallel_size=1),
        tp_config=TensorParallelConfig(dit_parallel_size=1),
        world_size=1,
        dit_parallel_size=1,
        vae_parallel_size=0,
    )
    assert parallel_config.dit_parallel_size == 1

    engine_config = EngineConfig(
        model_config=ModelConfig(model="PixArt-alpha/PixArt-XL-2-1024-MS"),
        runtime_config=RuntimeConfig(),
        parallel_config=parallel_config,
        fast_attn_config=FastAttnConfig(),
    )
    assert engine_config.model_config.model == "PixArt-alpha/PixArt-XL-2-1024-MS"


def test_launch_module_imports_on_cpu():
    """The HTTP service module must import on a CPU-only host."""
    import entrypoints.launch as launch

    assert launch.app is not None
    paths = {r.path for r in launch.app.routes}
    assert "/" in paths
    assert "/health" in paths
    assert "/generate" in paths


def test_mock_engine_generates_deterministic_image():
    """The MockEngine produces a deterministic, base64-decodable PNG."""
    import asyncio

    from entrypoints.launch import GenerateRequest, MockEngine

    engine = MockEngine()
    request = GenerateRequest(
        prompt="a cute rabbit",
        height=64,
        width=64,
        num_inference_steps=2,
        seed=42,
    )
    result = asyncio.run(engine.generate(request))
    assert result["save_to_disk"] is False
    assert result["output"]
    # The output must be a valid PNG.
    img_bytes = base64.b64decode(result["output"])
    from PIL import Image

    img = Image.open(io.BytesIO(img_bytes))
    assert img.size == (64, 64)
    assert img.format == "PNG"

    # Identical inputs must produce identical outputs (determinism).
    result2 = asyncio.run(engine.generate(request))
    assert result2["output"] == result["output"]


def test_mock_engine_save_to_disk(tmp_path):
    """The MockEngine honors save_disk_path and writes a PNG file."""
    import asyncio

    from entrypoints.launch import GenerateRequest, MockEngine

    engine = MockEngine()
    save_dir = str(tmp_path)
    request = GenerateRequest(
        prompt="a cute dog",
        height=32,
        width=32,
        num_inference_steps=1,
        seed=7,
        save_disk_path=save_dir,
    )
    result = asyncio.run(engine.generate(request))
    assert result["save_to_disk"] is True
    assert os.path.exists(result["output"])
    assert result["output"].endswith(".png")


def test_http_service_end_to_end_with_mock_engine():
    """Drive the FastAPI service end-to-end against the MockEngine.

    This exercises the full interactive-QA path: the ``/`` info endpoint, the
    ``/health`` probe, a successful ``/generate`` POST returning a 2xx, the
    ``save_disk_path`` branch, and input validation (400 on empty prompt).
    """
    import entrypoints.launch as launch

    launch.engine = launch.MockEngine()
    client = TestClient(launch.app)

    # Info endpoint.
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["service"] == "xDiT HTTP Service"
    assert body["mode"] == "mock"

    # Health/readiness probe.
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["mode"] == "mock"

    # A meaningful generation interaction.
    resp = client.post(
        "/generate",
        json={
            "prompt": "a cute rabbit",
            "num_inference_steps": 2,
            "seed": 42,
            "cfg": 7.5,
            "height": 64,
            "width": 64,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["save_to_disk"] is False
    assert data["output"]
    # Output must be a decodable PNG.
    img_bytes = base64.b64decode(data["output"])
    from PIL import Image

    img = Image.open(io.BytesIO(img_bytes))
    assert img.size == (64, 64)

    # Validation: empty prompt must yield 400.
    resp = client.post(
        "/generate",
        json={"prompt": "", "height": 64, "width": 64, "num_inference_steps": 2},
    )
    assert resp.status_code == 400

    # Validation: non-positive dimensions must yield 400.
    resp = client.post(
        "/generate",
        json={"prompt": "x", "height": 0, "width": 64, "num_inference_steps": 2},
    )
    assert resp.status_code == 400


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
