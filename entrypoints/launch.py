import argparse
import base64
import io
import logging
import os
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from xfuser import xFuserArgs

logger = logging.getLogger(__name__)


def _setup_logger():
    """Attach a console handler to the module logger if one is not present."""
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)


# Top-level dependencies that require a GPU are imported lazily so that the
# FastAPI application (and its mock/CPU-only mode) can be constructed and
# exercised on machines without CUDA, Ray, or model weights. This keeps the
# service importable for interactive QA and testing in CPU-only environments.


# Define request model
class GenerateRequest(BaseModel):
    prompt: str
    num_inference_steps: int | None = 50
    seed: int | None = 42
    cfg: float | None = 7.5
    save_disk_path: str | None = None
    height: int | None = 1024
    width: int | None = 1024

    # Add input validation
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "a beautiful landscape",
                "num_inference_steps": 50,
                "seed": 42,
                "cfg": 7.5,
                "height": 1024,
                "width": 1024,
            }
        }


app = FastAPI()

# ``engine`` is populated by the ``__main__`` block before the server starts.
# It is referenced lazily by the route handlers via ``get_engine()`` so that
# the module remains importable (and the FastAPI app constructable) even before
# an engine is bound, which is what the interactive-QA path relies on.
engine = None


def get_engine():
    """Return the active engine, raising an informative error if unset."""
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "xDiT engine is not initialized. Start the service with "
                "`python entrypoints/launch.py --mock` for a CPU-only mock "
                "backend, or with a real `--model_path` for GPU inference."
            ),
        )
    return engine


@app.get("/")
async def root():
    """Service info endpoint for interactive QA discovery."""
    return {
        "service": "xDiT HTTP Service",
        "endpoints": ["/", "/health", "/generate"],
        "mode": getattr(engine, "mode", "uninitialized"),
    }


@app.get("/health")
async def health():
    """Liveness/readiness probe used by interactive QA and deployment checks."""
    return {
        "status": "ok",
        "mode": getattr(engine, "mode", "uninitialized"),
    }


@app.post("/generate")
async def generate_image(request: GenerateRequest):
    try:
        # Add input validation
        if not request.prompt:
            raise HTTPException(
                status_code=400, detail="Prompt cannot be empty"
            )
        if request.height <= 0 or request.width <= 0:
            raise HTTPException(
                status_code=400, detail="Height and width must be positive"
            )
        if request.num_inference_steps <= 0:
            raise HTTPException(
                status_code=400,
                detail="num_inference_steps must be positive",
            )

        result = await get_engine().generate(request)
        return result
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e)) from e


class MockEngine:
    """CPU-only engine that produces a deterministic placeholder image.

    This backend exercises the full HTTP request/response path of the service
    without requiring CUDA, Ray, or multi-GB model weights. It is intended for
    interactive QA, smoke testing, and local development of the service layer.
    A real deployment uses the GPU-backed ``Engine`` instead.
    """

    mode = "mock"

    def __init__(self):
        _setup_logger()
        # PIL is part of the diffusers/torch ecosystem; import lazily so the
        # mock engine does not hard-depend on it at module import time.
        from PIL import Image, ImageDraw, ImageFilter

        self._Image = Image
        self._ImageDraw = ImageDraw
        self._ImageFilter = ImageFilter
        logger.info(
            "MockEngine initialized (CPU-only). No model weights or GPU "
            "required. Use --mock to enable this backend."
        )

    async def generate(self, request: GenerateRequest):
        start_time = time.time()
        # Clamp dimensions to keep the smoke test cheap while still honoring
        # the request shape.
        width = min(max(int(request.width), 1), 1024)
        height = min(max(int(request.height), 1), 1024)
        steps = max(int(request.num_inference_steps), 1)
        seed = int(request.seed) if request.seed is not None else 42

        # Deterministic pseudo-random fill derived from the prompt and seed so
        # that repeated requests with identical inputs yield identical output.
        prompt_hash = 0
        for ch in str(request.prompt):
            prompt_hash = (prompt_hash * 31 + ord(ch)) & 0xFFFFFFFF
        rng_state = (seed * 1103515245 + 12345 + prompt_hash) & 0x7FFFFFFF

        Image, ImageDraw = self._Image, self._ImageDraw
        img = Image.new("RGB", (width, height), color=(24, 24, 32))
        draw = ImageDraw.Draw(img)

        # Draw a few pseudo-random translucent bands to give the placeholder
        # structure that varies with the prompt/seed.
        num_bands = min(steps, 16)
        for _ in range(num_bands):
            rng_state = (rng_state * 1103515245 + 12345) & 0x7FFFFFFF
            r = (rng_state >> 16) & 0xFF
            rng_state = (rng_state * 1103515245 + 12345) & 0x7FFFFFFF
            g = (rng_state >> 16) & 0xFF
            rng_state = (rng_state * 1103515245 + 12345) & 0x7FFFFFFF
            b = (rng_state >> 16) & 0xFF
            x0 = (rng_state % max(width, 1))
            rng_state = (rng_state * 1103515245 + 12345) & 0x7FFFFFFF
            y0 = (rng_state >> 8) % max(height, 1)
            x1 = (x0 + max(1, width // num_bands)) % max(width, 1)
            y1 = (y0 + max(1, height // 2)) % max(height, 1)
            draw.rectangle(
                [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)],
                fill=(r, g, b),
            )

        # Soften the placeholder so it looks like a low-res diffusion output.
        import contextlib

        with contextlib.suppress(Exception):
            # Some Pillow builds/displays do not support GaussianBlur; the
            # placeholder is still valid without it.
            img = img.filter(self._ImageFilter.GaussianBlur(radius=2))

        elapsed_time = time.time() - start_time

        if request.save_disk_path:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"generated_image_{timestamp}.png"
            file_path = os.path.join(request.save_disk_path, filename)
            os.makedirs(request.save_disk_path, exist_ok=True)
            img.save(file_path)
            return {
                "message": "Image generated successfully (mock)",
                "elapsed_time": f"{elapsed_time:.4f} sec",
                "output": file_path,
                "save_to_disk": True,
            }

        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return {
            "message": "Image generated successfully (mock)",
            "elapsed_time": f"{elapsed_time:.4f} sec",
            "output": img_str,
            "save_to_disk": False,
        }


def build_gpu_engine(world_size: int, xfuser_args: xFuserArgs):
    """Construct the real GPU/Ray-backed engine.

    All GPU-only imports (``ray``, ``torch``, the xDiT pipeline wrappers) are
    performed here so they do not crash the module import on CPU-only hosts.
    """
    import ray
    import torch

    from xfuser import (
        xFuserFluxPipeline,
        xFuserHunyuanDiTPipeline,
        xFuserPixArtAlphaPipeline,
        xFuserPixArtSigmaPipeline,
        xFuserStableDiffusion3Pipeline,
    )

    @ray.remote(num_gpus=1)
    class ImageGenerator:
        def __init__(self, xfuser_args: xFuserArgs, rank: int, world_size: int):
            # Set PyTorch distributed environment variables
            os.environ["RANK"] = str(rank)
            os.environ["WORLD_SIZE"] = str(world_size)
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = "29500"

            self.rank = rank
            self.setup_logger()
            self.initialize_model(xfuser_args)

        def setup_logger(self):
            self.logger = logging.getLogger(__name__)
            # Add console handler if not already present
            if not self.logger.handlers:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.INFO)
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)
                self.logger.setLevel(logging.INFO)

        def initialize_model(self, xfuser_args: xFuserArgs):
            # init distributed environment in create_config
            self.engine_config, self.input_config = xfuser_args.create_config()

            model_name = self.engine_config.model_config.model.split("/")[-1]
            pipeline_map = {
                "PixArt-XL-2-1024-MS": xFuserPixArtAlphaPipeline,
                "PixArt-Sigma-XL-2-2K-MS": xFuserPixArtSigmaPipeline,
                "stable-diffusion-3-medium-diffusers": xFuserStableDiffusion3Pipeline,
                "HunyuanDiT-v1.2-Diffusers": xFuserHunyuanDiTPipeline,
                "FLUX.1-schnell": xFuserFluxPipeline,
                "FLUX.1-dev": xFuserFluxPipeline,
            }

            PipelineClass = pipeline_map.get(model_name)
            if PipelineClass is None:
                raise NotImplementedError(
                    f"{model_name} is currently not supported!"
                )

            self.logger.info(
                f"Initializing model {model_name} from {xfuser_args.model}"
            )

            self.pipe = PipelineClass.from_pretrained(
                pretrained_model_name_or_path=xfuser_args.model,
                engine_config=self.engine_config,
                torch_dtype=torch.float16,
            ).to("cuda")

            self.pipe.prepare_run(self.input_config)
            self.logger.info("Model initialization completed")

        def generate(self, request: GenerateRequest):
            try:
                start_time = time.time()
                output = self.pipe(
                    height=request.height,
                    width=request.width,
                    prompt=request.prompt,
                    num_inference_steps=request.num_inference_steps,
                    output_type="pil",
                    generator=torch.Generator(device="cuda").manual_seed(
                        request.seed
                    ),
                    guidance_scale=request.cfg,
                    max_sequence_length=self.input_config.max_sequence_length,
                )
                elapsed_time = time.time() - start_time

                if self.pipe.is_dp_last_group():
                    if request.save_disk_path:
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        filename = f"generated_image_{timestamp}.png"
                        file_path = os.path.join(
                            request.save_disk_path, filename
                        )
                        os.makedirs(request.save_disk_path, exist_ok=True)
                        output.images[0].save(file_path)
                        return {
                            "message": "Image generated successfully",
                            "elapsed_time": f"{elapsed_time:.2f} sec",
                            "output": file_path,
                            "save_to_disk": True,
                        }
                    else:
                        # Convert to base64
                        buffered = io.BytesIO()
                        output.images[0].save(buffered, format="PNG")
                        img_str = base64.b64encode(
                            buffered.getvalue()
                        ).decode()
                        return {
                            "message": "Image generated successfully",
                            "elapsed_time": f"{elapsed_time:.2f} sec",
                            "output": img_str,
                            "save_to_disk": False,
                        }
                return None

            except Exception as e:
                self.logger.error(f"Error generating image: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e)) from e

    class Engine:
        mode = "gpu"

        def __init__(self, world_size: int, xfuser_args: xFuserArgs):
            # Ensure Ray is initialized
            if not ray.is_initialized():
                ray.init()

            num_workers = world_size
            self.workers = [
                ImageGenerator.remote(
                    xfuser_args, rank=rank, world_size=world_size
                )
                for rank in range(num_workers)
            ]

        async def generate(self, request: GenerateRequest):
            results = ray.get(
                [worker.generate.remote(request) for worker in self.workers]
            )
            return next(path for path in results if path is not None)

    return Engine(world_size=world_size, xfuser_args=xfuser_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="xDiT HTTP Service")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the model. Required unless --mock is set.",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--pipefusion_parallel_degree",
        type=int,
        default=1,
        help="Degree of pipeline fusion parallelism",
    )
    parser.add_argument(
        "--ulysses_parallel_degree",
        type=int,
        default=1,
        help="Degree of Ulysses parallelism",
    )
    parser.add_argument(
        "--ring_degree",
        type=int,
        default=1,
        help="Degree of ring parallelism",
    )
    parser.add_argument(
        "--save_disk_path",
        type=str,
        default="output",
        help="Path to save generated images",
    )
    parser.add_argument(
        "--use_cfg_parallel",
        action="store_true",
        help="Whether to use CFG parallel",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help=(
            "Run the service in a CPU-only mock mode that produces a "
            "deterministic placeholder image without CUDA, Ray, or model "
            "weights. Intended for interactive QA and smoke testing."
        ),
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the HTTP service to.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6000,
        help="Port to bind the HTTP service to.",
    )
    args = parser.parse_args()

    if args.mock:
        engine = MockEngine()
    else:
        if not args.model_path:
            parser.error(
                "--model_path is required when not running in --mock mode."
            )
        xfuser_args = xFuserArgs(
            model=args.model_path,
            trust_remote_code=True,
            warmup_steps=1,
            use_parallel_vae=False,
            use_torch_compile=False,
            ulysses_degree=args.ulysses_parallel_degree,
            pipefusion_parallel_degree=args.pipefusion_parallel_degree,
            use_cfg_parallel=args.use_cfg_parallel,
            dit_parallel_size=0,
        )
        engine = build_gpu_engine(
            world_size=args.world_size, xfuser_args=xfuser_args
        )

    # Start the server
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
