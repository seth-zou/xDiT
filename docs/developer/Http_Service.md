## Launch a Text-to-Image Http Service

Launch an HTTP-based text-to-image service that generates images from textual descriptions (prompts) using the DiT model. 
The generated images can either be returned directly to users or saved to a specified disk location.
For example, the following command launches a HTTP service with 4 GPUs, 2 Ulysses parallel degree, 2 PipeFusion parallel degree, and the model path is `./models/FLUX.1-schnell`.

```bash
python ./entrypoints/launch.py --world_size 4 --ulysses_parallel_degree 2 --pipefusion_parallel_degree 2 --model_path /your_model_path/FLUX.1-schnell
```

An example HTTP request is shown below. The `save_disk_path` parameter is optional - if not set, the image will be returned directly; if set, the generated image will be saved to the specified directory on disk.

```bash
curl -X POST "http://localhost:6000/generate" \
     -H "Content-Type: application/json" \
     -d '{
           "prompt": "a cute rabbit",
           "num_inference_steps": 50,
           "seed": 42,
           "cfg": 7.5, 
           "save_disk_path": "/tmp"
         }'
```

### Endpoints

| Endpoint | Method | Description |
| --- | --- | --- |
| `/` | GET | Service info (service name, available endpoints, active mode). |
| `/health` | GET | Liveness/readiness probe. Returns `{"status": "ok", "mode": ...}`. |
| `/generate` | POST | Generate an image from a `prompt`. Returns a base64 PNG, or writes a PNG to `save_disk_path` when provided. |

### CPU-only mock mode (interactive QA without GPUs)

The GPU backend requires CUDA GPUs, Ray, and multi-GB model weights, none of
which are available in a CI or CPU-only environment. To let the service be
brought up and exercised end-to-end without those dependencies, launch it with
`--mock`:

```bash
python ./entrypoints/launch.py --mock --host 127.0.0.1 --port 6000
```

In mock mode the service runs a CPU-only `MockEngine` that produces a
deterministic placeholder PNG derived from the prompt and seed. No CUDA, Ray,
or model weights are required. The request/response contract (including
`save_disk_path` and input validation) is identical to the GPU backend, so
clients and tests developed against mock mode work unchanged against the real
backend.

A CPU-runnable pytest smoke suite covers this path:

```bash
pytest tests/test_cpu_smoke.py -v
```

