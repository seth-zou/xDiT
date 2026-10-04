# Dev container image for xDiT.
#
# This extends the same NVIDIA PyTorch base image used by docker/Dockerfile so
# the development container ships with a CUDA-enabled PyTorch build (CUDA,
# NCCL, cuDNN) and matches the runtime stack the project targets in production.
# The dev container adds the OS/Python prerequisites that xDiT needs for
# development (git, build tools, editable install deps).
#
# See .devcontainer/devcontainer.json for how this image is consumed by VS Code
# Dev Containers / GitHub Codespaces.

FROM nvcr.io/nvidia/pytorch:24.07-py3

# OS-level dependencies.
# - git: required by pre-commit and by cloning model repos at runtime.
# - build-essential / pkg-config: needed to build some Python wheels from
#   source (e.g. flash-attn) when prebuilt wheels are unavailable.
# - curl: used by VS Code's devcontainer bootstrapping.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        pkg-config \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Keep pip current and ensure editable installs work cleanly.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# NVIDIA's PyTorch image ships an older apex build that conflicts with newer
# PyTorch; remove it so the editable install of xfuser does not fail.
RUN pip uninstall -y apex || true

# Install the project in editable mode with the runtime extras that are
# installable in this base image. flash-attn is intentionally installed
# separately (and lazily) because building it from source is slow and may
# fail on non-GPU CI runners; devcontainer.json's postCreateCommand and the
# project's extras handle the common path.
COPY . /workspace/xdit
WORKDIR /workspace/xdit

RUN pip install --no-cache-dir -e ".[diffusers,flask]"

# Pre-install the dev toolchain used by pre-commit and CI so that the
# container is immediately usable for lint/format without a network round-trip.
# Versions are pinned to match .pre-commit-config.yaml and .github/workflows/
# lint.yml to avoid drift between local, pre-commit, and CI behavior.
RUN pip install --no-cache-dir \
        ruff==0.6.9 \
        black==24.2.0 \
        pre-commit

# Drop root for interactive development. VS Code's devcontainer feature will
# create this user automatically when "remoteUser": "vscode" is set; create it
# here as a fallback for plain `docker run` usage.
RUN id -u vscode >/dev/null 2>&1 || useradd --create-home --uid 1000 vscode
USER vscode
