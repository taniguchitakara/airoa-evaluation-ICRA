# Blackwell (RTX 5070 Ti / sm_120) support: CUDA 12.8 runtime base.
# The evaluation GPU is compute capability 12.0 (sm_120); older CUDA bases
# do not ship kernels for it and torch fails at runtime with
# "no kernel image is available for execution on the device".
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs build-essential clang ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock LICENSE README.md ./
COPY packages/policy-client ./packages/policy-client
COPY src ./src
COPY server ./server
COPY runtime_core ./runtime_core

# Create virtual environment and install the locked dependency set, except for
# the old LeRobot pin and its `av` sdist dependency. That lock entry only ships
# a source distribution for Linux, which forces a PyAV build against FFmpeg 7.
# We replace LeRobot later with the Python 3.11-compatible PI05 revision.
RUN uv venv --python 3.11 --seed /workspace/.venv
RUN GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-dev --no-install-project \
        --no-install-package lerobot \
        --no-install-package av
RUN uv pip install --python /workspace/.venv/bin/python pytest

# The locked workspace dependency currently resolves an older LeRobot revision
# without PI05. This checkpoint explicitly targets PI05, so override it with
# the latest Python 3.11-compatible upstream tag that ships `lerobot.policies.pi05`.
RUN /workspace/.venv/bin/pip install --force-reinstall \
        "lerobot[pi] @ git+https://github.com/huggingface/lerobot.git@v0.4.4"

# Replace the PyPI torch wheel (cu126, no sm_120 kernels) with the matching
# cu128 stack from the official PyTorch index. Keeping `--no-deps` here leaves
# the older CUDA 12.6 user-space libraries in place, which breaks import of the
# cu128 torch wheel with errors like:
#   ImportError: libnvshmem_host.so.3: cannot open shared object file
RUN uv pip install --python /workspace/.venv/bin/python --force-reinstall \
        torch==2.7.1 torchvision==0.22.1 \
        --index-url https://download.pytorch.org/whl/cu128

# Keep numpy on the 1.x line: LeRobot v0.4.4 SciPy stack expects < 2.3 and
# policy-client pins numpy<2.0. The torch reinstall above otherwise upgrades it.
RUN uv pip install --python /workspace/.venv/bin/python \
        numpy==1.26.4

# Some CUDA 12.8 components load NVSHMEM by soname at runtime; install the
# wheel that ships libnvshmem_host.so.3 and expose it via the linker path.
RUN uv pip install --python /workspace/.venv/bin/python \
        nvidia-nvshmem-cu12==3.2.5

ENV PATH="/workspace/.venv/bin:${PATH}"
ENV LD_LIBRARY_PATH="/workspace/.venv/lib/python3.11/site-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH}"
ENV PYTHONPATH="/workspace:/workspace/src:/workspace/packages/policy-client/src"

ENTRYPOINT ["/workspace/server/entrypoint.sh"]
