import contextlib
import os
import tempfile

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.vllm import get_current_vllm_config_or_none
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.platforms import current_platform

# Optional cap for runs that share a GPU with another process.
_frac = os.environ.get("TEST_GPU_FRACTION")
if _frac and torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(float(_frac))


@contextlib.contextmanager
def _ensure_current_vllm_config():
    if get_current_vllm_config_or_none() is not None:
        yield
    else:
        with set_current_vllm_config(VllmConfig()):
            yield


@pytest.fixture
def dist_init():
    """Same as upstream tests/conftest.py::dist_init (single-rank TP group)."""
    fd, temp_file = tempfile.mkstemp()
    os.close(fd)
    try:
        with _ensure_current_vllm_config():
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"file://{temp_file}",
                local_rank=0,
                backend=current_platform.dist_backend,
            )
            initialize_model_parallel(1, 1)
            yield
        cleanup_dist_env_and_memory()
    finally:
        with contextlib.suppress(OSError):
            os.unlink(temp_file)


@pytest.fixture
def enable_pickle(monkeypatch):
    """Same as upstream tests/conftest.py::enable_pickle."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


@pytest.fixture
def default_vllm_config():
    """Same as upstream tests/conftest.py::default_vllm_config."""
    with set_current_vllm_config(VllmConfig()):
        yield
