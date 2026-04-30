# Apache-2.0 — ported from vllm/v1/attention/backends/mla/sparse_mla_env.py
# (jasl/vllm@ds4-sm120-full, vLLM PR #40899). Env-var gating for the
# portable Triton sparse-MLA path. The kernels themselves are arch-portable;
# upstream gates them to sm_120 (Blackwell DGX Spark) by default. We extend
# the gate to also cover sm_86 (Ampere) and honor the existing
# VLLM_TRITON_MLA_SPARSE override on any device.
"""Environment controls for the portable Triton sparse MLA path."""

import logging
import os

import torch

logger = logging.getLogger(__name__)


_TRITON_MLA_SPARSE_ENV = "VLLM_TRITON_MLA_SPARSE"
_TRITON_MLA_SPARSE_TOPK_CHUNK_ENV = "VLLM_TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE"
_TRITON_MLA_SPARSE_QUERY_CHUNK_ENV = "VLLM_TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE"
_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH_ENV = "VLLM_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH"
_TRITON_MLA_SPARSE_HEAD_BLOCK_ENV = "VLLM_TRITON_MLA_SPARSE_HEAD_BLOCK_SIZE"
_TRITON_MLA_SPARSE_MATMUL_DECODE_ENV = "VLLM_TRITON_MLA_SPARSE_MATMUL_DECODE"

_ENV_TRUE_VALUES = {"1", "true", "yes", "on"}
_ENV_FALSE_VALUES = {"0", "false", "no", "off"}


def _optional_env_flag(name: str) -> bool | None:
    raw_value = os.getenv(name)
    if raw_value is None:
        return None
    value = raw_value.lower()
    if value in _ENV_TRUE_VALUES:
        return True
    if value in _ENV_FALSE_VALUES:
        return False
    return None


def _device_capability_major(device: torch.device | None = None) -> int:
    if not torch.cuda.is_available():
        return 0
    if device is None:
        index = torch.cuda.current_device()
    else:
        index = device.index if device.index is not None else torch.cuda.current_device()
    return torch.cuda.get_device_capability(index)[0]


def _is_sm12x_device(device: torch.device) -> bool:
    return _device_capability_major(device) == 12


def _is_sm8x_device(device: torch.device) -> bool:
    return _device_capability_major(device) == 8


def triton_sparse_mla_configured() -> bool | None:
    return _optional_env_flag(_TRITON_MLA_SPARSE_ENV)


def is_triton_sparse_mla_enabled_for_platform() -> bool:
    configured = triton_sparse_mla_configured()
    if configured is not None:
        return configured
    # Auto-enable on Ampere (sm_8x; native flash_mla is Hopper-only) and on
    # consumer Blackwell sm_12x (no FlashMLA Sparse native kernels yet).
    major = _device_capability_major()
    return major == 8 or major == 12


def is_triton_sparse_mla_enabled(device: torch.device) -> bool:
    configured = triton_sparse_mla_configured()
    if configured is not None:
        return configured
    return _is_sm8x_device(device) or _is_sm12x_device(device)


def triton_sparse_mla_cudagraphs_allowed(vllm_config=None) -> bool:
    configured = _optional_env_flag(_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH_ENV)
    if configured is not None:
        return configured
    return True


def disable_triton_sparse_mla_cudagraphs_if_enabled(vllm_config) -> None:
    # vLLM-specific compile mode toggling — not applicable in SGLang. Kept
    # as a no-op so the kernels module's import doesn't break; SGLang
    # disables CUDA graph capture via its own --disable-cuda-graph flag.
    return


def triton_sparse_mla_topk_chunk_size() -> int:
    raw_value = os.getenv(_TRITON_MLA_SPARSE_TOPK_CHUNK_ENV)
    if raw_value is None:
        return 512
    try:
        return max(1, int(raw_value))
    except ValueError:
        return 512


def triton_sparse_mla_query_chunk_size() -> int:
    raw_value = os.getenv(_TRITON_MLA_SPARSE_QUERY_CHUNK_ENV)
    if raw_value is None:
        return 256
    try:
        return max(1, int(raw_value))
    except ValueError:
        return 256


def triton_sparse_mla_head_block_size() -> int | None:
    raw_value = os.getenv(_TRITON_MLA_SPARSE_HEAD_BLOCK_ENV)
    if raw_value is None:
        return None
    try:
        value = int(raw_value)
    except ValueError:
        return None
    if value in (1, 2, 4):
        return value
    return None


def triton_sparse_mla_matmul_decode_enabled() -> bool:
    configured = _optional_env_flag(_TRITON_MLA_SPARSE_MATMUL_DECODE_ENV)
    if configured is not None:
        return configured
    # Default to on for sm_8x and sm_12x — the matmul decode path is the
    # one jasl9187 hardened upstream (commit d8fde51).
    major = _device_capability_major()
    return major == 8 or major == 12
