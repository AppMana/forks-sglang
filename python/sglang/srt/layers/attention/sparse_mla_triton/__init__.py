# Triton sparse-MLA path for V4-Flash on Ampere/sm_86 and Blackwell-consumer/sm_120.
# Kernels ported from vLLM PR #40899 (jasl/vllm@ds4-sm120-full).
"""Triton sparse-MLA backend for DeepSeek-V4 family.

Public entry point used by debug_flash_mla_adapter.py:

    flash_mla_with_kvcache_triton(**kwargs) -> (out, lse)

It mirrors DeepSeek's flash_mla_with_kvcache call signature so the
adapter can switch backends with a string toggle. Internally it dispatches
to the appropriate Triton kernel from `kernels.py` based on whether
secondary (compressed) indices are present.

Architecture support: Triton compiles for any compute capability (sm_8x and
up). Auto-enabled for sm_86 and sm_120; force on with VLLM_TRITON_MLA_SPARSE=1.
"""

from sglang.srt.layers.attention.sparse_mla_triton.dispatch import (
    flash_mla_with_kvcache_triton,
)

__all__ = ["flash_mla_with_kvcache_triton"]
