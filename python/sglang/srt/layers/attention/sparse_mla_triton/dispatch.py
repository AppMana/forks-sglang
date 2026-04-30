# Apache-2.0
"""Adapter from SGLang's flash_mla_with_kvcache call signature to the
Triton sparse-MLA kernels ported from vLLM PR #40899.

The SGLang V4 backend at deepseek_v4_backend_radix.py calls
flash_mla_with_kvcache_entrypoint with the following kwargs (excerpt
from the actual call site):

    q                          # (T, 1, H, 576)  — NoPE+RoPE concat
    k_cache (= swa_k_cache)    # uint8, page-banked: per-page layout is
                                #   [t0_kv (576B) | t1_kv | ... | t0_scales (8B) | ...]
    head_dim_v                 # 512
    block_table                # None (sparse path uses indices)
    cache_seqlens              # None (sparse path uses topk_length)
    tile_scheduler_metadata    # ignored on Triton path
    softmax_scale              # 1/sqrt(512)
    is_fp8_kvcache             # True (FP8 KV with E8M0 scales)
    indices                    # (T, 1, K_swa) page indices for SWA stream
    topk_length                # (T,) attended-token count per query, SWA
    attn_sink                  # (H,) float32
    extra_k_cache              # optional uint8 — secondary (compressed) cache
    extra_indices_in_kvcache   # optional (T, 1, K_extra) page indices
    extra_topk_length          # optional (T,) attended count, compressed

Returned: (out, lse) where out has shape (T, 1, H, head_dim_v=512). LSE is
optional in the upstream API; we return None for it (callers ignore).

Triton kernel mapping:
  - extra_indices is None  -> fp8ds_paged_sparse_mla_attention_with_sink_multihead
  - extra_indices is set   -> fp8ds_global_paged_sparse_mla_attention_with_sink_multihead

Both kernels write into the output tensor in-place. They expect q to be
the NoPE-only slice (head_dim=512); the kernels read RoPE bytes from the
K cache and add the RoPE QK term internally.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from sglang.srt.layers.attention.sparse_mla_triton.kernels import (
    fp8ds_global_paged_sparse_mla_attention_with_sink_multihead,
    fp8ds_paged_sparse_mla_attention_with_sink_multihead,
    sparse_mla_decode_head_block_size,
)


def _strip_seq_dim(t: torch.Tensor) -> torch.Tensor:
    """SGLang passes q / indices with a singleton seq dim at position 1.
    The Triton kernels expect [T, H, D] / [T, K] respectively. Squeeze it."""
    if t.dim() == 4:
        # (T, 1, H, D) -> (T, H, D)
        assert t.shape[1] == 1, f"unexpected seq dim {t.shape}"
        return t[:, 0]
    if t.dim() == 3:
        # (T, 1, K) -> (T, K)
        if t.shape[1] == 1:
            return t[:, 0]
    return t


def flash_mla_with_kvcache_triton(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    head_dim_v: int,
    softmax_scale: float,
    indices: torch.Tensor,
    topk_length: torch.Tensor,
    attn_sink: torch.Tensor,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    is_fp8_kvcache: bool = True,
    block_table=None,
    cache_seqlens=None,
    tile_scheduler_metadata=None,
    **_unused,
) -> Tuple[torch.Tensor, None]:
    assert is_fp8_kvcache, (
        "Triton sparse-MLA path requires is_fp8_kvcache=True (page-banked "
        "FP8+E8M0 KV layout)."
    )
    assert head_dim_v == 512, f"V4-Flash head_dim_v must be 512, got {head_dim_v}"
    # SGLang stores the page-banked KV with dtype torch.float8_e4m3fn while
    # the Triton kernels read raw bytes (uint8) and dequant internally. The
    # underlying memory is identical — view-cast in place. Same for the
    # optional secondary (compressed) cache below.
    if k_cache.dtype != torch.uint8:
        k_cache = k_cache.view(torch.uint8)
    if extra_k_cache is not None and extra_k_cache.dtype != torch.uint8:
        extra_k_cache = extra_k_cache.view(torch.uint8)

    q3 = _strip_seq_dim(q)              # (T, H, 512) — NoPE+RoPE concat (dim 448+64)
    indices2 = _strip_seq_dim(indices)  # (T, K_swa) page indices
    # The Triton kernels operate on q with head_dim=512: 448 NoPE + 64 RoPE.
    # The kernel reads NoPE FP8 (448 B -> 448 floats) and RoPE BF16
    # (128 B -> 64 BF16) from the cache; QK = Qn·Kn + Qr·Kr.
    assert q3.shape[-1] == 512, f"expected q with NoPE+RoPE element-dim=512, got {q3.shape}"

    num_tokens, num_heads, _ = q3.shape
    device = q3.device
    out = torch.empty(
        (num_tokens, num_heads, head_dim_v),
        dtype=q3.dtype,
        device=device,
    )

    head_block_size = sparse_mla_decode_head_block_size(num_tokens)

    # SGLang gives us `indices` as PAGE indices. The Triton kernels
    # `*_global_paged_sparse_mla_attention_with_sink_multihead` expect
    # TOKEN slot IDs (slot_id == page_idx * page_size + slot_within_page);
    # internally they recover (page_idx = slot_id // page_size,
    # pos_in_block = slot_id % page_size). Expand page indices to per-token
    # slots once on the host so the kernel can iterate token-by-token.
    page_size = _infer_page_size_bytes(k_cache)
    swa_slot_ids = _expand_page_indices_to_slots(indices2, page_size).to(torch.int32)
    swa_lens = topk_length.to(torch.int32)

    if extra_indices_in_kvcache is None or extra_k_cache is None:
        # Single-stream (SWA only). Use the global multihead kernel by
        # providing zero compressed candidates.
        empty_compressed_k = torch.zeros(
            1, page_size * 576, dtype=torch.uint8, device=device
        )
        empty_compressed_slots = torch.zeros(
            num_tokens, 1, dtype=torch.int32, device=device
        )
        empty_compressed_lens = torch.zeros(num_tokens, dtype=torch.int32, device=device)
        fp8ds_global_paged_sparse_mla_attention_with_sink_multihead(
            q=q3,
            compressed_k_cache=empty_compressed_k,
            slot_ids=empty_compressed_slots,
            topk_lens=empty_compressed_lens,
            compressed_block_size=page_size,
            swa_k_cache=k_cache,
            seq_lens=swa_lens,
            gather_lens=swa_lens,
            block_table=_identity_block_table(num_tokens, swa_slot_ids.shape[-1] // page_size, device),
            swa_block_size=page_size,
            num_compressed_candidates=0,
            num_swa_candidates=swa_slot_ids.shape[-1],
            scale=softmax_scale,
            attn_sink=attn_sink,
            output=out,
            head_block_size=head_block_size,
            num_heads=num_heads,
        )
    else:
        # Dual-stream: SWA + compressed.
        extra_indices2 = _strip_seq_dim(extra_indices_in_kvcache)
        extra_page_size = _infer_page_size_bytes(extra_k_cache)
        extra_slot_ids = _expand_page_indices_to_slots(extra_indices2, extra_page_size).to(torch.int32)
        extra_lens = extra_topk_length.to(torch.int32)
        fp8ds_global_paged_sparse_mla_attention_with_sink_multihead(
            q=q3,
            compressed_k_cache=extra_k_cache,
            slot_ids=extra_slot_ids,
            topk_lens=extra_lens,
            compressed_block_size=extra_page_size,
            swa_k_cache=k_cache,
            seq_lens=swa_lens,
            gather_lens=swa_lens,
            block_table=_identity_block_table(num_tokens, swa_slot_ids.shape[-1] // page_size, device),
            swa_block_size=page_size,
            num_compressed_candidates=extra_slot_ids.shape[-1],
            num_swa_candidates=swa_slot_ids.shape[-1],
            scale=softmax_scale,
            attn_sink=attn_sink,
            output=out,
            head_block_size=head_block_size,
            num_heads=num_heads,
        )

    # Add back the singleton seq dim SGLang's caller squeezes off.
    return out.unsqueeze(1), None


def _infer_page_size_bytes(k_cache: torch.Tensor) -> int:
    """V4 page-banked layout: per-page bytes = page_size * 576 + page_size * 8
    (KV bytes + scale bytes). The kernel uses the 576 stride internally;
    we recover page_size from the total bytes-per-page divided by 584."""
    bytes_per_page = k_cache.numel() * k_cache.element_size() // k_cache.shape[0]
    return bytes_per_page // 584


def _expand_page_indices_to_slots(
    page_indices: torch.Tensor, page_size: int
) -> torch.Tensor:
    """(T, K_pages) -> (T, K_pages * page_size). Each page index expands to
    page_size consecutive token slot IDs."""
    T, K = page_indices.shape
    arange = torch.arange(page_size, device=page_indices.device, dtype=page_indices.dtype)
    slots = page_indices[:, :, None] * page_size + arange[None, None, :]
    return slots.reshape(T, K * page_size)


def _identity_block_table(num_tokens: int, num_pages_per_token: int, device) -> torch.Tensor:
    """The SWA path of `*_global_paged_*` uses block_table[t][block_in_seq]
    -> physical_block. We've already expanded indices to slot_ids that
    encode the physical page (slot_id // page_size); for that path the
    kernel re-derives via slot_id arithmetic and doesn't actually need
    block_table. Pass an identity-style table to keep the signature."""
    return torch.arange(num_pages_per_token, device=device, dtype=torch.int32).expand(
        num_tokens, num_pages_per_token
    ).contiguous()
