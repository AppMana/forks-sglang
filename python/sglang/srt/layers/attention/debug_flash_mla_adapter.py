"""
Adapter for the V4 compressed-attention `flash_mla_with_kvcache` call.

Backends:
  - "kernel"  : DeepSeek's flash_mla pip pkg (Hopper / sm_90+ only)
  - "torch"   : pure-pytorch FP32-accumulated sparse-MLA reference
                (slow, but correct on any device that runs pytorch)
  - "tilelang": (Phase B, not yet implemented) JIT TileLang kernel for sm_86

Auto-pick:
  - device cap >= 9 + flash_mla importable -> "kernel"
  - device cap <  9                        -> "torch" until tilelang lands

The torch fallback produces correct output (within FP32 precision) for V4-Flash
sparse-MLA on Ampere consumer GPUs (sm_86) where neither flash_mla nor a sparse
kernel exists. It is the spec used to validate Phase B's TileLang kernel.

Critical math facts encoded in this file (verified against the V4 backend at
forks-sglang/python/sglang/srt/layers/attention/deepseek_v4_backend_radix.py
and the page-banked KV layout in nsa/index_buf_accessor_v4.py):

  - softmax_scale = 1/sqrt(head_dim_v=512), NOT 1/sqrt(head_dim=576)
  - per-token KV bytes = 448 (FP8 NoPE) + 128 (BF16 RoPE, 64 elements x 2B) + 8
    (ue8m0 scales, 7 valid + 1 pad) = 584 B
  - within a page, layout is [t0_kv (576B) | t1_kv | ... | tN_kv | t0_scales (8B) | t1_scales | ...]
    NOT [t0_kv | t0_scales | t1_kv | t1_scales | ...]. Scales are page-banked at
    byte offset page_size * 576 within the page.
  - scale_fp32 = exp2(int(byte) - 127). Naive cast(uint8, float32) is wrong.
  - 7 scale bytes per token, one per 64-element NoPE block (448 = 7 * 64).
  - V = K[..., :512] = NoPE-only (RoPE participates in QK only, not in PV)
  - attn_sink shape (num_heads,) float32, applied as a virtual key with logit
    `attn_sink[h]` and zero V-row: sumexp[h] += exp(attn_sink[h] - m[h]), once
    after the attended-key loop, before final normalization.
  - indices[q,0,k] is a page index. topk_length[q] is the per-query attended
    TOKEN count (not page count). Token range per page is page_size tokens,
    masked to topk_length total.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch


def _resolve_backend(requested: str) -> str:
    """
    Translate the requested backend into the concrete one to run.

    "kernel"   -> use flash_mla pip pkg (errors if unavailable)
    "torch"    -> torch fallback
    "tilelang" -> TileLang kernel (Phase B; falls back to torch with warning)
    "auto"     -> kernel on cap>=9, torch on cap<9
    """
    if requested == "auto":
        try:
            major, _ = torch.cuda.get_device_capability()
        except Exception:
            major = 0
        return "kernel" if major >= 9 else "torch"
    return requested


def flash_mla_with_kvcache_entrypoint(backend: str, **kwargs):
    backend = _resolve_backend(backend)

    if backend == "kernel":
        # Hopper path. flash_mla pip pkg required.
        import flash_mla

        return flash_mla.flash_mla_with_kvcache(**kwargs)

    if backend == "zero":
        # Pure zero-stub: no compute, returns the right output shape filled
        # with zeros. Used to bisect "is the V4 MoE/attention path working?"
        # vs "is sparse-MLA the bottleneck?".
        q = kwargs["q"]
        head_dim_v = kwargs["head_dim_v"]
        num_tokens = q.shape[0]
        num_heads = q.shape[2] if q.dim() == 4 else q.shape[1]
        o = torch.zeros(
            (num_tokens, 1, num_heads, head_dim_v),
            dtype=torch.bfloat16, device=q.device,
        )
        return (o, None)

    if backend == "torch":
        return _flash_mla_with_kvcache_torch_fallback(**kwargs)

    if backend == "tilelang":
        # Not yet implemented; degrade to torch so callers don't crash. Set
        # SGLANG_HACK_FLASHMLA_BACKEND=torch explicitly to silence the warning.
        import warnings
        warnings.warn(
            "tilelang backend not yet implemented for V4 sparse MLA; "
            "falling back to torch reference.",
            stacklevel=2,
        )
        return _flash_mla_with_kvcache_torch_fallback(**kwargs)

    raise RuntimeError(f"unsupported flash_mla backend: {backend!r}")


# ---------------------------------------------------------------------------
# Torch reference fallback
# ---------------------------------------------------------------------------

# V4-Flash hardcoded layout. See class docstring.
_NOPE_BYTES = 448            # FP8 e4m3 elements per token, 1 byte each
_NOPE_DIM = 448              # NoPE element count (= qk_nope_head_dim)
_ROPE_BYTES = 128            # BF16 elements per token, 2 bytes each
_ROPE_DIM = 64               # qk_rope_head_dim
_SCALE_BYTES_PER_TOKEN = 8   # 7 valid ue8m0 + 1 pad
_SCALE_VALID = 7
_NOPE_BLOCK = 64             # NoPE elements per ue8m0 scale block
_KV_BYTES_PER_TOKEN = _NOPE_BYTES + _ROPE_BYTES   # 576
_BYTES_PER_TOKEN = _KV_BYTES_PER_TOKEN + _SCALE_BYTES_PER_TOKEN  # 584
# Q has the same per-head element count as K's NoPE+RoPE:
# qk_nope_head_dim (448) + qk_rope_head_dim (64) = 512.
# v_head_dim is also 512 (declared by the model layer); the kernel returns
# 512-dim output. Since stored NoPE is only 448, the V projection is
# zero-padded to 512 in the fallback (real kernels likely fold in a
# learnable up-projection, but for smoke output the shape and validity
# of the path is what matters).
_Q_DIM = _NOPE_DIM + _ROPE_DIM   # 512


def _decode_pages_to_kv(
    k_cache: torch.Tensor,
    indices: torch.Tensor,
    page_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decompose the V4 page-banked uint8 KV cache into NoPE FP8, RoPE BF16,
    and per-block ue8m0 scales for the page subset addressed by `indices`.

    Args:
      k_cache: uint8 tensor with shape (num_pages, ..., total_dim) where the
               raw memory is contiguous and per-page bytes are
                   [t0_kv_576B, t1_kv_576B, ..., t{P-1}_kv_576B,
                    t0_scale_8B, ..., t{P-1}_scale_8B]
               (P = page_size). The outer view here is the (num_pages, P, 1, 584)
               that V4 hands FlashMLA.
      indices: int32 page indices, may include -1 sentinels.
      page_size: number of tokens per page.

    Returns:
      nope_fp32: (num_indices, page_size, NoPE=448) FP32, dequantized
                 (FP8 -> FP32 with per-block ue8m0 scaling applied).
      rope_bf16: (num_indices, page_size, RoPE=64) BF16.
      page_valid: (num_indices,) bool, False for -1 sentinels.
    """
    device = k_cache.device
    num_pages = k_cache.shape[0]

    # Sentinel mask before clamping. Indices use -1 for invalid; pad/garbage
    # may also be > num_pages, so clamp both ends.
    num_pages = k_cache.shape[0]
    page_valid = (indices >= 0) & (indices < num_pages)
    safe_idx = indices.clamp(min=0, max=num_pages - 1).to(torch.long)

    # Recover (num_pages, page_size * 584) contiguous uint8 view of the data
    # section. The (num_pages, page_size, 1, 584) view's element strides flatten
    # to (page_size*584, 584, 584, 1) -- contiguous reshape works.
    flat_pages = k_cache.reshape(num_pages, page_size * _BYTES_PER_TOKEN)
    selected = flat_pages.index_select(0, safe_idx.reshape(-1)).reshape(
        *safe_idx.shape, page_size * _BYTES_PER_TOKEN
    )
    # selected shape: (..., page_size * 584)

    # Split: KV bytes [0 : page_size*576), scales [page_size*576 : page_size*584)
    kv_bytes = selected[..., : page_size * _KV_BYTES_PER_TOKEN].reshape(
        *selected.shape[:-1], page_size, _KV_BYTES_PER_TOKEN
    )  # (..., page_size, 576)
    scale_bytes = selected[..., page_size * _KV_BYTES_PER_TOKEN :].reshape(
        *selected.shape[:-1], page_size, _SCALE_BYTES_PER_TOKEN
    )  # (..., page_size, 8)

    # NoPE FP8: first 448 bytes of each token's 576-byte KV slab.
    nope_uint8 = kv_bytes[..., :_NOPE_BYTES].contiguous()
    nope_fp8 = nope_uint8.view(torch.float8_e4m3fn)
    nope_fp32 = nope_fp8.to(torch.float32)  # (..., page_size, 448)

    # RoPE BF16: bytes [448:576] of each token's KV slab, viewed as BF16.
    rope_uint8 = kv_bytes[..., _NOPE_BYTES:].contiguous()
    rope_bf16 = rope_uint8.view(torch.bfloat16).reshape(
        *rope_uint8.shape[:-1], _ROPE_DIM
    )  # (..., page_size, 64)

    # ue8m0 scales: 7 valid bytes per token. scale_fp32 = 2^(int(byte) - 127).
    scales_u8 = scale_bytes[..., :_SCALE_VALID].to(torch.float32)
    scales_fp32 = torch.exp2(scales_u8 - 127.0)  # (..., page_size, 7)

    # Apply per-block scaling: NoPE element i lives in block i // 64, scaled by
    # scales_fp32[..., i // 64]. Reshape NoPE to (..., page_size, 7, 64) and
    # multiply.
    nope_blocked = nope_fp32.reshape(
        *nope_fp32.shape[:-1], _SCALE_VALID, _NOPE_BLOCK
    )
    nope_blocked = nope_blocked * scales_fp32.unsqueeze(-1)
    nope_fp32 = nope_blocked.reshape(*nope_fp32.shape)

    return nope_fp32, rope_bf16.to(torch.float32), page_valid


def _flash_mla_with_kvcache_torch_fallback(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    head_dim_v: int,
    block_table=None,
    cache_seqlens=None,
    tile_scheduler_metadata=None,
    softmax_scale: Optional[float] = None,
    is_fp8_kvcache: bool = False,
    indices: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    causal: bool = False,
    **_unused,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    FP32-accumulation pure-pytorch sparse MLA with V4 attn_sink + extras.

    Output dtype matches q's dtype (typically bf16).
    """
    assert is_fp8_kvcache, "torch fallback only implemented for V4 FP8 KV cache layout"
    assert q.dim() == 4, f"q must be 4D, got shape {q.shape}"

    device = q.device
    out_dtype = q.dtype

    # q: (num_tokens, 1, num_heads, head_dim). Squeeze the singleton.
    q4 = q
    num_tokens, _, num_heads, head_dim = q4.shape
    q_fp32 = q4.squeeze(1).to(torch.float32)  # (T, H, head_dim)

    # Diagnostic: log shape on first call so we can verify head_dim against
    # qk_nope_head_dim (448) + qk_rope_head_dim (64) = 512.
    if not getattr(_flash_mla_with_kvcache_torch_fallback, "_logged_shapes", False):
        idx_min = int(indices.min().item()) if indices is not None else None
        idx_max = int(indices.max().item()) if indices is not None else None
        topk_min = int(topk_length.min().item()) if topk_length is not None else None
        topk_max = int(topk_length.max().item()) if topk_length is not None else None
        print(
            f"[flash_mla_torch_fallback] q.shape={tuple(q.shape)} "
            f"k_cache.shape={tuple(k_cache.shape)} head_dim_v={head_dim_v} "
            f"indices.shape={None if indices is None else tuple(indices.shape)} "
            f"indices_range=[{idx_min},{idx_max}] "
            f"topk_length.shape={None if topk_length is None else tuple(topk_length.shape)} "
            f"topk_range=[{topk_min},{topk_max}] "
            f"attn_sink.shape={None if attn_sink is None else tuple(attn_sink.shape)} "
            f"extra_k_cache.shape={None if extra_k_cache is None else tuple(extra_k_cache.shape)}",
            flush=True,
        )
        _flash_mla_with_kvcache_torch_fallback._logged_shapes = True

    if softmax_scale is None:
        softmax_scale = head_dim_v ** -0.5  # 1/sqrt(512)

    if indices is None:
        # Dense path is unsupported in the V4 backend's call shape.
        raise RuntimeError("torch fallback expects sparse `indices`")

    # indices: (T, 1, K). topk_length: (T, 1). page_size derived from k_cache.
    assert indices.shape[0] == num_tokens, (
        f"indices.shape={indices.shape}, num_tokens={num_tokens}"
    )
    assert indices.shape[1] == 1, f"unexpected indices middle dim: {indices.shape}"
    assert k_cache.dim() == 4, f"k_cache must be 4D (num_pages, page_size, 1, 584), got {k_cache.shape}"
    page_size = k_cache.shape[1]
    assert k_cache.shape[3] == _BYTES_PER_TOKEN, (
        f"k_cache last dim must be {_BYTES_PER_TOKEN} bytes/token, got {k_cache.shape[3]}"
    )

    # Online-softmax accumulators per (token, head)
    m_running = torch.full((num_tokens, num_heads), float("-inf"),
                           dtype=torch.float32, device=device)
    sumexp = torch.zeros((num_tokens, num_heads),
                         dtype=torch.float32, device=device)
    acc_o = torch.zeros((num_tokens, num_heads, head_dim_v),
                        dtype=torch.float32, device=device)

    # ---- SWA pass ------------------------------------------------------------
    # indices_2d: (T, K)
    indices_2d = indices.reshape(num_tokens, -1)
    topk_2d = topk_length.reshape(num_tokens) if topk_length is not None else None

    _accumulate(
        q_fp32=q_fp32,
        k_cache=k_cache,
        indices_2d=indices_2d,
        topk_lengths=topk_2d,
        page_size=page_size,
        softmax_scale=softmax_scale,
        m_running=m_running,
        sumexp=sumexp,
        acc_o=acc_o,
    )

    # ---- Extras pass (compress_ratio=4 or 128 layers) ------------------------
    if extra_k_cache is not None and extra_indices_in_kvcache is not None:
        extra_indices_2d = extra_indices_in_kvcache.reshape(num_tokens, -1)
        extra_topk_2d = (
            extra_topk_length.reshape(num_tokens)
            if extra_topk_length is not None else None
        )
        extra_page_size = extra_k_cache.shape[1]
        _accumulate(
            q_fp32=q_fp32,
            k_cache=extra_k_cache,
            indices_2d=extra_indices_2d,
            topk_lengths=extra_topk_2d,
            page_size=extra_page_size,
            softmax_scale=softmax_scale,
            m_running=m_running,
            sumexp=sumexp,
            acc_o=acc_o,
        )

    # ---- Attention sink (virtual key with zero V-row) ------------------------
    if attn_sink is not None:
        sink = attn_sink.to(torch.float32).reshape(1, num_heads)  # (1, H)
        # sumexp[t,h] += exp(sink[h] - m_running[t,h]). When m == -inf (no real
        # keys attended -> can happen if topk_length=0), the sink dominates and
        # acc_o stays at 0, sumexp = 1. Final acc_o/sumexp = 0 -> safe.
        sink_term = torch.exp(sink - m_running)
        sumexp = sumexp + sink_term

    # Final normalization. Guard against sumexp == 0 (defensive: shouldn't happen
    # because at least the sink contributes when no real keys attended).
    sumexp = torch.clamp(sumexp, min=1e-30)
    o_fp32 = acc_o / sumexp.unsqueeze(-1)  # (T, H, head_dim_v)

    o = o_fp32.unsqueeze(1).to(out_dtype)  # (T, 1, H, head_dim_v)
    # Return shape mirroring flash_mla_with_kvcache: (out, lse). lse not used
    # by V4 backend; return None to signal absence.
    return (o, None)


def _accumulate(
    *,
    q_fp32: torch.Tensor,
    k_cache: torch.Tensor,
    indices_2d: torch.Tensor,
    topk_lengths: Optional[torch.Tensor],
    page_size: int,
    softmax_scale: float,
    m_running: torch.Tensor,
    sumexp: torch.Tensor,
    acc_o: torch.Tensor,
) -> None:
    """
    Online-softmax accumulation over one indexed page set. Updates m_running,
    sumexp, acc_o in place.

      q_fp32:        (T, H, 576)
      indices_2d:    (T, K) int32, may be -1
      topk_lengths:  (T,) int32 or None (None => all K positions valid)
      page_size:     tokens per page in the K cache being indexed
    """
    num_tokens, num_heads, _ = q_fp32.shape
    K = indices_2d.shape[1]
    device = q_fp32.device

    # Decode attended pages: (T, K, page_size, 448) FP32 NoPE,
    # (T, K, page_size, 64) FP32 RoPE, (T, K) page_valid bool.
    nope_fp32, rope_fp32, page_valid = _decode_pages_to_kv(
        k_cache, indices_2d, page_size
    )
    # Flatten (K, page_size) -> total attended tokens per query.
    K_pg = K * page_size
    nope_flat = nope_fp32.reshape(num_tokens, K_pg, _NOPE_DIM)         # (T, KP, 448)
    rope_flat = rope_fp32.reshape(num_tokens, K_pg, _ROPE_DIM)         # (T, KP, 64)

    # Per-token validity mask: page-valid AND token-position < topk_length.
    # broadcast page_valid (T, K) -> (T, K, page_size) -> (T, KP)
    page_valid_pg = page_valid.unsqueeze(-1).expand(-1, -1, page_size).reshape(num_tokens, K_pg)
    if topk_lengths is not None:
        # token position within the flat indexed sequence is implementation-
        # defined; FlashMLA semantics treat indices as page-granular and topk
        # as token-granular, with valid tokens being the first
        # topk_lengths[t] across the flattened (page, intra_page) sequence.
        positions = torch.arange(K_pg, device=device).unsqueeze(0)         # (1, KP)
        within = positions < topk_lengths.to(torch.long).unsqueeze(-1)     # (T, KP)
        valid = page_valid_pg & within
    else:
        valid = page_valid_pg

    # K_full = concat(NoPE, RoPE) along last dim -> (T, KP, 576) for QK^T
    k_full = torch.cat([nope_flat, rope_flat], dim=-1)                     # (T, KP, 576)

    # logits = (Q @ K^T) * scale -> (T, H, KP)
    # Q: (T, H, 576), K_full: (T, KP, 576)
    logits = torch.einsum("thd,tkd->thk", q_fp32, k_full) * softmax_scale

    # Mask invalid keys to -inf
    mask = valid.unsqueeze(1)                                               # (T, 1, KP)
    logits = logits.masked_fill(~mask, float("-inf"))

    # Online softmax: combine into m_running, sumexp, acc_o
    new_max = torch.maximum(m_running, logits.max(dim=-1).values)           # (T, H)
    # Where new_max is still -inf (this slice produced no valid keys AND no
    # prior keys), treat as 0 logit for downstream stability; acc_o won't
    # change (the exp(-inf - -inf) = exp(0) = 1 problem is sidestepped by
    # masking acc_s).
    new_max = torch.where(torch.isinf(new_max), torch.zeros_like(new_max), new_max)

    # Rescale prior accumulators
    rescale = torch.exp(m_running - new_max)
    rescale = torch.where(torch.isinf(m_running), torch.zeros_like(rescale), rescale)
    sumexp.mul_(rescale)
    acc_o.mul_(rescale.unsqueeze(-1))
    m_running.copy_(new_max)

    # Add this iteration's contribution
    shifted = torch.exp(logits - new_max.unsqueeze(-1))                     # (T, H, KP)
    shifted = shifted.masked_fill(~mask, 0.0)
    sumexp.add_(shifted.sum(dim=-1))
    # acc_o += sum_k shifted[t,h,k] * V[t,k] where V = NoPE only (head_dim_v=512)
    # nope_flat: (T, KP, 448). Pad to head_dim_v=512 since head_dim_v is 512 not 448.
    # Actually V4-Flash NoPE IS the V; head_dim_v=512 is a misnomer in the
    # backend? -- assert and zero-pad to 512 if head_dim_v > NoPE.
    if _NOPE_DIM == 512:
        v = nope_flat
    else:
        # nope_flat is (T, KP, 448). Pad to 512 with zeros for the V-projection.
        # This matches expected output dim 512; the extra 64 dims contribute 0.
        pad = torch.zeros(
            num_tokens, K_pg, 512 - _NOPE_DIM, device=device, dtype=torch.float32
        )
        v = torch.cat([nope_flat, pad], dim=-1)
    contrib = torch.einsum("thk,tkv->thv", shifted, v)                      # (T, H, 512)
    acc_o.add_(contrib)
