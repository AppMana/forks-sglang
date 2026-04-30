"""
Convert a DeepSeek-V4-Flash safetensors checkpoint to an Ampere-loadable
INT4/INT8 W*A16 layout, in-place per-shard.

For each shard:
  - Routed-expert MXFP4 weights -> INT4 packed + BF16 group_size=32 scales
    (`*.ffn.experts.E.{w1,w2,w3}.{weight,scale}`)
  - Attention FP8 weights        -> INT8 + BF16 128x128 block scales
    (`*.attn.{wq_a,wq_b,wkv,wo_a,wo_b}.{weight,scale}` and
     `*.ffn.shared_experts.{w1,w2,w3}.{weight,scale}`)
  - Everything else              -> passthrough

Output layout matches V4-Flash native naming so SGLang's V4 model loader can
iterate over the same per-expert and per-layer tensors. The new quantization
config tells SGLang which method to use:
  - routed-experts:    `gptq_marlin` W4A16 (group_size=32, sym=True)
  - attention/shared:  `gptq_marlin` W8A16 (group_size=128, sym=True)

Run on the existing local 1-layer snapshot:
    cd ~/Documents/appmana/forks-sglang
    source .venv/bin/activate
    python tools/ampere/dsv4_requant_shard.py \
        --src /var/lib/inference/v4-flash-1layer \
        --dst /var/lib/inference/v4-flash-1layer-int \
        --device cuda:1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from sglang.srt.layers.quantization.dsv4_aot_requantize import (  # noqa: E402
    dequantize_fp8_to_bf16,
    dequantize_mxfp4_to_bf16,
    is_routed_expert_weight,
    is_routed_expert_scale,
    is_fp8_weight,
    is_fp8_scale,
    matched_scale_name,
    requantize_fp8_to_int8_w8a16,
    requantize_mxfp4_to_int4_w4a16,
)


def _log(msg: str) -> None:
    print(f"[requant] {msg}", flush=True)


def _convert_shard(
    src_path: Path,
    dst_path: Path,
    device: str,
    out_scale_dtype: torch.dtype = torch.bfloat16,
    layer_remap: dict[int, int] | None = None,
) -> dict:
    """Read all tensors from src safetensors, requantize the relevant ones,
    write to dst. Returns a dict mapping HF-name -> {"dtype", "kind"} for the
    shard's quantization_config builder."""
    out_tensors: dict[str, torch.Tensor] = {}
    quant_info: dict[str, dict] = {}

    def _maybe_remap_layer(name: str) -> str | None:
        """Apply layer_remap to `layers.N.*` names. Returns the renamed name,
        or None if the source layer is filtered out (not in remap)."""
        if layer_remap is None or not name.startswith("layers."):
            return name
        parts = name.split(".", 2)
        try:
            n = int(parts[1])
        except (ValueError, IndexError):
            return name
        if n not in layer_remap:
            return None
        return f"layers.{layer_remap[n]}.{parts[2]}"

    with safe_open(src_path, framework="pt", device=device) as f:
        keys = sorted(f.keys())

        # Pre-build (weight, scale) pairs so we don't double-process scales.
        weight_to_scale: dict[str, str] = {}
        for k in keys:
            if is_routed_expert_weight(k) or is_fp8_weight(k):
                s = matched_scale_name(k)
                if s in keys:
                    weight_to_scale[k] = s
        handled_scales = set(weight_to_scale.values())

        n_int4 = n_int8 = n_pass = 0

        for k in keys:
            if k in handled_scales:
                continue  # consumed alongside its weight

            new_k = _maybe_remap_layer(k)
            if new_k is None:
                continue  # source layer filtered out by remap

            if k in weight_to_scale:
                s_name = weight_to_scale[k]
                new_s = _maybe_remap_layer(s_name)
                if new_s is None:
                    continue
                w = f.get_tensor(k)
                s = f.get_tensor(s_name)
                if is_routed_expert_weight(k):
                    out = requantize_mxfp4_to_int4_w4a16(
                        w, s, out_scale_dtype=out_scale_dtype
                    )
                    out_tensors[new_k] = out["qweight_packed"].cpu()
                    out_tensors[new_s] = out["scales"].cpu()
                    quant_info[new_k] = {"kind": "int4_w4a16", "group_size": 32}
                    n_int4 += 1
                else:
                    assert is_fp8_weight(k)
                    out = requantize_fp8_to_int8_w8a16(
                        w, s, out_scale_dtype=out_scale_dtype
                    )
                    out_tensors[new_k] = out["qweight"].cpu()
                    out_tensors[new_s] = out["scales"].cpu()
                    quant_info[new_k] = {"kind": "int8_w8a16", "block_size": [128, 128]}
                    n_int8 += 1
            else:
                # Passthrough: norms, RoPE, embed, head, hc_*, attn_sink, gate, etc.
                out_tensors[new_k] = f.get_tensor(k).cpu()
                n_pass += 1

    save_file(out_tensors, str(dst_path))
    _log(
        f"shard {src_path.name}: int4={n_int4}, int8={n_int8}, passthrough={n_pass}"
    )
    return quant_info


def _update_config(
    src_dir: Path, dst_dir: Path, all_quant_info: dict[str, dict]
) -> None:
    """Copy config.json -> dst_dir with a new quantization_config that lists
    per-tensor quant methods. SGLang's loader reads this to pick the right
    method per layer."""
    src_cfg = json.load(open(src_dir / "config.json"))
    cfg = dict(src_cfg)

    # Aggregate tensor patterns into the compressed-tensors-style config:
    # one group for INT4 (routed experts), one for INT8 (attention + shared).
    cfg["quantization_config"] = {
        "quant_method": "dsv4_int",
        "format": "int_packed",
        "config_groups": {
            "experts_w4a16": {
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": True,
                    "group_size": 32,
                    "strategy": "group",
                },
                "input_activations": {"num_bits": 16, "type": "float"},
                "targets": [
                    "*.ffn.experts.*.w1",
                    "*.ffn.experts.*.w2",
                    "*.ffn.experts.*.w3",
                ],
            },
            "attention_w8a16": {
                "weights": {
                    "num_bits": 8,
                    "type": "int",
                    "symmetric": True,
                    "block_size": [128, 128],
                    "strategy": "block",
                },
                "input_activations": {"num_bits": 16, "type": "float"},
                "targets": [
                    "*.attn.wq_a",
                    "*.attn.wq_b",
                    "*.attn.wkv",
                    "*.attn.wo_a",
                    "*.attn.wo_b",
                    "*.ffn.shared_experts.w1",
                    "*.ffn.shared_experts.w2",
                    "*.ffn.shared_experts.w3",
                ],
            },
        },
        "ignore": [
            "embed",
            "head",
            "norm",
            "lm_head",
            "*norm.weight",
            "attn.attn_sink",
            "*.gate.*",
            "hc_*",
            "*.hc_attn_*",
            "*.hc_ffn_*",
        ],
    }
    # Drop the fp8 quant config since we converted away from it.
    if "expert_dtype" in cfg:
        cfg["expert_dtype"] = "int4"  # was 'fp4'
    json.dump(cfg, open(dst_dir / "config.json", "w"), indent=2)
    _log(f"wrote {dst_dir / 'config.json'} with new quantization_config")

    # Copy the safetensors index, tokenizer, and generation_config as real
    # files (not symlinks) so the dst dir is a self-contained HF repo that can
    # be uploaded with `huggingface-cli upload` or mounted into pods.
    import shutil
    for fn in ("model.safetensors.index.json", "tokenizer.json",
               "tokenizer_config.json", "generation_config.json"):
        src_f = src_dir / fn
        if src_f.exists():
            dst_f = dst_dir / fn
            if dst_f.exists() or dst_f.is_symlink():
                dst_f.unlink()
            shutil.copy(src_f.resolve(), dst_f)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="V4-Flash snapshot dir")
    parser.add_argument("--dst", required=True, help="Output dir")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--layer-remap",
        default=None,
        help="JSON dict mapping src-layer-idx -> dst-layer-idx, e.g. "
        '\'{"0":0,"1":1,"41":2,"42":3}\' to use layers 0,1,41,42 of source as '
        "0,1,2,3 of dst. Tensors for source layers not in the dict are dropped.",
    )
    parser.add_argument(
        "--scale-dtype",
        choices=["bf16", "fp16"],
        default="bf16",
        help="Output scale dtype. BF16 covers e8m0's full range; FP16 saturates "
        "for very small/large blocks. Default BF16.",
    )
    args = parser.parse_args()

    src_dir = Path(args.src).resolve()
    dst_dir = Path(args.dst).resolve()
    assert src_dir.is_dir(), f"src not a dir: {src_dir}"
    dst_dir.mkdir(parents=True, exist_ok=True)

    out_scale_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.scale_dtype]

    layer_remap = None
    if args.layer_remap:
        layer_remap = {int(k): int(v) for k, v in json.loads(args.layer_remap).items()}
        _log(f"layer_remap: {layer_remap}")

    # Find safetensors shards in src (resolving symlinks if any).
    shards = sorted(src_dir.glob("*.safetensors"))
    if not shards:
        _log(f"no .safetensors files in {src_dir}")
        return 1
    _log(f"found {len(shards)} shards in {src_dir}")

    all_quant_info: dict[str, dict] = {}
    for shard in shards:
        dst_shard = dst_dir / shard.name
        _log(f"-> {shard.name}")
        info = _convert_shard(
            shard, dst_shard, args.device, out_scale_dtype,
            layer_remap=layer_remap,
        )
        all_quant_info.update(info)

    _update_config(src_dir, dst_dir, all_quant_info)
    _log(f"DONE  total quantized tensors: {len(all_quant_info)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
