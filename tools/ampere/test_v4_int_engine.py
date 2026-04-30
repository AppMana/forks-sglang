"""
Test V4-Flash with our INT4/INT8 quantization using SGLang's offline Engine
API. This bypasses HTTP/uvicorn/zmq entirely so we can see if the actual model
forward produces output -- isolated from any server-side IPC issues.
"""
import os
import sys

# Match env from our server test
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("SGLANG_DSV4_FP4_EXPERTS", "1")
os.environ.setdefault("SGLANG_JIT_DEEPGEMM_PRECOMPILE", "0")
os.environ.setdefault("SGLANG_APPLY_CONFIG_BACKUP", "none")
os.environ.setdefault("SGLANG_DSV4_MODE", "2604")
os.environ.setdefault("SGLANG_DSV4_2604_SUBMODE", "2604B")
os.environ.setdefault("SGLANG_FP8_PAGED_MQA_LOGITS_TORCH", "1")
os.environ.setdefault("SGLANG_OPT_FP8_WO_A_GEMM", "0")
os.environ.setdefault("SGLANG_OPT_FUSE_WQA_WKV", "0")
os.environ.setdefault("SGLANG_HACK_FLASHMLA_BACKEND", "zero")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MODEL = "/var/lib/inference/v4-flash-1layer-int"


def main():
    print(f"[engine-test] importing sglang.Engine...", flush=True)
    import sglang as sgl

    print(f"[engine-test] creating Engine for {MODEL}", flush=True)
    engine = sgl.Engine(
        model_path=MODEL,
        tokenizer_path=MODEL,
        trust_remote_code=True,
        mem_fraction_static=0.85,
        disable_cuda_graph=True,
        skip_server_warmup=True,
        context_length=4096,
        max_total_tokens=4096,
        max_running_requests=1,
        chunked_prefill_size=256,
        nsa_prefill_backend="tilelang",
        nsa_decode_backend="tilelang",
    )

    print(f"[engine-test] engine ready, generating...", flush=True)
    out = engine.generate(
        prompt="hi",
        sampling_params={"max_new_tokens": 4, "temperature": 0.0},
    )
    print(f"[engine-test] OUTPUT: {out!r}", flush=True)
    engine.shutdown()


if __name__ == "__main__":
    main()
