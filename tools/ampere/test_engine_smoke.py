"""
Smoke test: run SGLang Engine with a tiny non-V4 model to verify the fork
isn't broken at the engine/scheduler level.
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("SGLANG_APPLY_CONFIG_BACKUP", "none")


def main():
    import sglang as sgl
    print("[smoke] creating Engine with Qwen3-0.6B...", flush=True)
    llm = sgl.Engine(
        model_path="Qwen/Qwen3-0.6B",
        trust_remote_code=True,
        mem_fraction_static=0.3,
        disable_cuda_graph=True,
        skip_server_warmup=True,
        log_level="info",
    )
    print("[smoke] generating...", flush=True)
    out = llm.generate(
        prompt="The capital of France is",
        sampling_params={"max_new_tokens": 8, "temperature": 0.0},
    )
    print(f"[smoke] OUTPUT: {out!r}", flush=True)
    llm.shutdown()


if __name__ == "__main__":
    main()
