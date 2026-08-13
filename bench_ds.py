"""
bench_ds.py —— 实测 2xRTX4090 上的吞吐与显存
只喂随机 token，绕开数据文件，直接测量给定 ds_config 下
micro_batch 能否放得下、实际 tok/s、每步耗时、峰值显存。
"""
import argparse, time
import torch, deepspeed
from model import Qwen2Config, Qwen2ForCausalLM

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--micro", type=int, default=None)
    ap.add_argument("--block", type=int, default=2048)
    ap.add_argument("--vocab", type=int, default=151936, help="词表大小")
    ap.add_argument("--local_rank", type=int, default=-1)  # deepspeed launcher 注入
    ap = deepspeed.add_config_arguments(ap)
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg = Qwen2Config(vocab_size=args.vocab)
    model = Qwen2ForCausalLM(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    engine, _, _, _ = deepspeed.initialize(
        args=args, model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
    )
    device = engine.device
    mb = args.micro or engine.train_micro_batch_size_per_gpu()
    acc = engine.gradient_accumulation_steps()
    blk = args.block

    if engine.global_rank == 0:
        print(f"params={n_params/1e6:.0f}M | gpus={engine.world_size} | micro={mb} accum={acc} "
              f"block={blk} | global_batch_tokens={mb*acc*engine.world_size*blk/1e3:.0f}k | "
              f"gpu={torch.cuda.get_device_name()}", flush=True)

    for _ in range(2):
        for _ in range(acc):
            ids = torch.randint(0, cfg.vocab_size, (mb, blk), device=device)
            loss, logits, _ = engine(ids, labels=ids)
            engine.backward(loss)
        engine.step()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    toks = 0
    for step in range(args.steps):
        for _ in range(acc):
            ids = torch.randint(0, cfg.vocab_size, (mb, blk), device=device)
            loss, logits, _ = engine(ids, labels=ids)
            engine.backward(loss)
            toks += ids.numel()
        engine.step()
        if engine.global_rank == 0 and (step + 1) % 6 == 0:
            el = time.time() - t0
            print(f"  [step {step+1}/{args.steps}] {toks/el/1e3:.0f}k tok/s", flush=True)
    torch.cuda.synchronize()
    el = time.time() - t0

    if engine.global_rank == 0:
        print(f"RESULT steps={args.steps} elapsed={el:.1f}s | tok/s={toks/el:.0f} "
              f"({toks/el/1e3:.1f}k) | per_step={el/args.steps:.2f}s", flush=True)
        print(f"RESULT peak_alloc={torch.cuda.max_memory_allocated()/1e9:.2f}GB "
              f"peak_reserved={torch.cuda.max_memory_reserved()/1e9:.2f}GB", flush=True)

if __name__ == "__main__":
    main()
