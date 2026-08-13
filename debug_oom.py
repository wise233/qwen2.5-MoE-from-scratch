import argparse, torch, deepspeed
from model import Qwen2Config, Qwen2ForCausalLM

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--block", type=int, default=2048)
    ap.add_argument("--ckpt", action="store_true")
    ap.add_argument("--local_rank", type=int, default=-1)
    ap = deepspeed.add_config_arguments(ap)
    args = ap.parse_args()

    torch.manual_seed(0)
    cfg = Qwen2Config()
    if args.ckpt:
        cfg.use_act_ckpt = True
    model = Qwen2ForCausalLM(cfg)
    n = sum(p.numel() for p in model.parameters())

    engine, _, _, _ = deepspeed.initialize(
        args=args, model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad])
    r = engine.global_rank
    dev = engine.device

    bf16p = sum(1 for p in model.parameters() if p.dtype == torch.bfloat16)
    def mem(tag):
        a = torch.cuda.memory_allocated()/1e9
        if r == 0: print(f"  [{tag}] alloc={a:.2f}GB", flush=True)

    torch.cuda.reset_peak_memory_stats()
    mem("after-init")
    mb, blk = args.micro, args.block
    ids = torch.randint(0, cfg.vocab_size, (mb, blk), device=dev)
    loss, logits, _ = engine(ids, labels=ids)
    mem("after-forward")
    engine.backward(loss)
    torch.cuda.synchronize()
    mem("after-backward")
    peak = torch.cuda.max_memory_allocated()/1e9
    if r == 0:
        print(f"RESULT micro={mb} block={blk} | params_bf16={bf16p} | total={n/1e6:.0f}M | "
              f"peak_alloc={peak:.2f}GB | logits={logits.numel()*2/1e9:.2f}GB", flush=True)

if __name__ == "__main__":
    main()
