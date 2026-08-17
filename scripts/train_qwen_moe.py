"""
train_qwen_moe.py
============================================================================
数据：scripts/prepare_fineweb.py 产出的 train.bin / val.bin / meta.json（uint32）
模型：model.py（仓库根目录）里的 Qwen2ForCausalLM（MoE 魔改版）
DeepSpeed：
    bf16 混合精度          <-> configs/ds_config.json 的 "bf16"
    AdamW 优化器 + 权重衰减  <-> configs/ds_config.json 的 "optimizer"
    预热 + 余弦退火          <-> configs/ds_config.json 的 "scheduler"
    梯度累积（backward 自动按累积数缩放 loss）<-> configs/ds_config.json 的 "gradient_accumulation_steps"
    梯度裁剪                <-> configs/ds_config.json 的 "gradient_clipping"
    周期存档 / 断点续训       <-> configs/ds_config.json 的 "checkpoint"

用法（在仓库根目录下执行）：
    # 单卡冒烟（本机 4GB GPU，小模型）：
    deepspeed --num_gpus=1 scripts/train_qwen_moe.py --deepspeed --deepspeed_config configs/ds_config_tiny.json \
        --tiny --block_size 256 --max_steps 50

    # 正式多卡训练（配置与默认 --max_steps 20000 配套）：
    deepspeed --num_gpus=8 scripts/train_qwen_moe.py --deepspeed --deepspeed_config configs/ds_config.json \
        --hellaswag_every 1000 --hellaswag_samples 1000

    # 断点续训（DeepSpeed 自动恢复 model/optimizer/lr_scheduler/global_steps）：
    deepspeed --num_gpus=8 scripts/train_qwen_moe.py --deepspeed --deepspeed_config configs/ds_config.json --resume

注意：
- --max_steps 必须与 ds_config 里 scheduler 的 total_num_steps 一致（脚本会警告）。
- micro_batch 是"每卡"的微批大小，全局有效批 = micro_batch × accum × num_gpus；
  多卡时记得相应调小 ds_config 里的 train_micro_batch_size_per_gpu。
- 训练循环仍保留一个微批循环：DeepSpeed 无法替你决定一个数据块的边界，
  但 loss 缩放、梯度累积、优化器更新时机、裁剪、存档都交给 engine 内部处理。
- 数据 loader 按 rank 错开种子，多卡时各 rank 读不同批次；模型初始化种子各 rank 一致。
- 损失约定与 model.py 一致：model(x, labels=x)，模型内部自己移位，loader 只返回 x。
- 变长批量评测一律 batch=1（见 evaluate_hellaswag），不补零。
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import deepspeed

# 脚本在 scripts/ 下，把仓库根目录加进 sys.path 才能 import 根目录的 model.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Qwen2Config, Qwen2ForCausalLM

# HellaSwag 评测需要分词器把文本编成 token（本地缓存找不到就禁用该评测）
try:
    from transformers import AutoTokenizer

    TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", local_files_only=True)
except Exception:
    TOKENIZER = None


# ==================== 数据加载 ====================
class TokenLoader:
    """从 uint32 bin 文件里切 block_size 的 token 块"""

    def __init__(self, bin_path, block_size, device, seed=1337):
        self.tokens = np.memmap(bin_path, dtype=np.uint32, mode="r")
        self.n = len(self.tokens)
        assert self.n > block_size + 1, f"{bin_path} 的 token 数 {self.n} 小于 block_size+1"
        self.block_size = block_size
        self.device = device
        self.rng = np.random.default_rng(seed)  # 独立 RNG，不污染模型采样用的 torch RNG
        self.val_pos = 0  # 验证集顺序扫描指针

    def _chunk(self, start):
        # 取 [start, start+block_size) 的 token 并转成 torch.long
        return torch.from_numpy(self.tokens[start : start + self.block_size].astype(np.int64))

    def get_batch_train(self, batch_size):
        """训练：随机起点，一个 batch 的 token 块 [batch, block_size]"""
        ix = self.rng.integers(0, self.n - self.block_size, size=batch_size)
        xs = [self._chunk(int(i)) for i in ix]
        return torch.stack(xs).to(self.device)

    def get_batch_val(self, batch_size):
        """验证：顺序扫非重叠块（多次评测结果更稳定），扫完绕回"""
        if self.val_pos + batch_size * self.block_size > self.n - 1:
            self.val_pos = 0
        xs = []
        for _ in range(batch_size):
            xs.append(self._chunk(self.val_pos))
            self.val_pos += self.block_size
        return torch.stack(xs).to(self.device)


# ==================== 评测 ====================
@torch.no_grad()
def evaluate(model, loader, batch_size, num_iters):
    """在验证集上算平均 loss（含辅助损失）"""
    model.eval()
    total = 0.0
    for _ in range(num_iters):
        x = loader.get_batch_val(batch_size)
        loss, _, _ = model(x, labels=x)
        total += loss.item()
    model.train()
    return total / num_iters


@torch.no_grad()
def evaluate_hellaswag(model, tokenizer, ds, device):
    """在 HellaSwag 样本上算 4 选 1 准确率（官方 acc_norm：逐 token 平均 NLL，长度归一化，最小者胜）

    实现：每个样本的 prompt 前向一次拿 KV 缓存，各候选结尾逐个（batch=1）前向复用该缓存。
    已验证该方式与"prompt+结尾整串前向"逐位置完全一致；同时省掉 3 次 prompt 重复前向。
    不用批量补零结尾（实测该路径在部分场景下结果不稳定，batch=1 最稳妥）。
    """
    model.eval()
    try:
        correct = total = 0
        for ex in ds:
            prompt_ids = tokenizer.encode(ex["ctx"], add_special_tokens=False)
            if not prompt_ids:  # 空上下文，跳过
                continue
            # 1) prompt 前向一次：拿末尾 logits（预测结尾第一个 token）与各层 KV 缓存
            pt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            _, prompt_logits, kv = model(pt, use_cache=True)
            first_logit = prompt_logits[:, -1:, :]  # (1, 1, vocab)：预测 ending[0]

            # 2) 各候选结尾逐个前向（复用 prompt 的 KV），按真实长度算 mean-CE（官方 acc_norm）
            losses = []
            for e in ex["endings"]:
                ending_ids = tokenizer.encode(e, add_special_tokens=False)
                L = len(ending_ids)
                if L == 0:  # 空结尾：视作无穷差，不会胜出（与官方脚本一致）
                    losses.append(float("inf"))
                    continue
                et = torch.tensor([ending_ids], dtype=torch.long, device=device)
                _, end_logits, _ = model(et, past_key_values=kv)  # (1, L, vocab)
                # end_logits 位置 j 预测 ending[j+1]（取 [:, :-1]），拼上 first_logit 后位置 k 恰好预测 ending[k]
                all_logits = torch.cat([first_logit, end_logits[:, :-1, :]], dim=1)[0]  # (L, vocab)
                losses.append(
                    F.cross_entropy(all_logits.reshape(-1, all_logits.shape[-1]),
                                    et[0], reduction="mean").item()
                )
            pred = min(range(len(losses)), key=losses.__getitem__)
            correct += int(pred == int(ex["label"]))
            total += 1
        return correct / total if total else 0.0
    finally:
        model.train()  # 无论是否中途出错，评测完都切回训练模式


# ==================== 主流程 ====================
def parse_args():
    ap = argparse.ArgumentParser(description="train Qwen2.5-MoE (DeepSpeed)")
    ap.add_argument("--data_dir", default="fineweb-edu/bin", help="bin 数据目录（含 meta.json）")
    ap.add_argument("--tiny", action="store_true", help="用超小模型快速冒烟，验证整套流程")
    # 数据规模
    ap.add_argument("--block_size", type=int, default=2048, help="每块 token 数")
    ap.add_argument("--max_steps", type=int, default=20000, help="训练总步数（需与 ds_config 的 total_num_steps 一致）")
    ap.add_argument("--seed", type=int, default=1337)
    # 评测
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--eval_iters", type=int, default=20)
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--hellaswag_every", type=int, default=1000, help="每隔多少步跑一次 HellaSwag 准确率，0=关")
    ap.add_argument("--hellaswag_samples", type=int, default=200, help="HellaSwag 评测用样本数（验证集全量 10042，训练中评测子集即可）")
    ap.add_argument("--resume", action="store_true", help="从 DeepSpeed checkpoint（ds_config 的 checkpoint_dir）续训")
    ap.add_argument("--local_rank", type=int, default=-1, help="deepspeed launcher 注入")
    ap = deepspeed.add_config_arguments(ap)  # 追加 --deepspeed / --deepspeed_config
    return ap.parse_args()


def main():
    args = parse_args()
    if not args.deepspeed_config:
        raise SystemExit("必须指定 --deepspeed_config ds_config.json（配合 deepspeed 启动器使用）")

    # ---- 随机种子：模型初始化各 rank 保持一致；数据 loader 下面按 rank 错开 ----
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ---- 数据 meta ----
    meta = json.load(open(os.path.join(args.data_dir, "meta.json"), encoding="utf-8"))
    train_path = os.path.join(args.data_dir, "train.bin")
    val_path = os.path.join(args.data_dir, "val.bin")

    # ---- 模型（默认 0.55B MoE；--tiny 用小配置冒烟），先留 CPU 交给 DeepSpeed 搬上 GPU ----
    cfg = Qwen2Config()
    if args.tiny:
        cfg.hidden_size = 256
        cfg.mlp_mid_size = 1024
        cfg.layers = 4
        cfg.num_att_heads = 4
        cfg.num_kv_heads = 2
        cfg.moe_intermediate_size = 256
        cfg.num_local_experts = 4
        cfg.n_group = 1  # 与正式配置一致：去掉分组路由
        cfg.topk_group = 1
        cfg.first_k_dense_replace = 1

    model = Qwen2ForCausalLM(cfg)

    # ---- DeepSpeed 初始化：优化器/调度/AMP/裁剪/梯度累积/存档全由 engine 接管 ----
    engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
    )

    device = engine.device
    micro_batch = engine.train_micro_batch_size_per_gpu()
    accum_steps = engine.gradient_accumulation_steps()
    is_main = engine.global_rank == 0

    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        eff_tokens = micro_batch * accum_steps * args.block_size
        print(f"模型配置: layers={cfg.layers} hidden={cfg.hidden_size} | 参数量: {n_params/1e6:.1f}M")
        print(f"DeepSpeed: ZeRO stage={engine.zero_optimization_stage()} | micro_batch={micro_batch} "
              f"accum={accum_steps} | 有效批 {eff_tokens/1e3:.0f}k tokens/step | "
              f"全量 train 约 {meta['n_train_tokens']/eff_tokens:.0f} 步/epoch")

    # ---- 一致性检查：--max_steps 与配置里调度器的 total_num_steps 必须一致 ----
    ds_cfg = json.load(open(args.deepspeed_config, encoding="utf-8"))
    ckpt_dir = ds_cfg.get("checkpoint", {}).get("checkpoint_dir", "out-qwen-moe/ds-checkpoints")
    sched_total = ds_cfg.get("scheduler", {}).get("params", {}).get("total_num_steps")
    if sched_total is not None and sched_total != args.max_steps:
        print(f"[warn] ds_config scheduler.total_num_steps={sched_total} 与 --max_steps={args.max_steps} "
              f"不一致，余弦退火会按 {sched_total} 算，请改成一致再跑")

    # ---- 数据加载器（各 rank 种子错开，多卡读不同批次）----
    train_loader = TokenLoader(train_path, args.block_size, device, seed=args.seed + engine.global_rank)
    val_loader = TokenLoader(val_path, args.block_size, device, seed=args.seed + engine.global_rank)

    # ---- 断点续训（DeepSpeed 恢复 model/optimizer/lr_scheduler/global_steps）----
    if args.resume:
        path, _ = engine.load_checkpoint(ckpt_dir)
        if path is None:
            print("[warn] 未找到可恢复的 checkpoint，从头训练")
        elif is_main:
            print(f"已从 {path} 恢复：step={engine.global_steps}")

    # ---- HellaSwag 数据（训练中周期性评测准确率，仅主 rank 加载/跑）----
    hellaswag_ds = None
    if is_main and args.hellaswag_every > 0:
        try:
            from datasets import load_dataset

            hellaswag_ds = load_dataset("hellaswag", split="validation")
            if args.hellaswag_samples < len(hellaswag_ds):
                hellaswag_ds = hellaswag_ds.select(range(args.hellaswag_samples))
            print(f"HellaSwag 数据就绪：本次共评测 {len(hellaswag_ds)} 条")
        except Exception as e:
            print(f"[warn] HellaSwag 加载失败，本次训练跳过该评测: {e}")

    # ---- 训练循环 ----
    best_val = float("inf")  # 会话内的最佳验证损失（不跨断点持久化，仅用于日志标记）
    t0 = time.time()
    tokens_seen = 0  # 累计看过的 token（含梯度累积）
    for step in range(engine.global_steps, args.max_steps + 1):
        loss_accum = 0.0  # 各微批的平均损失（backward 已自动除 accum_steps，这里同步除一下便于显示）
        for _ in range(accum_steps):
            x = train_loader.get_batch_train(micro_batch)
            tokens_seen += x.numel()
            loss, logits, _ = engine(x, labels=x)  # HF 约定：labels=x，模型内部移位
            engine.backward(loss)  # DeepSpeed 内部按 grad_accum 缩放 loss
            loss_accum += loss.item() / accum_steps
        engine.step()  # 仅在累积边界真正更新优化器；并按 save_interval 自动存档

        # ---- 日志 ----
        if is_main and args.log_interval > 0 and step % args.log_interval == 0:
            el = time.time() - t0
            speed = tokens_seen / el
            lm = F.cross_entropy(logits[:, :-1].reshape(-1, cfg.vocab_size),
                                 x[:, 1:].reshape(-1)).item()  # 纯语言建模损失（辅助损失项 ≈ loss - lm）
            print(f"step {step:>6d} | loss {loss_accum:.4f} | lm {lm:.4f} "
                  f"| aux项 {loss_accum - lm:.4f} | {speed/1e3:.0f}k tok/s | {el/60:.1f}min")

        # ---- 验证 ----
        if is_main and step > 0 and args.eval_every > 0 and step % args.eval_every == 0:
            val_loss = evaluate(engine.module, val_loader, micro_batch, args.eval_iters)
            improved = val_loss < best_val
            best_val = min(best_val, val_loss)
            print(f"step {step:>6d} | val_loss {val_loss:.4f}" + ("  [新最佳]" if improved else ""))

        # ---- HellaSwag 准确率（替代采样）----
        if (is_main and TOKENIZER is not None and hellaswag_ds is not None
                and args.hellaswag_every > 0 and step > 0 and step % args.hellaswag_every == 0):
            acc = evaluate_hellaswag(engine.module, TOKENIZER, hellaswag_ds, device)
            print(f"step {step:>6d} | HellaSwag Acc {acc:.4f}（{len(hellaswag_ds)} 条）")

    if is_main:
        print(f"\n训练完成（{args.max_steps} 步）。周期存档由 DeepSpeed 写入 {ckpt_dir}，可用 --resume 续训")


if __name__ == "__main__":
    main()
