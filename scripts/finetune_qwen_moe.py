"""
finetune_qwen_moe.py —— 对魔改版 Qwen2.5-MoE 做 SFT / DPO 微调
============================================================================
直接吃 model.py 的 Qwen2ForCausalLM 和 DeepSpeed checkpoint，不需要转成 HF 格式。
分两步（与 scripts/prepare_hellaswag_sft.py / scripts/prepare_hellaswag_dpo.py 配套）。

在仓库根目录下执行：

    # 0) 数据准备
    python scripts/prepare_hellaswag_sft.py            # -> data/hellaswag_sft.jsonl
    python scripts/prepare_hellaswag_dpo.py            # -> data/hellaswag_dpo.jsonl

    # 1) SFT：ctx -> 正确结尾
    deepspeed --num_gpus=1 scripts/finetune_qwen_moe.py --stage sft \
        --data data/hellaswag_sft.jsonl \
        --deepspeed --deepspeed_config configs/ds_config_sft.json --max_steps 1500 \
        [--load_model_dir <预训练 DeepSpeed checkpoint 目录，可选>]

    # 2) DPO：正确结尾(chosen) > 错误结尾(rejected)，参考模型 = SFT 权重冻结副本
    deepspeed --num_gpus=1 scripts/finetune_qwen_moe.py --stage dpo \
        --data data/hellaswag_dpo.jsonl \
        --load_model_dir saves/finetune-sft/ds-checkpoints \
        --deepspeed --deepspeed_config configs/ds_config_dpo.json --max_steps 1900

本机没有 DeepSpeed 时用纯 torch 冒烟（--no_deepspeed，走 PlainRunner）：
    python scripts/finetune_qwen_moe.py --stage sft --tiny --no_deepspeed --batch_size 2 \
        --grad_accum_steps 4 --max_steps 5 --data data/hellaswag_sft.jsonl \
        --hellaswag_every 0 --save_every 0

要点：
- SFT：prompt(上下文) 的标签填 -100（模型 cross_entropy 已支持 ignore_index），只监督 response+eos。
- DPO：标准 DPO loss = -log σ( β*(πθ(y_w)-πref(y_w)-πθ(y_l)+πref(y_l)) )，
  logprob 按"response 区间逐 token 平均"（长度归一化，与 HellaSwag 的 acc_norm 哲学一致）。
- 两种后台共用同一套 Runner 接口：DeepSpeed 引擎 或 纯 torch（手动优化器/调度/裁剪）。
- 评测：周期性跑 HellaSwag 准确率（复用 evaluate_hellaswag 的 batch=1 + KV 缓存实现）。
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# 脚本在 scripts/ 下，把仓库根目录加进 sys.path 才能 import 根目录的 model.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Qwen2Config, Qwen2ForCausalLM

IGNORE_INDEX = -100  # 标签里不参与损失的 token id（与 model.py 的 config.ignore_index 一致）

# DeepSpeed 可选：本机没有时自动退化为纯 torch 路径
try:
    import deepspeed

    DS_AVAILABLE = True
except ImportError:
    deepspeed = None
    DS_AVAILABLE = False

# HellaSwag 评测需要分词器（本地缓存找不到就禁用该评测）
try:
    from transformers import AutoTokenizer

    TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", local_files_only=True)
    if TOKENIZER.pad_token is None:
        TOKENIZER.pad_token = TOKENIZER.eos_token  # Qwen2 约定：pad 复用 <|endoftext|>
except Exception:
    TOKENIZER = None


# ==================== 数据 ====================
class SFTDataset(Dataset):
    """SFT 数据：prompt(上下文) 掩码不参与损失，只监督 response(正确结尾)+eos"""

    def __init__(self, data, tokenizer, max_len):
        self.examples = []
        for r in data:
            pids = tokenizer.encode(r["prompt"], add_special_tokens=False)
            rids = tokenizer.encode(r["response"], add_special_tokens=False) + [tokenizer.eos_token_id]
            rids = rids[: max(0, max_len - len(pids))]  # 截断：优先保留 prompt
            if not rids:
                continue  # response 全被截掉，跳过
            input_ids = pids + rids
            labels = [IGNORE_INDEX] * len(pids) + rids
            self.examples.append((input_ids, labels))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def sft_collate(batch, pad_id):
    """右 padding 到 batch 内最长；labels 的 padding 位填 -100，attention_mask 记 0"""
    max_len = max(len(x) for x, _ in batch)
    ids, labels, masks = [], [], []
    for x, y in batch:
        p = max_len - len(x)
        ids.append(x + [pad_id] * p)
        labels.append(y + [IGNORE_INDEX] * p)
        masks.append([1] * len(x) + [0] * p)
    return (
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(masks, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


class DPODataset(Dataset):
    """DPO 数据：每个样本存 (prompt 长度, chosen 全长, rejected 全长)"""

    def __init__(self, data, tokenizer, max_len):
        self.examples = []
        for r in data:
            pids = tokenizer.encode(r["prompt"], add_special_tokens=False)
            if not pids:
                continue
            cids = tokenizer.encode(r["chosen"], add_special_tokens=False) + [tokenizer.eos_token_id]
            rids = tokenizer.encode(r["rejected"], add_special_tokens=False) + [tokenizer.eos_token_id]
            cids = cids[: max(0, max_len - len(pids))]
            rids = rids[: max(0, max_len - len(pids))]
            if not cids or not rids:
                continue
            self.examples.append((len(pids), pids + cids, pids + rids))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def dpo_collate(batch, pad_id):
    """chosen/rejected 各右 padding 到 batch 内最长，并记录每行的 response 起点与长度"""
    max_len = max(max(len(c), len(j)) for _, c, j in batch)
    ch_ids, ch_mask, rej_ids, rej_mask = [], [], [], []
    ch_start, ch_len, rej_start, rej_len = [], [], [], []
    for pstart, c, j in batch:
        ch_ids.append(c + [pad_id] * (max_len - len(c)))
        ch_mask.append([1] * len(c) + [0] * (max_len - len(c)))
        ch_start.append(pstart)
        ch_len.append(len(c) - pstart)
        rej_ids.append(j + [pad_id] * (max_len - len(j)))
        rej_mask.append([1] * len(j) + [0] * (max_len - len(j)))
        rej_start.append(pstart)
        rej_len.append(len(j) - pstart)
    return (
        torch.tensor(ch_ids, dtype=torch.long), torch.tensor(ch_mask, dtype=torch.long),
        torch.tensor(rej_ids, dtype=torch.long), torch.tensor(rej_mask, dtype=torch.long),
        torch.tensor(ch_start, dtype=torch.long), torch.tensor(ch_len, dtype=torch.long),
        torch.tensor(rej_start, dtype=torch.long), torch.tensor(rej_len, dtype=torch.long),
    )


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ==================== DPO 工具 ====================
def response_logprobs(model_call, input_ids, attention_mask, resp_start, resp_len):
    """对每行的 response 区间算平均 logprob（预测下一 token，长度归一化）

    model_call 接受 input_ids / attention_mask 关键字并返回元组，元组[1] 为 logits。
    输入 token 位置 i 的 logprob 来自 logits[i-1]（模型内部错位），因此 response 区间
    [resp_start, resp_start+resp_len) 的 logprob 落在 per_token 的 [resp_start-1, ...) 列。
    """
    logits = model_call(input_ids=input_ids, attention_mask=attention_mask)[1]  # (B, S, V)
    shift_logits = logits[:, :-1, :]  # (B, S-1, V)
    shift_targets = input_ids[:, 1:]  # (B, S-1)
    # logprob(target) = logit[target] - logsumexp(logits)，避免物化 [B,S,V] 的 log_softmax 省显存
    target_logits = shift_logits.gather(-1, shift_targets.unsqueeze(-1)).squeeze(-1).float()  # (B, S-1)
    logsumexp = torch.logsumexp(shift_logits.float(), dim=-1)  # (B, S-1)
    per_token = target_logits - logsumexp  # (B, S-1)
    out = torch.zeros(input_ids.shape[0], device=input_ids.device, dtype=torch.float32)
    for b in range(input_ids.shape[0]):
        s, L = int(resp_start[b]), int(resp_len[b])
        if L > 0:
            out[b] = per_token[b, s - 1 : s - 1 + L].mean()
    return out


def dpo_loss(fwd, ref_model, batch, beta):
    """标准 DPO loss：-log σ( β*(πθ(yw)-πref(yw)-πθ(yl)+πref(yl)) )，batch 平均"""
    ch_ids, ch_mask, rej_ids, rej_mask, ch_start, ch_len, rej_start, rej_len = batch
    pol_ch = response_logprobs(fwd, ch_ids, ch_mask, ch_start, ch_len)
    pol_re = response_logprobs(fwd, rej_ids, rej_mask, rej_start, rej_len)
    with torch.no_grad():
        ref_ch = response_logprobs(ref_model, ch_ids, ch_mask, ch_start, ch_len)
        ref_re = response_logprobs(ref_model, rej_ids, rej_mask, rej_start, rej_len)
    # 隐式奖励差：chosen 应比 rejected 更被策略偏爱（扣除参考基线）
    reward_diff = (pol_ch - ref_ch) - (pol_re - ref_re)
    return -F.logsigmoid(beta * reward_diff).mean()


# ==================== 训练后台（DeepSpeed / 纯 torch 统一接口） ====================
class DeepSpeedRunner:
    def __init__(self, args, model):
        self.engine, _, _, _ = deepspeed.initialize(
            args=args,
            model=model,
            model_parameters=[p for p in model.parameters() if p.requires_grad],
        )
        self.micro_batch = self.engine.train_micro_batch_size_per_gpu()
        self.accum_steps = self.engine.gradient_accumulation_steps()
        self.is_main = self.engine.global_rank == 0
        self.global_rank = self.engine.global_rank
        self.device = self.engine.device

    @property
    def module(self):
        return self.engine.module

    @property
    def global_steps(self):
        return self.engine.global_steps

    @property
    def lr(self):
        lrs = self.engine.get_lr()
        return lrs[0] if isinstance(lrs, list) else lrs

    def forward(self, **kw):
        return self.engine(**kw)

    def backward(self, loss):
        self.engine.backward(loss)

    def step(self):
        self.engine.step()

    def save(self, ckpt_dir, tag, step, best_acc):
        self.engine.save_checkpoint(
            ckpt_dir, tag=tag, client_state={"step": step, "best_acc": best_acc}
        )

    def load(self, ckpt_dir, load_opt=True):
        # tag=None 时 DeepSpeed 自动找最新的 checkpoint
        return self.engine.load_checkpoint(
            ckpt_dir, tag=None,
            load_optimizer_states=load_opt, load_lr_scheduler_states=load_opt,
        )


class PlainRunner:
    """纯 torch 后台：手动优化器/余弦调度/梯度裁剪，本机冒烟或不想用 DeepSpeed 时用"""

    def __init__(self, args, model):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.micro_batch = args.batch_size
        self.accum_steps = args.grad_accum_steps
        self.is_main = True
        self.global_rank = 0
        self.global_steps = 0
        self.dtype = self._resolve_dtype(args)

        # 2 维及以上权重做 weight_decay，1 维（bias / 归一化）不做
        decay, nodecay = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else nodecay).append(p)
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": args.weight_decay},
                {"params": nodecay, "weight_decay": 0.0},
            ],
            lr=args.max_lr, betas=tuple(args.betas),
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self._lr_lambda(args))
        self.grad_clip = args.grad_clip

    @staticmethod
    def _resolve_dtype(args):
        if args.dtype == "fp32":
            return torch.float32
        if args.dtype == "bf16":
            return torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float32

    def _lr_lambda(self, args):
        warmup, max_steps, max_lr, min_lr = args.warmup_steps, args.max_steps, args.max_lr, args.min_lr

        def f(step):
            if warmup > 0 and step < warmup:
                return (step + 1) / warmup
            if step >= max_steps:
                return min_lr / max_lr
            prog = (step - warmup) / max(1, max_steps - warmup)
            return min_lr / max_lr + 0.5 * (1 - min_lr / max_lr) * (1 + math.cos(math.pi * prog))

        return f

    @property
    def module(self):
        return self.model

    @property
    def lr(self):
        return self.scheduler.get_last_lr()[0]

    def forward(self, **kw):
        if self.dtype == torch.bfloat16 and self.device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return self.model(**kw)
        return self.model(**kw)

    def backward(self, loss):
        loss.backward()

    def step(self):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        self.global_steps += 1

    def save(self, ckpt_dir, tag, step, best_acc):
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "step": step, "global_steps": self.global_steps, "best_acc": best_acc,
            },
            os.path.join(ckpt_dir, f"{tag}.pt"),
        )

    def load(self, ckpt_dir, load_opt=True):
        path = os.path.join(ckpt_dir, "latest.pt")
        if not os.path.exists(path):
            return None, None
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model"])
        if load_opt:
            self.optimizer.load_state_dict(ck["optimizer"])
            self.scheduler.load_state_dict(ck["scheduler"])
            self.global_steps = ck.get("global_steps", 0)
        return path, ck


# ==================== HellaSwag 评测（复用 batch=1 + KV 缓存实现） ====================
@torch.no_grad()
def evaluate_hellaswag(model, tokenizer, ds, device):
    """官方 acc_norm：逐 token 平均 NLL，长度归一化，最小者胜；prompt 只前向一次拿 KV"""
    model.eval()
    try:
        correct = total = 0
        for ex in ds:
            prompt_ids = tokenizer.encode(ex["ctx"], add_special_tokens=False)
            if not prompt_ids:
                continue
            pt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            _, prompt_logits, kv = model(pt, use_cache=True)
            first_logit = prompt_logits[:, -1:, :]
            losses = []
            for e in ex["endings"]:
                ending_ids = tokenizer.encode(e, add_special_tokens=False)
                L = len(ending_ids)
                if L == 0:
                    losses.append(float("inf"))
                    continue
                et = torch.tensor([ending_ids], dtype=torch.long, device=device)
                _, end_logits, _ = model(et, past_key_values=kv)
                all_logits = torch.cat([first_logit, end_logits[:, :-1, :]], dim=1)[0]
                losses.append(
                    F.cross_entropy(all_logits.reshape(-1, all_logits.shape[-1]),
                                    et[0], reduction="mean").item()
                )
            pred = min(range(len(losses)), key=losses.__getitem__)
            correct += int(pred == int(ex["label"]))
            total += 1
        return correct / total if total else 0.0
    finally:
        model.train()


# ==================== 主流程 ====================
def parse_args():
    ap = argparse.ArgumentParser(description="SFT/DPO finetune Qwen2.5-MoE")
    ap.add_argument("--stage", choices=["sft", "dpo"], required=True)
    ap.add_argument("--data", default="data/hellaswag_sft.jsonl", help="数据 JSONL（SFT: prompt/response；DPO: prompt/chosen/rejected）")
    ap.add_argument("--out_dir", default="saves/finetune", help="checkpoint 输出目录")
    ap.add_argument("--load_model_dir", default=None, help="从已有 checkpoint 载入模型权重（SFT 用预训练、DPO 用 SFT 结果），只载权重不载优化器")
    ap.add_argument("--tiny", action="store_true", help="超小模型冒烟")
    ap.add_argument("--max_len", type=int, default=512, help="序列截断长度")
    ap.add_argument("--max_steps", type=int, default=None, help="训练总步数（默认 SFT=1500 / DPO=1900）")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--dpo_beta", type=float, default=0.1, help="DPO 温度系数 β")
    # 纯 torch 后台专用
    ap.add_argument("--no_deepspeed", action="store_true", help="强制走纯 torch 后台（无 DeepSpeed 时自动）")
    ap.add_argument("--batch_size", type=int, default=2, help="纯 torch 后台：每微批样本数")
    ap.add_argument("--grad_accum_steps", type=int, default=4, help="纯 torch 后台：梯度累积步数")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--max_lr", type=float, default=1e-4)
    ap.add_argument("--min_lr", type=float, default=1e-5)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.95])
    ap.add_argument("--dtype", choices=["auto", "fp32", "bf16"], default="auto", help="纯 torch 后台的计算精度")
    # 评测
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--hellaswag_every", type=int, default=200, help="每隔多少步跑一次 HellaSwag，0=关")
    ap.add_argument("--hellaswag_samples", type=int, default=200)
    ap.add_argument("--resume", action="store_true", help="从 out_dir 的 latest checkpoint 续训（含优化器状态）")
    ap.add_argument("--local_rank", type=int, default=-1, help="deepspeed launcher 注入，脚本本身不直接用")
    if DS_AVAILABLE:
        ap = deepspeed.add_config_arguments(ap)  # 追加 --deepspeed / --deepspeed_config
    return ap.parse_args()


def main():
    args = parse_args()
    if args.max_steps is None:
        args.max_steps = 1900 if args.stage == "dpo" else 1500

    # ---- DeepSpeed 后台判定 ----
    ds_config = getattr(args, "deepspeed_config", None)
    use_ds = DS_AVAILABLE and (not args.no_deepspeed) and ds_config is not None

    # ---- 模型（同一种子保证各 rank 初始化一致）----
    torch.manual_seed(args.seed)
    cfg = Qwen2Config()
    if args.tiny:
        cfg.hidden_size = 256
        cfg.mlp_mid_size = 1024
        cfg.layers = 4
        cfg.num_att_heads = 4
        cfg.num_kv_heads = 2
        cfg.moe_intermediate_size = 256
        cfg.num_local_experts = 4
        cfg.n_group = 1
        cfg.topk_group = 1
        cfg.first_k_dense_replace = 1
    model = Qwen2ForCausalLM(cfg)

    # ---- 训练后台 ----
    if use_ds:
        runner = DeepSpeedRunner(args, model)
    else:
        runner = PlainRunner(args, model)
    device = runner.device
    is_main = runner.is_main

    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        eff = runner.micro_batch * runner.accum_steps
        print(f"[{args.stage}] 模型 {n_params/1e6:.1f}M | 后台 {'DeepSpeed' if use_ds else '纯torch'} | "
              f"micro={runner.micro_batch} accum={runner.accum_steps} 每步样本 {eff} | "
              f"device {device}")
        # 一致性检查：--max_steps 与 DeepSpeed 调度器 total_num_steps 需一致
        if use_ds and ds_config:
            cfg_dict = json.load(open(ds_config, encoding="utf-8"))
            total = cfg_dict.get("scheduler", {}).get("params", {}).get("total_num_steps")
            if total is not None and total != args.max_steps:
                print(f"[warn] ds_config scheduler.total_num_steps={total} 与 --max_steps={args.max_steps} 不一致")

    # ---- 数据 ----
    rows = load_jsonl(args.data)
    if TOKENIZER is None:
        raise SystemExit("缺少分词器（Qwen/Qwen2.5-0.5B），无法微调")
    tokenizer = TOKENIZER
    pad_id = tokenizer.pad_token_id

    if args.stage == "sft":
        dataset = SFTDataset(rows, tokenizer, args.max_len)
        collate_fn = lambda b: sft_collate(b, pad_id)
    else:
        dataset = DPODataset(rows, tokenizer, args.max_len)
        collate_fn = lambda b: dpo_collate(b, pad_id)
    if is_main:
        print(f"数据 {args.data}：共 {len(rows)} 条，token 化后有效 {len(dataset)} 条")

    def batch_iter():
        while True:
            dl = DataLoader(dataset, batch_size=runner.micro_batch, shuffle=True,
                            collate_fn=collate_fn, num_workers=0)
            torch.manual_seed(args.seed + runner.global_rank)  # 每个 epoch 重新打乱种子
            for b in dl:
                yield b

    # ---- 载入已有权重 / 断点续训 ----
    if args.load_model_dir:
        path, _ = runner.load(args.load_model_dir, load_opt=False)
        if path is None:
            raise SystemExit(f"[error] --load_model_dir {args.load_model_dir} 里没有找到可载入的权重")
        if is_main:
            print(f"已载入模型权重（不含优化器）: {path}")

    ref_model = None
    if args.stage == "dpo":
        ref_model = Qwen2ForCausalLM(cfg).to(device)
        ref_model.load_state_dict(runner.module.state_dict())  # 参考模型 = 策略初值
        for p in ref_model.parameters():
            p.requires_grad_(False)
        ref_model.eval()
        ref_model = ref_model.to(next(runner.module.parameters()).dtype)  # 与策略同精度
        if is_main:
            print("已建立冻结的参考模型（= 策略初始权重）")

    if args.resume:
        path, client_state = runner.load(args.out_dir, load_opt=True)
        if path is None:
            print("[warn] 未找到可恢复的 checkpoint，从头训练")
        elif is_main:
            print(f"已恢复: {path}（step={runner.global_steps}）")

    # ---- HellaSwag 评测数据 ----
    hellaswag_ds = None
    if is_main and args.hellaswag_every > 0 and TOKENIZER is not None:
        try:
            from datasets import load_dataset

            hellaswag_ds = load_dataset("hellaswag", split="validation")
            if args.hellaswag_samples < len(hellaswag_ds):
                hellaswag_ds = hellaswag_ds.select(range(args.hellaswag_samples))
            print(f"HellaSwag 就绪：本次评测 {len(hellaswag_ds)} 条")
        except Exception as e:
            print(f"[warn] HellaSwag 加载失败，跳过评测: {e}")

    # ---- 训练循环 ----
    best_acc = 0.0
    t0 = time.time()
    tokens_seen = 0
    train_iter = batch_iter()
    for step in range(runner.global_steps, args.max_steps + 1):
        loss_accum = 0.0
        for _ in range(runner.accum_steps):
            batch = next(train_iter)
            if args.stage == "sft":
                x, mask, y = (t.to(device) for t in batch)
                tokens_seen += x.numel()
                loss, _, _ = runner.forward(input_ids=x, attention_mask=mask, labels=y)
            else:
                batch = tuple(t.to(device) for t in batch)  # DataLoader 产出 CPU 张量，搬到训练设备
                tokens_seen += sum(t.numel() for t in batch[:4])
                loss = dpo_loss(runner.forward, ref_model, batch, args.dpo_beta)
            runner.backward(loss)
            loss_accum += loss.item() / runner.accum_steps
        runner.step()

        # ---- 日志 ----
        if is_main and args.log_interval > 0 and step % args.log_interval == 0:
            el = time.time() - t0
            speed = tokens_seen / el
            print(f"step {step:>6d} | loss {loss_accum:.4f} | lr {runner.lr:.2e} "
                  f"| {speed/1e3:.0f}k tok/s | {el/60:.1f}min")

        # ---- HellaSwag 准确率 ----
        if (is_main and hellaswag_ds is not None and args.hellaswag_every > 0
                and step > 0 and step % args.hellaswag_every == 0):
            acc = evaluate_hellaswag(runner.module, TOKENIZER, hellaswag_ds, device)
            improved = acc > best_acc
            best_acc = max(best_acc, acc)
            print(f"step {step:>6d} | HellaSwag Acc {acc:.4f}" + ("  [新最佳]" if improved else ""))

        # ---- 存档 ----
        if step > 0 and args.save_every > 0 and step % args.save_every == 0:
            runner.save(args.out_dir, "latest", step, best_acc)
            if is_main:
                print(f"step {step}: 已存档 -> {args.out_dir}")

    if is_main:
        runner.save(args.out_dir, "latest", args.max_steps, best_acc)
        print(f"\n{args.stage} 训练完成（{args.max_steps} 步），存档 -> {args.out_dir}")


if __name__ == "__main__":
    main()
