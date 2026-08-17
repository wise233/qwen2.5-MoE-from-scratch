# 微调（SFT → DPO）运行说明

目标：先预训练出 `model.py` 的 MoE 基座，再用 HellaSwag 数据做两阶段微调，提升 HellaSwag 准确率。

```
预训练 → SFT（ctx → 正确结尾）→ DPO（正确结尾 > 错误结尾）
```

> 所有命令都在**仓库根目录**下执行。

## 0. 环境与数据

```bash
# 数据（本机或服务器均可，hellaswag 会下载）
python scripts/prepare_hellaswag_sft.py                 # -> data/hellaswag_sft.jsonl（39,905 条）
python scripts/prepare_hellaswag_dpo.py                 # -> data/hellaswag_dpo.jsonl（119,715 条，abacusai）
# 备用：不连 HF 时从 hellaswag 本地构造 DPO 对
python scripts/prepare_hellaswag_dpo.py --source hellaswag
```

## 1. SFT

```bash
deepspeed --num_gpus=1 scripts/finetune_qwen_moe.py --stage sft \
    --data data/hellaswag_sft.jsonl \
    --load_model_dir out-qwen-moe/ds-checkpoints \   # 预训练 checkpoint 目录（可省，省则随机初始化）
    --deepspeed --deepspeed_config configs/ds_config_sft.json \
    --max_steps 1500 \
    --out_dir saves/finetune-sft \
    --hellaswag_every 200 --hellaswag_samples 500
```

- `configs/ds_config_sft.json`：micro 8 × accum 8 = 每步 64 样本，bf16，AdamW lr 1e-4，ZeRO-2。
- `--max_steps` 必须与配置里 `scheduler.total_num_steps` 一致（默认 1500，约 2.4 个 epoch）。
- 日志里的 `HellaSwag Acc` 是周期评测的训练指标。

## 2. DPO

```bash
deepspeed --num_gpus=1 scripts/finetune_qwen_moe.py --stage dpo \
    --data data/hellaswag_dpo.jsonl \
    --load_model_dir saves/finetune-sft/ds-checkpoints \  # SFT 结果目录
    --deepspeed --deepspeed_config configs/ds_config_dpo.json \
    --max_steps 1900 \
    --out_dir saves/finetune-dpo \
    --hellaswag_every 200 --hellaswag_samples 500
```

- 脚本自动把 `--load_model_dir` 的权重复制一份冻结作为参考模型，策略模型在上面继续训。
- DPO loss：`-log σ( β·(πθ(yw)-πref(yw)-πθ(yl)+πref(yl)) )`，β 默认 0.1（`--dpo_beta`）。
- `configs/ds_config_dpo.json`：lr 5e-5，total 1900（约 1 epoch）。

## 3. 断点续训 / 查看结果

- 训练中每 `--save_every`（默认 200）步往 `--out_dir` 写一个 DeepSpeed checkpoint（`latest/`）。
- 续训（同一 stage，恢复优化器/调度器/步数）：
  ```bash
  deepspeed --num_gpus=1 scripts/finetune_qwen_moe.py --stage dpo \
      --data data/hellaswag_dpo.jsonl --deepspeed --deepspeed_config configs/ds_config_dpo.json \
      --resume --out_dir saves/finetune-dpo
  ```
- 用 `scripts/hellaswag_qwen2.py` 或训练日志里的 HellaSwag Acc 对比基座（0.4809）与微调后的提升。

## 本机冒烟（无 DeepSpeed）

```bash
python scripts/finetune_qwen_moe.py --stage sft --tiny --no_deepspeed \
    --data data/hellaswag_sft.jsonl --max_steps 5 --batch_size 2 --grad_accum_steps 4 \
    --hellaswag_every 0 --out_dir saves/smoke-sft
python scripts/finetune_qwen_moe.py --stage dpo --tiny --no_deepspeed \
    --data data/hellaswag_dpo.jsonl --load_model_dir saves/smoke-sft \
    --max_steps 5 --batch_size 2 --grad_accum_steps 4 \
    --hellaswag_every 0 --out_dir saves/smoke-dpo
```

## 注意事项

- 参考模型 / 策略模型同卡各占一份：577M 模型下 DPO 显存约 8-9GB（bf16 + AdamW），单张 24GB 卡足够。
- 多卡时 `train_micro_batch_size_per_gpu` 是每卡值，全局有效批 = micro × accum × 卡数，记得相应调小。
- SFT/DPO 都只监督 response 区间（prompt 用 `-100` 掩码），padding 用 attention_mask 屏蔽。
- 脚本在 `scripts/` 下会自动把仓库根目录加进 `sys.path`，`model.py` 的 import 不受目录结构影响。
