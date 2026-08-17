"""
prepare_hellaswag_sft.py —— 用 HellaSwag 原始数据构建 SFT 微调数据对
============================================================================
SFT 目标：给定上下文 ctx（prompt），让模型续写正确结尾 endings[label]（response）。
每条样本写成一行 JSONL：{"prompt": "<上下文>", "response": "<正确结尾>"}

用法（在仓库根目录下执行）：
    python scripts/prepare_hellaswag_sft.py                          # -> data/hellaswag_sft.jsonl
    python scripts/prepare_hellaswag_sft.py --split validation       # 用验证集（做小规模冒烟）
    python scripts/prepare_hellaswag_sft.py --max_samples 1000       # 只取前 1000 条

数据来自 HF datasets 的 hellaswag（无网络时会用本地缓存）。
"""

import argparse
import json
import os

from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser(description="从 HellaSwag 构建 SFT 数据对")
    ap.add_argument("--split", default="train", help="hellaswag 的 split（默认 train）")
    ap.add_argument("--out", default="data/hellaswag_sft.jsonl", help="输出 JSONL 路径")
    ap.add_argument("--max_samples", type=int, default=None, help="最多取多少条（None=全量）")
    args = ap.parse_args()

    ds = load_dataset("hellaswag", split=args.split)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for ex in ds:
            if args.max_samples is not None and n >= args.max_samples:
                break
            label = int(ex["label"])
            record = {"prompt": ex["ctx"], "response": ex["endings"][label]}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1

    print(f"已写入 {n} 条 SFT 数据 -> {args.out}")
    print(f"示例：{json.dumps(record, ensure_ascii=False)[:200]}...")


if __name__ == "__main__":
    main()
