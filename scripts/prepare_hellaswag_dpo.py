"""
prepare_hellaswag_dpo.py —— 构建 DPO 微调数据
============================================================================
DPO 目标：偏好对 = (正确结尾, 错误结尾)。两种来源：
  1) 默认 abacusai/HellaSwag_DPO_FewShot（HF 上现成的 prompt/chosen/rejected 三元组，
     每个 HellaSwag 样本造 3 条偏好对，train 119715 / eval 30126）。
  2) --source hellaswag：本地用 HellaSwag 原始数据构造，chosen=正确结尾、rejected=错误结尾
     （与 abacusai 的构造方式一致，可离线用）。

每条样本写成一行 JSONL：{"prompt": "...", "chosen": "...", "rejected": "..."}

用法（在仓库根目录下执行）：
    python scripts/prepare_hellaswag_dpo.py                              # abacusai -> data/hellaswag_dpo.jsonl
    python scripts/prepare_hellaswag_dpo.py --source hellaswag           # 本地从 hellaswag 构造
    python scripts/prepare_hellaswag_dpo.py --split validation           # 用验证集（小规模冒烟）
"""

import argparse
import json
import os

from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser(description="构建 DPO 微调数据")
    ap.add_argument("--source", choices=["abacusai", "hellaswag"], default="abacusai",
                    help="数据来源：HF 现成数据集 或 本地 hellaswag 构造")
    ap.add_argument("--split", default="train", help="split（默认 train）")
    ap.add_argument("--out", default="data/hellaswag_dpo.jsonl", help="输出 JSONL 路径")
    ap.add_argument("--max_samples", type=int, default=None, help="最多取多少条（None=全量）")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        if args.source == "abacusai":
            ds = load_dataset("abacusai/HellaSwag_DPO_FewShot", split=args.split)
            for ex in ds:
                if args.max_samples is not None and n >= args.max_samples:
                    break
                record = {"prompt": ex["prompt"], "chosen": ex["chosen"], "rejected": ex["rejected"]}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n += 1
        else:  # hellaswag：正确结尾 vs 每个错误结尾，构造偏好对
            ds = load_dataset("hellaswag", split=args.split)
            for ex in ds:
                label = int(ex["label"])
                correct = ex["endings"][label]
                for k, ending in enumerate(ex["endings"]):
                    if k == label:
                        continue
                    if args.max_samples is not None and n >= args.max_samples:
                        break
                    record = {"prompt": ex["ctx"], "chosen": correct, "rejected": ending}
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n += 1
                if args.max_samples is not None and n >= args.max_samples:
                    break

    print(f"已写入 {n} 条 DPO 数据（来源 {args.source}） -> {args.out}")
    print(f"示例：{json.dumps(record, ensure_ascii=False)[:200]}...")


if __name__ == "__main__":
    main()
