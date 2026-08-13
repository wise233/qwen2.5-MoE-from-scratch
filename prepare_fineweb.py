"""
fineweb-edu -> tokens -> .bin
=============================
读取 fineweb-edu 目录下所有 *.parquet（HuggingFace FineWeb-Edu 格式），
用 transformers 加载 Qwen2.5 tokenizer，把每个文档 tokenize 成 token id 流，
保存为 nanoGPT 风格的 .bin 文件（flat uint32 数组，可直接 mmap 读取训练）。

重要：Qwen2.5 词表大小 151643 > 65535，uint16 装不下，必须用 uint32。
      （nanoGPT 的 prepare.py 用 uint16 是因为 GPT-2 词表只有 50257。）

输出（--out_dir 下）：
    all.bin    —— 临时文件，token 按文档顺序拼接的完整流，处理完即删
    train.bin  —— 前 (1-val_frac) 比例的 token
    val.bin    —— 最后 val_frac 比例的 token
    meta.json  —— tokenizer / 词表 / token 数等元信息

用法：
    python prepare_fineweb.py [--data_dir fineweb-edu] [--out_dir fineweb-edu/bin]
                             [--tokenizer Qwen/Qwen2.5-0.5B] [--val_frac 0.01]
                             [--batch_size 2000] [--limit -1] [--with_eos]
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer


def load_tokenizer(name: str):
    """先试本地缓存，失败再联网下载（可能触发网络请求）"""
    try:
        return AutoTokenizer.from_pretrained(name, local_files_only=True)
    except Exception as e:
        print(f"[warn] 本地缓存没有 {name}，尝试联网下载：{e}")
        return AutoTokenizer.from_pretrained(name)


def main():
    ap = argparse.ArgumentParser(description="把 fineweb-edu parquet 预处理成 token 的 bin 文件")
    ap.add_argument("--data_dir", default="fineweb-edu/sample/10BT", help="存放 *.parquet 的目录")
    ap.add_argument("--out_dir", default="fineweb-edu/bin", help="输出目录（train.bin / val.bin / meta.json）")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B", help="HuggingFace 上的 tokenizer 名称")
    ap.add_argument("--val_frac", type=float, default=0.01, help="验证集 token 占比（取整段流的最后一部分）")
    ap.add_argument("--batch_size", type=int, default=2000, help="每批处理的文档数（大一些 tokenize 更快）")
    ap.add_argument("--limit", type=int, default=-1, help="只处理前 N 篇文档（调试用），-1 表示全部")
    ap.add_argument("--with_eos", action="store_true", help="在每篇文档后面插入 <|endoftext|> 作为文档分隔符")
    args = ap.parse_args()

    # ---- 0. 参数检查 ----
    if not 0.0 <= args.val_frac < 1.0:
        sys.exit("--val_frac 必须在 [0, 1) 之间")
    os.makedirs(args.out_dir, exist_ok=True)
    parquet_files = sorted(glob.glob(os.path.join(args.data_dir, "*.parquet")))
    if not parquet_files:
        sys.exit(f"[error] 目录 {args.data_dir} 下没有 *.parquet 文件")

    # ---- 1. 加载 tokenizer ----
    tok = load_tokenizer(args.tokenizer)
    eos_id = tok.eos_token_id
    # 我们只做 tokenize 不跑模型，屏蔽"序列超过 model_max_length"的告警（超长文档是正常的）
    tok.model_max_length = 10**9
    print(f"tokenizer: {args.tokenizer} | vocab={tok.vocab_size} | eos_id={eos_id}")

    # ---- 2. 流式读 parquet + tokenize + 写 all.bin ----
    all_path = os.path.join(args.out_dir, "all.bin")
    total_tokens, total_docs = 0, 0
    t0 = time.time()

    with open(all_path, "wb") as out:
        for pf_path in parquet_files:
            pf = pq.ParquetFile(pf_path)
            print(f"处理文件: {pf_path}（{pf.metadata.num_rows:,} 行）")
            for batch in pf.iter_batches(batch_size=args.batch_size, columns=["text"]):
                # 过滤掉 None / 空白文本（避免 tokenizer 报错）
                texts = [t for t in batch["text"].to_pylist() if isinstance(t, str) and t.strip()]
                if args.limit >= 0 and total_docs >= args.limit:  # 调试：截断
                    texts = texts[: args.limit - total_docs]
                if not texts:
                    continue

                enc = tok(texts, add_special_tokens=False)  # 批量 tokenize，不加 BOS/EOS
                # 每篇文档一个 uint32 数组，可选在末尾补 eos，最后拼成一批一次性写盘
                arrays = []
                n_nonempty = 0
                for toks in enc["input_ids"]:
                    if not toks:
                        continue
                    n_nonempty += 1
                    a = np.asarray(toks, dtype=np.uint32)
                    if args.with_eos:
                        a = np.concatenate([a, np.array([eos_id], dtype=np.uint32)])
                    arrays.append(a)
                if not arrays:
                    continue
                flat = np.concatenate(arrays)  # 本批所有 token 拼成一段
                flat.tofile(out)  # 追加写盘
                total_tokens += flat.size
                total_docs += n_nonempty

                # 进度：每 10 秒或每 10 万篇打一行
                if time.time() - t0 >= 10 or total_docs % 100_000 == 0:
                    el = time.time() - t0
                    print(f"[{el:7.1f}s] docs={total_docs:>8,d} tokens={total_tokens:>12,d} "
                          f"speed={total_tokens/el/1000:>6.0f}k tok/s")
            if args.limit >= 0 and total_docs >= args.limit:
                break

    if total_tokens == 0:
        os.remove(all_path)
        sys.exit("[error] 没有产出任何 token，请检查数据与 --limit")

    # ---- 3. 按 token 比例切分 train / val（流最后 val_frac 给验证集）----
    n_val = int(total_tokens * args.val_frac)
    n_train = total_tokens - n_val
    all_mem = np.memmap(all_path, dtype=np.uint32, mode="r")
    assert all_mem.size == total_tokens, f"写盘 token 数 {total_tokens} 与文件大小 {all_mem.size} 不一致"

    train_path = os.path.join(args.out_dir, "train.bin")
    val_path = os.path.join(args.out_dir, "val.bin")
    all_mem[:n_train].tofile(train_path)  # memmap 切片 + tofile，边读边写，不占内存
    if n_val > 0:
        all_mem[n_train:].tofile(val_path)
    del all_mem
    os.remove(all_path)  # 删掉临时 all.bin

    # ---- 4. 写元信息 ----
    meta = {
        "tokenizer": args.tokenizer,
        "vocab_size": tok.vocab_size,
        "dtype": "uint32",
        "val_frac": args.val_frac,
        "with_eos": args.with_eos,
        "n_docs": int(total_docs),
        "n_train_tokens": int(n_train),
        "n_val_tokens": int(n_val),
        "files": [os.path.basename(p) for p in parquet_files],
    }
    meta_path = os.path.join(args.out_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    el = time.time() - t0
    print(f"\n完成，耗时 {el:.1f}s")
    print(f"  总 token: {total_tokens:,} | 文档: {total_docs:,}")
    print(f"  train.bin: {n_train:,} tokens ({(os.path.getsize(train_path) if os.path.exists(train_path) else 0)/1e9:.2f} GB)")
    print(f"  val.bin:   {n_val:,} tokens ({(os.path.getsize(val_path) if os.path.exists(val_path) else 0)/1e9:.2f} GB)")
    print(f"  meta.json: {meta_path}")
    print("  抽查 train.bin 前 20 个 token id:", np.fromfile(train_path, dtype=np.uint32, count=20).tolist())


if __name__ == "__main__":
    main()
