# Qwen2.5-MoE

<p align="center">
  <img src="Qwen2.5-MoE.jpg" alt="Qwen2.5-MoE" width="820">
</p>

基于 **Qwen2.5-0.5B** 的纯 PyTorch 自包含实现，借鉴 **DeepSeek-V3** 的 MoE 架构改造成的混合专家（Mixture-of-Experts）因果语言模型。

- **零外部依赖**：模型推理不依赖 `transformers`，只用 `torch`，便于独立运行、学习和部署
- **GQA + RoPE + FlashAttention**：14 个 Q 头、2 个 KV 头，注意力走 `F.scaled_dot_product_attention`
- **DeepSeek-V3 风格 MoE**：sigmoid 路由打分 + top-2 稀疏专家 + 共享专家 + 负载均衡辅助损失
- **DeepSpeed 训练**：bf16 混合精度、ZeRO-2、梯度累积、WarmupCosineLR 调度、断点续训

## 架构

| 组件 | 配置 |
|------|------|
| 词表 | 151,936（Qwen2.5 tokenizer，uint32） |
| 隐藏层 / 层数 | 896 / 24 |
| 注意力 | 14 Q 头，2 KV 头（GQA），head_dim=64，RoPE θ=1e6 |
| 最大序列 | 32,768 |
| 前馈 | 前 2 层稠密 SwiGLU，后 22 层 MoE |
| MoE | 6 路由专家 / top-2 激活 / 专家中间维 896 / 1 共享专家 / aux_loss 系数 0.01 |
| 参数量 | 总 **577M**，单 token 激活 **365M** |

## 文件结构

```
├── model.py              # 模型定义（Qwen2 + DeepSeek-V3 MoE，纯 PyTorch）
├── train_qwen_moe.py     # DeepSpeed 训练脚本（数据加载 / 训练循环 / HellaSwag 评测）
├── prepare_fineweb.py    # fineweb-edu parquet → uint32 token .bin 预处理
├── hellaswag_qwen2.py    # HellaSwag 评测脚本
├── ds_config.json        # 正式训练 DeepSpeed 配置
├── ds_config_tiny.json   # 冒烟测试用超小配置
├── bench_ds.py           # 吞吐/显存基准测试工具（实测硬件容量用）
├── debug_oom.py          # 内存分阶段诊断工具
└── play.ipynb            # 交互实验 notebook
```

## 快速开始

```bash
# 1. 环境（需要 CUDA + PyTorch + DeepSpeed + numpy + pyarrow + transformers）
pip install torch deepspeed numpy pyarrow transformers datasets

# 2. 数据预处理：fineweb-edu parquet → train.bin / val.bin / meta.json
#    （默认读 fineweb-edu/sample/10BT/*.parquet，产物写入 fineweb-edu/bin/）
python prepare_fineweb.py --data_dir fineweb-edu/sample/10BT --out_dir fineweb-edu/bin

# 3. 冒烟测试（单卡、超小模型，验证全流程）
deepspeed --num_gpus=1 train_qwen_moe.py --deepspeed --deepspeed_config ds_config_tiny.json \
    --tiny --block_size 256 --max_steps 50

# 4. 正式训练（--max_steps 必须与 ds_config.json 的 total_num_steps 一致）
deepspeed --num_gpus=2 train_qwen_moe.py --deepspeed --deepspeed_config ds_config.json \
    --max_steps 76294 --hellaswag_every 1000 --hellaswag_samples 1000

# 5. 断点续训
deepspeed --num_gpus=2 train_qwen_moe.py --deepspeed --deepspeed_config ds_config.json --resume
```

## 硬件实测结论（2×RTX 4090 / 24GB）

本项目在 2×4090 上做过完整基准测试，供参考：

| 配置 | 峰值显存 | 吞吐 |
|------|---------|------|
| micro=16, block=2048（原默认） | ❌ OOM | — |
| micro=2, block=2048 | 16.2 GB | ~8.0k tok/s |
| micro=4, block=1024 | 16.1 GB | ~9.3k tok/s |

- **显存瓶颈是 lm_head 的 logits 张量**（词表 151,936 过大）与 24 层前向激活，二者随 `batch×block` 线性增长；模型权重本身仅约 2.3 GB
- 每 micro 批 token 数上限约 **4096**（micro=2×block2048 或 micro=4×block1024）
- 激活检查点（gradient checkpointing）可省 ~60% 激活显存，但反向重算开销让吞吐**不升反降**（实测 5.9k vs 9.3k tok/s），本项目默认关闭
- 换更小词表（如 GPT-2 的 50257）能省显存，但**不会提速**——瓶颈是内存带宽而非 lm_head 计算量

## 注意事项

- `train_micro_batch_size_per_gpu × gradient_accumulation_steps × num_gpus × block_size` = 全局有效批；改 `total_num_steps` 时需同步 `--max_steps`，否则余弦退火调度与训练步数不一致（脚本会告警）
- DeepSpeed 启动器会注入 `--local_rank`，`train_qwen_moe.py` 已显式接收
- 数据文件（`.bin`、parquet）与训练输出（`out-qwen-moe/`）不入库，见 `.gitignore`

## 参考

- [Qwen2.5](https://github.com/QwenLM/Qwen2.5)（Apache-2.0）
- [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)（MoE 架构参考）
- [nanoGPT](https://github.com/karpathy/nanoGPT)（数据格式与训练循环风格）
