# Qwen2.5-MoE

<p align="center">
  <img src="Qwen2.5-MoE.jpg" alt="Qwen2.5-MoE" width="820">
</p>

基于 **Qwen2.5-0.5B** 的纯 PyTorch 实现，借鉴 DeepSeek-V3 的 MoE 架构改造成的混合专家语言模型。**本项目将包含大模型的架构定义、预训练、微调（SFT、DPO）、知识蒸馏、多模态的全链路脚本文件。**

- **零外部依赖**：不依赖 `transformers`、LlamaFactory，只用 `torch`，便于独立运行、学习和部署
- **前沿架构选择**：GQA + RoPE + FlashAttention
- **DeepSeek-V3 MoE**：sigmoid 路由打分 + top-2 稀疏专家 + 共享专家 + 负载均衡辅助损失
- **DeepSpeed 训练**：bf16 混合精度、ZeRO-2、梯度累积、WarmupCosineLR 调度、断点续训

## 基模架构

| 组件 | 配置 |
|------|------|
| 词表 | 151,936（Qwen2.5 tokenizer，uint32） |
| 隐藏层 / 层数 | 896 / 24 |
| 注意力 | 14 Q 头，2 KV 头，head_dim=64，RoPE θ=1e6 |
| 最大序列 | 32,768 |
| 前馈 | 前 2 层稠密 SwiGLU，后 22 层 MoE |
| MoE | 6 路由专家 / top-2 激活 / 专家中间维 896 / 1 共享专家 / aux_loss 系数 0.01 |
| 参数量 | 总 **577M**，单 token 激活 **365M** |

## 文件结构

```
├── model.py              # 模型定义（纯 torch MoE，零外部依赖）
├── scripts/              # 训练与数据脚本（在仓库根目录下执行）
│   ├── train_qwen_moe.py        # 预训练（DeepSpeed）
│   ├── finetune_qwen_moe.py     # SFT / DPO 微调
│   ├── prepare_fineweb.py       # fineweb-edu parquet → uint32 token .bin
│   ├── prepare_hellaswag_sft.py # HellaSwag → SFT 数据对
│   ├── prepare_hellaswag_dpo.py # HellaSwag → DPO 偏好对
│   └── hellaswag_qwen2.py       # HellaSwag 评测脚本
├── configs/              # DeepSpeed 配置
│   ├── ds_config.json           # 预训练（正式）
│   ├── ds_config_tiny.json      # 预训练（测试）
│   ├── ds_config_sft.json       # SFT
│   └── ds_config_dpo.json       # DPO
├── docs/                 # 文档
│   ├── FINETUNE_README.md       # 微调（SFT → DPO）运行说明
│   └── hellaswag.txt            # 基线评测结果
├── data/                 # 生成的数据（gitignored）
└── play.ipynb            # 交互实验 notebook
```

## 快速开始

```bash
# 1. 环境
pip install torch deepspeed numpy pyarrow transformers datasets

# 2. 数据预处理：fineweb-edu parquet → train.bin / val.bin / meta.json
python scripts/prepare_fineweb.py --data_dir fineweb-edu/sample/10BT --out_dir fineweb-edu/bin

# 3. 单卡测试
deepspeed --num_gpus=1 scripts/train_qwen_moe.py --deepspeed --deepspeed_config configs/ds_config_tiny.json \
    --tiny --block_size 256 --max_steps 50

# 4. 正式训练（--max_steps 必须与 ds_config.json 的 total_num_steps 一致）
deepspeed --num_gpus=2 scripts/train_qwen_moe.py --deepspeed --deepspeed_config configs/ds_config.json \
    --max_steps 76294 --hellaswag_every 1000 --hellaswag_samples 1000

# 5. 断点续训
deepspeed --num_gpus=2 scripts/train_qwen_moe.py --deepspeed --deepspeed_config configs/ds_config.json --resume

# 6. SFT / DPO 微调（用 HellaSwag 数据，详见 docs/FINETUNE_README.md）
python scripts/prepare_hellaswag_sft.py                 # -> data/hellaswag_sft.jsonl
python scripts/prepare_hellaswag_dpo.py                 # -> data/hellaswag_dpo.jsonl
deepspeed --num_gpus=1 scripts/finetune_qwen_moe.py --stage sft \
    --data data/hellaswag_sft.jsonl --deepspeed --deepspeed_config configs/ds_config_sft.json
```

## 后续工作

| 组件 | 配置 |
|------|------|
| 在Hellaswag上微调（SFT → DPO） | ✅ |
| 蒸馏 DeepSeek-R1 | ❌ |
| 为模型添加视觉头与投影层 | ❌ |

## 参考

- [Qwen2.5](https://github.com/QwenLM/Qwen2.5)（Apache-2.0）
- [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)（MoE 架构参考）
- [nanoGPT](https://github.com/karpathy/nanoGPT)（数据格式与训练循环风格）
