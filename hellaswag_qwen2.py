
import argparse  # 命令行参数解析

import torch  # 张量运算
import torch.nn.functional as F  # 函数式接口（cross_entropy 等）
from datasets import load_dataset  # 加载 HellaSwag 数据集
from transformers import AutoModelForCausalLM, AutoTokenizer  # 加载模型与分词器
def compute_ending_loss(model, tokenizer, prompt, ending, device, reduction="mean"):
    """计算"在上下文后续写该结尾"的损失，越小越好

    Args:
        prompt: 上下文文本
        ending: 候选结尾文本
        reduction: "mean"=官方指标（逐 token 平均 NLL，按结尾长度归一化）；"sum"=未归一化之和（会偏向短结尾）
    Returns:
        float 损失
    """
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)  # 上下文 -> token id
    ending_ids = tokenizer.encode(ending, add_special_tokens=False)  # 结尾 -> token id
    if not ending_ids:  # 空结尾（数据里几乎不会出现，防御一下）
        return float("inf")  # 视为无穷差，不会胜出

    prompt_len = len(prompt_ids)  # 上下文长度
    full_ids = prompt_ids + ending_ids  # 拼接：上下文 + 结尾
    full_tensor = torch.tensor([full_ids], device=device)  # 加 batch 维 -> (1, seq_len)

    with torch.no_grad():  # 评估阶段不需要梯度
        logits = model(full_tensor).logits[0]  # (seq_len, vocab)；logits[i] 预测第 i+1 个 token

    # 只统计"预测结尾部分"的损失：
    #   结尾 token 位于 full_ids[prompt_len..seq_len-1]，
    #   它们分别由 logits[prompt_len-1 .. seq_len-2] 预测，即 logits[prompt_len-1 : -1]
    shift_logits = logits[prompt_len - 1 : -1]  # (ending_len, vocab)
    shift_labels = torch.tensor(full_ids[prompt_len:], device=device)  # 结尾的真实 token id
    # mean：逐 token 平均负对数似然（按结尾长度归一化），对应官方 HellaSwag 的 acc_norm 指标
    loss = F.cross_entropy(shift_logits, shift_labels, reduction=reduction)
    return loss.item()  # 转成 Python float
def evaluate_example(example, model, tokenizer, device, reduction="sum"):
    """对一个样本，选出 4 个候选结尾里损失最小的那个，返回其索引"""
    prompt = example["ctx"]  # 上下文
    losses = []  # 依次存放 4 个结尾的损失
    for ending in example["endings"]:  # 遍历候选结尾
        loss = compute_ending_loss(model, tokenizer, prompt, ending, device, reduction)  # 单结尾损失
        losses.append(loss)  # 收集
    return {"pred": losses.index(min(losses))}  # 取损失最小者（=续写概率最高者）


def main():
    parser = argparse.ArgumentParser(description="HellaSwag 评估")  # 参数解析器
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")  # 模型标识（HF 仓库名或本地路径）
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")  # 计算设备
    parser.add_argument("--num-samples", type=int, default=10000, help="只评估前 N 条；不传则评估全量验证集(10042)")  # 样本数
    parser.add_argument("--reduction", choices=["sum", "mean"], default="mean", help="官方 acc_norm 指标用 mean（长度归一化）；sum 未归一化、会偏向短结尾")  # 损失聚合方式
    args = parser.parse_args()  # 解析命令行参数

    tokenizer = AutoTokenizer.from_pretrained(args.model)  # 加载分词器
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)  # 加载模型并搬到指定设备
    model.eval()  # 切到推理模式（关闭 dropout）

    dataset = load_dataset("hellaswag", split="validation")  # 加载 HellaSwag 验证集
    if args.num_samples is not None:  # 用户指定了条数
        dataset = dataset.select(range(args.num_samples))  # 只取前 N 条

    # 逐条评估并显示进度条；结果里新增 "pred" 列（预测的结尾索引）
    results = dataset.map(
        lambda ex: evaluate_example(ex, model, tokenizer, args.device, args.reduction)  # 闭包捕获模型等
    )

    # 关键点：该数据集的 label 是字符串（如 '3'），pred 是 int，必须 int() 转换后比较
    correct = sum(1 for r in results if r["pred"] == int(r["label"]))  # 预测正确的样本数
    accuracy = correct / len(results)  # 准确率 = 正确数 / 总数
    print(f"HellaSwag Accuracy ({len(results)} samples): {accuracy:.4f}")


if __name__ == "__main__":  # 直接运行本文件时的入口
    main()  # 执行主函数