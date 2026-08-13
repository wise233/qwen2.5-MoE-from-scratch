"""
Qwen2.5-0.5B —— 纯 PyTorch 自包含实现（MoE 魔改版）
==================================================
原版与 modeling_qwen2.py（HuggingFace transformers 的 Qwen2 实现）结构完全一致，
但去掉了所有 transformers 依赖，只依赖 torch，便于独立运行和学习。
本版在此基础上把 FFN 层借鉴 modeling_deepseek_v3.py 的 DeepseekV3MoE /
DeepseekV3TopkRouter / DeepseekV3Experts 改造成 MoE 混合专家架构。

架构要点：
- Pre-Norm + 残差：归一化放在子层之前
- GQA 分组查询注意力：14 个 Q 头，仅 2 个 K/V 头（保持不变）
- RoPE 旋转位置编码：基频 theta = 1e6（保持不变）
- 注意力用 F.scaled_dot_product_attention（config.use_sdpa=False 可切回 eager 手写路径对照）
- lm_head 与 embedding 共享权重（tied weights）
- FFN 改造成 DeepSeekV3 风格 MoE：
    * 前 first_k_dense_replace 层保留稠密 SwiGLU MLP（训练早期更稳）
    * 其余层 = 稀疏路由专家（sigmoid 打分 + 分组限定 + top-k 选择）
      + 1 个所有 token 都经过的共享专家
    * 可选负载均衡辅助损失（MoE 训练稳定性的关键）

超参数（基础沿用 Qwen2.5-0.5B config.json，MoE 部分为本项目拟定）：
    vocab=151936  hidden=896  layers=24  heads=14  kv_heads=2  head_dim=64
    dense_intermediate=4864  max_seq=32768  rope_theta=1e6  eps=1e-6
    MoE: 8 个路由专家 / top-2 激活 / 专家中间维 640 / 1 个共享专家
         n_group=4, topk_group=2, routed_scaling_factor=2.5,
         first_k_dense_replace=2（前 2 层稠密，后 22 层 MoE）
    总参数量约 0.55B，单 token 激活参数约 0.32B。
"""

from dataclasses import dataclass  
# 数据类：@dataclass 是 Python 3.7 引入的装饰器，它的核心作用是自动为类生成 __init__、__repr__、__eq__ 等特殊方法，从而减少大量样板代码。

import torch  # 张量运算主库
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 无状态的函数式接口（softmax、dropout 等）


# ==================== 1. 配置 ====================
@dataclass
class Qwen2Config:
    """Qwen2.5-0.5B 的全部超参数（MoE 魔改版：新增 MoE 相关字段）"""
    # ---------- 与原版 Qwen2.5-0.5B 一致的部分 ----------
    vocab_size: int = 151936  # 词表大小（嵌入表行数 / lm_head 输出维度）
    hidden_size: int = 896  # 隐藏层维度，即模型的"宽度"
    mlp_mid_size: int = 4864  # 稠密层 MLP 中间维度（约 hidden 的 5.4 倍，仅前几层用）
    layers: int = 24  # 解码器层数（模型深度）
    num_att_heads: int = 14  # Query 头数
    num_kv_heads: int = 2  # Key/Value 头数（GQA：远小于 Q 头数以省内存）
    head_dim: int = 64  # 每个注意力头的维度: 896 / 14
    max_pos_emb: int = 32768  # 最大序列长度
    rope_theta: float = 1000000.0  # RoPE 基频 theta
    eps: float = 1e-6  # RMSNorm 分母防除零常数
    att_dropout: float = 0.0  # 注意力权重 dropout 概率
    use_sdpa: bool = True  # 注意力实现：True 用 F.scaled_dot_product_attention，False 用 eager 手写路径
    initial_range: float = 0.02  # 权重初始化的标准差
    use_cache: bool = True  # 是否默认使用 KV 缓存（生成加速）
    tie_word_embeddings: bool = True  # lm_head 与 embedding 是否共享权重

    # ---------- MoE 魔改新增（借鉴 DeepSeekV3） ----------
    first_k_dense_replace: int = 2  # 前几层保留稠密 MLP，其余层换 MoE（对应 DeepseekV3.first_k_dense_replace）
    num_local_experts: int = 6  # 每层路由专家总数（方案A：6 个更宽的专家，每个训练得更充分）
    num_experts_per_tok: int = 2  # 每个 token 实际激活的专家数（top-2 稀疏）
    moe_intermediate_size: int = 896  # 单个路由专家的中间维度（方案A：从 640 加宽到 896，提升单专家容量）
    n_shared_experts: int = 1  # 共享专家数量（每个 token 都经过的稠密 MLP）
    n_group: int = 1  # 分组路由：1 = 退化为普通 top-k（0.5B 规模不需要 DeepSeekV3 的分组限定，徒增约束）
    topk_group: int = 1  # 与 n_group=1 配合：放行全部专家，去掉分组限定
    routed_scaling_factor: float = 2.5  # 路由权重缩放系数（补偿稀疏激活带来的幅值损失）
    norm_topk_prob: bool = True  # 是否对 top-k 路由权重归一化（和为 1）
    use_aux_loss: bool = True  # 是否计算负载均衡辅助损失（MoE 训练稳定性的关键）
    aux_loss_coef: float = 0.01  # 辅助损失在总损失中的权重系数


# ==================== 2. RMSNorm ====================
class Qwen2RMSNorm(nn.Module):
    """RMSNorm：按均方根归一化，不减去均值（比 LayerNorm 少一次统计、训练更稳）"""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))  # 可学习缩放权重，初始全 1
        self.eps = eps  # 保存防除零常数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype  # 记录输入精度，最后再转回去
        x =x.to(torch.float32)  # 先提升到 float32 计算，避免精度损失
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)  # 最后一维求均方（keepdim 保留维度以便广播）
        return self.weight * x.to(input_dtype)  # 逐元素乘可学习权重，并恢复原精度


# ==================== 3. 旋转位置编码（RoPE） ====================
class Qwen2RotaryEmbedding(nn.Module):
    """RoPE：根据位置 id 预计算整段序列的 cos/sin 表"""

    def __init__(self, dim: int, theta: float = 1000000.0, max_seq_len: int = 32768):
        super().__init__()
        # 经典 RoPE 逆频率：每对维度一个频率，随维度序号指数递减
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))  # 形状 [dim/2]
        self.register_buffer("inv_freq", inv_freq, persistent=False)  # 注册为 buffer；persistent=False 表示不存进 state_dict
        self.max_seq_len = max_seq_len  #记录最大序列长度（本实现未用到动态外推）

    @torch.no_grad()  # 该前向只做查表运算，不需要梯度
    def forward(self, x, position_ids):  # x 仅用于取 dtype；position_ids 形状 [batch, seq]
        # 把 inv_freq 展开到 batch 维：inv_freq[None,:,None] -> [1, dim/2, 1] -> expand 成 [batch, dim/2, 1]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()  # 形状 [batch, 1, seq]
        # 矩阵乘：每个位置的 位置值 × 对应频率，得到该位置每个频率维的弧度
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)  # 形状 [batch, seq, dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # 弧度复制两份，凑满 head_dim 长度 [batch, seq, dim]
        cos = emb.cos().to(x.dtype)  # 余弦表
        sin = emb.sin().to(x.dtype)  # 正弦表
        return cos, sin  # 返回 [batch, seq, dim] 的 cos/sin


def rotate_half(x):
    """把最后一维切成两半，交换位置并给后半取负（旋转矩阵的核心变换）"""
    x1 = x[..., : x.shape[-1] // 2]  # 前半维
    x2 = x[..., x.shape[-1] // 2 :]  # 后半维
    return torch.cat((-x2, x1), dim=-1)  # 拼接 [-x2, x1]，等价于复数平面上的旋转


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """对 query/key 施加旋转位置编码"""
    cos = cos.unsqueeze(unsqueeze_dim)  # 在"头"维度处加一维，使 cos/sin 可从 [batch, seq, dim] 广播到 [batch, heads, seq, dim]
    sin = sin.unsqueeze(unsqueeze_dim)  # sin 同理
    q_embed = (q * cos) + (rotate_half(q) * sin)  # 旋转公式：q*cos + rotate_half(q)*sin
    k_embed = (k * cos) + (rotate_half(k) * sin)  # 对 key 做同样旋转
    return q_embed, k_embed  # 返回旋转后的 q、k


# ==================== 4. GQA 辅助函数 ====================
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA：把 K/V 头重复 n_rep 次，与 Q 的头数对齐

    等价于 torch.repeat_interleave(x, dim=1, repeats=n_rep)，
    把 [batch, num_kv_heads, seq, head_dim] 变成 [batch, num_heads, seq, head_dim]。
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape  # 解构输入形状
    if n_rep == 1:  # 每组恰好 1 个 KV 头，无需重复
        return hidden_states  # 原样返回
    hidden_states = hidden_states[:, :, None, :, :].expand(  # 在头维度后插入一维并扩展（expand 不复制内存，只是视图）
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)  # 展平回正常的头维度


# ==================== 5. 前馈网络（SwiGLU） ====================
class Qwen2MLP(nn.Module):
    """SwiGLU 门控前馈网络：三个线性层 + silu 激活

    intermediate_size 可覆盖默认的 mlp_mid_size（共享专家复用它，
    传入 moe_intermediate_size * n_shared_experts）。
    """

    def __init__(self, config: Qwen2Config, intermediate_size: int | None = None):
        super().__init__()  # 初始化父类
        self.hidden_size = config.hidden_size  # 输入/输出维度
        self.mlp_mid_size = config.mlp_mid_size if intermediate_size is None else intermediate_size  # 中间扩展维度
        self.gate_proj = nn.Linear(self.hidden_size, self.mlp_mid_size, bias=False)  # 门控投影：决定保留多少信息
        self.up_proj = nn.Linear(self.hidden_size, self.mlp_mid_size, bias=False)  # 上投影：提供待激活的值
        self.down_proj = nn.Linear(self.mlp_mid_size, self.hidden_size, bias=False)  # 下投影：投影回原维度

    def forward(self, x):  # 输入 x 形状 [batch, seq, hidden_size]
        # SwiGLU 核心公式：silu(gate(x)) * up(x)，再 down 投影回 hidden_size
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))  # 输出 [batch, seq, hidden_size]


# ==================== 5.5 MoE 混合专家（借鉴 DeepSeekV3） ====================
class Qwen2TopkRouter(nn.Module):
    """门控路由：DeepSeekV3 风格 —— sigmoid 打分 + 分组限定 + top-k 选择

    与常见 softmax 路由不同，DeepSeekV3 用 sigmoid 对每个专家独立打分，
    并引入"分组"：先把专家分成 num_group 组，每组内取 top-2 分数之和作为
    组分数，只放行 topk_group 组，再在剩余专家里做 top-k。这样每个 token
    只需和少数专家计算路由分数，降低路由开销，也让专家分布更均匀。
    """

    def __init__(self, config: Qwen2Config):
        super().__init__()  # 初始化父类
        self.top_k = config.num_experts_per_tok  # 每个 token 激活的专家数
        self.num_experts = config.num_local_experts  # 路由专家总数
        self.hidden_dim = config.hidden_size  # 路由输入的维度
        # 路由权重：[num_experts, hidden]，每一行是一个专家的打分向量
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))
        self.routed_scaling_factor = config.routed_scaling_factor  # 输出缩放系数
        self.num_group = config.n_group  # 分组数
        self.topk_group = config.topk_group  # 放行的组数
        self.norm_topk_prob = config.norm_topk_prob  # 是否归一化路由权重
        # DeepSeekV3 无辅助损失负载均衡用的偏置：官方在训练循环里动态更新它，
        # 这里先注册为零 buffer（register_buffer 而非 Parameter，不参与梯度计算）。
        self.register_buffer("e_score_correction_bias", torch.zeros(self.num_experts))

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.hidden_dim)  # 展平成 [n_tokens, hidden]
        # 用 fp32 打分，避免低精度下 sigmoid 数值不稳定
        router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))  # [n_tokens, num_experts]
        scores = router_logits.sigmoid()  # 每个专家独立打分（0~1）
        scores_for_choice = scores + self.e_score_correction_bias  # 加偏置后再用于"选择"

        # ---- 分组限定：只考虑 topk_group 组专家 ----
        group_scores = (  # 每组分数 = 组内 top-2 专家分数之和
            scores_for_choice.view(-1, self.num_group, self.num_experts // self.num_group)
            .topk(2, dim=-1)[0]  # 每组取最高的 2 个（DeepSeekV3 固定取 2）
            .sum(dim=-1)  # 求和得组分数 [n_tokens, num_group]
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]  # 选分数最高的几组
        group_mask = torch.zeros_like(group_scores)  # 组掩码
        group_mask.scatter_(1, group_idx, 1)  # 选中的组置 1
        score_mask = (  # 把组掩码还原成专家掩码 [n_tokens, num_experts]
            group_mask.unsqueeze(-1)
            .expand(-1, self.num_group, self.num_experts // self.num_group)  # 组内每个专家复制一份
            .reshape(-1, self.num_experts)
        )
        # 未放行的专家分数置 -inf，top-k 一定不会选中它们
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))

        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]  # 最终激活的专家 id
        topk_weights = scores.gather(1, topk_indices)  # 权重取"未加偏置"的原始 sigmoid 分数
        if self.norm_topk_prob:  # 归一化：让 top-k 权重和为 1
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor  # 缩放，补偿稀疏激活的幅值损失
        return router_logits, topk_weights, topk_indices  # 返回 (原始logits, 权重, 专家id)


class Qwen2Experts(nn.Module):
    """全部路由专家的权重打包成三维张量，逐个被命中的专家按 token 计算（DeepSeekV3 风格）"""

    def __init__(self, config: Qwen2Config):
        super().__init__()  # 初始化父类
        self.num_experts = config.num_local_experts  # 专家数
        self.hidden_dim = config.hidden_size  # 输入/输出维度
        self.intermediate_dim = config.moe_intermediate_size  # 每个专家的中间维度
        # 三维权重 [num_experts, 2*mid, hidden]：每行的前一半是 gate、后一半是 up，一次线性后切半
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))  # [num_experts, hidden, mid]

    def forward(self, hidden_states, top_k_index, top_k_weights):
        """hidden_states: [n_tokens, hidden]; 返回同形状的加权专家输出"""
        final_hidden_states = torch.zeros_like(hidden_states)  # 先建全零输出，再把各专家结果累加回去
        with torch.no_grad():  # 掩码/索引只是元数据，不需要梯度
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)  # [n_tokens, top_k, num_experts]
            expert_mask = expert_mask.permute(2, 1, 0)  # 换成 [num_experts, top_k, n_tokens]，便于按专家取
            expert_hit = (expert_mask.sum(dim=(-1, -2)) > 0).nonzero()  # 有哪些专家被至少一个 token 选中

        for expert_idx in expert_hit:  # 逐个被命中的专家
            expert_idx = expert_idx[0]  # 取出专家编号
            # top_k_pos: 该 token 是第几个被选槽位；token_idx: 哪些 token 选了这个专家
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]  # 取出这批 token 的输入
            gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)  # 一次算 gate/up 再切半
            current_hidden_states = F.silu(gate) * up  # SwiGLU 激活
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])  # down 投影
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]  # 乘路由权重
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))  # 累加回对应位置

        return final_hidden_states


class Qwen2MoE(nn.Module):
    """MoE 模块：稀疏路由专家 + 共享专家（DeepSeekV3 结构）"""

    def __init__(self, config: Qwen2Config):
        super().__init__()  # 初始化父类
        self.config = config  # 保存配置
        self.experts = Qwen2Experts(config)  # 稀疏路由专家
        self.gate = Qwen2TopkRouter(config)  # 路由门控
        # 共享专家：每个 token 都经过的稠密 MLP（保证每层表达能力的"下限"）
        self.shared_experts = Qwen2MLP(config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts)

    def forward(self, hidden_states):
        """返回 (MoE 输出, 负载均衡辅助损失)"""
        residuals = hidden_states  # 共享专家要吃残差输入（MoE 整体外面还有一层残差连接）
        orig_shape = hidden_states.shape  # 记下原始 [batch, seq, hidden]
        router_logits, topk_weights, topk_indices = self.gate(hidden_states)  # 路由：打分 + 选专家
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])  # 展平成 token 序列
        hidden_states = self.experts(hidden_states, topk_indices, topk_weights).view(*orig_shape)  # 稀疏专家部分
        hidden_states = hidden_states + self.shared_experts(residuals)  # 叠加共享专家输出
        return hidden_states, self._load_balancing_loss(router_logits, topk_indices)  # 输出 + 辅助损失

    def _load_balancing_loss(self, router_logits, topk_indices):
        """经典负载均衡辅助损失（Mixtral 风格）：让各专家分派更均匀

        loss = num_experts * sum_e ( f_e * P_e )
        - f_e: 被分派到专家 e 的 token 占比
        - P_e: 专家 e 的平均路由概率
        只有 use_aux_loss 为真时才计算，否则返回 0。
        """
        if not self.config.use_aux_loss:  # 开关关闭则不计算（DeepSeekV3 官方走无辅助损失方案）
            return torch.tensor(0.0, device=router_logits.device)
        num_experts = self.config.num_local_experts  # 专家数
        routing_probs = router_logits.softmax(dim=-1)  # 路由 logits 转概率 [n_tokens, num_experts]
        expert_mask = F.one_hot(topk_indices, num_classes=num_experts)  # [n_tokens, top_k, num_experts]
        frac_dispatch = expert_mask.sum(dim=1).float().mean(dim=0)  # f_e：[num_experts]
        mean_prob = routing_probs.float().mean(dim=0)  # P_e：[num_experts]
        return (frac_dispatch * mean_prob).sum() * num_experts  # 加权和乘专家数


# ==================== 6. 多头注意力（GQA + RoPE） ====================
class Qwen2Attention(nn.Module):
    """多头注意力层：Q/K/V 投影 -> RoPE -> GQA 展开 -> 缩放点积注意力 -> O 投影"""

    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()  # 初始化父类
        self.config = config  # 保存配置
        self.layer_idx = layer_idx  # 记录层号（KV 缓存按层存放）
        self.head_dim = config.head_dim  # 每头维度
        self.num_att_heads = config.num_att_heads  # Q 头数 = 14
        self.num_kv_heads = config.num_kv_heads  # K/V 头数 = 2
        # GQA 比例：每组 KV 头要服务多少个 Q 头（14 / 2 = 7）
        self.num_key_value_groups = self.num_att_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5  # 缩放因子 1/sqrt(head_dim)，防止点积数值过大
        self.attention_dropout = config.att_dropout  # 注意力 dropout 概率
        # Q 投影到全部头：896 -> 14*64 = 896（Qwen2 的 Q/K/V 都带 bias，但我们这里修改为不需要bias）
        self.q_proj = nn.Linear(config.hidden_size, self.num_att_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)  # K 只需 KV 头数：896 -> 2*64 = 128
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)  # V 同样 128
        self.o_proj = nn.Linear(self.num_att_heads * self.head_dim, config.hidden_size, bias=False)  # 输出投影：合并所有头，无 bias

    def forward(  # 前向：返回注意力输出 +（可选的）更新后 KV 缓存
        self,
        hidden_states: torch.Tensor,  # 输入 [batch, seq, hidden_size]
        position_embeddings: tuple,  # 预计算的 (cos, sin) 表
        past_key_value: tuple | None = None,  # 该层历史的 (K, V) 缓存
        use_cache: bool = False,  # 是否返回更新后的缓存
    ):
        bsz, q_len, _ = hidden_states.shape  # batch 大小、当前 query 序列长度
        # 三个投影 + 拆成多头 + 头维度移到第 1 维（位置在倒数第 2 维）
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_att_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)  # 头数是 2
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings  # 解包 cos/sin
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)  # 对 Q、K 施加旋转位置编码

        # KV 缓存：把当前步的 K/V 拼接到历史后面，得到完整的 K/V 序列（生成时不用重算旧 token）
        if past_key_value is not None:  # 存在历史缓存
            past_key, past_value = past_key_value  # 解包历史 K/V
            key_states = torch.cat([past_key, key_states], dim=-2)  # 按序列维拼接 K
            value_states = torch.cat([past_value, value_states], dim=-2)  # 按序列维拼接 V
        present = (key_states, value_states) if use_cache else None  # 需要缓存时，返回更新后的 K/V

        # GQA：把 K/V 头重复 7 次，使头数与 Q 对齐
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # ---- 右对齐因果掩码：allowed[i][j]=True 表示 query i 允许看 key j ----
        # （兼容两种场景：prefill 时 q_len==kv_len 退化为标准下三角；解码时 q_len==1 全部放行）
        kv_len = key_states.shape[-2]  # 完整 key 序列长度
        q_start = kv_len - q_len  # 本次 query 的绝对起始位置（生成时 kv_len>q_len，起始位置非 0）
        allowed = (torch.arange(q_len, device=query_states.device)[:, None] + q_start) >= torch.arange(
            kv_len, device=query_states.device
        )[None, :]

        if self.config.use_sdpa:  #FlashAttention
            # SDPA 返回的是"已和 V 加权求和"的最终输出
            # 布尔掩码语义与手写相反：True=允许看，False=屏蔽。
            # 不能用 is_causal=True：它假设 q_len==kv_len，生成时(q_len=1<kv_len)
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=allowed,  # 显式传右对齐因果掩码（True=允许）
                dropout_p=self.attention_dropout if self.training else 0.0,  # 训练时才做 dropout
                scale=self.scaling,  # 缩放因子 1/sqrt(head_dim)
            )
        else:  #手写Attention
            attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scaling  # 缩放点积
            causal = ~allowed  # True=未来位置，需要屏蔽
            attn_weights = attn_weights.masked_fill(causal[None, None], float("-inf"))  # 未来位置填 -inf
            # 用 float32 做 softmax 提升数值稳定性，再转回原精度
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)  # 训练时对注意力权重做 dropout
            attn_output = torch.matmul(attn_weights, value_states)  # 按注意力权重加权求和 V


        # 头维度挪回中间并展开：[batch, seq, heads*head_dim]，再经过输出投影
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)  # 输出投影 [batch, seq, hidden_size]
        return attn_output, present  # 返回输出与（可选）缓存


# ==================== 7. 解码器层 ====================
class Qwen2DecoderLayer(nn.Module):
    """单个解码器层 = 归一化 -> 自注意力 -> 残差 -> 归一化 -> (稠密 MLP | MoE) -> 残差（Pre-Norm 结构）"""

    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()  # 初始化父类
        self.self_attn = Qwen2Attention(config=config, layer_idx=layer_idx)  # 自注意力子层
        # 前 first_k_dense_replace 层用稠密 MLP，其余层换成 MoE（对应 DeepSeekV3 的设计）
        if layer_idx < config.first_k_dense_replace:
            self.mlp = Qwen2MLP(config)  # 稠密前馈网络
        else:
            self.mlp = Qwen2MoE(config)  # 混合专家
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.eps)  # 注意力前的归一化
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.eps)  # FFN 前的归一化

    def forward(  # 前向：返回该层输出 + 更新后的缓存 + 本层辅助损失
        self,
        hidden_states: torch.Tensor,  # 输入 [batch, seq, hidden_size]
        position_embeddings: tuple,  # cos/sin
        past_key_value: tuple | None = None,  # 该层 KV 缓存
        use_cache: bool = False,  # 缓存开关
    ):
        residual = hidden_states  # 保存残差（Pre-Norm：先归一化再进子层）
        hidden_states = self.input_layernorm(hidden_states)  # 第一步归一化
        hidden_states, present = self.self_attn(  # 自注意力
            hidden_states, position_embeddings, past_key_value, use_cache  # 透传缓存
        )
        hidden_states = residual + hidden_states  # 注意力残差连接
        residual = hidden_states  # 再次保存残差
        hidden_states = self.post_attention_layernorm(hidden_states)  # 第二步归一化
        out = self.mlp(hidden_states)  # 前馈子层（稠密 MLP 或 MoE）
        if isinstance(self.mlp, Qwen2MoE):  # MoE 额外返回负载均衡辅助损失
            hidden_states, aux_loss = out  # 解包 (输出, 辅助损失)
        else:  # 稠密 MLP 只返回输出
            hidden_states, aux_loss = out, torch.tensor(0.0, device=out.device)
        hidden_states = residual + hidden_states  # 前馈残差连接
        return hidden_states, present, aux_loss  # 返回输出、缓存、辅助损失


# ==================== 8. 完整主干（嵌入 + 24 层 + 最终归一化） ====================
class Qwen2Model(nn.Module):
    """Transformer 主干：词嵌入 -> 24 层解码器 -> 最终归一化（不含 lm_head）"""

    def __init__(self, config: Qwen2Config):
        super().__init__()  # 初始化父类
        self.config = config  # 保存配置
        self.padding_idx = 0  # padding token 的 id（对应嵌入行固定为 0，训练时不会更新）
        # 词嵌入表：行数=词表，列数=hidden_size；padding_idx 指定了那一行保持为 0
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(  # ModuleList：让每层的参数都被注册进模型
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.layers)]  # 建 24 层
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.eps)  # 最后一层之后的最终归一化
        self.rotary_emb = Qwen2RotaryEmbedding(config.head_dim, config.rope_theta)  # 全局共享一份 RoPE 表

    def forward(  # 前向：返回 (隐状态, KV 缓存列表, 辅助损失总和)
        self,
        input_ids: torch.LongTensor,  # token id 序列 [batch, seq]
        position_ids: torch.LongTensor | None = None,  # 位置 id（不传则自动计算）
        past_key_values: list | None = None,  # 各层的 KV 缓存列表，长度 = 层数
        use_cache: bool = False,  # 是否返回缓存
    ):
        batch, seq_len = input_ids.shape  # batch 大小、序列长度
        hidden_states = self.embed_tokens(input_ids)  # 查嵌入表 -> [batch, seq, hidden_size]

        # 计算本次 query 的绝对起始位置：若带缓存，则从历史长度开始
        if past_key_values is not None and past_key_values[0] is not None:  # 有缓存
            past_len = past_key_values[0][0].shape[-2]  # 历史 key 的长度
        else:
            past_len = 0  # 无缓存，从 0 开始
        if position_ids is None:  # 未显式给出位置 id
            # 当前位置 id = 历史长度 + 段内序号（生成时每步都从最新位置开始）
            position_ids = torch.arange(past_len, past_len + seq_len, device=input_ids.device)[None, :].expand(batch, -1)

        cos, sin = self.rotary_emb(hidden_states, position_ids)  # 预计算整段 cos/sin
        position_embeddings = (cos, sin)  # 打包成元组传给各层

        presents = [] if use_cache else None  # 收集各层更新后的缓存
        total_aux_loss = torch.tensor(0.0, device=input_ids.device)  # 累加所有 MoE 层的辅助损失
        for i, layer in enumerate(self.layers):  # 逐层前向
            pv = None if past_key_values is None else past_key_values[i]  # 取该层的历史缓存
            hidden_states, present, aux_loss = layer(hidden_states, position_embeddings, pv, use_cache)  # 前向一层
            total_aux_loss = total_aux_loss + aux_loss  # 累加辅助损失
            if use_cache:  # 需要缓存时记录
                presents.append(present)  # 存下该层更新后的 K/V

        hidden_states = self.norm(hidden_states)  # 最终归一化
        return hidden_states, presents, total_aux_loss  # 返回隐状态、缓存、辅助损失


# ==================== 9. 因果语言模型（lm_head + 损失 + 生成） ====================
class Qwen2ForCausalLM(nn.Module):
    """因果语言模型：主干 + 词表映射头，支持训练损失计算与自回归生成"""

    def __init__(self, config: Qwen2Config):
        super().__init__()  # 初始化父类
        self.config = config  # 保存配置
        self.model = Qwen2Model(config)  # 主干
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)  # 隐状态 -> 词表 logits（无 bias）
        self.post_init()  # 权重初始化 + 权重共享

    def post_init(self):
        """HF 风格权重初始化，并在需要时绑定 lm_head 与 embedding 的权重"""
        for _, module in self.named_modules():  # 遍历所有子模块
            if isinstance(module, nn.Linear):  # 线性层
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initial_range)  # 权重：正态分布 N(0, 0.02)
                if module.bias is not None:  # 有 bias 的层
                    nn.init.zeros_(module.bias)  # bias 初始化为 0
            elif isinstance(module, nn.Embedding):  # 嵌入表
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initial_range)  # 同样是 N(0, 0.02)
            elif isinstance(module, Qwen2TopkRouter):  # MoE 路由权重（专家打分向量）
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initial_range)
                nn.init.zeros_(module.e_score_correction_bias)  # 负载均衡偏置置 0
            elif isinstance(module, Qwen2Experts):  # MoE 专家的三维权重（不是 nn.Linear，需单独初始化）
                nn.init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initial_range)
                nn.init.normal_(module.down_proj, mean=0.0, std=self.config.initial_range)
        if self.config.tie_word_embeddings:  # 若开启权重共享（Qwen2 默认开启）
            # 让 lm_head 与 embed_tokens 指向同一个 Parameter 对象（节省 ~136M 参数）
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(  # 前向：返回 (loss, logits, 缓存)
        self,
        input_ids: torch.LongTensor,  # token id [batch, seq]
        position_ids: torch.LongTensor | None = None,  # 位置 id
        past_key_values: list | None = None,  # KV 缓存
        use_cache: bool = False,  # 缓存开关
        labels: torch.LongTensor | None = None,  # 训练标签（可选，用于计算损失）
    ):
        hidden_states, presents, aux_loss = self.model(input_ids, position_ids, past_key_values, use_cache)  # 主干前向
        logits = self.lm_head(hidden_states)  # 映射到词表 -> [batch, seq, vocab_size]

        loss = None  # 默认无损失
        if labels is not None:  # 训练模式
            # 语言模型标准做法：用当前 token 预测下一个 token，所以输入和标签错开一位
            shift_logits = logits[:, :-1, :].contiguous()  # 去掉最后一个位置的 logits
            shift_labels = labels[:, 1:].contiguous()  # 去掉第一个位置的标签
            lm_loss = F.cross_entropy(  # 交叉熵损失
                shift_logits.view(-1, self.config.vocab_size),  # 展平成 [N, vocab]
                shift_labels.view(-1),  # 展平成 [N]
            )
            # 总损失 = 语言建模损失 + 负载均衡辅助损失（MoE 训练稳定性的关键）
            loss = lm_loss + self.config.aux_loss_coef * aux_loss
        return loss, logits, presents  # 返回损失、logits、缓存

    @torch.no_grad()  # 生成过程不需要梯度
    def generate(  # 自回归生成（KV 缓存加速）
        self,
        input_ids: torch.LongTensor,  # 输入的 prompt [batch, seq]
        max_new_tokens: int = 32,  # 最多生成多少个新 token
        temperature: float = 1.0,  # 采样温度：越大越随机
        top_k: int = 50,  # top-k 采样：只从概率最高的 k 个里选
        top_p: float = 0.9,  # top-p（核采样）：累计概率达到 p 的集合
        eos_token_id: int | None = None,  # 结束符 id，生成到它则提前停止
    ):
        self.eval()  # 切换到推理模式（关闭 dropout）
        # 预填充（prefill）：把整个 prompt 一次性前向，得到每层的 K/V 缓存
        _, _, past_key_values = self(input_ids, use_cache=True)  # 首次前向建立缓存
        for _ in range(max_new_tokens):  # 逐 token 生成
            # 每次只喂最后一个 token，其余上下文从 KV 缓存读取（避免重复计算）
            _, logits, past_key_values = self(input_ids[:, -1:], past_key_values=past_key_values, use_cache=True)
            next_logits = logits[:, -1, :] / temperature  # 取最后一个位置的 logits 并除以温度

            if top_k > 0:  # top-k 截断
                topk_logits, topk_idx = next_logits.topk(top_k, dim=-1)  # 取概率最高的 k 个
                next_logits = torch.full_like(next_logits, float("-inf"))  # 全部先置为 -inf
                next_logits = next_logits.scatter(-1, topk_idx, topk_logits)  # 只把 top-k 的位置填回原值

            probs = F.softmax(next_logits, dim=-1)  # softmax 得概率分布
            if top_p < 1.0:  # top-p（核采样）截断
                sorted_probs, sorted_idx = probs.sort(descending=True)  # 概率从大到小排序
                cumsum = sorted_probs.cumsum(-1)  # 累计概率
                remove = cumsum - sorted_probs > top_p  # 累计概率超过阈值的"尾部"要被滤掉
                sorted_probs = sorted_probs.masked_fill(remove, 0.0)  # 尾部置零
                probs = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)  # 还原到原始索引位置
                probs = probs / probs.sum(-1, keepdim=True)  # 重新归一化，保证和为 1

            next_token = torch.multinomial(probs, num_samples=1)  # 按概率采样一个 token
            input_ids = torch.cat([input_ids, next_token], dim=1)  # 拼接到输出序列
            if eos_token_id is not None and (next_token == eos_token_id).all():  # 遇到结束符
                break  # 提前停止生成
        return input_ids  # 返回完整输出序列 [batch, prompt_len + 生成数]


# ==================== 10. 冒烟测试 ====================
if __name__ == "__main__":
    config = Qwen2Config()  # 用 MoE 版默认超参数
    model = Qwen2ForCausalLM(config)  # 构建模型

    n_params = sum(p.numel() for p in model.parameters())  # 统计参数个数
    print(f"总参数量: {n_params / 1e6:.1f} M（约 {n_params / 1e9:.2f} B）")

    # 估算单 token 实际激活的参数（MoE 层只激活 top-2 专家 + 共享专家）
    n_emb = config.vocab_size * config.hidden_size  # 词嵌入（共享后只计一次）
    n_attn = sum(p.numel() for p in model.model.layers[0].self_attn.parameters()) * config.layers  # 所有注意力层
    n_dense_mlp = sum(p.numel() for p in model.model.layers[0].mlp.parameters()) * config.first_k_dense_replace  # 前几层稠密 MLP
    n_moe_active = (  # 每层 MoE：top-2 专家 + 共享专家
        config.num_experts_per_tok * 3 * config.hidden_size * config.moe_intermediate_size  # 路由专家（SwiGLU 三投影）
        + 3 * config.hidden_size * config.moe_intermediate_size  # 共享专家
    ) * (config.layers - config.first_k_dense_replace)
    print(f"单 token 激活参数量: {(n_emb + n_attn + n_dense_mlp + n_moe_active) / 1e6:.1f} M")

    # 前向 + 损失测试（用随机 id 当输入和标签；loss 已包含负载均衡辅助损失）
    ids = torch.randint(0, config.vocab_size, (1, 8))  # 随机 token 序列 [1, 8]
    loss, logits, past = model(ids, labels=ids, use_cache=True)  # 前向（含损失、缓存）
    print("logits:", tuple(logits.shape), "| loss:", round(loss.item(), 4))  # 打印输出形状与损失

    # 生成测试：给 3 个 token 作为 prompt，生成 5 个新 token
    out = model.generate(torch.tensor([[0, 1, 2]]), max_new_tokens=5, eos_token_id=None)
    print("generate 输出形状:", tuple(out.shape), "(3 个 prompt + 5 个新 token)")  # 应为 [1, 8]
