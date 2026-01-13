import torch
import torch.nn as nn
import torch.nn.functional as F

class ViewPooling(nn.Module):
    def __init__(self, mode='attn', dim=768, num_heads=8):
        super().__init__()
        self.mode = mode
        self.dim = dim
        self.num_heads = num_heads
        
        self.proj_in = nn.Linear(dim,dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.659 / 2)

        if mode == 'attn':
            self.q_proj = nn.Linear(dim, dim)
            self.k_proj = nn.Linear(dim, dim)
            self.v_proj = nn.Linear(dim, dim)
            self.out_proj = nn.Linear(dim, dim)
            self.query_token = nn.Parameter(torch.randn(1, 1, dim)) # 学习一个全局 Query

    def forward(self, x, k):
        # 输入 x 形状: (B*k, C)
        Bk, C = x.shape
        B = Bk // k
        
        x = x.view(B, k, C) # (B, k, C)
        x = self.proj_in(x)
        
        if self.mode == 'mean':
            x_mean = x.mean(dim=1, keepdim=True)
            x_mean_norm = torch.nn.functional.normalize(x_mean, dim=-1)
            logit_scale = self.logit_scale.exp().clamp(max=100, min=1e-4)
            return logit_scale * x_mean_norm  # (B, 1, C)

        
        elif self.mode == 'attn':
            # q: (B, 1, C), k/v: (B, k, C)
            q = self.q_proj(self.query_token.expand(B, -1, -1))
            k_f = self.k_proj(x)
            v_f = self.v_proj(x)
            
            # 使用 PyTorch 高效的注意力实现
            # 注意：此处 q 为 1 个 token，k 为 k 个视角
            attn_out = F.scaled_dot_product_attention(
                q.unsqueeze(1), k_f.unsqueeze(1), v_f.unsqueeze(1)
            ).squeeze(1) # (B, 1, C)
            
            attn_out = self.out_proj(attn_out)  # (B, 1, C)
            attn_out_norm = torch.nn.functional.normalize(attn_out, dim=-1)
            logit_scale = self.logit_scale.exp().clamp(max=100, min=1e-4)
            return logit_scale * attn_out_norm  # (B, 1, C)

class ConvResidualBlock(nn.Module):
    def __init__(self, in_d, out_d, zero_init=False):
        super().__init__()
        self.conv = nn.Conv2d(in_d, out_d, kernel_size=3, padding=1, bias=False)
        self.norm = nn.GroupNorm(1, out_d) 
        self.act = nn.GELU()
        self.proj = nn.Conv2d(in_d, out_d, 1) if in_d != out_d else nn.Identity()
        
        # 如果是 Zero Init 模式，将卷积和缩放层设为 0
        if zero_init:
            nn.init.zeros_(self.conv.weight)
            if hasattr(self.proj, 'weight'):
                nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        return self.act(self.norm(self.conv(x))) + self.proj(x)
    
class MultiViewToEmbedding(nn.Module):
    def __init__(self, dim, pool_mode='attn', num_heads=12):
        super().__init__()
        self.dim = dim
        
        self.enc1 = ConvResidualBlock(dim, dim)
        self.enc2 = ConvResidualBlock(dim, dim)

        # 4. 视角聚合 (View Pooling)
        self.view_aggregation = ViewPooling(pool_mode, dim, num_heads)

    def forward(self, x, k):
        # x: (B*k, C, H, W)

        x = self.enc1(x)  # (B*k, C, H, W)
        x = self.enc2(x)  # (B*k, C, H, W
        x = x.mean(dim=[2,3])  # 全局平均池化，得到 (B*k, C)        
        
        # --- 视角聚合 ---
        # 核心：将 B*k 个向量聚合为 B 个向量
        embedding = self.view_aggregation(x, k) # (B, 1, dim)
        
        return embedding


class LayerBudgetController(nn.Module):
    def __init__(self, num_chunk_layers):
        super().__init__()
        self.num_chunk_layers = num_chunk_layers
        
        # 可学习参数：每一层争夺"剩余预算"的权重，什么初始化合理，能给softmax合适的分布？
        self.layer_logits = nn.Parameter(torch.ones(num_chunk_layers) * 10.0)


    def forward(self, total_budget):
        """
        Args:
            total_budget (float or tensor): 总预算数量
        """
        # --- 1. 预处理总预算 ---
        if isinstance(total_budget, torch.Tensor):
            budget_val = total_budget.float() # 保持梯度图（虽然通常它是常数）
        else:
            budget_val = torch.tensor(float(total_budget), device=self.layer_logits.device)

        # 确保总预算至少能覆盖每层 1 个 (Base Constraint)
        # 使用 clamp 替代 if-else，代码更简洁且兼容 Tensor 操作
        effective_total_budget = torch.max(
            budget_val, 
            torch.tensor(float(self.num_chunk_layers), device=budget_val.device)
        )

        # --- 2. 计算剩余预算 (Residual Budget) ---
        # 既然每层至少要 1，那就先扣除掉必须的
        residual_budget = effective_total_budget - self.num_chunk_layers
        
        # --- 3. 计算连续分配 (Continuous Allocation) ---
        probs = F.softmax(self.layer_logits, dim=0)
        
        # 理想的浮点数分配结果 (Float)
        # 例如: [2.5, 3.5, 1.0] -> 剩余预算分配
        allocated_residual_float = probs * residual_budget
        
        # --- 4. 关键修改：CumSum Trick 保证整数和守恒 ---
        # 技巧：对 PDF 做累积求和变成 CDF -> 对 CDF 取整 -> 做差分还原 PDF
        # 这能保证 sum(integer_parts) === round(sum(float_parts))
        
        # a. 累积求和
        cum_alloc = torch.cumsum(allocated_residual_float, dim=0)
        
        # b. 最后一个元素强制等于 residual_budget，消除累积浮点误差
        # (虽然 cumsum 后最后一个值理论上就是 residual_budget，但为了数值稳定强行赋值)
        cum_alloc_target = cum_alloc.clone()
        cum_alloc_target[-1] = residual_budget
        
        # c. 对累积值取整
        cum_alloc_rounded = torch.round(cum_alloc_target)
        
        # d. 差分还原 (第 i 个 = cum[i] - cum[i-1])
        # 补一个 0 在最前面方便做差分
        padded_cum = torch.cat([
            torch.zeros(1, device=cum_alloc.device), 
            cum_alloc_rounded
        ])
        allocated_residual_int = padded_cum[1:] - padded_cum[:-1]
        
        # --- 5. STE (Straight-Through Estimator) ---
        # 前向传播用整数 (int)，反向传播传给浮点 (float)
        # y = x_float + (x_int - x_float).detach()
        allocated_residual_ste = allocated_residual_float + (allocated_residual_int - allocated_residual_float).detach()
        
        # --- 6. 加上 Base 1 ---
        # 最终预算 = 基础预算(1) + 剩余分配(>=0)
        final_budgets = 1.0 + allocated_residual_ste
        
        return final_budgets