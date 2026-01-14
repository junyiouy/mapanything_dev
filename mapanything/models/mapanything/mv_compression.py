import torch
import torch.nn as nn
import torch.nn.functional as F

class ViewPooling(nn.Module):
    def __init__(self, mode='attn', dim=768, num_heads=8):
        super().__init__()
        self.mode = mode
        self.dim = dim
        self.num_heads = num_heads
        
        self.norm = nn.LayerNorm(dim)
        self.output_norm = nn.LayerNorm(dim)
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
        
        
        if self.mode == 'mean':
            return self.output_norm(x.mean(dim=1, keepdim=True)) # (B, 1, C)
        
        elif self.mode == 'attn':
            # q: (B, 1, C), k/v: (B, k, C)
            q = self.q_proj(self.query_token.expand(B, -1, -1))
            x = self.norm(x)
            k_f = self.k_proj(x)
            v_f = self.v_proj(x)
            
            # 使用 PyTorch 高效的注意力实现
            # 注意：此处 q 为 1 个 token，k 为 k 个视角
            attn_out = F.scaled_dot_product_attention(
                q.unsqueeze(1), k_f.unsqueeze(1), v_f.unsqueeze(1)
            ).squeeze(1) # (B, 1, C)
            
            return self.output_norm(self.out_proj(attn_out))

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
