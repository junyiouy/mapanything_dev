import torch
import torch.nn as nn
import torch.nn.functional as F

class ViewPooling(nn.Module):
    def __init__(self, mode='mean', dim=768, num_heads=8):
        super().__init__()
        self.mode = mode
        self.dim = dim
        self.num_heads = num_heads
        
        if mode == 'attn':
            self.q_proj = nn.Linear(dim, dim)
            self.k_proj = nn.Linear(dim, dim)
            self.v_proj = nn.Linear(dim, dim)
            self.out_proj = nn.Linear(dim, dim)
            # 初始化 query 为一个小值或 0
            self.query_token = nn.Parameter(torch.zeros(1, 1, dim))
            self.norm = nn.LayerNorm(dim)
            
            # 零初始化 Attention 的输出投影，确保初始时聚合结果为 0
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, x, k):
        Bk, C, H, W = x.shape
        B = Bk // k
        
        if self.mode == 'mean':
            return x.view(B, k, C, H, W).mean(dim=1)
        
        elif self.mode == 'attn':
            x_flat = x.view(B, k, C, H, W).permute(0, 3, 4, 1, 2).reshape(B * H * W, k, C)
            q = self.q_proj(self.query_token.expand(B * H * W, -1, -1)) 
            k_f = self.k_proj(x_flat) 
            v_f = self.v_proj(x_flat) 
            
            h = self.num_heads
            d_h = self.dim // h
            q = q.view(-1, 1, h, d_h).transpose(1, 2)
            k_f = k_f.view(-1, k, h, d_h).transpose(1, 2)
            v_f = v_f.view(-1, k, h, d_h).transpose(1, 2)
            
            attn_out = F.scaled_dot_product_attention(q, k_f, v_f, scale=d_h ** -0.5)
            attn_out = attn_out.transpose(1, 2).reshape(B * H * W, 1, self.dim)
            x_comp = self.out_proj(attn_out)
            
            out = x_comp.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            out = self.norm(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
            return out

class ConvResidualBlock(nn.Module):
    def __init__(self, in_d, out_d, zero_init=False):
        super().__init__()
        self.conv = nn.Conv2d(in_d, out_d, kernel_size=3, padding=1, bias=False)
        self.norm = nn.GroupNorm(min(32, out_d), out_d) 
        self.act = nn.GELU()
        self.proj = nn.Conv2d(in_d, out_d, 1) if in_d != out_d else nn.Identity()
        
        # 如果是 Zero Init 模式，将卷积和缩放层设为 0
        if zero_init:
            nn.init.zeros_(self.conv.weight)
            if hasattr(self.proj, 'weight'):
                nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        return self.act(self.norm(self.proj(x) + self.conv(x)))
    
class MultiScaleViewCompressionUNet(nn.Module):
    def __init__(self, dim, pool_mode='mean', num_heads=8):
        super().__init__()
        self.dim = dim
        self.pool_top = ViewPooling(pool_mode, dim, num_heads)

        # Encoder/Decoder 保持正常初始化
        self.enc1 = ConvResidualBlock(dim, dim)
        self.down1 = nn.Conv2d(dim, dim, 3, stride=2, padding=1)
        self.enc2 = ConvResidualBlock(dim, dim)
        self.down2 = nn.Conv2d(dim, dim, 3, stride=2, padding=1)
        
        self.bottleneck = ConvResidualBlock(dim, dim)
        
        self.up2 = nn.ConvTranspose2d(dim, dim, 2, stride=2)
        self.dec2 = ConvResidualBlock(dim * 2, dim)
        self.up1 = nn.ConvTranspose2d(dim, dim, 2, stride=2)
        
        # 关键修改：dec1 使用零初始化，确保初始 y1 为 0
        self.dec1 = ConvResidualBlock(dim * 2, dim, zero_init=True)
        
        # 如果 pool_mode 是 mean，我们需要一个可学习的缩放因子来在初始化时置零
        self.alpha = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x, k):
        # x: (B*k, C, H, W)
        
        # --- 保存 Residual 路径：提取第 0 个视图 (B, C, H, W) ---
        # 假设 x 的排列是 [B1_v0, B1_v1, ..., B1_vk, B2_v0, ...]
        # 我们取每个 batch 的第一个 view
        res_connection = x.view(-1, k, self.dim, x.shape[-2], x.shape[-1])[:, 0, :, :, :]

        # --- ENCODER ---
        e1 = self.enc1(x)                
        d1 = self.down1(e1)              
        e2 = self.enc2(d1)               
        d2 = self.down2(e2)              

        # --- BOTTLENECK ---
        b = self.bottleneck(d2)          

        # --- DECODER Level 2 ---
        y2_up = self.up2(b)                         
        # 这里简化了逻辑，直接对所有 view 做 mean/attn 聚合
        y2_global = y2_up.view(-1, k, self.dim, y2_up.shape[-2], y2_up.shape[-1]).mean(dim=1)
        y2_ext = torch.repeat_interleave(y2_global, repeats=k, dim=0)
        
        if y2_ext.shape[-2:] != e2.shape[-2:]:
            y2_ext = F.interpolate(y2_ext, size=e2.shape[-2:], mode='bilinear', align_corners=False)
        y2 = self.dec2(torch.cat([y2_ext, e2], dim=1))

        # --- DECODER Level 1 ---
        y1_up = self.up1(y2)                        
        y1_global = y1_up.view(-1, k, self.dim, y1_up.shape[-2], y1_up.shape[-1]).mean(dim=1)
        y1_ext = torch.repeat_interleave(y1_global, repeats=k, dim=0)
        
        if y1_ext.shape[-2:] != e1.shape[-2:]:
            y1_ext = F.interpolate(y1_ext, size=e1.shape[-2:], mode='bilinear', align_corners=False)
        
        # y1 的最后一层被初始化为 0
        y1 = self.dec1(torch.cat([y1_ext, e1], dim=1))

        # --- OUTPUT & RESIDUAL ---
        y1_global = self.pool_top(y1, k)
        
        # 最终输出 = 第一个视角 + alpha * 聚合后的增量
        # 初始化时 alpha=0 且 dec1=0，输出纯粹是第一个视角
        return res_connection + self.alpha * y1_global

# 运行对比
if __name__ == "__main__":

    # ---------------------------------------------------------
    # 性能测试函数
    # ---------------------------------------------------------
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    def profile_memory_and_flops(mode='mean', dim=768, num_heads=12):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # 1. 静态模型显存
        mem_base = torch.cuda.memory_allocated(device) / (1024**2)
        model = MultiScaleViewCompressionUNet(dim, mode, num_heads).to(device)
        mem_after_init = torch.cuda.memory_allocated(device) / (1024**2)
        static_mem = mem_after_init - mem_base

        # 模拟输入
        B, k, H, W = 2, 12, 18, 37
        inputs = [torch.randn(B, dim, H, W).to(device) for _ in range(k)]

        # 2. FLOPs 统计 (使用 fvcore)
        flops_counter = FlopCountAnalysis(model, (inputs,))
        total_flops = flops_counter.total() / 1e9 # GFLOPs

        # 3. 前向传播显存
        torch.cuda.reset_peak_memory_stats()
        output = model(inputs)
        forward_peak_mem = torch.cuda.max_memory_allocated(device) / (1024**2)
        
        # 4. 后向传播显存
        loss = output.sum()
        loss.backward()
        backward_peak_mem = torch.cuda.max_memory_allocated(device) / (1024**2)

        return {
            "Params (M)": sum(p.numel() for p in model.parameters()) / 1e6,
            "GFLOPs": total_flops,
            "Static VRAM (MB)": static_mem,
            "Forward Peak VRAM (MB)": forward_peak_mem,
            "Total Peak VRAM (MB)": backward_peak_mem
        }
    results = {}
    for mode in ['mean', 'attn']:
        print(f"Profiling {mode} mode...")
        results[mode] = profile_memory_and_flops(mode)

    # 打印结果表格
    header = f"{'Metric':<25} | {'Mean':<12} | {'Attention':<12} | {'Diff'}"
    print("\n" + "="*70)
    print(header)
    print("-" * 70)
    for key in results['mean'].keys():
        v_m = results['mean'][key]
        v_a = results['attn'][key]
        diff = ((v_a - v_m) / v_m * 100) if v_m != 0 else 0
        print(f"{key:<25} | {v_m:<12.2f} | {v_a:<12.2f} | {diff:>+7.1f}%")
    print("="*70)