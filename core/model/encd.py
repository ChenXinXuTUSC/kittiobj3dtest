import torch
import torch.nn as nn

from .attn import MultiHeadAttention

class Encoder(nn.Module):
    def __init__(self, num_heads: int, hidn_size: int):
        super().__init__()
        self.attention = None

        self.ma = MultiHeadAttention(num_heads, hidn_size)
        self.ln_1 = nn.LayerNorm(hidn_size)
        self.ln_2 = nn.LayerNorm(hidn_size)
        self.fc = nn.Sequential(
            nn.Linear(hidn_size, hidn_size * 4),
            nn.GELU(),
            nn.Linear(hidn_size * 4, hidn_size),
            nn.Dropout(p=1e-2)
        )

    def forward(self, x: torch.Tensor, retain_attn: bool = False):
        x = x + self.ma(self.ln_1(x), retain_attn)
        x = x + self.fc(self.ln_2(x))

        if retain_attn:
            self.attention = self.ma.attention
        
        return x

class CascadedEncoder(nn.Module):
    def __init__(self, num_casd: int, num_heads: int, hidn_size: int):
        super().__init__()
        self.blocks = nn.ModuleList([
            Encoder(num_heads, hidn_size) for _ in range(num_casd)
        ])
    
    def forward(self, x:torch.Tensor, retain_attn: bool = False):
        if retain_attn:
            self.attention = []

        for enc in self.blocks:
            x = enc(x, retain_attn)
            if retain_attn:
                self.attention.append(enc.attention)
        return x
