import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidn_size: int, head_size: int):
        '''
        Args
        -
        - hidn_size: int, number of embedded token feature channels
        - head_size: int, attention output feature channels
        '''
        super().__init__()
        self.attention = None

        self.hidn_size = hidn_size
        self.head_size = head_size

        self.Wq = nn.Linear(hidn_size, head_size)
        self.Wk = nn.Linear(hidn_size, head_size)
        self.Wv = nn.Linear(hidn_size, head_size)
    
    def forward(self, x: torch.Tensor, retain_attn: bool = False):
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        # (A)ttention scores
        # 1. Q x K^T to get attention scores matrix
        # 2. scale
        # 3. softmax to get probability from scores
        A = torch.matmul(q, k.transpose(-1, -2)) # transpose k to perform matrix multiplication
        A = A / (self.head_size**0.5) # A / sqrt(h)
        P = F.softmax(A, dim=-1)
        P = F.dropout(P, 1e-2)

        if retain_attn:
            self.attention = P
        # compute weighted output value
        x = torch.matmul(P, v)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, hidn_size: int):
        super().__init__()

        self.num_heads = num_heads
        self.hidn_size = hidn_size

        one_head_hidn_size = hidn_size // num_heads
        all_head_hidn_size = num_heads * one_head_hidn_size

        # all heads are parallel, do not use nn.Sequential for serial
        self.heads = nn.ModuleList([
            Attention(
                hidn_size,
                one_head_hidn_size
            ) for _ in range(num_heads)
        ])

        self.backp = nn.Linear(all_head_hidn_size, hidn_size)

    def forward(self, x: torch.Tensor, retain_attn: bool=False):
        #compute each head's attention output
        x = [head(x, retain_attn) for head in self.heads]
        x = torch.cat(x, dim=-1)
        x = self.backp(x)
        x = F.dropout(x, p=1e-2)

        if retain_attn:
            self.attention = torch.stack([
                head.attention for head in self.heads
            ], dim=1)

        return x
