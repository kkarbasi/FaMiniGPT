import torch
import torch.nn as nn
from torch.nn import functional as F


# H: head_size
# E: embedding_size
class Head(nn.Module):

  def __init__(self, head_size, embedding_size, block_size, dropout):
    super().__init__()
    self.key = nn.Linear(embedding_size, head_size, bias=False) # AK ends up using embedding_size//n_heads as the head_size in each head. Turns out so does pytorch MHA
    self.query = nn.Linear(embedding_size, head_size, bias=False)
    self.value = nn.Linear(embedding_size, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B, T, C = x.shape # Batch x Time x Embedding of each element
    k = self.key(x) # Turns (B, T, E) to (B, T, H)
    q = self.query(x) # Turns (B, T, E) to (B, T, H)
    v = self.query(x) # Turns B, T, E to B, T, H
    kq = k @ q.transpose(-2, -1) * C**-0.5
    kq = kq.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
    kq = F.softmax(kq, dim=-1) # (B, T, T)
    kq = self.dropout(kq)
    kqv = kq @ v # B, T, H
    return kqv

  
class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads, head_size, embedding_size, block_size, droppout): # head_size is the size of single head
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size, embedding_size, block_size, droppout) for _ in range(num_heads)])

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    return out # output size here would be num_heads * head_size (e.g. 4*16 = 64)


class MLP(nn.Module):
  def __init__(self, multihead_size):
    super().__init__()
    self.ff = nn.Sequential(
        nn.Linear(multihead_size, multihead_size),
        nn.ReLU()
        )

  def forward(self, x):
    return self.ff(x)


class TransformerBlock(nn.Module):
  def __init__(self, num_heads,
                     head_size,
                     embedding_size,
                     multihead_size,
                     block_size,
                     dropout
                     ):
    super().__init__()
    self.mha = MultiHeadAttention(num_heads, head_size, embedding_size, block_size, dropout) # output: B, T, multihead_size
    self.mlp = MLP(multihead_size)
    self.residual_proj = nn.Linear(embedding_size, multihead_size, bias=False)
    self.dropout = nn.Dropout(dropout)
    self.ln1 = nn.LayerNorm(embedding_size)
    self.ln2 = nn.LayerNorm(multihead_size)

  def forward(self, x):
    x = self.residual_proj(x) + self.mha(self.ln1(x))
    x = x + self.mlp(self.ln2(x))
    x = self.dropout(x)
    return x
