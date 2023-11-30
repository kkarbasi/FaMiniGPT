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