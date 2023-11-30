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


class PoetryModel(nn.Module):
  def __init__(self, vocab_size,
                     num_heads,
                     head_size,
                     embedding_size,
                     multihead_size,
                     block_size,
                     num_transformers,
                     dropout, 
                     device):
    super().__init__()
    self.block_size = block_size
    self.device = device
    self.token_embedding  = nn.Embedding(vocab_size, embedding_size) # for each vocab we have an embedding
    self.position_embedding = nn.Embedding(block_size, embedding_size)
    self.head_transformer = TransformerBlock(num_heads, head_size, embedding_size,
                                             multihead_size, block_size, dropout)
    self.transformer_blocks = nn.ModuleList([TransformerBlock(num_heads, head_size, multihead_size, multihead_size, block_size, dropout) for _ in range(num_transformers)])
    self.ln = nn.LayerNorm(multihead_size)
    self.mha_proj = nn.Linear(multihead_size, vocab_size)

  def forward(self, x, y=None):
    B, T = x.shape
    t_emb = self.token_embedding(x)
    p_emb = self.position_embedding(torch.arange(T, device=self.device))
    embeddings = t_emb + p_emb
    mha_out = self.head_transformer(embeddings)
    for t in self.transformer_blocks:
      mha_out = t(mha_out)
    logits = self.mha_proj(self.ln(mha_out))
    loss = None
    if y is not None:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      y = y.view(B*T)
      loss = F.cross_entropy(logits, y)

    return logits, loss

  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      idx_cropped = idx[:, -self.block_size:]
      logits, loss = self(idx_cropped)
      last_frame = logits[:, -1, :]
      probs = torch.softmax(last_frame, dim=-1)
      next_token = torch.multinomial(probs, num_samples=1)
      idx = torch.concat((idx, next_token), dim=1)
    return idx

