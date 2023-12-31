import torch

import torch.nn as nn


"""
https://arxiv.org/pdf/1706.03762.pdf

Attention Is All You Need
"""


class SinePositionalEncoding(nn.Module):
    """
    From paper "Attention Is All You Need", section 3.5
    """
    def __init__(self, dimension, max_timesteps=1000):
        super(SinePositionalEncoding, self).__init__()
        assert dimension % 2 == 0, "Embedding dimension must be even"
        self.dimension = dimension
        self.pe_matrix = torch.zeros(max_timesteps, dimension)
        # Gather all the even dimensions across the embedding vector
        even_indices = torch.arange(0, self.dimension, 2)
        # Calculate the term using log transforms for faster calculations
        # (https://stackoverflow.com/questions/17891595/pow-vs-exp-performance)
        log_term = torch.log(torch.tensor(10000.0)) / self.dimension
        div_term = torch.exp(even_indices * -log_term)

        # Precompute positional encoding matrix based on odd/even timesteps
        timesteps = torch.arange(max_timesteps).unsqueeze(1)
        self.pe_matrix[:, 0::2] = torch.sin(timesteps * div_term)
        self.pe_matrix[:, 1::2] = torch.cos(timesteps * div_term)

    def forward(self, timestep):
        # [bs, d_model]
        return self.pe_matrix[timestep].to(timestep.device)


class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, embedding_dim=64):
        super().__init__()
        # For each of heads use d_k = d_v = d_model / num_heads
        # self.num_heads = num_heads
        # self.d_model = embedding_dim
        self.d_keys = embedding_dim
        self.d_values = embedding_dim

        # Linear projections of Q, K and V, where
        # W_k is [d_model, d_k], W_q is [d_model, d_k], W_v is [d_model, d_v]
        self.query_projection = nn.Linear(d_model, embedding_dim, bias=False)
        self.key_projection = nn.Linear(d_model, embedding_dim, bias=False)
        self.value_projection = nn.Linear(d_model, embedding_dim, bias=False)

        self.final_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, input_tensor):
        # Get linear projections of K, Q and V according to Fig. 2 in the original Transformer paper
        queries = self.query_projection(input_tensor)  # [b, seqlen, d_k]
        keys = self.key_projection(input_tensor)       # [b, seqlen, d_k]
        values = self.value_projection(input_tensor)   # [b, seqlen, d_v]

        # Perform Scaled Dot-Product Attention (eq. 1 in the Transformer paper).
        # Each SDPA block yields tensor of size d_v.
        scale = self.d_keys ** -0.5
        attention_scores = torch.softmax(torch.matmul(queries, keys.transpose(-1, -2)) * scale, dim=-1) # [b, seqlen, seqlen]
        attention_scores = torch.matmul(attention_scores, values)   # [b, seqlen, d_v]

        return attention_scores
    

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads=8, d_model=512):
        super().__init__()
        head_size = d_model // num_heads
        self.mha = nn.ModuleList([SelfAttentionBlock(d_model, head_size) for _ in range(num_heads)])

    def forward(self, x):
        x = torch.cat([head(x) for head in self.mha], dim=-1)
        
        return x
    