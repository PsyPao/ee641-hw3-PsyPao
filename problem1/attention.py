"""
Attention mechanisms for sequence-to-sequence modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Args:
        Q: Query tensor [batch, ..., seq_len_q, d_k]
        K: Key tensor [batch, ..., seq_len_k, d_k]
        V: Value tensor [batch, ..., seq_len_v, d_k]
        mask: Optional mask [batch, ..., seq_len_q, seq_len_k]
              Values: 1 for positions to attend, 0 for positions to mask

    Returns:
        output: Attention output [batch, ..., seq_len_q, d_k]
        attention_weights: Attention weights [batch, ..., seq_len_q, seq_len_k]
    """
    d_k = Q.size(-1)

    scores = torch.matmul(Q, K.transpose(-2, -1))
    scores = scores / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -float('inf'))

    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)

    return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.

    Splits d_model into num_heads, applies attention in parallel,
    then concatenates and projects the results.
    """

    def __init__(self, d_model, num_heads):
        """
        Initialize multi-head attention.

        Args:
            d_model: Model dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        """
        Split tensor into multiple heads.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Tensor with shape [batch, num_heads, seq_len, d_k]
        """
        batch_size, seq_len, _ = x.size()

        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x):
        """
        Combine multiple heads back into single tensor.

        Args:
            x: Input tensor [batch, num_heads, seq_len, d_k]

        Returns:
            Tensor with shape [batch, seq_len, d_model]
        """
        if x.dim() == 5:
            x = x.squeeze(1)
        batch_size, num_heads, seq_len, d_k = x.size()

        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def forward(self, query, key, value, mask=None):
        """
        Forward pass of multi-head attention.

        Args:
            query: Query tensor [batch, seq_len_q, d_model]
            key: Key tensor [batch, seq_len_k, d_model]
            value: Value tensor [batch, seq_len_v, d_model]
            mask: Optional attention mask

        Returns:
            output: Attention output [batch, seq_len_q, d_model]
            attention_weights: Attention weights [batch, num_heads, seq_len_q, seq_len_k]
        """
        batch_size = query.size(0)

        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            elif mask.dim() == 4 and mask.size(1) == 1:
                pass
            else:
                mask = mask.unsqueeze(1) if mask.size(1) != self.num_heads else mask

        attn_output, attention_weights = scaled_dot_product_attention(Q, K, V, mask)

        attn_output = self.combine_heads(attn_output)
        output = self.W_o(attn_output)

        return output, attention_weights


def create_causal_mask(seq_len, device=None):
    """
    Create causal mask to prevent attending to future positions.

    Args:
        seq_len: Sequence length
        device: Device to create tensor on

    Returns:
        Mask tensor [1, 1, seq_len, seq_len] lower triangular matrix
    """
    # Lower triangular matrix
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)