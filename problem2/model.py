"""
Transformer encoder model for sequence classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from positional_encoding import get_positional_encoding


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    """

    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        # Linear projections in batch from d_model => h x d_k
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)

        # Concatenate heads and project
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        output = self.W_o(context)

        return output, attention_weights


class TransformerEncoderLayer(nn.Module):
    """
    Single transformer encoder layer.
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_output, _ = self.self_attn(x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # FFN with residual
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)

        return x


class SortingClassifier(nn.Module):
    """
    Transformer encoder for sorting detection.

    Uses different positional encoding strategies to analyze
    length generalization capabilities.
    """

    def __init__(
        self,
        vocab_size=101,  # 0-99 for integers + 100 for padding
        d_model=128,
        num_heads=4,
        num_layers=4,
        d_ff=512,
        dropout=0.1,
        max_len=300,
        encoding_type='sinusoidal'
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding (configurable)
        self.pos_encoding = get_positional_encoding(encoding_type, d_model, max_len)
        self.encoding_type = encoding_type

        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)  # Binary classification
        )

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask=None):
        """
        Forward pass.

        Args:
            x: Input sequence [batch, seq_len]
            mask: Padding mask [batch, seq_len]

        Returns:
            Logits for binary classification [batch, 2]
        """
        # Embed tokens
        x = self.embedding(x) * math.sqrt(self.d_model)

        # Add positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # Reshape mask for attention: [batch, seq_len] -> [batch, 1, 1, seq_len]
        attn_mask = None
        if mask is not None:
            attn_mask = mask.unsqueeze(1).unsqueeze(1)

        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x, attn_mask)

        # Global average pooling
        if mask is not None:
            # Mask out padding tokens
            mask = mask.unsqueeze(-1).float()
            x = x * mask
            lengths = mask.sum(dim=1)
            x = x.sum(dim=1) / lengths
        else:
            x = x.mean(dim=1)

        # Classification
        logits = self.classifier(x)

        return logits

    def predict(self, x, mask=None):
        """
        Make predictions.

        Args:
            x: Input sequence [batch, seq_len]
            mask: Padding mask

        Returns:
            Predicted class [batch]
        """
        logits = self.forward(x, mask)
        return logits.argmax(dim=-1)


def create_model(encoding_type='sinusoidal', **kwargs):
    """
    Create model with specified positional encoding.

    Args:
        encoding_type: One of 'sinusoidal', 'learned', 'none'
        **kwargs: Additional model arguments

    Returns:
        SortingClassifier model
    """
    return SortingClassifier(encoding_type=encoding_type, **kwargs)