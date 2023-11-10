import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config.config import read_config

config = read_config('config/config.json')


class DigitEmbedding(nn.Module):
    def __init__(self):
        super(DigitEmbedding, self).__init__()
        # config.vocab_size-1 digits + 1 padding token "0"
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embed_dim,
            padding_idx=config.padding_idx
        )

    def forward(self, x):
        return self.embedding(x)


class SimpleAttentionModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, mlp_hidden_dim, padding_idx=0):
        super(SimpleAttentionModel, self).__init__()
        self.padding = padding_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.q_matrix = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_matrix = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_matrix = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, vocab_size)
        )

    def forward(self, x):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        # Compute padding mask
        padding_mask = (x == self.padding)  # [batch_size, seq_len]
        # Compute q, k, v matrices
        q = self.q_matrix(embedded)  # [batch_size, seq_len, embed_dim]
        k = self.k_matrix(embedded)  # [batch_size, seq_len, embed_dim]
        v = self.v_matrix(embedded)  # [batch_size, seq_len, embed_dim]

        # Compute attention weights
        attn_weights = torch.matmul(q, k.transpose(-2, -1))  # [batch_size, seq_len, seq_len]
        attn_weights.masked_fill_(padding_mask.unsqueeze(1), 0)  # [batch_size, seq_len, seq_len]

        # Apply attention weights to v
        v_out = torch.matmul(attn_weights, v)  # [batch_size, seq_len, embed_dim]

        # Sum over the sequence length dimension to get a single vector per batch
        v_out = v_out.sum(dim=1)  # [batch_size, embed_dim]

        # Pass through the MLP to get the final probability distribution
        logits = self.mlp(v_out)  # [batch_size, vocab_size]
        probs = F.softmax(logits, dim=-1)
        return probs


class SelfAttentionModel(nn.Module):
    # The only difference between this model and SimpleAttentionModel is that
    # the query vector is computed from the last token that is not padding.
    def __init__(self, vocab_size, embed_dim, mlp_hidden_dim, padding_idx=0, pos_embed=False):
        super(SelfAttentionModel, self).__init__()
        self.padding = padding_idx
        self.pos_embed = pos_embed
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.q_matrix = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_matrix = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_matrix = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, vocab_size)
        )

    def forward(self, x):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        if self.pos_embed:
            # PE(pos, 2i) = sin(pos/10000^(2i/d_model))
            # PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
            pos_embed_vec = torch.zeros(embedded.shape[1], embedded.shape[2])
            for pos in range(embedded.shape[1]):
                for i in range(embedded.shape[2] // 2):
                    pos_embed_vec[pos, 2 * i] = np.sin(pos / 10000 ** (2 * i / embedded.shape[2]))
                    pos_embed_vec[pos, 2 * i + 1] = np.cos(pos / 10000 ** (2 * i / embedded.shape[2]))
            pos_embed_vec = pos_embed_vec.unsqueeze(0).repeat(embedded.shape[0], 1, 1)
            embedded = embedded + pos_embed_vec
        # Compute padding mask
        padding_mask = (x == self.padding)  # [batch_size, seq_len]
        # get last token that is not padding
        last_token_idx = (x != self.padding).sum(dim=1) - 1  # [batch_size]
        # Compute q, k, v matrices
        q = self.q_matrix(embedded[:, last_token_idx, :])  # [batch_size, embed_dim]
        k = self.k_matrix(embedded)  # [batch_size, seq_len, embed_dim]
        v = self.v_matrix(embedded)  # [batch_size, seq_len, embed_dim]

        # Compute attention weights
        attn_weights = torch.matmul(q, k.transpose(-2, -1))  # [batch_size, 1, seq_len]
        attn_weights.masked_fill_(padding_mask.unsqueeze(1), 0)  # [batch_size, 1, seq_len]

        # Apply attention weights to v
        v_out = torch.matmul(attn_weights, v)  # [batch_size, 1, embed_dim]

        # Sum over the sequence length dimension to get a single vector per batch
        v_out = v_out.sum(dim=1)  # [batch_size, embed_dim]

        # Pass through the MLP to get the final probability distribution
        logits = self.mlp(v_out)  # [batch_size, vocab_size]
        probs = F.softmax(logits, dim=-1)
        return probs
