import torch
import torch.nn as nn
import torch.nn.functional as F



class SimpleAttentionModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, mlp_hidden_dim):
        super(SimpleAttentionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
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
        q = self.q_matrix(embedded)  # [batch_size, seq_len, embed_dim]
        k = self.k_matrix(embedded)  # [batch_size, seq_len, embed_dim]
        v = self.v_matrix(embedded)  # [batch_size, seq_len, embed_dim]

        # Compute attention weights
        attn_weights = torch.matmul(q, k.transpose(-2, -1))  # [batch_size, seq_len, seq_len]
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Apply attention weights to v
        v_out = torch.matmul(attn_weights, v)  # [batch_size, seq_len, embed_dim]

        # Sum over the sequence length dimension to get a single vector per batch
        v_out = v_out.sum(dim=1)  # [batch_size, embed_dim]

        # Pass through the MLP to get the final probability distribution
        logits = self.mlp(v_out)  # [batch_size, vocab_size]
        probs = F.softmax(logits, dim=-1)
        return probs