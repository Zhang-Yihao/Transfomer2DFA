import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from train.config import read_config

try:
    config = read_config('config.json')
except FileNotFoundError:
    config = read_config('train/config.json')


class PositionalEncoding(nn.Module):
    # pos_embed_vec = torch.zeros(embedded.shape[1], embedded.shape[2])
    # for pos in range(embedded.shape[1]):
    #     for i in range(embedded.shape[2] // 2):
    #         pos_embed_vec[pos, 2 * i] = np.sin(pos / 10000 ** (2 * i / embedded.shape[2]))
    #         pos_embed_vec[pos, 2 * i + 1] = np.cos(pos / 10000 ** (2 * i / embedded.shape[2]))
    # pos_embed_vec = pos_embed_vec.unsqueeze(0).repeat(embedded.shape[0], 1, 1)
    def __init__(self, embed_dim, Embedding):
        super(PositionalEncoding, self).__init__()
        self.pos_embed_vec = torch.zeros(config.seq_length, embed_dim)
        # self.embedding is the embedding layer of the model
        self.embedding = Embedding
        # if embedding is a matrix, then we need to compute the positional encoding
        # if embedding is a nn.Embedding layer, then we don't need to compute the positional encoding
        if isinstance(self.embedding, nn.Embedding):
            for pos in range(config.seq_length):
                for i in range(embed_dim // 2):
                    self.pos_embed_vec[pos, 2 * i] = np.sin(pos / 10000 ** (2 * i / embed_dim))
                    self.pos_embed_vec[pos, 2 * i + 1] = np.cos(pos / 10000 ** (2 * i / embed_dim))
            self.pos_embed_vec = self.pos_embed_vec.unsqueeze(0)

    def forward(self, x):
        embedded = self.embedding(x)
        return embedded + self.pos_embed_vec.repeat(embedded.shape[0], 1, 1)


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
    def __init__(self, vocab_size, embed_dim, mlp_hidden_dim, output_size, padding_idx=0):
        super(SimpleAttentionModel, self).__init__()
        self.padding = padding_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.q_matrix = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_matrix = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_matrix = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, output_size)
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
        logits = self.mlp(v_out)  # [batch_size, output_size]
        probs = F.softmax(logits, dim=-1)
        return probs


class SelfAttentionModel(nn.Module):
    # The only difference between this model and SimpleAttentionModel is that
    # the query vector is computed from the last token that is not padding.
    def __init__(self, vocab_size, embed_dim, mlp_hidden_dim, output_size, padding_idx=0):
        super(SelfAttentionModel, self).__init__()
        self.padding = padding_idx
        self.pos_embed = config.pos_embed
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        if self.pos_embed:
            self.embedding = PositionalEncoding(embed_dim, self.embedding)
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
        logits = self.mlp(v_out)  # [batch_size, output_size]
        probs = F.softmax(logits, dim=-1)
        return probs


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_linear = nn.Linear(embed_dim, 3 * embed_dim)  # 生成查询(Q), 键(K), 值(V)
        self.fc_out = nn.Linear(embed_dim, embed_dim)  # 最后的全连接层

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))  # 缩放因子

    def forward(self, value, key, query, mask=None):
        batch_size = query.shape[0]
        # 1) Linear projections in batch from embed_dim => num_heads x head_dim
        qkv = self.qkv_linear(query).view(batch_size, -1, self.num_heads, 3 * self.head_dim)
        query, key, value = qkv.chunk(3, dim=-1)

        # 2) Transpose for attention dot product: bsz x num_heads x seq_len x head_dim
        query, key, value = [x.transpose(1, 2) for x in [query, key, value]]

        # 3) Calculate and scale scores.
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 4) Apply softmax to get the weights.
        attention_weights = F.softmax(scores, dim=-1)

        # 5) Apply attention weights to value vectors.
        context = torch.matmul(attention_weights, value)

        # 6) Concatenate heads and put through final linear layer.
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        output = self.fc_out(context)

        return output


class EnhancedAttentionModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, mlp_hidden_dim, output_size, num_heads=1, padding_idx=0, dropout_rate=0.1):
        super(EnhancedAttentionModel, self).__init__()
        self.padding = padding_idx
        self.pos_embed = config.pos_embed
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        if self.pos_embed:
            self.embedding = PositionalEncoding(embed_dim, self.embedding)
        self.attention_layers = nn.ModuleList([MultiHeadAttention(embed_dim, num_heads) for _ in range(3)])  # 三层多头注意力
        self.norm_layers = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(3)])  # 每个注意力层后的归一化层
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # 添加dropout正则化
            nn.Linear(mlp_hidden_dim, output_size)
        )

    def forward(self, x):
        mask = (x == self.padding)
        x = self.embedding(x)  # [batch_size, seq_len, embed_dim]

        for attn, norm in zip(self.attention_layers, self.norm_layers):
            x = norm(attn(x, x, x, mask))  # 应用多头注意力并归一化

        x = torch.mean(x, dim=1)  # 在序列长度上取平均，也可以根据需要改用其他方法聚合
        x = self.mlp(x)  # 通过前馈网络

        return F.log_softmax(x, dim=-1)
