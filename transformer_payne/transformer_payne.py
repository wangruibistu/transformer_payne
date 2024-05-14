import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# 定义频率编码函数
def frequency_encoding(x, min_period, max_period, dimension):
    # 定义频率编码的范围和维度
    positions = x.unsqueeze(-1) / torch.linspace(
        min_period, max_period, dimension // 2, device=x.device
    )
    encodings = torch.cat([torch.sin(positions), torch.cos(positions)], dim=-1)
    return encodings


# 定义模型中的各个子模块
class SinusoidalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(SinusoidalEmbedding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

    def forward(self, x):
        return frequency_encoding(
            x, min_period=1e-6, max_period=10, dimension=self.d_model
        )


class MLPEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(MLPEmbedding, self).__init__()
        self.fc1 = nn.Linear(input_dim, 4 * embed_dim)
        self.fc2 = nn.Linear(4 * embed_dim, embed_dim)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, query, key, value):
        attn_output, _ = self.multihead_attn(query, key, value)
        return attn_output


class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, enc_p):
        attn_output = self.attention(x, enc_p, enc_p)
        x = self.norm1(x + attn_output)
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        return x


class TransformerPayneModelWave(nn.Module):
    def __init__(self, no_layer, no_token, d_att, d_ff, no_head, out_dimensionality):
        super(TransformerPayneModelWave, self).__init__()
        self.no_layer = no_layer
        self.no_token = no_token
        self.d_att = d_att
        self.d_ff = d_ff
        self.no_head = no_head
        self.out_dimensionality = out_dimensionality

        self.sinusoidal_embedding = SinusoidalEmbedding(d_att)
        self.mlp_embedding = MLPEmbedding(d_att, no_token * d_att)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_att, no_head, d_ff) for _ in range(no_layer)]
        )
        self.final_norm = nn.LayerNorm(d_att)
        self.decoder_0 = nn.Linear(d_att, 256)
        self.decoder_1 = nn.Linear(256, out_dimensionality)

    def forward(self, x):
        p, w = x

        # Frequency encoding for wavelength
        enc_w = self.sinusoidal_embedding(w).unsqueeze(0)

        # MLP embedding for spectrum parameters
        enc_p = self.mlp_embedding(p)
        enc_p = enc_p.view(self.no_token, self.d_att)

        # Initial values for transformer blocks
        x_pre = enc_w
        x_post = enc_w

        for transformer_block in self.transformer_blocks:
            x_post = transformer_block(x_post, enc_p)
            x_pre = x_pre + x_post

        x_pre = self.final_norm(x_pre)
        x = x_pre + x_post

        x = F.gelu(self.decoder_0(x[0]))
        x = self.decoder_1(x)

        # Post-processing
        x[1] = torch.pow(10, x[1])
        x[0] = x[0] * x[1]
        return x