import copy
import math

import torch
from torch import nn
from torch.functional import F

import configs

MAX_SEQ_LEN = configs.MAX_SEQ_LEN


def clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_emb, tgt_emb, generator):
        super().__init__()
        self.src_emb = src_emb
        self.tgt_emb = tgt_emb
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator

    def encode(self, x, src_mask):
        x = self.src_emb(x)
        x = self.encoder(x, src_mask)
        return x

    def decode(self, enc_out, src_mask, y, tgt_mask):
        y = self.tgt_emb(y)
        y = self.decoder(y, enc_out, src_mask, tgt_mask)
        return y

    def forward(self, x, y, src_mask, tgt_mask):
        x = self.encode(x, src_mask)
        y = self.decode(x, src_mask, y, tgt_mask)
        return y


class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x, raw_outputs=False):
        x = self.linear(x)
        if raw_outputs:
            return x
        else:
            return F.log_softmax(x, dim=-1)


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        x_mean = x.mean(dim=-1, keepdim=True)
        x_std = x.std(dim=-1, keepdim=True)
        x = (x - x_mean) / (x_std + self.eps)
        x = self.gamma * x + self.beta
        return x


class SublayerConnection(nn.Module):
    def __init__(self, d_model, p):
        super().__init__()
        self.dropout = nn.Dropout(p)
        self.layernorm = LayerNorm(d_model)

    def forward(self, x, sublayer):
        x_ = x.clone()
        x = self.layernorm(x)
        x = sublayer(x)
        x = self.dropout(x)
        x += x_
        return x


class EncoderLayer(nn.Module):
    def __init__(self, mha, ffn, d_model, p):
        super().__init__()
        self.mha = mha
        self.ffn = ffn
        self.sublayers = clones(SublayerConnection(d_model, p), 2)
        self.d_model = d_model

    def forward(self, x, enc_self_mask):
        x = self.sublayers[0](x, lambda x: self.mha(x, x, x, enc_self_mask))
        x = self.sublayers[1](x, lambda x: self.ffn(x))
        return x


class Encoder(nn.Module):
    def __init__(self, layer, n_layers):
        super().__init__()
        self.layers = clones(layer, n_layers)
        self.layernorm = LayerNorm(layer.d_model)

    def forward(self, x, enc_self_mask):
        for layer in self.layers:
            x = layer(x, enc_self_mask)
        x = self.layernorm(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, self_mha, cross_mha, ffn, d_model, p):
        super().__init__()
        self.self_mha = self_mha
        self.cross_mha = cross_mha
        self.ffn = ffn
        self.sublayers = clones(SublayerConnection(d_model, p), 3)
        self.d_model = d_model

    def forward(self, y, x, src_mask, tgt_mask):
        y = self.sublayers[0](y, lambda y: self.self_mha(y, y, y, tgt_mask))
        y = self.sublayers[1](y, lambda y: self.cross_mha(y, x, x, src_mask))
        y = self.sublayers[2](y, lambda y: self.ffn(y))
        return y


class Decoder(nn.Module):
    def __init__(self, layer, n_layers):
        super().__init__()
        self.layers = clones(layer, n_layers)
        self.layernorm = LayerNorm(layer.d_model)

    def forward(self, y, x, src_mask, tgt_mask):
        for layer in self.layers:
            y = layer(y, x, src_mask, tgt_mask)
        y = self.layernorm(y)
        return y


def scaled_dot_attention(q, k, v, mask, dropout):
    d_k = q.shape[-1]
    attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        attn = attn.masked_fill(mask == 0, -1e9)
    attn = F.softmax(attn, -1)
    if dropout is not None:
        attn = dropout(attn)

    value = torch.matmul(attn, v)
    return value, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, p=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.dropout = nn.Dropout(p)
        self.linears = clones(nn.Linear(d_model, d_model), 4)

    def forward(self, q, k, v, mask=None):
        batch_size, max_seq_len, _ = q.shape
        q = self.linears[0](q)
        k = self.linears[1](k)
        v = self.linears[2](v)
        q = q.reshape(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = k.reshape(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = v.reshape(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        value, attn = scaled_dot_attention(q, k, v, mask, self.dropout)
        value = value.transpose(1, 2).reshape(batch_size, max_seq_len, self.n_heads * self.d_k)
        value = self.linears[3](value)

        del q
        del k
        del v
        return value


class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, p=0.1):
        super().__init__()
        self.expand_linear = nn.Linear(d_model, d_model * 4)
        self.shrink_linear = nn.Linear(d_model * 4, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x = self.expand_linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.shrink_linear(x)
        return x


class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        x = self.emb(x) * math.sqrt(self.d_model)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, d_model, p=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p)
        pe = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len).unsqueeze(-1)
        even_i = torch.arange(0, d_model, 2)
        denom = torch.pow(10000, even_i / d_model)
        pe[:, 0::2] = torch.sin(pos / denom)
        pe[:, 1::2] = torch.cos(pos / denom)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, : x.shape[1]]
        x = self.dropout(x)
        return x


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, h=8, dropout=0.1):
    c = copy.deepcopy
    position = PositionalEncoding(MAX_SEQ_LEN, d_model, dropout)
    src_emb = nn.Sequential(Embeddings(src_vocab, d_model), c(position))
    tgt_emb = nn.Sequential(Embeddings(tgt_vocab, d_model), c(position))

    mha = MultiHeadAttention(d_model, h, dropout)
    pos_ffn = PositionWiseFFN(d_model, dropout)

    encoder_layer = EncoderLayer(c(mha), c(pos_ffn), d_model, dropout)
    encoder = Encoder(encoder_layer, N)

    decoder_layer = DecoderLayer(c(mha), c(mha), c(pos_ffn), d_model, dropout)
    decoder = Decoder(decoder_layer, N)

    generator = Generator(d_model, tgt_vocab)

    model = Transformer(encoder, decoder, src_emb, tgt_emb, generator)

    for i, p in enumerate(model.parameters()):
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
