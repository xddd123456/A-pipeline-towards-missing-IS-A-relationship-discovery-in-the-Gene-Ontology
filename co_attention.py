# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------


import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MultiHandAtt(nn.Module):
    def __init__(self, input_size=768, output_size=768, hidden_size=768, num_heads=16):
        super(MultiHandAtt, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.linear_v = nn.Linear(input_size, output_size)
        self.linear_k = nn.Linear(input_size, output_size)
        self.linear_q = nn.Linear(input_size, output_size)
        self.linear_merge = nn.Linear(input_size, output_size)

        self.dropout = nn.Dropout(0.1)

    def forward(self, v, k, q):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.num_heads,
            self.hidden_size
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.num_heads,
            self.hidden_size
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.num_heads,
            self.hidden_size
        ).transpose(1, 2)

        atted = self.att(v, k, q)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.hidden_size
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask=None):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, input_size=768, output_size=768):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=input_size,
            mid_size=input_size*4,
            out_size=output_size,
            dropout_r=0.01,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, hidden_size=768, dropout_rate=0.01):
        super(SA, self).__init__()

        self.mhatt = MultiHandAtt()
        self.ffn = FFN()

        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_size)

        self.dropout2 = nn.Dropout(dropout_rate)
        self.norm2 = LayerNorm(hidden_size)

    def forward(self, x):
        x = self.norm1(x + self.dropout1(
            self.mhatt(x, x, x)
        ))

        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self, input_size=768): #  input_tensor_x=None,input_tensor_y=None,
        super(SGA, self).__init__()

        self.mhatt1 = MultiHandAtt()
        self.mhatt2 = MultiHandAtt()
        self.ffn = FFN()
        # self.x = input_tensor_x
        # self.y = input_tensor_y
        self.dropout1 = nn.Dropout(0.1)
        self.norm1 = LayerNorm(input_size)

        self.dropout2 = nn.Dropout(0.1)
        self.norm2 = LayerNorm(input_size)

        self.dropout3 = nn.Dropout(0.1)
        self.norm3 = LayerNorm(input_size)

    def forward(self, x, y):
        # x = self.x
        # y = self.y
        x = self.norm1(x + self.dropout1(
            self.mhatt1(x, x, x)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(y, y, x)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x

