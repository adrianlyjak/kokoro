# https://github.com/yl4579/StyleTTS2/blob/main/models.py
from .istftnet import AdainResBlk1d
from torch.nn.utils import weight_norm
from transformers import AlbertModel
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearNorm(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.linear_layer.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class TextEncoder(nn.Module):
    def __init__(self, channels, kernel_size, depth, n_symbols, actv=nn.LeakyReLU(0.2)):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)
        padding = (kernel_size - 1) // 2
        self.cnn = nn.ModuleList()
        for _ in range(depth):
            self.cnn.append(nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)),
                LayerNorm(channels),
                actv,
                nn.Dropout(0.2),
            ))
        self.lstm = nn.LSTM(channels, channels//2, 1, batch_first=True, bidirectional=True)
        self.hidden_size = channels // 2

    def forward(self, x, input_lengths, m):
        x = self.embedding(x)  # [B, T, emb]
        x = x.transpose(1, 2)  # [B, emb, T]
        m = m.unsqueeze(1)
        x.masked_fill_(m, 0.0)
        for c in self.cnn:
            x = c(x)
            x.masked_fill_(m, 0.0)
        x = x.transpose(1, 2)  # [B, T, chn]
        # Create sequence mask for LSTM
        batch_size = x.size(0)
        max_length = x.size(1)
        seq_mask = (
            torch.arange(max_length, device=x.device)[None, :] < input_lengths[:, None]
        )

        # Initialize LSTM states
        h0 = torch.zeros(2, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(2, batch_size, self.hidden_size, device=x.device)

        # Run LSTM with mask
        x, _ = self.lstm(x * seq_mask.unsqueeze(2), (h0, c0))
        x = x * seq_mask.unsqueeze(2)

        # Replace negative indices with positive ones
        x = x.transpose(2, 1)  # [B, chn, T]
        x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[2]], device=x.device)
        x_pad[:, :, : x.shape[2]] = x
        x = x_pad
        x.masked_fill_(m, 0.0)
        return x


class AdaLayerNorm(nn.Module):
    def __init__(self, style_dim, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.fc = nn.Linear(style_dim, channels*2)

    def forward(self, x, s):
        x = x.transpose(-1, -2)
        x = x.transpose(1, -1)
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        gamma, beta = gamma.transpose(1, -1), beta.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        x = (1 + gamma) * x + beta
        return x.transpose(1, -1).transpose(-1, -2)


class ProsodyPredictor(nn.Module):
    def __init__(self, style_dim, d_hid, nlayers, max_dur=50, dropout=0.1):
        super().__init__()
        self.text_encoder = DurationEncoder(sty_dim=style_dim, d_model=d_hid,nlayers=nlayers, dropout=dropout)
        self.lstm = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.duration_proj = LinearNorm(d_hid, max_dur)
        self.shared = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.F0 = nn.ModuleList()
        self.F0.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.F0.append(AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout))
        self.F0.append(AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout))
        self.N = nn.ModuleList()
        self.N.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.N.append(AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout))
        self.N.append(AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout))
        self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)
        self.N_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)
        self.hidden_size = d_hid // 2

    def forward(self, texts, style, text_lengths, alignment, m):
        d = self.text_encoder(texts, style, text_lengths, m)

        # Create sequence mask
        batch_size = d.size(0)
        max_length = d.size(1)
        seq_mask = (
            torch.arange(max_length, device=d.device)[None, :] < text_lengths[:, None]
        )

        # Initialize LSTM states
        h0 = torch.zeros(2, batch_size, self.hidden_size, device=d.device)
        c0 = torch.zeros(2, batch_size, self.hidden_size, device=d.device)

        # Run LSTM with mask
        x, _ = self.lstm(d * seq_mask.unsqueeze(-1), (h0, c0))
        x = x * seq_mask.unsqueeze(-1)

        x_pad = torch.zeros([x.shape[0], m.shape[-1], x.shape[-1]], device=x.device)
        x_pad[:, : x.shape[1], :] = x
        x = x_pad

        duration = self.duration_proj(nn.functional.dropout(x, 0.5, training=False))
        en = (d.transpose(-1, -2) @ alignment)
        return duration.squeeze(-1), en

    def F0Ntrain(self, x, s):
        x, _ = self.shared(x.transpose(-1, -2))
        F0 = x.transpose(-1, -2)
        for block in self.F0:
            F0 = block(F0, s)
        F0 = self.F0_proj(F0)
        N = x.transpose(-1, -2)
        for block in self.N:
            N = block(N, s)
        N = self.N_proj(N)
        return F0.squeeze(1), N.squeeze(1)


class DurationEncoder(nn.Module):
    def __init__(self, sty_dim, d_model, nlayers, dropout=0.1):
        super().__init__()
        self.lstms = nn.ModuleList()
        self.hidden_size = d_model // 2
        for _ in range(nlayers):
            self.lstms.append(nn.LSTM(d_model + sty_dim, d_model // 2, num_layers=1, batch_first=True, bidirectional=True, dropout=dropout))
            self.lstms.append(AdaLayerNorm(sty_dim, d_model))
        self.dropout = dropout
        self.d_model = d_model
        self.sty_dim = sty_dim

    def forward(self, x, style, text_lengths, m):
        masks = m
        x = x.permute(2, 0, 1)
        s = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, s], axis=-1)
        x.masked_fill_(masks.unsqueeze(2).transpose(0, 1), 0.0)
        x = x.transpose(0, 1)  # [B, T, chn]
        x = x.transpose(2, 1)  # [B, chn, T]

        # Create sequence mask once
        batch_size = text_lengths.size(0)
        max_length = x.size(2)
        seq_mask = (
            torch.arange(max_length, device=x.device)[None, :] < text_lengths[:, None]
        )

        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                x = block(x.transpose(2, 1), style).transpose(2, 1)
                x = torch.cat([x, s.permute(1, 2, 0)], axis=1)
                x.masked_fill_(masks.unsqueeze(2).transpose(1, 2), 0.0)
            else:
                x = x.transpose(2, 1)

                # Initialize LSTM states
                h0 = torch.zeros(2, batch_size, self.hidden_size, device=x.device)
                c0 = torch.zeros(2, batch_size, self.hidden_size, device=x.device)

                # Run LSTM with mask
                x, _ = block(x * seq_mask.unsqueeze(2), (h0, c0))
                x = x * seq_mask.unsqueeze(2)

                x = F.dropout(x, p=self.dropout, training=False)
                x = x.transpose(2, 1)
                x_pad = torch.zeros(
                    [x.shape[0], x.shape[1], m.shape[1]], device=x.device
                )
                x_pad[:, :, : x.shape[2]] = x
                x = x_pad

        return x.transpose(2, 1)


# https://github.com/yl4579/StyleTTS2/blob/main/Utils/PLBERT/util.py
class CustomAlbert(AlbertModel):
    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        return outputs.last_hidden_state
