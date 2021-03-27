import torch
import torch.nn as nn
import math


class InnerProd(nn.Module):
    def __init__(self, fc_dim):
        super(InnerProd, self).__init__()
        self.scale = nn.Parameter(torch.ones(fc_dim))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, feat_img, feat_sound):
        sound_size = feat_sound.size()
        B, C = sound_size[0], sound_size[1]
        feat_img = feat_img.view(B, 1, C)
        z = torch.bmm(feat_img * self.scale, feat_sound.view(B, C, -1)) \
            .view(B, 1, *sound_size[2:])
        z = z + self.bias
        return z

    def forward_nosum(self, feat_img, feat_sound):
        (B, C, H, W) = feat_sound.size()
        feat_img = feat_img.view(B, C)
        z = (feat_img * self.scale).view(B, C, 1, 1) * feat_sound
        z = z + self.bias
        return z

    # inference purposes
    def forward_pixelwise(self, feats_img, feat_sound):
        (B, C, HI, WI) = feats_img.size()
        (B, C, HS, WS) = feat_sound.size()
        feats_img = feats_img.view(B, C, HI*WI)
        feats_img = feats_img.transpose(1, 2)
        feat_sound = feat_sound.view(B, C, HS * WS)
        # => HI * WI x HS * WS => (Global AVG по HS * WS то получим где звук проявляется на картинке.
        z = torch.bmm(feats_img * self.scale, feat_sound)\
            .view(B, HI, WI, HS, WS)
        z = z + self.bias
        return z


class Bias(nn.Module):
    def __init__(self):
        super(Bias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        # self.bias = nn.Parameter(-torch.ones(1))

    def forward(self, feat_img, feat_sound):
        (B, C, H, W) = feat_sound.size()
        feat_img = feat_img.view(B, 1, C)
        z = torch.bmm(feat_img, feat_sound.view(B, C, H * W)).view(B, 1, H, W)
        z = z + self.bias
        return z

    def forward_nosum(self, feat_img, feat_sound):
        (B, C, H, W) = feat_sound.size()
        z = feat_img.view(B, C, 1, 1) * feat_sound
        z = z + self.bias
        return z

    # inference purposes
    def forward_pixelwise(self, feats_img, feat_sound):
        (B, C, HI, WI) = feats_img.size()
        (B, C, HS, WS) = feat_sound.size()
        feats_img = feats_img.view(B, C, HI*WI)
        feats_img = feats_img.transpose(1, 2)
        feat_sound = feat_sound.view(B, C, HS * WS)
        z = torch.bmm(feats_img, feat_sound) \
           .view(B, HI, WI, HS, WS)
        z = z + self.bias
        return z


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, channels, num_encoder_layers=2, nhead=8, dim_feedforward=1024, dropout=0.1, activation='relu'):
        super(Transformer, self).__init__()
        self.pos_encoder = PositionalEncoding(channels, dropout, max_len=256*256+14*14)
        encoder_layer = torch.nn.TransformerEncoderLayer(channels, nhead, dim_feedforward, dropout, activation)
        encoder_norm = torch.nn.LayerNorm(channels)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.decoder = torch.nn.Linear(channels, 1)

    def forward(self, feat_img, feat_sound):
        (B, C, HS, WS) = feat_sound.size()

        feat_img = feat_img.view(B, C, 1)
        feat_img = feat_img.transpose(1, 2)  # (B, 1, C)
        feat_img = feat_img.transpose(0, 1)  # (1, B, C)

        feat_sound = feat_sound.view(B, C, HS * WS)
        feat_sound = feat_sound.transpose(1, 2)  # (B, HS * WS, C)
        feat_sound = feat_sound.transpose(0, 1)  # (HS * WS, B, C)

        feat = torch.cat([feat_img, feat_sound], dim=0)   # (1 + HS * WS, B, C)
        feat = self.pos_encoder(feat)   # (1 + HS * WS, B, C)
        x = self.encoder(feat)  # (1 + HS * WS, B, C)
        x = self.fc(x)  # (1 + HS * WS, B, 1)
        x = x[:, :, 0]  # (1 + HS * WS, B)
        x = x.transpose(1, 0)  # (B, 1 + HS * WS)
        img = x[:, :1, None]  # (B, 1)
        sound = x[:, None, 1:]  # (B, HS * WS)
        x = torch.bmm(img, sound)  # (B, 1, HS * WS)
        x = x.view(B, 1, HS, WS)

        return x

    # inference purposes
    def forward_pixelwise(self, feat_img, feat_sound):
        (B, C, HI, WI) = feat_img.size()
        (B, C, HS, WS) = feat_sound.size()

        feat_img = feat_img.view(B, C, HI * WI)
        feat_img = feat_img.transpose(1, 2)  # (B, HI * WI, C)
        feat_img = feat_img.transpose(0, 1)  # (HI * WI, B, C)

        feat_sound = feat_sound.view(B, C, HS * WS)
        feat_sound = feat_sound.transpose(1, 2)  # (B, HS * WS, C)
        feat_sound = feat_sound.transpose(0, 1)  # (HS * WS, B, C)

        feat = torch.cat([feat_img, feat_sound], dim=0)  # (HI * WI + HS * WS, B, C)
        feat = self.pos_encoder(feat)  # (HI * WI + HS * WS, B, C)
        x = self.encoder(feat)  # (HI * WI + HS * WS, B, C)
        x = self.fc(x)  # (HI * WI + HS * WS, B, 1)
        x = x[:, :, 0]  # (HI * WI + HS * WS, B)
        x = x.transpose(1, 0)  # (B, HI * WI + HS * WS)
        img = x[:, :HI*WI, None]  # (B, HI * WI)
        sound = x[:, None, HI*WI:]  # (B, HS * WS)
        x = torch.bmm(img, sound)  # (B, HI * WI, HS * WS)
        x = x.view(B, HI, WI, HS, WS)

        return x
