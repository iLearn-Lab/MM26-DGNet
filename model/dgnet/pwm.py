from functools import partial

import torch
import torch.nn as nn

from model.dgnet.wavelet import (
    create_wavelet_filter,
    inverse_wavelet_transform,
    wavelet_transform,
)


class BackgroundKnowledgeGuidedModulation(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, text_bg):
        l = self.conv(x * text_bg)
        avg_out = self.fc2(self.relu1(self.fc1(nn.functional.avg_pool2d(l, kernel_size=(l.shape[2], l.shape[3])))))
        text_out = self.fc2(self.relu1(self.fc1(text_bg)))
        out = 1 - self.sigmoid(avg_out + text_out)
        return out


class TargetKnowledgeGuidedModulation(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, text_fg):
        l = self.conv(x * text_fg)
        max_out = self.fc2(self.relu1(self.fc1(nn.functional.max_pool2d(l, kernel_size=(l.shape[2], l.shape[3])))))
        text_out = self.fc2(self.relu1(self.fc1(text_fg)))
        out = self.sigmoid(max_out + text_out)
        return out


class PriorKnowledgeWaveletModulation(nn.Module):
    def __init__(self, in_channels, out_channels, wt_type="haar"):
        super().__init__()
        self.in_channels = in_channels
        self.wt_filter, self.iwt_filter = create_wavelet_filter(
            wt_type, out_channels, out_channels, torch.float
        )
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
        self.inconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0), nn.BatchNorm2d(out_channels)
        )
        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)
        self.outconv = nn.Sequential(
            nn.Conv2d(out_channels, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        self.lang_proj_targets = nn.Sequential(
            nn.Linear(512, out_channels), nn.LayerNorm(out_channels), nn.GELU()
        )
        self.lang_proj_bg = nn.Sequential(
            nn.Linear(512, out_channels), nn.LayerNorm(out_channels), nn.GELU()
        )

        self.attn_LL = BackgroundKnowledgeGuidedModulation(out_channels)
        self.attn_HL = TargetKnowledgeGuidedModulation(out_channels)
        self.attn_LH = TargetKnowledgeGuidedModulation(out_channels)
        self.attn_HH = TargetKnowledgeGuidedModulation(out_channels)

    def forward(self, x, text_sequence_targets, text_sequence_bg):
        l2 = self.inconv(x)
        dwt_x = self.wt_function(l2)
        LL = dwt_x[:, :, 0, :, :]
        HL = dwt_x[:, :, 1, :, :]
        LH = dwt_x[:, :, 2, :, :]
        HH = dwt_x[:, :, 3, :, :]

        targets_global = text_sequence_targets
        targets_global = (
            self.lang_proj_targets(targets_global).unsqueeze(-1).unsqueeze(-1)
        )
        bg_global = text_sequence_bg
        bg_global = self.lang_proj_bg(bg_global).unsqueeze(-1).unsqueeze(-1)

        LL_new = self.attn_LL(LL, bg_global) * LL
        HL_new = self.attn_HL(HL, targets_global) * HL
        LH_new = self.attn_LH(LH, targets_global) * LH
        HH_new = self.attn_HH(HH, targets_global) * HH

        idwt = torch.stack([LL_new, HL_new, LH_new, HH_new], dim=2)
        idwt = self.iwt_function(idwt)
        out = self.outconv(idwt + l2)
        return out
