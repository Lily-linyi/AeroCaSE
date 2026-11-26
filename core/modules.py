

import torch
from torch import nn
from torch.nn import functional as F


def _valid_gn_groups(channels, max_groups):
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvGNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, *, groups, kernel_size=3, stride=1):
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=kernel_size // 2, bias=False,
            ),
            nn.GroupNorm(_valid_gn_groups(out_channels, groups), out_channels),
            nn.ReLU(inplace=True),
        )


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(
            2, 1, kernel_size, padding=kernel_size // 2, bias=False,
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class AuxGuidedCausalFrequencyFilter(nn.Module):

    def __init__(
        self, channels, *, norm_groups, low_radius, high_radius,
        low_weight, mid_weight, high_weight, learnable_weights,
        gate_temperature, min_gate, high_bg_weight=None,
    ):
        super().__init__()
        if not 0.0 <= low_radius < high_radius <= 1.0:
            raise ValueError("Require 0 <= low_radius < high_radius <= 1")
        self.channels = int(channels)
        self.low_radius = float(low_radius)
        self.high_radius = float(high_radius)
        self.gate_temperature = float(gate_temperature)
        self.min_gate = float(min_gate)

        self.high_bg_weight = high_bg_weight

        weights = torch.tensor([low_weight, mid_weight, high_weight], dtype=torch.float32)
        if learnable_weights:
            self.band_weights = nn.Parameter(weights)
        else:
            self.register_buffer("band_weights", weights)

        self.out_proj = nn.Sequential(
            ConvGNAct(channels, channels, kernel_size=1, groups=norm_groups),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def _frequency_masks(self, height, width, device, dtype):
        y = torch.fft.fftfreq(height, device=device, dtype=dtype).view(height, 1)
        x = torch.fft.fftfreq(width, device=device, dtype=dtype).view(1, width)
        radius = torch.sqrt(x.square() + y.square())
        radius = radius / radius.max().clamp_min(1e-6)

        low = (radius <= self.low_radius).to(dtype)
        high = (radius >= self.high_radius).to(dtype)
        mid = 1.0 - low - high
        return low.view(1, 1, height, width), mid.view(1, 1, height, width), high.view(1, 1, height, width)

    def _positive_weights(self, dtype, device):
        weights = self.band_weights.to(device=device, dtype=dtype)
        return (
            torch.sigmoid(weights[0]),
            1.0 + F.softplus(weights[1]),
            F.softplus(weights[2]),
        )

    def forward(self, feat, small_prob):
        dtype = feat.dtype
        b, c, h, w = feat.shape
        feat_fp32 = feat.float()
        prob = F.interpolate(
            small_prob.float(),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        prob = torch.sigmoid(torch.logit(prob.clamp(1e-4, 1.0 - 1e-4)) / self.gate_temperature)
        prob = prob.clamp_min(self.min_gate)

        low_mask, mid_mask, high_mask = self._frequency_masks(
            h, w, feat_fp32.device, feat_fp32.dtype
        )
        low_w, mid_w, high_w = self._positive_weights(feat_fp32.dtype, feat_fp32.device)

        fft_feat = torch.fft.fft2(feat_fp32, dim=(-2, -1), norm="ortho")
        low = torch.fft.ifft2(fft_feat * low_mask, dim=(-2, -1), norm="ortho").real
        mid = torch.fft.ifft2(fft_feat * mid_mask, dim=(-2, -1), norm="ortho").real
        high = torch.fft.ifft2(fft_feat * high_mask, dim=(-2, -1), norm="ortho").real

        causal_feat = low_w * low + mid_w * mid + (1.0 + high_w * prob) * high
        return self.out_proj(causal_feat).to(dtype)


class CausalSmallObjectActivation(nn.Module):
    """DSSOE shallow attention/prior branch followed by CFEM.

    ``filter_options`` supplies the CFEM settings explicitly. Input images are
    the same normalized tensors supplied to the external semantic backbone.
    """

    def __init__(
        self, in_channels, stem_channels, mid_channels, *,
        norm_groups, attention_kernel_size, filter_options,
    ):
        super().__init__()
        self.stem_x4 = nn.Sequential(
            ConvGNAct(in_channels, stem_channels, stride=2, groups=norm_groups),
            ConvGNAct(stem_channels, mid_channels, stride=2, groups=norm_groups),
            ConvGNAct(mid_channels, mid_channels, groups=mid_channels),
        )
        self.stem_x8 = nn.Sequential(
            ConvGNAct(mid_channels, mid_channels, stride=2, groups=norm_groups),
            ConvGNAct(mid_channels, mid_channels, groups=mid_channels),
        )
        self.attention = SpatialAttention(kernel_size=attention_kernel_size)
        self.aux_head = nn.Sequential(
            ConvGNAct(mid_channels * 2, mid_channels, groups=norm_groups),
            nn.Conv2d(mid_channels, 2, kernel_size=1),
        )
        self.causal_filter = AuxGuidedCausalFrequencyFilter(
            channels=mid_channels, norm_groups=norm_groups, **filter_options,
        )

    def forward(self, x):
        f4 = self.stem_x4(x)
        f8 = self.stem_x8(f4)

        att8 = self.attention(f8)
        f8_att = f8 * att8
        f4_att = f4 * F.interpolate(
            att8, size=f4.shape[-2:], mode="bilinear", align_corners=False
        )
        aux_small_logits = self.aux_head(
            torch.cat(
                [
                    f4_att,
                    F.interpolate(
                        f8_att,
                        size=f4_att.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ),
                ],
                dim=1,
            )
        )
        small_prob = F.softmax(aux_small_logits.float(), dim=1)[:, 1:2]
        causal_f8 = self.causal_filter(f8, small_prob)
        return f4_att, f8_att, causal_f8, aux_small_logits, small_prob
