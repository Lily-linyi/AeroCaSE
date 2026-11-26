"""Minimal AeroCaSE enhancement interface for an external semantic stream."""

import torch
from torch import nn
from torch.nn import functional as F

from .modules import CausalSmallObjectActivation


class AeroCaSECore(nn.Module):


    def __init__(self, *, embed_dim, soa_init_scale, branch_options):
        super().__init__()
        self.soa_enhancer = CausalSmallObjectActivation(**branch_options)
        self.causal_sos_to_embed = nn.Conv2d(
            branch_options["mid_channels"], embed_dim, kernel_size=1,
        )
        self.sos_scale = nn.Parameter(torch.tensor(float(soa_init_scale)))

    def forward(self, image, semantic_features):
        """Return enhanced feature maps, binary-prior logits, and prior probability."""
        _, _, causal_f8, aux_small_logits, small_prob = self.soa_enhancer(image)
        causal_embed = self.causal_sos_to_embed(causal_f8)

        enhanced_features = []
        for feat in semantic_features:
            causal_resized = F.interpolate(
                causal_embed,
                size=feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            enhanced_features.append(feat + self.sos_scale * causal_resized)

        return enhanced_features, aux_small_logits, small_prob
