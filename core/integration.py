"""Integration with the REIN semantic stream."""
import torch
from torch import nn
from torch.nn import functional as F
from mmseg.registry import MODELS
from rein.models.backbones.reins_dinov2 import ReinsDinoVisionTransformer
from rein.models.backbones.utils import set_requires_grad, set_train
from .modules import CausalSmallObjectActivation


@MODELS.register_module()
class AeroCaSEBackbone(ReinsDinoVisionTransformer):
    def __init__(self, *, core_settings, query_settings, **kwargs):
        super().__init__(**kwargs)
        options = core_settings['branch_options']
        channels = options['mid_channels']
        self.soa_enhancer = CausalSmallObjectActivation(**options)
        self.causal_sos_to_embed = nn.Conv2d(channels, self.embed_dim, kernel_size=1)
        self.sos_scale = nn.Parameter(torch.tensor(float(core_settings['soa_init_scale'])))
        self.query_sos_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, self.reins.query_dims),
        )
        self.query_sos_scale = nn.Parameter(torch.tensor(float(query_settings['init_scale'])))
        self.enable_query_guidance = bool(query_settings['enabled'])

    def forward_features(self, x, masks=None):
        feats, query_embed = super().forward_features(x, masks)
        _, _, causal_f8, aux_logits, small_prob = self.soa_enhancer(x)
        causal_embed = self.causal_sos_to_embed(causal_f8)
        enhanced = [feat + self.sos_scale * F.interpolate(
            causal_embed, size=feat.shape[-2:], mode='bilinear', align_corners=False,
        ) for feat in feats]
        if query_embed.ndim == 2:
            query_embed = query_embed.unsqueeze(0).expand(x.shape[0], -1, -1)
        if self.enable_query_guidance:
            gate = F.interpolate(small_prob, size=causal_f8.shape[-2:],
                                 mode='bilinear', align_corners=False)
            query_embed = query_embed + self.query_sos_scale * self.query_sos_proj(
                causal_f8 * gate).unsqueeze(1)
        return enhanced, query_embed, aux_logits

    @staticmethod
    def _adaptation_names():
        return ('reins', 'soa_enhancer', 'causal_sos_to_embed',
                'query_sos_proj', 'sos_scale', 'query_sos_scale')

    def train(self, mode=True):
        nn.Module.train(self, mode)
        if mode:
            set_requires_grad(self, self._adaptation_names())
            set_train(self, self._adaptation_names())
        return self

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state = nn.Module.state_dict(self, destination=destination,
                                     prefix=prefix, keep_vars=keep_vars)
        for key in list(state):
            if key.startswith(prefix) and not any(
                name in key[len(prefix):] for name in self._adaptation_names()
            ):
                state.pop(key)
        return state
