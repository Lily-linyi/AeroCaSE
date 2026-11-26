"""Construct AeroCaSE and restore its model state."""
from copy import deepcopy
from pathlib import Path
import tempfile
import torch


def _state_dict(checkpoint):
    state = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    if not isinstance(state, dict) or not all(torch.is_tensor(v) for v in state.values()):
        raise ValueError('Expected a tensor state_dict in the checkpoint')
    return {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}


def read_config(checkpoint, config_path=None):
    from mmengine.config import Config
    if config_path:
        return Config.fromfile(str(config_path), import_custom_modules=False)
    text = checkpoint.get('meta', {}).get('cfg')
    if not isinstance(text, str):
        raise ValueError('Checkpoint has no embedded configuration; use --config')
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'model_config.py'
        path.write_text(text, encoding='utf-8')
        return Config.fromfile(str(path), import_custom_modules=False)


def _model_options(config, state):
    options = deepcopy(config.model)
    backbone = options['backbone']
    if 'core_settings' not in backbone:
        stem = state['backbone.soa_enhancer.stem_x4.0.0.weight']
        mid = state['backbone.causal_sos_to_embed.weight'].shape[1]
        kernel = state['backbone.soa_enhancer.attention.conv.weight'].shape[-1]
        groups = backbone.pop('soa_norm_groups', 8)
        frequency = {name: backbone.pop(name) for name in (
            'low_radius', 'high_radius', 'low_weight', 'mid_weight', 'high_weight',
            'gate_temperature', 'min_gate',
        )}
        frequency['learnable_weights'] = backbone.pop('learnable_freq_weights')
        frequency['high_bg_weight'] = backbone.pop('high_bg_weight', None)
        backbone['core_settings'] = dict(
            soa_init_scale=backbone.pop('soa_init_scale'),
            branch_options=dict(in_channels=stem.shape[1], stem_channels=stem.shape[0],
                                mid_channels=mid, norm_groups=groups,
                                attention_kernel_size=kernel, filter_options=frequency),
        )
        backbone['query_settings'] = dict(
            init_scale=backbone.pop('query_init_scale'),
            enabled=backbone.pop('enable_query_guidance'),
        )
        declared_channels = backbone.pop('soa_channels')
        if declared_channels != mid:
            raise ValueError('Configuration and checkpoint branch dimensions differ')
    backbone['type'] = 'AeroCaSEBackbone'
    backbone['init_cfg'] = None
    options['decode_head']['type'] = 'AeroCaSEHead'
    options['pretrained'] = None
    return options


def load_model(checkpoint_path, backbone_path=None, config_path=None):
    from mmengine.registry import init_default_scope
    import rein  # registers the external semantic stream
    from mmseg.registry import MODELS
    from . import integration, head  # registers the AeroCaSE components

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state = _state_dict(checkpoint)
    config = read_config(checkpoint, config_path)
    init_default_scope('mmseg')
    model = MODELS.build(_model_options(config, state))
    if backbone_path:
        frozen = _state_dict(torch.load(backbone_path, map_location='cpu'))
        frozen = {k if k.startswith('backbone.') else 'backbone.' + k: v
                  for k, v in frozen.items()}
        frozen.update(state)
        state = frozen
    result = model.load_state_dict(state, strict=True)
    report = dict(strict_load=True, missing_keys=list(result.missing_keys),
                  unexpected_keys=list(result.unexpected_keys), loaded_tensors=len(state),
                  backbone_class=type(model.backbone).__module__ + '.' + type(model.backbone).__name__,
                  head_class=type(model.decode_head).__module__ + '.' + type(model.decode_head).__name__,
                  inference_executed=False)
    model.dataset_meta = checkpoint.get('meta', {}).get('dataset_meta', {})
    return model, config, report
