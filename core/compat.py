
from types import MethodType

from .modules import AuxGuidedCausalFrequencyFilter, CausalSmallObjectActivation


def align_legacy_model_with_paper(model):
    branch_fields = (
        "stem_x4", "stem_x8", "attention", "aux_head", "causal_filter",
    )
    filter_fields = (
        "band_weights", "out_proj", "_frequency_masks",
        "gate_temperature", "min_gate", "low_radius", "high_radius",
    )
    branches = [
        module for module in model.modules()
        if all(hasattr(module, field) for field in branch_fields)
    ]
    if not branches:
        raise ValueError("Expected a complete AeroCaSE shallow attention/prior branch")

    for branch in branches:
        frequency_filter = branch.causal_filter
        if not all(hasattr(frequency_filter, field) for field in filter_fields):
            raise ValueError("Unsupported AeroCaSE frequency-filter interface")
        if tuple(frequency_filter.band_weights.shape) != (3,):
            raise ValueError("Expected the legacy three-element band_weights tensor")

    for branch in branches:
        frequency_filter = branch.causal_filter
        frequency_filter._positive_weights = MethodType(
            AuxGuidedCausalFrequencyFilter._positive_weights, frequency_filter,
        )
        frequency_filter.forward = MethodType(
            AuxGuidedCausalFrequencyFilter.forward, frequency_filter,
        )
        branch.forward = MethodType(CausalSmallObjectActivation.forward, branch)
    return model
