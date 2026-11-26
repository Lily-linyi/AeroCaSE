"""Public AeroCaSE core modules."""

from .aerocase import AeroCaSECore
from .compat import align_legacy_model_with_paper
from .modules import (
    AuxGuidedCausalFrequencyFilter,
    CausalSmallObjectActivation,
    SpatialAttention,
)

__all__ = [
    "AeroCaSECore",
    "align_legacy_model_with_paper",
    "AuxGuidedCausalFrequencyFilter",
    "CausalSmallObjectActivation",
    "SpatialAttention",
]
