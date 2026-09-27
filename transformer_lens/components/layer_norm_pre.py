"""Hooked Transformer Layer Norm Pre Component.

This module contains all the component :class:`LayerNormPre`.
"""

from typing import Dict, Union

import torch
import torch.nn as nn
from jaxtyping import Float

from transformer_lens.config.hooked_transformer_config import HookedTransformerConfig
from transformer_lens.hook_points import HookPoint


# LayerNormPre
# I fold the LayerNorm weights and biases into later weights and biases.
# This is just the 'center and normalise' part of LayerNorm
# Centering is equivalent to just deleting one direction of residual space,
# and is equivalent to centering the weight matrices of everything writing to the residual stream
# Normalising is a funkier non-linear operation, that projects the residual stream onto the unit hypersphere
class LayerNormPre(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        """LayerNormPre - the 'center and normalise' part of LayerNorm. Length is
        normally d_model, but is d_mlp for softmax. Not needed as a parameter. This
        should only be used in inference mode after folding in LayerNorm weights"""
        super().__init__()
        self.cfg = HookedTransformerConfig.unwrap(cfg)
        self.eps = self.cfg.eps

        # Adds a hook point for the normalisation scale factor
        self.hook_scale = HookPoint()  # [batch, pos]
        # Hook Normalized captures LN output - here it's a vector with std 1 and mean 0
        self.hook_normalized = HookPoint()  # [batch, pos, length]

    def _hooked_scale(self, x_fp32: torch.Tensor) -> torch.Tensor:
        """Fire hook_scale in cfg.dtype; return the fp32 scale used for the division.

        Fixes #1108: hooks must observe the dtype the model consumes. If no hook edits the
        scale (identity check) the fp32 value keeps feeding the division so the unhooked
        forward is bit-identical to the pre-fix numerics; a returned tensor is honored.
        """
        scale_fp32 = (x_fp32.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        scale_model = scale_fp32.to(self.cfg.dtype)
        hooked = self.hook_scale(scale_model)
        if hooked is scale_model:
            return scale_fp32
        return hooked.to(scale_fp32.dtype)

    def forward(
        self,
        x: Union[
            Float[torch.Tensor, "batch pos d_model"],
            Float[torch.Tensor, "batch pos head_index d_model"],
        ],
    ) -> Union[
        Float[torch.Tensor, "batch pos d_model"],
        Float[torch.Tensor, "batch pos head_index d_model"],
    ]:
        if self.cfg.dtype not in [torch.float32, torch.float64]:
            x = x.to(torch.float32)

        x = x - x.mean(-1, keepdim=True)  # [batch, pos, length]
        scale = self._hooked_scale(x)  # hook_scale fires in cfg.dtype (#1108)
        # Cast BEFORE the hook so hook_normalized observes what the model consumes (#1108).
        return self.hook_normalized((x / scale).to(self.cfg.dtype))
