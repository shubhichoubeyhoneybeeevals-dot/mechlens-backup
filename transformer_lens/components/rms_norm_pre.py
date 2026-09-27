"""Hooked Transformer RMS Norm Pre Component.

This module contains all the component :class:`RMSNormPre`.
"""

from typing import Dict, Union

import torch
import torch.nn as nn
from jaxtyping import Float

from transformer_lens.config.hooked_transformer_config import HookedTransformerConfig
from transformer_lens.hook_points import HookPoint


class RMSNormPre(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        """RMSNormPre - LayerNormPre without the centering and bias (RMS = Root Mean Square)"""
        super().__init__()
        self.cfg = HookedTransformerConfig.unwrap(cfg)
        self.eps = self.cfg.eps

        # Adds a hook point for the normalisation scale factor
        self.hook_scale = HookPoint()  # [batch, pos]
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
        self, x: Float[torch.Tensor, "batch pos length"]
    ) -> Float[torch.Tensor, "batch pos length"]:
        if self.cfg.dtype not in [torch.float32, torch.float64]:
            x = x.to(torch.float32)

        scale = self._hooked_scale(x)  # hook_scale fires in cfg.dtype (#1108)
        # Cast BEFORE the hook so hook_normalized observes what the model consumes (#1108).
        return self.hook_normalized((x / scale).to(self.cfg.dtype))  # [batch, pos, length]
