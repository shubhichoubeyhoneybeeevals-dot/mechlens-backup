"""Regression tests for #1108: norm hooks must fire in ``cfg.dtype``.

Under bfloat16, ``LayerNorm`` / ``LayerNormPre`` / ``RMSNorm`` / ``RMSNormPre`` upcast to
float32 internally. Previously ``hook_scale`` and ``hook_normalized`` fired on the float32
intermediates and the cast to ``cfg.dtype`` happened *after* the hook, so ``run_with_cache``
stored float32 values the model never consumed. Re-deriving ``blocks.N.mlp.hook_post`` from the
cached ``ln2.hook_normalized`` (the SAE-dashboard pipeline in the report) then diverged from a
live run. Invisible in float32 because there is no rounding step to skip.
"""

import pytest
import torch

from transformer_lens import HookedTransformer
from transformer_lens.components.layer_norm import LayerNorm
from transformer_lens.components.layer_norm_pre import LayerNormPre
from transformer_lens.components.rms_norm import RMSNorm
from transformer_lens.components.rms_norm_pre import RMSNormPre
from transformer_lens.config.hooked_transformer_config import HookedTransformerConfig

MODEL = "gpt2"  # 12 layers, so blocks.8 exists; CI-cached
LAYER = 8
HOOK_POST = f"blocks.{LAYER}.mlp.hook_post"
HOOK_LN2_NORM = f"blocks.{LAYER}.ln2.hook_normalized"
HOOK_LN2_SCALE = f"blocks.{LAYER}.ln2.hook_scale"
PROMPT = "The quick brown fox jumps over the lazy dog because the dog was asleep."


# ---------------------------------------------------------------------------------------------
# Component-level: no model download needed
# ---------------------------------------------------------------------------------------------


def _cfg(normalization_type: str, dtype: torch.dtype) -> HookedTransformerConfig:
    return HookedTransformerConfig(
        n_layers=1,
        d_model=16,
        n_ctx=8,
        d_head=4,
        n_heads=4,
        d_vocab=32,
        act_fn="gelu",
        normalization_type=normalization_type,
        dtype=dtype,
    )


def _reference_forward(module, x):
    """The pre-fix computation: fp32 norm, cast after. Used to pin the unhooked numerics."""
    dtype = module.cfg.dtype
    if dtype not in (torch.float32, torch.float64):
        x = x.to(torch.float32)
    if isinstance(module, (LayerNorm, LayerNormPre)):
        x = x - x.mean(-1, keepdim=True)
    scale = (x.pow(2).mean(-1, keepdim=True) + module.eps).sqrt()
    x = (x / scale).to(dtype)
    if isinstance(module, LayerNorm):
        return x * module.w + module.b
    if isinstance(module, RMSNorm):
        return x * module.w
    return x


@pytest.mark.parametrize(
    "cls,norm_type",
    [(LayerNorm, "LN"), (LayerNormPre, "LNPre"), (RMSNorm, "RMS"), (RMSNormPre, "RMSPre")],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_norm_hooks_observe_model_dtype_and_unhooked_forward_is_unchanged(cls, norm_type, dtype):
    torch.manual_seed(0)
    module = cls(_cfg(norm_type, dtype))
    if hasattr(module, "w"):
        with torch.no_grad():
            module.w.copy_(torch.rand_like(module.w) + 0.5)
    x = torch.randn(2, 5, 16).to(dtype)

    seen = {}

    def grab(name):
        def hook(act, hook):
            seen[name] = act.detach().clone()

        return hook

    module.hook_scale.add_hook(grab("scale"))
    module.hook_normalized.add_hook(grab("normalized"))
    try:
        out = module(x)
    finally:
        module.hook_scale.remove_hooks()
        module.hook_normalized.remove_hooks()

    # Hooks fire in cfg.dtype, like every other hook in the model.
    assert seen["scale"].dtype == dtype
    assert seen["normalized"].dtype == dtype
    # Unhooked numerics are pinned: the fp32 scale still feeds the division.
    assert torch.equal(out, module(x))
    assert torch.equal(out, _reference_forward(module, x))
    # hook_normalized observed exactly what the module consumed downstream.
    if cls is LayerNorm:
        assert torch.equal(out, seen["normalized"] * module.w + module.b)
    elif cls is RMSNorm:
        assert torch.equal(out, seen["normalized"] * module.w)
    else:
        assert torch.equal(out, seen["normalized"])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_hook_scale_edit_is_honored(dtype):
    """Returning a new tensor from hook_scale must still change the output (identity path)."""
    torch.manual_seed(0)
    module = LayerNormPre(_cfg("LNPre", dtype))
    x = torch.randn(2, 5, 16).to(dtype)
    base = module(x)

    module.hook_scale.add_hook(lambda scale, hook: scale * 2)
    try:
        edited = module(x)
    finally:
        module.hook_scale.remove_hooks()

    assert not torch.equal(edited, base)
    assert torch.allclose(edited.float(), base.float() / 2, atol=2e-2, rtol=2e-2)


# ---------------------------------------------------------------------------------------------
# Model-level: the #1108 scenario on GPT-2 small in bf16 (CPU)
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bf16_model():
    model = HookedTransformer.from_pretrained(MODEL, dtype=torch.bfloat16, device="cpu")
    model.eval()
    return model


@pytest.fixture(scope="module")
def tokens(bf16_model):
    return bf16_model.to_tokens(PROMPT)


def _live_capture(model, tokens, hook_name):
    captured = {}

    def grab(act, hook):
        captured["x"] = act.detach().clone()

    model.run_with_hooks(tokens, fwd_hooks=[(hook_name, grab)], return_type=None)
    return captured["x"]


@torch.no_grad()
def test_cached_vs_live_hook_post_bf16(bf16_model, tokens):
    """run_with_cache and a live hook at blocks.8.mlp.hook_post agree exactly in bf16."""
    _, cache = bf16_model.run_with_cache(tokens, names_filter=HOOK_POST, return_type=None)
    cached = cache[HOOK_POST]
    live = _live_capture(bf16_model, tokens, HOOK_POST)
    assert cached.dtype == torch.bfloat16
    assert live.dtype == torch.bfloat16
    max_diff = (cached.float() - live.float()).abs().max().item()
    assert torch.equal(cached, live), f"max |diff| = {max_diff}"


@torch.no_grad()
def test_model_norm_hooks_cached_in_model_dtype(bf16_model, tokens):
    _, cache = bf16_model.run_with_cache(
        tokens, names_filter=[HOOK_LN2_NORM, HOOK_LN2_SCALE], return_type=None
    )
    assert cache[HOOK_LN2_NORM].dtype == torch.bfloat16
    assert cache[HOOK_LN2_SCALE].dtype == torch.bfloat16


@torch.no_grad()
def test_recompute_hook_post_from_cached_ln2_matches_live(bf16_model, tokens):
    """Re-running blocks.8.mlp on the cached ln2.hook_normalized reproduces the live hook_post."""
    _, cache = bf16_model.run_with_cache(
        tokens, names_filter=[HOOK_POST, HOOK_LN2_NORM], return_type=None
    )
    captured = {}

    def grab(act, hook):
        captured["x"] = act.detach().clone()

    with bf16_model.hooks(fwd_hooks=[(HOOK_POST, grab)]):
        bf16_model.blocks[LAYER].mlp(cache[HOOK_LN2_NORM])

    assert torch.equal(captured["x"], cache[HOOK_POST])
