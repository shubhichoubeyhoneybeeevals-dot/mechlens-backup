# TransformerLens 2.17.0

Patch release focused on hook correctness. No new adapters, no API changes.

## Fixes

### `HookedTransformer` norm hooks now fire in the model's dtype (#1108)

Reported by @nkechi-of: on a bfloat16 `HookedTransformer`, SAE feature activations computed from a
`run_with_cache` cache at `blocks.8.mlp.hook_post` differed slightly from a live run, while float32
matched exactly.

**Root cause.** `LayerNorm`, `LayerNormPre`, `RMSNorm` and `RMSNormPre` upcast to float32
internally for numerical stability. `hook_scale` and `hook_normalized` were fired on those float32
intermediates, and the cast back to `cfg.dtype` happened *after* the hook. Every other hook in the
model fires in `cfg.dtype`, so a bf16 cache contained float32 norm activations that the model never
consumed downstream: attention and the MLP had seen the bf16-rounded values. Anything that re-derives
later activations from the cached norm output, such as re-running `blocks.N.mlp` on
`cache["blocks.N.ln2.hook_normalized"]`, skipped that rounding step and drifted from the live
forward. In float32 there is no rounding to skip, which is why the mismatch was invisible there.

**Fix.** The cast now happens before the hook, so hooks observe exactly what the model consumes:

- `hook_normalized` receives `(x / scale).to(cfg.dtype)`, and the model uses its return value as-is.
- `hook_scale` receives `scale.to(cfg.dtype)`. If no hook edits it, the float32 scale still feeds
  the division internally, so the **unhooked forward pass is bit-identical to 2.16.0**. If a hook
  returns a new tensor, that edit is honored.
- float32 and float64 models are unaffected: the casts are no-ops.

**What you may notice.** On reduced-precision models, `cache["blocks.N.ln1.hook_scale"]`,
`ln2.hook_scale`, `ln_final.hook_scale` and the matching `hook_normalized` entries are now stored in
`cfg.dtype` (bf16/fp16) instead of float32. Logits, loss and every other cached activation are
unchanged. If you trained an SAE or probe on float32 `hook_normalized` activations pulled from a
bf16 model, its inputs will now be bf16-rounded; in our testing that is rounding-scale noise, but
it is a change in what the cache holds, so we are calling it out.

Regression coverage: `tests/unit/components/test_norm_hook_dtype.py` pins the unhooked numerics
for all four norms in bf16 and fp32, and reproduces the report end to end on GPT-2 small
(cached vs. live `blocks.8.mlp.hook_post`, including the recompute-from-cache path).

### `TransformerBridge`: `hook_scale` / `hook_normalized` edits and backward hooks were silently dropped on the native-autograd path

Ported from upstream TransformerLens PR #1527 (upstream issue #1526). On `NormalizationBridge`
with `use_native_layernorm_autograd=True`, which covers ten LayerNorm adapters (OPT, Phi, StableLM,
OLMo, GPT-OSS, GPT-NeoX, BART, BERT, HuBERT, XGLM) and, through `RMSNormalizationBridge`'s default,
every RMSNorm model, the two norm hooks were fired on detached side computations and their return
values discarded before HF's own forward ran on the raw input. Forward-hook edits (ablation,
patching, steering) had no effect and produced no warning; backward hooks never fired.

Now, observation-only hooks such as `run_with_cache` are unchanged and keep bit-identical native HF
output. A forward hook that returns a new tensor causes the output to be reconstructed from the
hooked values, and attaching a backward hook routes the layer through the grad-connected
from-scratch path. Both fallbacks emit a `UserWarning` because their numerics differ from native HF
at rounding scale.

Regression coverage: `tests/unit/model_bridge/generalized_components/test_normalization_hook_semantics.py`.

## Upgrade notes

- No API changes. `pip install -U transformer-lens`.
- If you compare caches across versions, expect the norm-hook dtype change described above on
  bf16/fp16 `HookedTransformer` models only.

## Thanks

Thanks to @nkechi-of for the clear report and reproducer on #1108, and to the upstream
TransformerLens contributors whose fix for #1526 is included here.
