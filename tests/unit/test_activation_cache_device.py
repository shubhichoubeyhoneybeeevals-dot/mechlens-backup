"""Device regressions using real tensors.

In a full checkout, imports TransformerLens normally. For the supplied single-file
snapshot, set ACTIVATION_CACHE_SOURCE to its path. That mode substitutes only the
unavailable Slice/get_act_name/warn_if_mps utility boundary; all cache methods and
PyTorch allocation, movement and arithmetic execute unchanged.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import warnings
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


def _load_cache():
    source = os.environ.get("ACTIVATION_CACHE_SOURCE")
    if not source:
        from transformer_lens.ActivationCache import ActivationCache

        return ActivationCache

    class Slice:
        def __init__(self, value=None):
            self.value = value
            self.mode = "empty" if value is None else "int" if isinstance(value, int) else "slice"

        def apply(self, tensor, dim=0):
            if self.value is None:
                return tensor
            indices = [slice(None)] * tensor.ndim
            indices[dim] = self.value
            return tensor[tuple(indices)]

    def get_act_name(name, layer=None, layer_type=None):
        if layer is None:
            return "hook_" + name
        if name == "result":
            return f"blocks.{layer}.attn.hook_result"
        return f"blocks.{layer}.hook_{name}"

    package = ModuleType("transformer_lens")
    package.__path__ = []
    utilities = ModuleType("transformer_lens.utilities")
    utilities.Slice = Slice
    utilities.SliceInput = object
    utilities.get_act_name = get_act_name
    utilities.warn_if_mps = lambda device: None
    package.utilities = utilities
    replacements = {"transformer_lens": package, "transformer_lens.utilities": utilities}
    previous = {key: sys.modules.get(key) for key in replacements}
    try:
        sys.modules.update(replacements)
        spec = importlib.util.spec_from_file_location("cache_under_test", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.ActivationCache
    finally:
        for key, value in previous.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


ActivationCache = _load_cache()


def model(config_device="cpu"):
    return SimpleNamespace(
        cfg=SimpleNamespace(
            device=config_device,
            n_layers=1,
            n_heads=2,
            d_mlp=5,
            d_model=4,
            normalization_type="LN",
            attn_only=True,
        )
    )


def tensors(device="cpu", dtype=torch.float64):
    return {
        "hook_embed": torch.arange(24, device=device, dtype=dtype).reshape(2, 3, 4),
        "blocks.0.hook_resid_pre": torch.ones(2, 3, 4, device=device, dtype=dtype),
        "blocks.0.attn.hook_result": torch.ones(2, 3, 2, 4, device=device, dtype=dtype),
        "blocks.0.ln1.hook_scale": torch.ones(2, 3, 1, device=device, dtype=dtype),
    }


def invoke(api, device_kwargs):
    if api == "constructor":
        return ActivationCache(tensors(), model(), **device_kwargs).cache_dict["hook_embed"]
    cache = ActivationCache(tensors(), model(), device="cpu")
    if api == "heads":
        return cache.stack_head_results(layer=1, **device_kwargs)
    return cache.stack_activation("resid_pre", layer=1, **device_kwargs)


@pytest.mark.parametrize("api", ["constructor", "heads", "activations"])
@pytest.mark.parametrize("kwargs", [{}, {"device": None}], ids=["omitted", "none"])
def test_inference_warns_at_user_call(api, kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = invoke(api, kwargs)
    assert result.device == torch.device("cpu")
    assert len(caught) == 1
    assert caught[0].category is FutureWarning
    assert "device inference" in str(caught[0].message)
    assert Path(caught[0].filename).resolve() == Path(__file__).resolve()


@pytest.mark.parametrize("api", ["constructor", "heads", "activations"])
@pytest.mark.parametrize(
    "device", ["cpu", torch.device("cpu"), "meta"], ids=["string", "device-object", "meta"]
)
def test_explicit_device_moves_output_without_warning(api, device):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = invoke(api, {"device": device})
    assert not caught
    assert result.device == torch.device(device)
    assert result.dtype == torch.float64


@pytest.mark.parametrize("path", ["heads", "neurons", "projected-ln"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("remove_batch", [False, True], ids=["batched", "unbatched"])
def test_layer_zero_allocates_on_cache_not_model_device(path, dtype, remove_batch):
    # Real CPU tensors versus a meta model device catches misplaced allocations
    # on a CPU-only runner. This is not a simulation of CUDA execution.
    cache = ActivationCache(tensors(dtype=dtype), model("meta"), device="cpu")
    if remove_batch:
        cache.cache_dict = {key: value[0] for key, value in cache.cache_dict.items()}
        cache.has_batch_dim = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if path == "heads":
            result, labels = cache.stack_head_results(
                layer=0, apply_ln=True, return_labels=True, device="cpu"
            )
            last_dim = 4
        else:
            kwargs = {}
            if path == "projected-ln":
                kwargs = {"apply_ln": True, "project_output_onto": torch.ones(4, 2, dtype=dtype)}
            result, labels = cache.stack_neuron_results(layer=0, return_labels=True, **kwargs)
            last_dim = 2 if path == "projected-ln" else 4
    assert not caught
    assert result.device == torch.device("cpu")
    assert result.dtype == dtype
    assert result.shape == ((0, 3, last_dim) if remove_batch else (0, 2, 3, last_dim))
    assert result.numel() == 0
    assert labels == []


@pytest.mark.parametrize("apply_ln", [False, True])
@pytest.mark.parametrize("one_dimensional", [False, True])
def test_empty_projection_shape_and_dtype_promotion(apply_ln, one_dimensional):
    cache = ActivationCache(tensors(dtype=torch.float32), model("meta"), device="cpu")
    projection = torch.ones((4,) if one_dimensional else (4, 2), dtype=torch.float64)
    result = cache.stack_neuron_results(layer=0, apply_ln=apply_ln, project_output_onto=projection)
    assert result.device.type == "cpu"
    assert result.dtype == torch.float64
    assert result.shape == ((0, 2, 3) if one_dimensional else (0, 2, 3, 2))


def test_slice_forwards_device_after_move_and_for_empty_cache():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cache = ActivationCache(tensors(), model(), device="cpu").to("meta")
        sliced = cache.apply_slice_to_batch_dim(None)
        empty = ActivationCache({}, model(), device="meta").apply_slice_to_batch_dim(None)
    assert not caught
    assert sliced.device.type == "meta"
    assert empty.device.type == "meta"


def test_full_decomposition_forwards_device_without_warning():
    m = model()
    m.accumulated_bias = lambda *args, **kwargs: torch.zeros(4, dtype=torch.float64)
    cache = ActivationCache(tensors(), m, device="cpu")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result, labels = cache.get_full_resid_decomposition(layer=0, return_labels=True)
    assert not caught
    assert result.device.type == "cpu"
    assert result.shape == (2, 2, 3, 4)
    assert labels == ["embed", "bias"]


@pytest.mark.parametrize("path", ["heads", "neurons", "projected-ln"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_layer_zero_cuda_cache_cpu_model(path):
    cache = ActivationCache(tensors(), model("cpu"), device="cuda")
    if path == "heads":
        result = cache.stack_head_results(layer=0, apply_ln=True, device="cuda")
    else:
        kwargs = {}
        if path == "projected-ln":
            kwargs = {
                "apply_ln": True,
                "project_output_onto": torch.ones(4, 2, dtype=torch.float64, device="cuda"),
            }
        result = cache.stack_neuron_results(layer=0, **kwargs)
    assert result.device.type == "cuda"
    assert result.dtype == torch.float64
    assert result.shape[0] == 0
