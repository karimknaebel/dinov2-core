# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the upstream DINOv2 repository.

from functools import partial
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import torch
from torch import nn

from .layers import Mlp, SwiGLUFFN
from .vision_transformer import (
    DinoVisionTransformer,
    vit_base,
    vit_giant2,
    vit_large,
    vit_small,
)

_DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
_GIANT_SWIGLU_FFN = partial(SwiGLUFFN, adjust_hidden_features=True)


def _load_state_dict(weights: str | Path, check_hash: bool) -> dict[str, torch.Tensor]:
    location = str(weights)
    if urlparse(location).scheme in {"http", "https"}:
        return torch.hub.load_state_dict_from_url(
            location,
            map_location="cpu",
            check_hash=check_hash,
            weights_only=True,
        )
    return torch.load(
        Path(location).expanduser(), map_location="cpu", weights_only=True
    )


def _make_model(
    model_fn: Callable[..., DinoVisionTransformer],
    model_base_name: str,
    model_full_name: str,
    *,
    pretrained: bool,
    weights: str | Path | None,
    check_hash: bool,
    ffn_layer: Callable[..., nn.Module] = Mlp,
    num_register_tokens: int = 0,
    interpolate_antialias: bool = False,
    interpolate_offset: float = 0.1,
    **kwargs: Any,
) -> DinoVisionTransformer:
    model = model_fn(
        img_size=518,
        patch_size=14,
        init_values=1.0,
        ffn_layer=ffn_layer,
        num_register_tokens=num_register_tokens,
        interpolate_antialias=interpolate_antialias,
        interpolate_offset=interpolate_offset,
        **kwargs,
    )
    if pretrained:
        state_dict = _load_state_dict(
            weights
            or f"{_DINOV2_BASE_URL}/{model_base_name}/{model_full_name}_pretrain.pth",
            check_hash=check_hash,
        )
        state_dict.pop("mask_token", None)
        model.load_state_dict(state_dict, strict=True)
    return model


def dinov2_vits14(
    *,
    pretrained: bool = True,
    weights: str | Path | None = None,
    check_hash: bool = False,
    **kwargs: Any,
):
    return _make_model(
        vit_small,
        "dinov2_vits14",
        "dinov2_vits14",
        pretrained=pretrained,
        weights=weights,
        check_hash=check_hash,
        **kwargs,
    )


def dinov2_vitb14(
    *,
    pretrained: bool = True,
    weights: str | Path | None = None,
    check_hash: bool = False,
    **kwargs: Any,
):
    return _make_model(
        vit_base,
        "dinov2_vitb14",
        "dinov2_vitb14",
        pretrained=pretrained,
        weights=weights,
        check_hash=check_hash,
        **kwargs,
    )


def dinov2_vitl14(
    *,
    pretrained: bool = True,
    weights: str | Path | None = None,
    check_hash: bool = False,
    **kwargs: Any,
):
    return _make_model(
        vit_large,
        "dinov2_vitl14",
        "dinov2_vitl14",
        pretrained=pretrained,
        weights=weights,
        check_hash=check_hash,
        **kwargs,
    )


def dinov2_vitg14(
    *,
    pretrained: bool = True,
    weights: str | Path | None = None,
    check_hash: bool = False,
    **kwargs: Any,
):
    return _make_model(
        vit_giant2,
        "dinov2_vitg14",
        "dinov2_vitg14",
        pretrained=pretrained,
        weights=weights,
        check_hash=check_hash,
        ffn_layer=_GIANT_SWIGLU_FFN,
        **kwargs,
    )


def dinov2_vits14_reg(
    *,
    pretrained: bool = True,
    weights: str | Path | None = None,
    check_hash: bool = False,
    **kwargs: Any,
):
    return _make_model(
        vit_small,
        "dinov2_vits14",
        "dinov2_vits14_reg4",
        pretrained=pretrained,
        weights=weights,
        check_hash=check_hash,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )


def dinov2_vitb14_reg(
    *,
    pretrained: bool = True,
    weights: str | Path | None = None,
    check_hash: bool = False,
    **kwargs: Any,
):
    return _make_model(
        vit_base,
        "dinov2_vitb14",
        "dinov2_vitb14_reg4",
        pretrained=pretrained,
        weights=weights,
        check_hash=check_hash,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )


def dinov2_vitl14_reg(
    *,
    pretrained: bool = True,
    weights: str | Path | None = None,
    check_hash: bool = False,
    **kwargs: Any,
):
    return _make_model(
        vit_large,
        "dinov2_vitl14",
        "dinov2_vitl14_reg4",
        pretrained=pretrained,
        weights=weights,
        check_hash=check_hash,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )


def dinov2_vitg14_reg(
    *,
    pretrained: bool = True,
    weights: str | Path | None = None,
    check_hash: bool = False,
    **kwargs: Any,
):
    return _make_model(
        vit_giant2,
        "dinov2_vitg14",
        "dinov2_vitg14_reg4",
        pretrained=pretrained,
        weights=weights,
        check_hash=check_hash,
        ffn_layer=_GIANT_SWIGLU_FFN,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )
