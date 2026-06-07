# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the upstream DINOv2 repository.

from .hub import (
    dinov2_vitb14,
    dinov2_vitb14_reg,
    dinov2_vitg14,
    dinov2_vitg14_reg,
    dinov2_vitl14,
    dinov2_vitl14_reg,
    dinov2_vits14,
    dinov2_vits14_reg,
)
from .vision_transformer import (
    DinoVisionTransformer,
    vit_base,
    vit_giant2,
    vit_large,
    vit_small,
)

__all__ = [
    "DinoVisionTransformer",
    "dinov2_vitb14",
    "dinov2_vitb14_reg",
    "dinov2_vitg14",
    "dinov2_vitg14_reg",
    "dinov2_vitl14",
    "dinov2_vitl14_reg",
    "dinov2_vits14",
    "dinov2_vits14_reg",
    "vit_base",
    "vit_giant2",
    "vit_large",
    "vit_small",
]
