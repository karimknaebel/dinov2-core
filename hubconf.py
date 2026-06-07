# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the upstream DINOv2 repository.

from pathlib import Path
import sys

_src = Path(__file__).resolve().parent / "src"
if _src.exists():
    sys.path.insert(0, str(_src))

from dinov2_core.hub import (
    dinov2_vitb14,
    dinov2_vitb14_reg,
    dinov2_vitg14,
    dinov2_vitg14_reg,
    dinov2_vitl14,
    dinov2_vitl14_reg,
    dinov2_vits14,
    dinov2_vits14_reg,
)

dependencies = ["torch"]
