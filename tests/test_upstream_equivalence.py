import os
import gc
import functools
from io import BytesIO
from urllib.request import urlopen

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

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


pytestmark = pytest.mark.upstream

assert_equal = functools.partial(torch.testing.assert_close, rtol=0, atol=0)
FEATURE_KEYS = (
    "x_norm_clstoken",
    "x_norm_regtokens",
    "x_norm_patchtokens",
    "x_prenorm",
)
REAL_IMAGE_URL = "https://github.com/karimknaebel/storage/releases/download/example-images/pipeline-cat-chonk.jpeg"
UPSTREAM_REPO = "facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8"
COMMON_BACKBONES = [
    ("dinov2_vits14", dinov2_vits14),
    ("dinov2_vitb14", dinov2_vitb14),
    ("dinov2_vitl14", dinov2_vitl14),
    ("dinov2_vits14_reg", dinov2_vits14_reg),
    ("dinov2_vitb14_reg", dinov2_vitb14_reg),
    ("dinov2_vitl14_reg", dinov2_vitl14_reg),
]
GIANT_BACKBONES = [
    pytest.param(
        "dinov2_vitg14",
        dinov2_vitg14,
        marks=pytest.mark.skipif(
            not os.environ.get("DINOV2_CORE_TEST_GIANT"),
            reason="set DINOV2_CORE_TEST_GIANT=1",
        ),
    ),
    pytest.param(
        "dinov2_vitg14_reg",
        dinov2_vitg14_reg,
        marks=pytest.mark.skipif(
            not os.environ.get("DINOV2_CORE_TEST_GIANT"),
            reason="set DINOV2_CORE_TEST_GIANT=1",
        ),
    ),
]
BACKBONES = COMMON_BACKBONES + GIANT_BACKBONES


def load_upstream(name, pretrained):
    os.environ["XFORMERS_DISABLED"] = "1"
    return torch.hub.load(
        UPSTREAM_REPO,
        name,
        pretrained=pretrained,
        trust_repo=True,
        skip_validation=True,
    ).eval()


def detach_features(features):
    return {key: features[key].detach().cpu() for key in FEATURE_KEYS}


def load_real_image(device):
    with urlopen(REAL_IMAGE_URL) as response:
        image = Image.open(BytesIO(response.read())).convert("RGB")
    x = (
        torch.from_numpy(np.array(image, copy=True))
        .permute(2, 0, 1)
        .float()
        .div(255)
        .unsqueeze(0)
    )
    _, _, h, w = x.shape
    assert (h, w) == (686, 960)
    x = F.pad(x, (0, -w % 14, 0, -h % 14))
    assert x.shape[-2:] != (518, 518)
    assert x.shape[-2] % 14 == 0
    assert x.shape[-1] % 14 == 0
    return x.to(device)


@pytest.mark.skipif(
    not os.environ.get("DINOV2_CORE_TEST_UPSTREAM"),
    reason="set DINOV2_CORE_TEST_UPSTREAM=1",
)
@pytest.mark.parametrize(("name", "ours_fn"), BACKBONES)
def test_random_initialized_outputs_match_upstream(name, ours_fn):
    torch.manual_seed(1)
    x = torch.randn(1, 3, 28, 28)

    torch.manual_seed(0)
    ours = ours_fn(pretrained=False).eval()
    with torch.no_grad():
        ours_forward = detach_features(ours(x))
        ours_features = detach_features(ours.forward_features(x))
    del ours
    gc.collect()

    torch.manual_seed(0)
    upstream = load_upstream(name, pretrained=False)
    with torch.no_grad():
        upstream_forward = detach_features(upstream(x, is_training=True))
        upstream_features = detach_features(upstream.forward_features(x))

    for key in FEATURE_KEYS:
        assert_equal(ours_forward[key], upstream_forward[key])
        assert_equal(ours_features[key], upstream_features[key])


@pytest.mark.skipif(
    not os.environ.get("DINOV2_CORE_TEST_UPSTREAM"),
    reason="set DINOV2_CORE_TEST_UPSTREAM=1",
)
@pytest.mark.parametrize(("name", "ours_fn"), COMMON_BACKBONES)
def test_initialized_weights_match_upstream(name, ours_fn):
    torch.manual_seed(0)
    ours = ours_fn(pretrained=False).state_dict()
    torch.manual_seed(0)
    upstream = load_upstream(name, pretrained=False).state_dict()

    assert set(ours) == set(upstream) - {"mask_token"}
    for key, value in ours.items():
        assert_equal(value, upstream[key])


@pytest.mark.pretrained
@pytest.mark.skipif(
    not os.environ.get("DINOV2_CORE_TEST_PRETRAINED"),
    reason="set DINOV2_CORE_TEST_PRETRAINED=1",
)
def test_pretrained_vits14_outputs_match_upstream():
    ours = dinov2_vits14(pretrained=True).eval()
    upstream = load_upstream("dinov2_vits14", pretrained=True)
    torch.manual_seed(0)
    x = torch.randn(1, 3, 28, 28)

    ours_forward = ours(x)
    upstream_forward = upstream(x, is_training=True)
    for key in FEATURE_KEYS:
        assert_equal(ours_forward[key], upstream_forward[key])


@pytest.mark.cuda
@pytest.mark.skipif(
    not os.environ.get("DINOV2_CORE_TEST_UPSTREAM"),
    reason="set DINOV2_CORE_TEST_UPSTREAM=1",
)
@pytest.mark.skipif(
    not os.environ.get("DINOV2_CORE_TEST_CUDA"), reason="set DINOV2_CORE_TEST_CUDA=1"
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize(
    ("name", "ours_fn"),
    [
        ("dinov2_vits14", dinov2_vits14),
        ("dinov2_vits14_reg", dinov2_vits14_reg),
    ],
)
def test_cuda_real_image_outputs_match_upstream(name, ours_fn):
    device = torch.device("cuda")
    x = load_real_image(device)

    torch.manual_seed(0)
    ours = ours_fn(pretrained=False).eval().to(device)
    with torch.no_grad():
        ours_forward = detach_features(ours(x))
    del ours
    gc.collect()
    torch.cuda.empty_cache()

    torch.manual_seed(0)
    upstream = load_upstream(name, pretrained=False).to(device)
    with torch.no_grad():
        upstream_forward = detach_features(upstream(x, is_training=True))

    for key in FEATURE_KEYS:
        assert_equal(ours_forward[key], upstream_forward[key])
