import os

import pytest
import torch

from dinov2_core.hub import dinov2_vits14, dinov2_vits14_reg


pytestmark = pytest.mark.upstream


def load_upstream(name, pretrained):
    os.environ["XFORMERS_DISABLED"] = "1"
    return torch.hub.load(
        "facebookresearch/dinov2",
        name,
        pretrained=pretrained,
        trust_repo=True,
        skip_validation=True,
    ).eval()


@pytest.mark.skipif(not os.environ.get("DINOV2_CORE_TEST_UPSTREAM"), reason="set DINOV2_CORE_TEST_UPSTREAM=1")
@pytest.mark.parametrize(
    ("name", "ours_fn"),
    [
        ("dinov2_vits14", dinov2_vits14),
        ("dinov2_vits14_reg", dinov2_vits14_reg),
    ],
)
def test_random_initialized_outputs_match_upstream(name, ours_fn):
    torch.manual_seed(0)
    ours = ours_fn(pretrained=False).eval()
    torch.manual_seed(0)
    upstream = load_upstream(name, pretrained=False)
    x = torch.randn(1, 3, 28, 28)

    ours_forward = ours(x)
    upstream_forward = upstream(x, is_training=True)
    for key in ("x_norm_clstoken", "x_norm_regtokens", "x_norm_patchtokens", "x_prenorm"):
        torch.testing.assert_close(ours_forward[key], upstream_forward[key])
    ours_features = ours.forward_features(x)
    upstream_features = upstream.forward_features(x)
    for key in ("x_norm_clstoken", "x_norm_regtokens", "x_norm_patchtokens", "x_prenorm"):
        torch.testing.assert_close(ours_features[key], upstream_features[key])


@pytest.mark.skipif(not os.environ.get("DINOV2_CORE_TEST_UPSTREAM"), reason="set DINOV2_CORE_TEST_UPSTREAM=1")
@pytest.mark.parametrize(
    ("name", "ours_fn"),
    [
        ("dinov2_vits14", dinov2_vits14),
        ("dinov2_vits14_reg", dinov2_vits14_reg),
    ],
)
def test_initialized_weights_match_upstream(name, ours_fn):
    torch.manual_seed(0)
    ours = ours_fn(pretrained=False).state_dict()
    torch.manual_seed(0)
    upstream = load_upstream(name, pretrained=False).state_dict()

    assert set(ours) == set(upstream) - {"mask_token"}
    for key, value in ours.items():
        torch.testing.assert_close(value, upstream[key])


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
    for key in ("x_norm_clstoken", "x_norm_regtokens", "x_norm_patchtokens", "x_prenorm"):
        torch.testing.assert_close(ours_forward[key], upstream_forward[key], rtol=1e-5, atol=2e-5)
