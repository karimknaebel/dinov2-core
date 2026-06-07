import torch
import pytest

from dinov2_core import DinoVisionTransformer
from dinov2_core.hub import dinov2_vits14
from dinov2_core.layers import SwiGLUFFN


def make_model(num_register_tokens=0):
    torch.manual_seed(0)
    model = DinoVisionTransformer(
        img_size=28,
        patch_size=14,
        embed_dim=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2,
        init_values=1.0,
        num_register_tokens=num_register_tokens,
        interpolate_antialias=bool(num_register_tokens),
        interpolate_offset=0.0 if num_register_tokens else 0.1,
    )
    model.eval()
    return model


def test_forward_features_without_registers():
    model = make_model()
    x = torch.randn(2, 3, 28, 28)

    assert model(x)["x_norm_clstoken"].shape == (2, 32)
    features = model.forward_features(x)
    assert features["x_norm_clstoken"].shape == (2, 32)
    assert features["x_norm_regtokens"].shape == (2, 0, 32)
    assert features["x_norm_patchtokens"].shape == (2, 4, 32)
    assert features["x_prenorm"].shape == (2, 5, 32)


def test_forward_features_with_registers():
    model = make_model(num_register_tokens=4)
    features = model.forward_features(torch.randn(2, 3, 28, 28))

    assert features["x_norm_clstoken"].shape == (2, 32)
    assert features["x_norm_regtokens"].shape == (2, 4, 32)
    assert features["x_norm_patchtokens"].shape == (2, 4, 32)
    assert features["x_prenorm"].shape == (2, 9, 32)


def test_get_intermediate_layers_shapes():
    model = make_model(num_register_tokens=4)
    x = torch.randn(2, 3, 28, 28)

    outputs = model.get_intermediate_layers(x, n=2)
    assert len(outputs) == 2
    assert outputs[0].shape == (2, 4, 32)

    outputs = model.get_intermediate_layers(
        x, n=[0], reshape=True, return_class_token=True
    )
    assert outputs[0][0].shape == (2, 32, 2, 2)
    assert outputs[0][1].shape == (2, 32)


def test_get_intermediate_layers_sequence_n_compiles_dynamic():
    model = make_model(num_register_tokens=4)
    compiled = torch.compile(
        model.get_intermediate_layers, dynamic=True, fullgraph=True
    )

    with torch.no_grad():
        for h, w in ((28, 28), (42, 56)):
            x = torch.randn(2, 3, h, w)
            outputs = compiled(x, [0], True, True)
            assert outputs[0][0].shape == (2, 32, h // 14, w // 14)
            assert outputs[0][1].shape == (2, 32)


def test_get_intermediate_layers_rejects_duplicate_indices():
    model = make_model()

    with pytest.raises(AssertionError, match="block indices must be unique"):
        model.get_intermediate_layers(torch.randn(2, 3, 28, 28), n=[0, 0])


def test_swiglu_hidden_features_sizing():
    ffn = SwiGLUFFN(in_features=32, hidden_features=128, adjust_hidden_features=True)

    assert ffn.w12.out_features == 176
    assert ffn.w3.in_features == 88


def test_state_dict_loads_strictly_without_mask_token():
    model = make_model(num_register_tokens=4)
    clone = make_model(num_register_tokens=4)

    clone.load_state_dict(model.state_dict(), strict=True)


def test_hub_small_constructor_defaults():
    model = dinov2_vits14(pretrained=False)

    assert model.patch_size == 14
    assert model.patch_embed.img_size == (518, 518)
    assert model.num_register_tokens == 0
    assert model.embed_dim == 384
    assert model.n_blocks == 12
