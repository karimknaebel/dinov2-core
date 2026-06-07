import importlib.util
from pathlib import Path

import torch


def test_hubconf_exports_only_backbones():
    spec = importlib.util.spec_from_file_location("hubconf", Path(__file__).parents[1] / "hubconf.py")
    hubconf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hubconf)

    exported = {name for name in dir(hubconf) if name.startswith("dinov2_")}
    assert exported == {
        "dinov2_vitb14",
        "dinov2_vitb14_reg",
        "dinov2_vitg14",
        "dinov2_vitg14_reg",
        "dinov2_vitl14",
        "dinov2_vitl14_reg",
        "dinov2_vits14",
        "dinov2_vits14_reg",
    }
    assert hubconf.dependencies == ["torch"]


def test_torch_hub_local_loads_from_checkout():
    model = torch.hub.load(
        str(Path(__file__).parents[1]),
        "dinov2_vits14",
        source="local",
        pretrained=False,
    )

    assert model.embed_dim == 384
