Minimal DINOv2 backbones for downstream fine-tuning.

This package keeps the torch.hub backbone API from
`facebookresearch/dinov2` and removes training-only features such as mask
tokens, xFormers paths, and task-specific heads.

## Torch hub

```python
import torch

model = torch.hub.load("karimknaebel/dinov2-core", "dinov2_vits14", trust_repo=True, pretrained=True)
features = model(torch.randn(1, 3, 518, 518))
cls = features["x_norm_clstoken"]
```

With registers:

```python
model = torch.hub.load("karimknaebel/dinov2-core", "dinov2_vits14_reg", trust_repo=True, pretrained=True)
```

Available backbones:

```text
dinov2_vits14      dinov2_vits14_reg
dinov2_vitb14      dinov2_vitb14_reg
dinov2_vitl14      dinov2_vitl14_reg
dinov2_vitg14      dinov2_vitg14_reg
```

Run the default offline tests with:

```sh
uv run pytest
```

Optional upstream equivalence tests require network access:

```sh
DINOV2_CORE_TEST_UPSTREAM=1 uv run pytest -m upstream
DINOV2_CORE_TEST_PRETRAINED=1 uv run pytest -m pretrained
DINOV2_CORE_TEST_UPSTREAM=1 DINOV2_CORE_TEST_CUDA=1 uv run pytest -m cuda
```
