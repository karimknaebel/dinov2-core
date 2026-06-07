Minimal DINOv2 backbones for downstream fine-tuning.

Based on
[`facebookresearch/dinov2@7764ea0`](https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8),
with the upstream torch.hub API kept intact. Training-only pieces are removed:
mask tokens, xFormers paths and warnings, and task-specific heads. The backbones
support `torch.compile(dynamic=True)` for varying input sizes.

## Torch hub

```python
import torch

model = torch.hub.load("karimknaebel/dinov2-core", "dinov2_vits14", trust_repo=True, pretrained=True)
model_reg = torch.hub.load("karimknaebel/dinov2-core", "dinov2_vits14_reg", trust_repo=True, pretrained=True)

features = model(torch.randn(1, 3, 518, 518))
cls = features["x_norm_clstoken"]
```

Available backbones:

```text
dinov2_vits14      dinov2_vits14_reg
dinov2_vitb14      dinov2_vitb14_reg
dinov2_vitl14      dinov2_vitl14_reg
dinov2_vitg14      dinov2_vitg14_reg
```

## Tests

```sh
uv run pytest
```

Optional upstream equivalence tests:

```sh
DINOV2_CORE_TEST_UPSTREAM=1 uv run pytest -m upstream
DINOV2_CORE_TEST_PRETRAINED=1 uv run pytest -m pretrained
DINOV2_CORE_TEST_UPSTREAM=1 DINOV2_CORE_TEST_CUDA=1 uv run pytest -m cuda
```

The upstream tests compare against the pinned DINOv2 commit and require bitwise
equality.
