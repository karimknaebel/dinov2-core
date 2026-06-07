Minimal DINOv2 backbones for downstream fine-tuning.

This package keeps the torch.hub backbone API from
`facebookresearch/dinov2` and removes training-only features such as mask
tokens, xFormers paths, and task-specific heads.

```python
import torch

model = torch.hub.load(".", "dinov2_vits14", source="local", pretrained=False)
features = model(torch.randn(1, 3, 518, 518))
cls = features["x_norm_clstoken"]
```

Run the default offline tests with:

```sh
uv run pytest
```

Optional upstream equivalence tests require network access:

```sh
DINOV2_CORE_TEST_UPSTREAM=1 uv run pytest -m upstream
DINOV2_CORE_TEST_PRETRAINED=1 uv run pytest -m pretrained
```
