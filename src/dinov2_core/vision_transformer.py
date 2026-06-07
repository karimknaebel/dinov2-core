# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the upstream DINOv2 repository.

from functools import partial
import math
from typing import Callable, Sequence

import torch
from torch import Tensor, nn
from torch.nn.init import trunc_normal_

from .layers import Block, Mlp, PatchEmbed


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        ffn_bias: bool = True,
        proj_bias: bool = True,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        init_values: float | None = None,
        embed_layer: Callable[..., nn.Module] = PatchEmbed,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        block_fn: Callable[..., nn.Module] = Block,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        num_register_tokens: int = 0,
        interpolate_antialias: bool = False,
        interpolate_offset: float = 0.1,
    ) -> None:
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.num_features = self.embed_dim = embed_dim
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim)
        )
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
            if num_register_tokens
            else None
        )

        if drop_path_uniform:
            drop_rates = [drop_path_rate] * depth
        else:
            drop_rates = torch.linspace(0, drop_path_rate, depth).tolist()
        blocks = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=drop_rates[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()
        self.init_weights()

    @staticmethod
    @torch.compiler.disable
    def _interpolate_with_offset(
        patch_pos_embed: Tensor,
        scale_factor: tuple[float, float],
        interpolate_antialias: bool,
    ) -> Tensor:
        # Keep upstream's offset scale_factor behavior without Dynamo specializing on each image size.
        return nn.functional.interpolate(
            patch_pos_embed,
            mode="bicubic",
            antialias=interpolate_antialias,
            scale_factor=scale_factor,
        )

    def init_weights(self) -> None:
        def init_weights_vit_timm(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        self.apply(init_weights_vit_timm)

    def interpolate_pos_encoding(self, x: Tensor, w: int, h: int) -> Tensor:
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        n = self.pos_embed.shape[1] - 1
        if npatch == n and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        m = int(math.sqrt(n))
        assert n == m * m
        patch_pos_embed = patch_pos_embed.reshape(1, m, m, dim).permute(0, 3, 1, 2)
        if self.interpolate_offset:
            patch_pos_embed = self._interpolate_with_offset(
                patch_pos_embed,
                (
                    (w0 + self.interpolate_offset) / m,
                    (h0 + self.interpolate_offset) / m,
                ),
                self.interpolate_antialias,
            )
        else:
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed,
                mode="bicubic",
                antialias=self.interpolate_antialias,
                size=(w0, h0),
            )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(
            previous_dtype
        )

    def prepare_tokens(self, x: Tensor) -> Tensor:
        b, _, w, h = x.shape
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(b, -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        if self.register_tokens is not None:
            x = torch.cat(
                (x[:, :1], self.register_tokens.expand(b, -1, -1), x[:, 1:]), dim=1
            )
        return x

    def forward_features(self, x: Tensor) -> dict[str, Tensor]:
        x = self.prepare_tokens(x)
        for block in self.blocks:
            x = block(x)
        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,
        }

    def _get_intermediate_layers(
        self, x: Tensor, n: int | Sequence[int] = 1
    ) -> list[Tensor]:
        x = self.prepare_tokens(x)
        output = []
        if isinstance(n, int):
            blocks_to_take = set(range(len(self.blocks) - n, len(self.blocks)))
        else:
            # Dynamo handles set membership here better than tuple/list membership with dynamic shapes.
            blocks_to_take = set(n)
            assert len(blocks_to_take) == len(n), "block indices must be unique"
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), (
            f"only {len(output)} / {len(blocks_to_take)} blocks found"
        )
        return output

    def get_intermediate_layers(
        self,
        x: Tensor,
        n: int | Sequence[int] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        norm: bool = True,
    ) -> tuple[Tensor, ...] | tuple[tuple[Tensor, Tensor], ...]:
        outputs = self._get_intermediate_layers(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens :] for out in outputs]
        if reshape:
            b, _, w, h = x.shape
            outputs = [
                out.reshape(b, w // self.patch_size, h // self.patch_size, -1).permute(
                    0, 3, 1, 2
                )
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        return self.forward_features(x)


def vit_small(
    patch_size: int = 16, num_register_tokens: int = 0, in_chans: int = 3, **kwargs
) -> DinoVisionTransformer:
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        in_chans=in_chans,
        **kwargs,
    )


def vit_base(
    patch_size: int = 16, num_register_tokens: int = 0, in_chans: int = 3, **kwargs
) -> DinoVisionTransformer:
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        in_chans=in_chans,
        **kwargs,
    )


def vit_large(
    patch_size: int = 16, num_register_tokens: int = 0, in_chans: int = 3, **kwargs
) -> DinoVisionTransformer:
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        in_chans=in_chans,
        **kwargs,
    )


def vit_giant2(
    patch_size: int = 16, num_register_tokens: int = 0, in_chans: int = 3, **kwargs
) -> DinoVisionTransformer:
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        in_chans=in_chans,
        **kwargs,
    )
