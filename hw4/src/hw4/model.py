"""PromptIR restoration network.

Adapted from the official PromptIR implementation:
https://github.com/va1shn9v/PromptIR

PromptIR: Prompting for All-in-One Image Restoration, Potlapalli et al.,
NeurIPS 2023. This project trains the architecture from scratch on HW4 data.
"""

from __future__ import annotations

import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def to_3d(x: torch.Tensor) -> torch.Tensor:
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return rearrange(x, "b (h w) c -> b c h w", h=height, w=width)


class BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(torch.Size(normalized_shape)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(variance + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        shape = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        variance = x.var(-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(variance + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim: int, norm_type: str) -> None:
        super().__init__()
        self.body = (
            BiasFreeLayerNorm(dim)
            if norm_type == "BiasFree"
            else WithBiasLayerNorm(dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), height, width)


class FeedForward(nn.Module):
    def __init__(self, dim: int, expansion_factor: float, bias: bool) -> None:
        super().__init__()
        hidden = int(dim * expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.depthwise = nn.Conv2d(
            hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2, bias=bias
        )
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.depthwise(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, bias: bool) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_depthwise = nn.Conv2d(
            dim * 3, dim * 3, 3, padding=1, groups=dim * 3, bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        query, key, value = self.qkv_depthwise(self.qkv(x)).chunk(3, dim=1)
        query = rearrange(query, "b (h c) x y -> b h c (x y)", h=self.num_heads)
        key = rearrange(key, "b (h c) x y -> b h c (x y)", h=self.num_heads)
        value = rearrange(value, "b (h c) x y -> b h c (x y)", h=self.num_heads)
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)
        attention = (query @ key.transpose(-2, -1) * self.temperature).softmax(dim=-1)
        output = attention @ value
        output = rearrange(
            output, "b h c (x y) -> b (h c) x y", h=self.num_heads, x=height, y=width
        )
        return self.project_out(output)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        expansion_factor: float,
        bias: bool,
        norm_type: str,
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm(dim, norm_type)
        self.attention = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, norm_type)
        self.feed_forward = FeedForward(dim, expansion_factor, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.feed_forward(self.norm2(x))


class Downsample(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(features, features // 2, 3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(features, features * 2, 3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class PromptGenBlock(nn.Module):
    """Generate a learned prompt mixture conditioned on image features."""

    def __init__(
        self, prompt_dim: int, prompt_size: int, input_dim: int, prompt_len: int = 5
    ) -> None:
        super().__init__()
        self.prompt = nn.Parameter(
            torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size)
        )
        self.weights = nn.Linear(input_dim, prompt_len)
        self.conv = nn.Conv2d(prompt_dim, prompt_dim, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        weights = F.softmax(self.weights(x.mean(dim=(-2, -1))), dim=1)
        prompt = weights[:, :, None, None, None] * self.prompt.expand(
            batch, -1, -1, -1, -1
        )
        prompt = prompt.sum(dim=1)
        return self.conv(
            F.interpolate(prompt, (height, width), mode="bilinear", align_corners=False)
        )


def blocks(
    count: int,
    dim: int,
    heads: int,
    expansion_factor: float,
    bias: bool,
    norm_type: str,
) -> nn.Sequential:
    return nn.Sequential(
        *[
            TransformerBlock(dim, heads, expansion_factor, bias, norm_type)
            for _ in range(count)
        ]
    )


class PromptIR(nn.Module):
    """All-in-one blind image restoration network with learned prompts."""

    def __init__(
        self,
        dim: int = 48,
        num_blocks: tuple[int, int, int, int] = (4, 6, 6, 8),
        refinement_blocks: int = 4,
        heads: tuple[int, int, int, int] = (1, 2, 4, 8),
        expansion_factor: float = 2.66,
        bias: bool = False,
        norm_type: str = "WithBias",
        prompt_len: int = 5,
    ) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, 3, padding=1, bias=bias)
        self.encoder1 = blocks(
            num_blocks[0], dim, heads[0], expansion_factor, bias, norm_type
        )
        self.down12 = Downsample(dim)
        self.encoder2 = blocks(
            num_blocks[1], dim * 2, heads[1], expansion_factor, bias, norm_type
        )
        self.down23 = Downsample(dim * 2)
        self.encoder3 = blocks(
            num_blocks[2], dim * 4, heads[2], expansion_factor, bias, norm_type
        )
        self.down34 = Downsample(dim * 4)
        self.latent = blocks(
            num_blocks[3], dim * 8, heads[3], expansion_factor, bias, norm_type
        )

        self.prompt3 = PromptGenBlock(320, 16, dim * 8, prompt_len)
        self.prompt2 = PromptGenBlock(128, 32, dim * 4, prompt_len)
        self.prompt1 = PromptGenBlock(64, 64, dim * 2, prompt_len)

        self.prompt_block3 = TransformerBlock(
            dim * 8 + 320, heads[2], expansion_factor, bias, norm_type
        )
        self.prompt_reduce3 = nn.Conv2d(dim * 8 + 320, dim * 4, 1, bias=bias)
        self.up43 = Upsample(dim * 4)
        self.skip_reduce3 = nn.Conv2d(dim * 6, dim * 4, 1, bias=bias)
        self.decoder3 = blocks(
            num_blocks[2], dim * 4, heads[2], expansion_factor, bias, norm_type
        )

        self.prompt_block2 = TransformerBlock(
            dim * 4 + 128, heads[2], expansion_factor, bias, norm_type
        )
        self.prompt_reduce2 = nn.Conv2d(dim * 4 + 128, dim * 4, 1, bias=bias)
        self.up32 = Upsample(dim * 4)
        self.skip_reduce2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=bias)
        self.decoder2 = blocks(
            num_blocks[1], dim * 2, heads[1], expansion_factor, bias, norm_type
        )

        self.prompt_block1 = TransformerBlock(
            dim * 2 + 64, heads[2], expansion_factor, bias, norm_type
        )
        self.prompt_reduce1 = nn.Conv2d(dim * 2 + 64, dim * 2, 1, bias=bias)
        self.up21 = Upsample(dim * 2)
        self.decoder1 = blocks(
            num_blocks[0], dim * 2, heads[0], expansion_factor, bias, norm_type
        )
        self.refinement = blocks(
            refinement_blocks, dim * 2, heads[0], expansion_factor, bias, norm_type
        )
        self.output = nn.Conv2d(dim * 2, 3, 3, padding=1, bias=bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        enc1 = self.encoder1(self.patch_embed(image))
        enc2 = self.encoder2(self.down12(enc1))
        enc3 = self.encoder3(self.down23(enc2))
        latent = self.latent(self.down34(enc3))

        latent = torch.cat([latent, self.prompt3(latent)], dim=1)
        latent = self.prompt_reduce3(self.prompt_block3(latent))
        dec3 = torch.cat([self.up43(latent), enc3], dim=1)
        dec3 = self.decoder3(self.skip_reduce3(dec3))

        dec3 = torch.cat([dec3, self.prompt2(dec3)], dim=1)
        dec3 = self.prompt_reduce2(self.prompt_block2(dec3))
        dec2 = torch.cat([self.up32(dec3), enc2], dim=1)
        dec2 = self.decoder2(self.skip_reduce2(dec2))

        dec2 = torch.cat([dec2, self.prompt1(dec2)], dim=1)
        dec2 = self.prompt_reduce1(self.prompt_block1(dec2))
        dec1 = torch.cat([self.up21(dec2), enc1], dim=1)
        dec1 = self.refinement(self.decoder1(dec1))
        return self.output(dec1) + image
