"""DiT-style action denoiser for ManiSkill diffusion policy.

This module keeps the same public forward signature as
``diffusion_policy.conditional_unet1d.ConditionalUnet1D`` so the existing
4D-map diffusion policy can swap UNet and DiT backbones without changing the
training or sampling loops.
"""

from typing import Union

import torch
import torch.nn as nn

from map4d.backbone.model.diffusion.positional_embedding import SinusoidalPosEmb


def _modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock1D(nn.Module):
    """Transformer block with adaptive layer norm conditioning."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 6 * embed_dim),
        )

        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(cond).chunk(6, dim=-1)
        )
        attn_in = _modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * attn_out
        mlp_in = _modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(mlp_in)
        return x


class ConditionalDiT1D(nn.Module):
    """DiT action denoiser with global observation conditioning.

    Args:
        input_dim: Action dimension.
        global_cond_dim: Flattened observation conditioning dimension.
        diffusion_step_embed_dim: Size of diffusion timestep embedding.
        embed_dim: Transformer token dimension.
        depth: Number of DiT blocks.
        num_heads: Attention heads.
        mlp_ratio: Feed-forward expansion ratio.
        dropout: Dropout used by attention and MLP.
        max_horizon: Maximum action horizon supported by learned position tokens.
    """

    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int = 256,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_horizon: int = 128,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.input_dim = input_dim
        self.global_cond_dim = global_cond_dim
        self.embed_dim = embed_dim
        self.max_horizon = max_horizon

        self.action_proj = nn.Linear(input_dim, embed_dim)
        self.position_embed = nn.Parameter(torch.zeros(1, max_horizon, embed_dim))

        self.time_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.cond_encoder = nn.Sequential(
            nn.Linear(global_cond_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock1D(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.final_norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 2 * embed_dim),
        )
        self.output_proj = nn.Linear(embed_dim, input_dim)

        nn.init.normal_(self.position_embed, std=0.02)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"ConditionalDiT1D parameters: {n_params / 1e6:.2f}M")

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        global_cond: torch.Tensor = None,
    ) -> torch.Tensor:
        """Predict diffusion noise.

        Args:
            sample: Noisy action sequence with shape ``(B, T, action_dim)``.
            timestep: Diffusion timestep scalar or ``(B,)`` tensor.
            global_cond: Flattened observation conditioning with shape
                ``(B, global_cond_dim)``.
        """
        if global_cond is None:
            raise ValueError("ConditionalDiT1D requires global_cond")
        if sample.shape[1] > self.max_horizon:
            raise ValueError(
                f"Action horizon {sample.shape[1]} exceeds max_horizon={self.max_horizon}"
            )

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif timesteps.ndim == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        cond = self.time_encoder(timesteps) + self.cond_encoder(global_cond)
        x = self.action_proj(sample)
        x = x + self.position_embed[:, : sample.shape[1]].to(dtype=x.dtype)
        for block in self.blocks:
            x = block(x, cond)
        shift, scale = self.final_modulation(cond).chunk(2, dim=-1)
        x = _modulate(self.final_norm(x), shift, scale)
        return self.output_proj(x)
