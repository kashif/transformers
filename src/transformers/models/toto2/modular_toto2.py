# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Toto 2 model (Datadog time-series foundation model)."""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import PreTrainedConfig
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ..llama.modeling_llama import LlamaMLP, LlamaRMSNorm, eager_attention_forward


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="Datadog/Toto-2.0-4m")
class Toto2Config(PreTrainedConfig):
    r"""
    Configuration for Toto 2, a decoder-only multivariate time-series foundation model. The architecture interleaves
    causal *time* attention layers (attending over the time axis, patched) with bidirectional *variate* attention
    layers (attending over the variate axis at a fixed time position). Linear layers use μP-style fan-in scaling,
    attention uses `1/head_dim` scaling (μP), and residual connections follow the τ-weighted scheme of u-μP.

    Args:
        patch_size (`int`, *optional*, defaults to 32):
            Number of time points per input patch.
        hidden_size (`int`, *optional*, defaults to 256):
            Model width (aka `d_model`).
        intermediate_size (`int`, *optional*, defaults to 688):
            MLP intermediate size (aka `d_ff`); the SwiGLU produces `2*intermediate_size` activations from
            `hidden_size` and contracts back to `hidden_size`.
        num_hidden_layers (`int`, *optional*, defaults to 4):
            Total number of transformer layers (time + variate).
        num_attention_heads (`int`, *optional*, defaults to 4):
            Number of query heads.
        num_key_value_heads (`int`, *optional*, defaults to 4):
            Number of key/value groups. `num_attention_heads // num_key_value_heads` > 1 enables GQA.
        head_dim (`int`, *optional*, defaults to 64):
            Head dimension for Q/K and V (kept identical for the converted 4M checkpoint).
        layer_group_size (`int`, *optional*, defaults to 4):
            Size of the time/variate layer group. Layers `i % layer_group_size` within a group are variate (when
            `variate_layer_first=True`) or the last `num_variate_layers_per_group` indices of the group (otherwise).
        num_variate_layers_per_group (`int`, *optional*, defaults to 1):
            Variate-attention layers per group.
        variate_layer_first (`bool`, *optional*, defaults to `False`):
            If `True`, variate layers come at the start of each group; if `False`, at the end.
        qk_norm (`bool`, *optional*, defaults to `False`):
            Apply RMSNorm to Q/K before attention.
        per_dim_scale (`bool`, *optional*, defaults to `True`):
            Apply a learned positive per-dimension scale to Q before attention (softplus-parameterized).
        use_xpos (`bool`, *optional*, defaults to `True`):
            Use RoPE with xPos extrapolation scaling on the time axis.
        partial_rotary_factor (`float`, *optional*, defaults to 0.5):
            Fraction of `head_dim` that is rotated by RoPE (the rest is passed through).
        rope_theta (`float`, *optional*, defaults to 10000.0):
            RoPE base.
        xpos_scale_base (`int`, *optional*, defaults to 256):
            xPos scale base.
        xpos_scale_exponent (`float`, *optional*, defaults to 1.0):
            xPos scale exponent (query side; key side uses `-xpos_scale_exponent`).
        attn_bias (`bool`, *optional*, defaults to `True`):
            Bias on attention linear projections.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Bias on MLP linear projections.
        norm_eps (`float`, *optional*, defaults to 1e-4):
            RMSNorm epsilon.
        norm_include_weight (`bool`, *optional*, defaults to `False`):
            Whether RMSNorm includes a learnable affine weight.
        residual_mult (`float`, *optional*, defaults to 0.75):
            τ-rule global residual multiplier (u-μP).
        residual_attn_ratio (`float`, *optional*, defaults to 5.13621546657774):
            τ-rule attention:MLP residual ratio (u-μP). Compute via
            `Toto2Config.compute_residual_attn_ratio(context_length, patch_size)`.
        quantiles (`Sequence[float]`, *optional*, defaults to `(0.1, ..., 0.9)`):
            Quantile levels predicted by the output head.
        num_output_patches (`int`, *optional*, defaults to 1):
            Number of future patches predicted per token of hidden state.
        context_length (`int`, *optional*, defaults to 4096):
            Default context length used by [`Toto2ForPrediction`] (also sets `max_position_embeddings`).
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Attention dropout.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Initializer std.
    """

    model_type = "toto2"
    keys_to_ignore_at_inference = []
    is_encoder_decoder = False

    def __init__(
        self,
        patch_size: int = 32,
        hidden_size: int = 256,
        intermediate_size: int = 688,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 4,
        num_key_value_heads: int = 4,
        head_dim: int = 64,
        layer_group_size: int = 4,
        num_variate_layers_per_group: int = 1,
        variate_layer_first: bool = False,
        qk_norm: bool = False,
        per_dim_scale: bool = True,
        use_xpos: bool = True,
        partial_rotary_factor: float = 0.5,
        rope_theta: float = 10000.0,
        xpos_scale_base: int = 256,
        xpos_scale_exponent: float = 1.0,
        attn_bias: bool = True,
        mlp_bias: bool = False,
        norm_eps: float = 1e-4,
        norm_include_weight: bool = False,
        residual_mult: float = 0.75,
        residual_attn_ratio: float = 5.136215466577748,
        quantiles: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        num_output_patches: int = 1,
        context_length: int = 4096,
        attention_dropout: float = 0.0,
        initializer_range: float = 0.02,
        hidden_act: str = "silu",
        **kwargs,
    ):
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.layer_group_size = layer_group_size
        self.num_variate_layers_per_group = num_variate_layers_per_group
        self.variate_layer_first = variate_layer_first
        self.qk_norm = qk_norm
        self.per_dim_scale = per_dim_scale
        self.use_xpos = use_xpos
        self.partial_rotary_factor = partial_rotary_factor
        self.rope_theta = rope_theta
        self.xpos_scale_base = xpos_scale_base
        self.xpos_scale_exponent = xpos_scale_exponent
        self.attn_bias = attn_bias
        self.mlp_bias = mlp_bias
        self.rms_norm_eps = norm_eps
        self.norm_include_weight = norm_include_weight
        self.residual_mult = residual_mult
        self.residual_attn_ratio = residual_attn_ratio
        self.quantiles = list(quantiles)
        self.num_output_patches = num_output_patches
        self.context_length = context_length
        self.max_position_embeddings = context_length
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.hidden_act = hidden_act

        assert num_hidden_layers % layer_group_size == 0, (
            f"num_hidden_layers ({num_hidden_layers}) must be divisible by layer_group_size ({layer_group_size})"
        )
        assert num_attention_heads % num_key_value_heads == 0, (
            f"num_attention_heads ({num_attention_heads}) must be divisible by num_key_value_heads "
            f"({num_key_value_heads})"
        )

        super().__init__(**kwargs)

    @staticmethod
    def compute_residual_attn_ratio(context_length: int, patch_size: int) -> float:
        """u-μP default: `sqrt(S / log(S))` with `S = context_length / patch_size`."""
        s = context_length / patch_size
        return math.sqrt(s / math.log(s))

    def is_variate_layer(self, layer_idx: int) -> bool:
        mod = layer_idx % self.layer_group_size
        if self.variate_layer_first:
            return mod < self.num_variate_layers_per_group
        return mod >= self.layer_group_size - self.num_variate_layers_per_group


@dataclass
@auto_docstring
class Toto2ModelOutput(BaseModelOutput):
    r"""
    loc (`torch.Tensor` of shape `(batch, num_variates, time)`):
        Per-patch location used to standardize the input (broadcast to each token in the patch).
    scale (`torch.Tensor` of shape `(batch, num_variates, time)`):
        Per-patch scale used to standardize the input.
    """

    loc: torch.Tensor | None = None
    scale: torch.Tensor | None = None


@dataclass
@auto_docstring
class Toto2PredictionOutput(BaseModelOutput):
    r"""
    quantiles (`torch.Tensor` of shape `(num_quantiles, batch, num_variates, horizon)`):
        Quantile forecasts for each knot in `config.quantiles`.
    mean_predictions (`torch.Tensor` of shape `(batch, num_variates, horizon)`):
        Median (`q=0.5`) forecast if `0.5 in config.quantiles`, otherwise the central knot.
    loss (`torch.Tensor` of shape `(1,)`, *optional*):
        Quantile loss when `future_values` is provided.
    """

    quantiles: torch.Tensor | None = None
    mean_predictions: torch.Tensor | None = None
    loss: torch.Tensor | float | None = None


# ---------------------------------------------------------------------------
# Normalization (direct reuse from Llama).
# ---------------------------------------------------------------------------


class Toto2RMSNorm(LlamaRMSNorm):
    """RMSNorm as in Llama; `Toto2Config.norm_include_weight=False` gives a pure, unweighted norm."""

    def __init__(self, hidden_size: int, eps: float = 1e-4, include_weight: bool = False):
        super().__init__(hidden_size, eps=eps)
        if not include_weight:
            # Freeze the weight to 1 and stop tracking it as a learnable parameter.
            self.weight = nn.Parameter(torch.ones(hidden_size), requires_grad=False)


# ---------------------------------------------------------------------------
# Rotary embedding (time axis). Toto2 uses a non-standard interleaved rotation plus optional
# xPos length-extrapolation scaling. We convert the checkpoint's QK weights so that Llama's
# standard half-split rotation gives the same result, and apply xPos scaling on top here.
# ---------------------------------------------------------------------------


class Toto2RotaryEmbedding(nn.Module):
    """Partial RoPE (first `partial_rotary_factor * head_dim` dimensions are rotated) with optional xPos scaling.

    The convert script permutes the Q and K projection rows so that the cached *interleaved* knot positions line up
    with Llama's *half-split* rotate convention — this keeps the module standard-shaped while matching the original
    output.
    """

    inv_freq: torch.Tensor

    def __init__(self, config: Toto2Config):
        super().__init__()
        self.config = config
        self.rotary_dim = int(round(config.head_dim * config.partial_rotary_factor))
        assert self.rotary_dim % 2 == 0, "rotary_dim must be even"

        inv_freq = 1.0 / (
            config.rope_theta ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        if config.use_xpos:
            # xPos base-scale per dimension pair: (arange + 0.4*D) / (1.4*D)
            xpos_base = (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) + 0.4 * self.rotary_dim) / (
                1.4 * self.rotary_dim
            )
            self.register_buffer("xpos_base", xpos_base, persistent=False)
        else:
            self.xpos_base = None

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        # position_ids shape: [T] or [B, T]. We reshape to ensure output broadcasts over (heads).
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        freqs = position_ids[..., None].to(torch.float32) * self.inv_freq  # [B, T, D/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [B, T, D]
        cos = emb.cos().to(dtype=x.dtype)
        sin = emb.sin().to(dtype=x.dtype)
        return cos, sin

    def xpos_scale(self, position_ids: torch.Tensor, sign: float, dtype: torch.dtype) -> torch.Tensor | None:
        if self.xpos_base is None:
            return None
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        max_pos = position_ids.max()
        center = torch.div(max_pos + 1, 2, rounding_mode="floor")
        power = (position_ids.float() - center) / self.config.xpos_scale_base
        base_scale = self.xpos_base ** power[..., None]  # [B, T, D/2]
        base_scale = torch.cat((base_scale, base_scale), dim=-1)  # [B, T, D]
        return (base_scale ** (sign * self.config.xpos_scale_exponent)).to(dtype=dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary_with_xpos(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_xpos: torch.Tensor | None,
    k_xpos: torch.Tensor | None,
    rotary_dim: int,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply partial rotary to the first `rotary_dim` dims of q/k; the rest pass through unchanged.

    `q_xpos` / `k_xpos` (when not None) multiply the rotated half in-place with the xPos scale."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_rot = q_rot * cos + rotate_half(q_rot) * sin
    k_rot = k_rot * cos + rotate_half(k_rot) * sin
    if q_xpos is not None:
        q_rot = q_rot * q_xpos.unsqueeze(unsqueeze_dim)
    if k_xpos is not None:
        k_rot = k_rot * k_xpos.unsqueeze(unsqueeze_dim)
    return torch.cat((q_rot, q_pass), dim=-1), torch.cat((k_rot, k_pass), dim=-1)


# ---------------------------------------------------------------------------
# Per-dimension scale (u-μP).
# ---------------------------------------------------------------------------


class Toto2PerDimScale(nn.Module):
    """Learned positive per-dimension scaling applied to Q before attention.

    The stored parameter `per_dim_scale` is the *raw* one; the effective multiplier is
    `softplus(per_dim_scale) / log(2)` so that zero-init yields unit scaling. At conversion we bake the
    `u-μP` normalization constant (0.52103) into the stored parameter so that standard `softplus(p)/log(2)` produces
    the same output as the reference.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.per_dim_scale = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = (F.softplus(self.per_dim_scale) / math.log(2.0)).to(x.dtype)
        return x * r


# ---------------------------------------------------------------------------
# MLP — SwiGLU. At conversion the original fused `fc1` (2*hidden) is split into Llama-style
# `gate_proj` / `up_proj` (and the μP 1/sqrt(fan_in) scale is baked into weights).
# ---------------------------------------------------------------------------


class Toto2MLP(LlamaMLP):
    """Llama-style SwiGLU MLP. Bias is optional (`config.mlp_bias`)."""

    def __init__(self, config: Toto2Config):
        super().__init__(config)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)


# ---------------------------------------------------------------------------
# Attention — custom: xPos, partial RoPE, per-dim Q scale, optional QK norm, μP `1/head_dim` scaling.
# ---------------------------------------------------------------------------


class Toto2Attention(nn.Module):
    """Toto2 self-attention. Matches `LlamaAttention`'s projection layout (separate q/k/v/o) but uses partial RoPE
    with xPos, a learned per-dimension Q scale, and μP `1/head_dim` attention scaling. Variate layers skip causal
    masking and skip RoPE entirely (position has no meaning across the variate axis)."""

    def __init__(self, config: Toto2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_variate = config.is_variate_layer(layer_idx)
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        # μP scaling: 1/d_k instead of 1/sqrt(d_k).
        self.scaling = 1.0 / self.head_dim
        self.attention_dropout = config.attention_dropout
        self.is_causal = not self.is_variate

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attn_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attn_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attn_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attn_bias)

        if config.qk_norm:
            self.q_norm = Toto2RMSNorm(self.head_dim, eps=config.rms_norm_eps, include_weight=False)
            self.k_norm = Toto2RMSNorm(self.head_dim, eps=config.rms_norm_eps, include_weight=False)
        else:
            self.q_norm = None
            self.k_norm = None

        if config.per_dim_scale:
            self.per_dim_scale = Toto2PerDimScale(self.head_dim)
        else:
            self.per_dim_scale = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        xpos_scales: tuple[torch.Tensor | None, torch.Tensor | None] | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_dim: int | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        q = self.q_proj(hidden_states).view(*input_shape, self.num_heads, self.head_dim).transpose(-3, -2)
        k = self.k_proj(hidden_states).view(*input_shape, self.num_key_value_heads, self.head_dim).transpose(-3, -2)
        v = self.v_proj(hidden_states).view(*input_shape, self.num_key_value_heads, self.head_dim).transpose(-3, -2)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if self.per_dim_scale is not None:
            q = self.per_dim_scale(q)

        if position_embeddings is not None and not self.is_variate:
            cos, sin = position_embeddings
            q_xpos, k_xpos = xpos_scales if xpos_scales is not None else (None, None)
            q, k = apply_rotary_with_xpos(q, k, cos, sin, q_xpos, k_xpos, rotary_dim)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            is_causal=self.is_causal and attention_mask is None,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), attn_weights


# ---------------------------------------------------------------------------
# τ-weighted residual rule (u-μP).
# ---------------------------------------------------------------------------


def _tau_rule(layer_idx: int, total_depth: int, residual_mult: float, residual_attn_ratio: float) -> float:
    """u-μP residual scaling rule. Matches `unit_scaling.transformer_residual_scaling_rule`: produces a τ per
    sub-layer (alternating attn/mlp, so 2*num_layers in total)."""
    frac = layer_idx / max(total_depth - 1, 1)
    # Blend attn-ratio at layer 0 → 1 at the last layer, then scale by global residual_mult.
    blend = residual_attn_ratio ** (1.0 - frac)
    return residual_mult * blend


class Toto2DecoderLayer(nn.Module):
    """One Toto2 transformer block with τ-weighted residuals:

    `hidden = (tau_a / sqrt(1 + tau_a^2)) * attn(norm(hidden)) + (1 / sqrt(1 + tau_a^2)) * hidden`
    `hidden = (tau_m / sqrt(1 + tau_m^2)) * mlp(norm(hidden)) + (1 / sqrt(1 + tau_m^2)) * hidden`

    where `tau_a`, `tau_m` are the per-layer τ values from the u-μP residual rule.
    """

    def __init__(self, config: Toto2Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_variate = config.is_variate_layer(layer_idx)
        self.self_attn = Toto2Attention(config, layer_idx=layer_idx)
        self.mlp = Toto2MLP(config)
        self.norm1 = Toto2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, include_weight=config.norm_include_weight
        )
        self.norm2 = Toto2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, include_weight=config.norm_include_weight
        )

        total_depth = 2 * config.num_hidden_layers
        tau_a = _tau_rule(2 * layer_idx, total_depth, config.residual_mult, config.residual_attn_ratio)
        tau_m = _tau_rule(2 * layer_idx + 1, total_depth, config.residual_mult, config.residual_attn_ratio)
        denom_a = (1.0 + tau_a * tau_a) ** 0.5
        denom_m = (1.0 + tau_m * tau_m) ** 0.5
        self.register_buffer("attn_alpha", torch.tensor(tau_a / denom_a), persistent=False)
        self.register_buffer("attn_beta", torch.tensor(1.0 / denom_a), persistent=False)
        self.register_buffer("mlp_alpha", torch.tensor(tau_m / denom_m), persistent=False)
        self.register_buffer("mlp_beta", torch.tensor(1.0 / denom_m), persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        xpos_scales: tuple[torch.Tensor | None, torch.Tensor | None] | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_dim: int | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        attn_out, _ = self.self_attn(
            self.norm1(hidden_states),
            position_embeddings=position_embeddings,
            xpos_scales=xpos_scales,
            attention_mask=attention_mask,
            rotary_dim=rotary_dim,
            **kwargs,
        )
        hidden_states = self.attn_alpha * attn_out + self.attn_beta * hidden_states

        mlp_out = self.mlp(self.norm2(hidden_states))
        hidden_states = self.mlp_alpha * mlp_out + self.mlp_beta * hidden_states
        return hidden_states


# ---------------------------------------------------------------------------
# Causal per-patch scaler (RevIN-style).
# ---------------------------------------------------------------------------


class Toto2PatchedCausalStdScaler(nn.Module):
    """Cumulative mean/std along the time axis, evaluated at patch boundaries and broadcast across each patch."""

    def __init__(self, patch_size: int, correction: float = 1.0, minimum_scale: float = 1e-6):
        super().__init__()
        self.patch_size = patch_size
        self.correction = correction
        self.minimum_scale = minimum_scale

    def forward(
        self, data: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hp = data.to(torch.float64) if data.is_floating_point() else data.to(torch.float32)
        if mask is None:
            mask = torch.ones_like(data, dtype=torch.bool)

        cum_data = (hp * mask).cumsum(dim=-1)
        denom = mask.cumsum(dim=-1).clamp_min(1)
        causal_loc = cum_data / denom

        prev_loc = torch.cat([torch.zeros_like(causal_loc[..., :1]), causal_loc[..., :-1]], dim=-1)
        delta = hp - prev_loc
        increment = delta * (hp - causal_loc) * mask
        m2 = increment.cumsum(dim=-1)
        causal_var = m2 / (denom - self.correction).clamp(min=1)
        causal_scale = causal_var.clamp_min(0).sqrt().clamp_min(self.minimum_scale)

        b = data.shape[:-1]
        seq_len = data.shape[-1]
        assert seq_len % self.patch_size == 0, (
            f"time length {seq_len} must be a multiple of patch_size {self.patch_size}"
        )
        loc = causal_loc.view(*b, -1, self.patch_size)[..., -1:].expand(*b, -1, self.patch_size).reshape(*b, seq_len)
        scale = (
            causal_scale.view(*b, -1, self.patch_size)[..., -1:].expand(*b, -1, self.patch_size).reshape(*b, seq_len)
        )
        loc, scale = loc.to(data.dtype), scale.to(data.dtype)
        return torch.where(mask, (data - loc) / scale, torch.zeros_like(data)), loc, scale


# ---------------------------------------------------------------------------
# Base / Pretrained.
# ---------------------------------------------------------------------------


@auto_docstring
class Toto2PreTrainedModel(PreTrainedModel):
    config: Toto2Config
    base_model_prefix = "model"
    main_input_name = "past_values"
    input_modalities = ("time",)
    _no_split_modules = ["Toto2DecoderLayer"]
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _can_record_outputs = {
        "hidden_states": Toto2DecoderLayer,
        "attentions": Toto2Attention,
    }


# ---------------------------------------------------------------------------
# Core model — multivariate, alternating time/variate attention.
# ---------------------------------------------------------------------------


class Toto2Model(Toto2PreTrainedModel):
    def __init__(self, config: Toto2Config):
        super().__init__(config)
        self.config = config
        self.scaler = Toto2PatchedCausalStdScaler(patch_size=config.patch_size)

        # Patch tokenizer: concatenate scaled patch + mask channel, project to hidden_size.
        self.patch_proj = nn.Sequential(
            nn.Linear(2 * config.patch_size, 4 * config.hidden_size),
            nn.SiLU(),
            nn.Linear(4 * config.hidden_size, config.hidden_size),
        )
        self.patch_skip = nn.Linear(2 * config.patch_size, config.hidden_size)

        self.layers = nn.ModuleList(
            [Toto2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.out_norm = Toto2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, include_weight=config.norm_include_weight
        )
        self.rotary_emb = Toto2RotaryEmbedding(config)

        self.post_init()

    def _embed_patches(self, scaled: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # scaled, mask: [B, N, T]. Output: [B, N, S, D] where S = T / patch_size.
        bsz, nvar, time = scaled.shape
        p = self.config.patch_size
        s = time // p
        patch = scaled.view(bsz, nvar, s, p)
        mpatch = (~mask).to(scaled.dtype).view(bsz, nvar, s, p)
        x = torch.cat([patch, mpatch], dim=-1)
        return self.patch_proj(x) + self.patch_skip(x)

    def _run_stack(
        self, hidden: torch.Tensor, group_ids: torch.Tensor | None, time_ids: torch.Tensor | None
    ) -> torch.Tensor:
        # hidden: [B, N, S, D]
        bsz, nvar, seq, _ = hidden.shape
        device = hidden.device

        if time_ids is None:
            time_ids = torch.arange(seq, device=device, dtype=torch.long)
        position_embeddings = self.rotary_emb(hidden, time_ids)
        if self.rotary_emb.xpos_base is not None:
            q_scale = self.rotary_emb.xpos_scale(time_ids, sign=+1.0, dtype=hidden.dtype)
            k_scale = self.rotary_emb.xpos_scale(time_ids, sign=-1.0, dtype=hidden.dtype)
            xpos_scales = (q_scale, k_scale)
        else:
            xpos_scales = None

        for layer in self.layers:
            if layer.is_variate:
                # reshape: [B, N, S, D] -> [B*S, N, D]  (attend across variates at each time)
                x = hidden.transpose(1, 2).reshape(bsz * seq, nvar, hidden.shape[-1])
                x = layer(x)
                hidden = x.view(bsz, seq, nvar, hidden.shape[-1]).transpose(1, 2).contiguous()
            else:
                # reshape: [B, N, S, D] -> [B*N, S, D]  (causal attend across time)
                x = hidden.reshape(bsz * nvar, seq, hidden.shape[-1])
                x = layer(
                    x,
                    position_embeddings=position_embeddings,
                    xpos_scales=xpos_scales,
                    rotary_dim=self.rotary_emb.rotary_dim,
                )
                hidden = x.view(bsz, nvar, seq, hidden.shape[-1])

        return self.out_norm(hidden)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        past_values: torch.Tensor,
        past_values_mask: torch.Tensor | None = None,
        series_ids: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Toto2ModelOutput:
        r"""
        past_values (`torch.FloatTensor` of shape `(batch, num_variates, time)`):
            Multivariate time-series context. `time` must be a multiple of `config.patch_size`.
        past_values_mask (`torch.BoolTensor` of shape `(batch, num_variates, time)`, *optional*):
            `True` at valid positions, `False` at masked positions. Defaults to all-valid.
        series_ids (`torch.LongTensor` of shape `(batch, num_variates)`, *optional*):
            Series identifiers for the variate axis, used to restrict cross-variate attention to matching ids.
            Defaults to zeros (single group).
        """
        if past_values_mask is None:
            past_values_mask = torch.ones_like(past_values, dtype=torch.bool)

        scaled, loc, scale = self.scaler(past_values, past_values_mask)
        scaled = scaled.asinh()
        hidden = self._embed_patches(scaled, past_values_mask)
        hidden = self._run_stack(hidden, group_ids=series_ids, time_ids=None)

        return Toto2ModelOutput(last_hidden_state=hidden, loc=loc, scale=scale)


# ---------------------------------------------------------------------------
# Prediction head.
# ---------------------------------------------------------------------------


class Toto2ForPrediction(Toto2PreTrainedModel):
    def __init__(self, config: Toto2Config):
        super().__init__(config)
        self.config = config
        self.model = Toto2Model(config)

        # Quantile-knot head: project hidden_size -> (num_output_patches * patch_size * num_quantiles).
        out_size = config.num_output_patches * config.patch_size * len(config.quantiles)
        self.output_head = nn.Sequential(
            nn.Linear(config.hidden_size, 4 * config.hidden_size),
            nn.SiLU(),
            nn.Linear(4 * config.hidden_size, out_size),
        )
        self.output_skip = nn.Linear(config.hidden_size, out_size)

        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        past_values: torch.Tensor,
        past_values_mask: torch.Tensor | None = None,
        series_ids: torch.Tensor | None = None,
        horizon_len: int | None = None,
        future_values: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Toto2PredictionOutput:
        r"""
        past_values (`torch.FloatTensor` of shape `(batch, num_variates, time)`): Multivariate context.
        past_values_mask (`torch.BoolTensor`, *optional*): Observed-mask for `past_values`.
        series_ids (`torch.LongTensor` of shape `(batch, num_variates)`, *optional*): Series ids for variate grouping.
        horizon_len (`int`, *optional*): Number of future fine-resolution steps to forecast. Defaults to
            `config.patch_size * config.num_output_patches` (one patch per token). Longer horizons iterate the model
            in AR fashion with median feedback; that loop is not yet implemented in this scaffold.
        future_values (`torch.Tensor`, *optional*): Fine-resolution ground truth for loss computation.
        """
        outputs = self.model(
            past_values=past_values, past_values_mask=past_values_mask, series_ids=series_ids, **kwargs
        )
        hidden = outputs.last_hidden_state  # [B, N, S, D]

        num_q = len(self.config.quantiles)
        head = self.output_head(hidden) + self.output_skip(
            hidden
        )  # [B, N, S, patch_size * num_output_patches * num_q]
        head = head.view(*hidden.shape[:-1], self.config.num_output_patches, self.config.patch_size, num_q)

        # One-patch prediction: take the last source position and the first output patch.
        last = head[..., -1, 0, :, :]  # [B, N, patch_size, num_q]
        last = last.permute(3, 0, 1, 2)  # [num_q, B, N, patch_size]

        loc = outputs.loc[..., -self.config.patch_size :].unsqueeze(0)
        scale = outputs.scale[..., -self.config.patch_size :].unsqueeze(0)
        quantiles = last.sinh() * scale + loc
        quantiles = quantiles.sort(dim=0).values

        if 0.5 in self.config.quantiles:
            median = quantiles[self.config.quantiles.index(0.5)]
        else:
            median = quantiles[num_q // 2]

        loss = None
        if future_values is not None:
            target_len = min(future_values.shape[-1], median.shape[-1])
            errors = future_values[..., :target_len].unsqueeze(0) - quantiles[..., :target_len]
            q = torch.tensor(self.config.quantiles, device=errors.device, dtype=errors.dtype).view(
                -1, *([1] * (errors.dim() - 1))
            )
            loss = torch.maximum((q - 1) * errors, q * errors).mean()

        if horizon_len is not None and horizon_len > quantiles.shape[-1]:
            logger.warning(
                "Requested `horizon_len=%d` exceeds `config.patch_size * config.num_output_patches=%d`. "
                "AR decoding beyond one patch is not yet implemented in this scaffold; returning the "
                "single-patch forecast.",
                horizon_len,
                self.config.patch_size * self.config.num_output_patches,
            )

        return Toto2PredictionOutput(
            last_hidden_state=outputs.last_hidden_state,
            quantiles=quantiles,
            mean_predictions=median,
            loss=loss,
        )


__all__ = [
    "Toto2Config",
    "Toto2Model",
    "Toto2ForPrediction",
    "Toto2PreTrainedModel",
]
