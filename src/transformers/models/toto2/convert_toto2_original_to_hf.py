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

"""Convert a Datadog/Toto-2.0-* checkpoint to the transformers layout.

What the conversion does (boundaries of the translation, in order):

1. Bake the u-μP forward scaling into every `nn.Linear` weight/bias. The original uses
   `uu.Linear` which multiplies its output by `1/sqrt(fan_in)` (or `1/fan_in` for the readout linear in
   the output head) at forward time. `Toto2Attention` / `Toto2MLP` / the patch/head ResidualMLPs use plain
   `nn.Linear`, so we pre-scale weights and biases by that factor.
2. Split the fused `in_proj` (`[Q | K | V]` packed along dim 0) into `q_proj` / `k_proj` / `v_proj`.
3. Split the fused MLP `fc1` (2 * intermediate_size along dim 0) into Llama-style `gate_proj` / `up_proj`.
   Upstream computes `gate * silu(x)` where `gate=first_half, x=second_half`, while Llama computes
   `silu(gate_proj(x)) * up_proj(x)` — so the mapping is `up_proj = first_half`, `gate_proj = second_half`.
4. Permute Q and K rows within the rotated sub-head so that transformers' half-split rotary (first and
   second half of the rotary dim are treated as the real/imag pair) matches Datadog's interleaved rotary
   (consecutive even/odd pairs are the real/imag pair).
5. Fold the u-μP `per_dim_scale` constant (≈0.52103) into the stored raw scale parameter so that
   `softplus(p)/log(2)` in `Toto2PerDimScale` directly reproduces the original's effective multiplier.
6. Bake the `OutputResidualMLP` trailing `silu_glu` forward scale (only applies to the output head). Not
   required for Toto-2.0-4m which uses a plain `gate * silu(x)` in the ResidualMLP (no `silu_glu`). We
   still emit the factor explicitly as a TODO marker for the larger checkpoints if they use it.

Sample usage:

    python src/transformers/models/toto2/convert_toto2_original_to_hf.py \\
        --output_dir /tmp/toto2-4m-hf \\
        --huggingface_repo_id Datadog/Toto-2.0-4m
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as load_safetensors

from transformers import Toto2Config, Toto2ForPrediction


# u-μP constant that `Toto2PerDimScale.forward` no longer applies (bake it into the stored parameter).
PER_DIM_SCALE_OUTPUT_SCALE = 0.52103
# Forward multiplier that `unit_scaling.functional.silu` applies on top of `F.silu` (empirical, constant).
U_SILU_FORWARD = 1.7667829990386963
# τ = 1 residual_add forward factor `1 / sqrt(1 + 1^2) = 1/sqrt(2)`.
RESIDUAL_ADD_TAU_1 = 1.0 / math.sqrt(2.0)


def _bake_linear(
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    scale_power: float = 0.5,
    extra_weight_factor: float = 1.0,
    extra_bias_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return (W', b') that reproduce `uu.Linear` forward as plain `nn.Linear`.

    `uu.Linear` output = `F.linear(input, W, b) * (1 / fan_in ** scale_power)`.
    `extra_*_factor` bake additional downstream forward factors that cannot be represented as a plain
    `nn.Linear` (e.g. a trailing `residual_add(tau=1)` divides by `sqrt(2)`; a preceding `U.silu` multiplies
    by ~1.767 **before** the next linear, which is equivalent to baking that factor into this linear's `W`
    but *not* its `b`).
    """
    fan_in = weight.shape[1]
    base = 1.0 / (fan_in**scale_power)
    new_w = weight.clone() * base * extra_weight_factor
    new_b = None if bias is None else bias.clone() * base * extra_bias_factor
    return new_w, new_b


def _interleaved_to_half_split(weight: torch.Tensor, rotary_dim: int, head_dim: int, num_heads: int) -> torch.Tensor:
    """Permute the first `rotary_dim` dims of each head in `weight[out, in]` so that transformers' half-split
    rotate pattern (rotate_half(x) = cat(-x[:, half:], x[:, :half])) produces the same output as Datadog's
    interleaved rotate (pair consecutive even/odd coordinates)."""
    out, inp = weight.shape
    assert out == num_heads * head_dim, (out, num_heads, head_dim)
    w = weight.view(num_heads, head_dim, inp)
    rot = w[:, :rotary_dim, :]  # (H, rotary_dim, in)
    pass_through = w[:, rotary_dim:, :]

    # Datadog layout: rot[:, 2i]    is the "real" part of pair i
    #                 rot[:, 2i + 1] is the "imag" part of pair i
    # Llama layout:   rot[:, i]           is the "real" part of pair i
    #                 rot[:, rotary_dim/2 + i] is the "imag" part
    # so we gather rows [0, 2, 4, ..., 1, 3, 5, ...].
    even_idx = torch.arange(0, rotary_dim, 2)
    odd_idx = torch.arange(1, rotary_dim, 2)
    perm = torch.cat([even_idx, odd_idx], dim=0)  # length = rotary_dim
    rot = rot.index_select(dim=1, index=perm)

    return torch.cat([rot, pass_through], dim=1).reshape(out, inp)


def _build_hf_config(orig_cfg: dict) -> Toto2Config:
    """Map Datadog config.json fields to Toto2Config."""
    return Toto2Config(
        patch_size=orig_cfg["patch_size"],
        hidden_size=orig_cfg["d_model"],
        intermediate_size=orig_cfg["d_ff"],
        num_hidden_layers=orig_cfg["num_layers"],
        num_attention_heads=orig_cfg["num_heads"],
        num_key_value_heads=orig_cfg["num_groups"],
        head_dim=orig_cfg["qk_dim"],
        layer_group_size=orig_cfg["layer_group_size"],
        num_variate_layers_per_group=orig_cfg["num_variate_layers_per_group"],
        variate_layer_first=orig_cfg["variate_layer_first"],
        qk_norm=orig_cfg["qk_norm"],
        per_dim_scale=orig_cfg.get("per_dim_scale", False),
        use_xpos=orig_cfg.get("use_xpos", False),
        attn_bias=orig_cfg["attn_bias"],
        mlp_bias=orig_cfg["mlp_bias"],
        norm_eps=orig_cfg["norm_eps"],
        norm_include_weight=orig_cfg["norm_include_weight"],
        residual_mult=orig_cfg["residual_mult"],
        residual_attn_ratio=orig_cfg["residual_attn_ratio"],
        num_output_patches=orig_cfg.get("num_output_patches", 1),
    )


def convert_state_dict(orig_sd: dict[str, torch.Tensor], cfg: Toto2Config) -> dict[str, torch.Tensor]:
    """Rewrite the Datadog state dict into the transformers key layout."""
    num_heads = cfg.num_attention_heads
    num_kv = cfg.num_key_value_heads
    qk_dim = cfg.head_dim
    v_dim = cfg.head_dim
    rotary_dim = int(round(cfg.head_dim * cfg.partial_rotary_factor))

    new_sd: dict[str, torch.Tensor] = {}

    # ---- Patch projection (InputResidualMLP: linear1 → U.silu → linear2; + skip_proj; wrapped in residual_add τ=1) ----
    # linear1 feeds into U.silu → linear2, so its own μP factor is the only thing to bake here.
    w, b = _bake_linear(orig_sd["patch_proj.linear1.weight"], orig_sd["patch_proj.linear1.bias"])
    new_sd["model.patch_proj.0.weight"], new_sd["model.patch_proj.0.bias"] = w, b
    # linear2: bake μP factor *and* the U.silu forward scale (applied to the input activation → equivalent
    # to multiplying W but not b) *and* the τ=1 residual_add factor (multiplies the whole `y = W x + b`).
    w, b = _bake_linear(
        orig_sd["patch_proj.linear2.weight"],
        orig_sd["patch_proj.linear2.bias"],
        extra_weight_factor=U_SILU_FORWARD * RESIDUAL_ADD_TAU_1,
        extra_bias_factor=RESIDUAL_ADD_TAU_1,
    )
    new_sd["model.patch_proj.2.weight"], new_sd["model.patch_proj.2.bias"] = w, b
    # skip_proj: no silu upstream, only μP + residual_add.
    w, b = _bake_linear(
        orig_sd["patch_proj.skip_proj.weight"],
        orig_sd["patch_proj.skip_proj.bias"],
        extra_weight_factor=RESIDUAL_ADD_TAU_1,
        extra_bias_factor=RESIDUAL_ADD_TAU_1,
    )
    new_sd["model.patch_skip.weight"], new_sd["model.patch_skip.bias"] = w, b

    # ---- Output head (OutputResidualMLP) — linear2 / skip_proj use scale_power=1.0 (LinearReadout). ----
    w, b = _bake_linear(
        orig_sd["output_head.param_projection.proj.linear1.weight"],
        orig_sd["output_head.param_projection.proj.linear1.bias"],
        scale_power=0.5,  # hidden linear
    )
    new_sd["output_head.0.weight"], new_sd["output_head.0.bias"] = w, b
    w, b = _bake_linear(
        orig_sd["output_head.param_projection.proj.linear2.weight"],
        orig_sd["output_head.param_projection.proj.linear2.bias"],
        scale_power=1.0,  # readout linear
        extra_weight_factor=U_SILU_FORWARD * RESIDUAL_ADD_TAU_1,
        extra_bias_factor=RESIDUAL_ADD_TAU_1,
    )
    new_sd["output_head.2.weight"], new_sd["output_head.2.bias"] = w, b
    w, b = _bake_linear(
        orig_sd["output_head.param_projection.proj.skip_proj.weight"],
        orig_sd["output_head.param_projection.proj.skip_proj.bias"],
        scale_power=1.0,
        extra_weight_factor=RESIDUAL_ADD_TAU_1,
        extra_bias_factor=RESIDUAL_ADD_TAU_1,
    )
    new_sd["output_skip.weight"], new_sd["output_skip.bias"] = w, b

    # ---- Transformer layers ----
    for i in range(cfg.num_hidden_layers):
        # Attention: split in_proj into q/k/v, bake μP factor, permute rotary-dim indices for q/k.
        inw = orig_sd[f"transformer.layers.{i}.attn.in_proj.weight"]  # (q + k + v, hidden)
        inb = orig_sd[f"transformer.layers.{i}.attn.in_proj.bias"]
        q_size = num_heads * qk_dim
        k_size = num_kv * qk_dim
        q_w = inw[:q_size]
        k_w = inw[q_size : q_size + k_size]
        v_w = inw[q_size + k_size :]
        q_b = inb[:q_size]
        k_b = inb[q_size : q_size + k_size]
        v_b = inb[q_size + k_size :]

        # Rotary permutation on Q/K (V is not rotated).
        q_w = _interleaved_to_half_split(q_w, rotary_dim, qk_dim, num_heads)
        k_w = _interleaved_to_half_split(k_w, rotary_dim, qk_dim, num_kv)

        # Bias: same permutation applied to the head-dim axis.
        def _perm_bias(bias, rdim, hd, nh):
            b = bias.view(nh, hd)
            even = torch.arange(0, rdim, 2)
            odd = torch.arange(1, rdim, 2)
            perm = torch.cat([even, odd, torch.arange(rdim, hd)])
            return b.index_select(dim=1, index=perm).reshape(-1)

        q_b = _perm_bias(q_b, rotary_dim, qk_dim, num_heads)
        k_b = _perm_bias(k_b, rotary_dim, qk_dim, num_kv)

        for proj_name, raw_w, raw_b in [("q_proj", q_w, q_b), ("k_proj", k_w, k_b), ("v_proj", v_w, v_b)]:
            w, b = _bake_linear(raw_w, raw_b)
            new_sd[f"model.layers.{i}.self_attn.{proj_name}.weight"] = w
            new_sd[f"model.layers.{i}.self_attn.{proj_name}.bias"] = b

        ow = orig_sd[f"transformer.layers.{i}.attn.out_proj.weight"]
        ob = orig_sd[f"transformer.layers.{i}.attn.out_proj.bias"]
        w, b = _bake_linear(ow, ob)
        new_sd[f"model.layers.{i}.self_attn.o_proj.weight"] = w
        new_sd[f"model.layers.{i}.self_attn.o_proj.bias"] = b

        # Per-dim scale: no baking needed. The reference composes `unit_scaling.softplus`
        # (forward scale `1/0.52103`) with `per_dim_scale` (forward scale `0.52103`) — the two cancel, so
        # the effective multiplier applied to Q is `F.softplus(p) / log(2)`, which is exactly what
        # `Toto2PerDimScale.forward` computes with plain `F.softplus`. Store the raw parameter as-is.
        new_sd[f"model.layers.{i}.self_attn.per_dim_scale.per_dim_scale"] = orig_sd[
            f"transformer.layers.{i}.attn._pds.per_dim_scale"
        ].clone()

        # MLP: fc1 packs [gate | up] (Datadog order) → transformers needs up_proj, gate_proj.
        fc1 = orig_sd[f"transformer.layers.{i}.ffn.fc1.weight"]  # (2 * inter, hidden)
        fc2 = orig_sd[f"transformer.layers.{i}.ffn.fc2.weight"]  # (hidden, inter)
        inter = fc1.shape[0] // 2
        datadog_gate, datadog_x = fc1[:inter], fc1[inter:]
        up_w, _ = _bake_linear(datadog_gate, None)  # Datadog's "gate" term, multiplied linearly
        gate_w, _ = _bake_linear(datadog_x, None)  # Datadog's "x" term, fed through silu
        new_sd[f"model.layers.{i}.mlp.up_proj.weight"] = up_w
        new_sd[f"model.layers.{i}.mlp.gate_proj.weight"] = gate_w
        down_w, _ = _bake_linear(fc2, None)
        new_sd[f"model.layers.{i}.mlp.down_proj.weight"] = down_w

        # τ scalars go into our persistent buffers.
        new_sd[f"model.layers.{i}.attn_tau"] = orig_sd[f"transformer.layers.{i}.attn_tau"].clone()
        new_sd[f"model.layers.{i}.mlp_tau"] = orig_sd[f"transformer.layers.{i}.mlp_tau"].clone()

    return new_sd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--huggingface_repo_id", default="Datadog/Toto-2.0-4m")
    parser.add_argument("--safe_serialization", type=bool, default=True)
    args = parser.parse_args()

    local = snapshot_download(repo_id=args.huggingface_repo_id)
    with open(os.path.join(local, "config.json")) as f:
        orig_cfg = json.load(f)
    orig_sd = load_safetensors(os.path.join(local, "model.safetensors"))

    cfg = _build_hf_config(orig_cfg)
    new_sd = convert_state_dict(orig_sd, cfg)

    model = Toto2ForPrediction(cfg)
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    # The RMSNorm layers (`norm_include_weight=False`) have a non-learnable `.weight` buffer that is not in
    # the Datadog state dict — filter those missing keys out of the warning.
    missing = [
        k
        for k in missing
        if not k.endswith("norm1.weight") and not k.endswith("norm2.weight") and not k.endswith("out_norm.weight")
    ]
    if missing:
        print(f"[warn] {len(missing)} missing keys after load (first 5): {missing[:5]}")
    if unexpected:
        print(f"[warn] {len(unexpected)} unexpected keys after load (first 5): {unexpected[:5]}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.save_pretrained(out_dir)
    model.save_pretrained(out_dir, safe_serialization=args.safe_serialization)
    print(f"Saved converted Toto2 checkpoint to {out_dir}")


if __name__ == "__main__":
    main()
