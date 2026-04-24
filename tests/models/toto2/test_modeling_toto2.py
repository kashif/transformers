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

"""Tests for the Toto2 scaffold. Coverage is intentionally minimal: the scaffold can be imported, an
instance can be built with a tiny config, and forward on random multivariate input produces the expected
quantile / median shapes. Conversion of the official `Datadog/Toto-2.0-4m` weights is TODO — richer
coverage (kernel parity, slow integration test with the real checkpoint) will land alongside that."""

import unittest

import torch

from transformers import Toto2Config, is_torch_available
from transformers.testing_utils import require_flash_attn, require_torch, require_torch_accelerator, torch_device


if is_torch_available():
    from transformers import Toto2ForPrediction, Toto2Model


def _tiny_config(**overrides):
    defaults = {
        "patch_size": 16,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 16,
        "layer_group_size": 4,
        "num_variate_layers_per_group": 1,
        "variate_layer_first": False,
        "qk_norm": False,
        "per_dim_scale": True,
        "use_xpos": True,
        "context_length": 256,
    }
    defaults.update(overrides)
    return Toto2Config(**defaults)


@require_torch
class Toto2ScaffoldTest(unittest.TestCase):
    def test_build_and_forward(self):
        config = _tiny_config()
        model = Toto2ForPrediction(config).to(torch_device).eval()

        batch, time, num_var = 2, 128, 3
        past_values = torch.randn(batch, time, num_var, device=torch_device)
        with torch.no_grad():
            out = model(past_values=past_values, prediction_length=config.patch_size)

        num_q = len(config.quantiles)
        # (B, prediction_length, N, num_q) — matches PatchTST-style axis order.
        self.assertEqual(tuple(out.quantiles.shape), (batch, config.patch_size, num_var, num_q))
        # (B, prediction_length, N) — median/point forecast.
        self.assertEqual(tuple(out.prediction_outputs.shape), (batch, config.patch_size, num_var))
        # Quantiles must be sorted along the quantile axis (last dim).
        diffs = out.quantiles.diff(dim=-1)
        self.assertTrue((diffs >= -1e-5).all(), f"quantiles should be monotone; min diff {diffs.min().item():.2e}")

    def test_alternating_stack_layer_types(self):
        config = _tiny_config()
        model = Toto2Model(config).eval()
        # With num_variate_layers_per_group=1 and variate_layer_first=False, the last layer of each group is variate.
        expected = [False, False, False, True]  # single group of 4 layers
        actual = [layer.is_variate for layer in model.layers]
        self.assertEqual(actual, expected)

    def _attn_kernel_equivalence(self, impl: str, dtype=torch.float32, tolerance: float = 1e-4):
        config = _tiny_config()
        eager = Toto2ForPrediction._from_config(config, attn_implementation="eager")
        eager.to(dtype=dtype, device=torch_device).eval()
        other = Toto2ForPrediction._from_config(config, attn_implementation=impl)
        other.load_state_dict(eager.state_dict())
        other.to(dtype=dtype, device=torch_device).eval()

        past_values = torch.randn(2, 128, 3, dtype=dtype, device=torch_device)
        with torch.no_grad():
            out_eager = eager(past_values=past_values)
            out_other = other(past_values=past_values)
        diff = (out_eager.prediction_outputs - out_other.prediction_outputs).abs().max().item()
        self.assertLess(diff, tolerance, f"{impl} vs eager prediction_outputs max diff {diff:.2e}")

    def test_eager_matches_sdpa(self):
        self._attn_kernel_equivalence("sdpa", dtype=torch.float32, tolerance=1e-4)

    @require_flash_attn
    @require_torch_accelerator
    def test_eager_matches_flash_attention_2(self):
        self._attn_kernel_equivalence("flash_attention_2", dtype=torch.bfloat16, tolerance=5e-3)

    def test_past_observed_mask_passed_through(self):
        config = _tiny_config()
        model = Toto2ForPrediction(config).eval()

        # (B, sequence_length, num_input_channels) per transformers convention.
        past_values = torch.randn(1, 64, 2)
        mask = torch.ones_like(past_values, dtype=torch.bool)
        mask[:, :16, :] = False  # mask the first patch

        with torch.no_grad():
            out = model(past_values=past_values, past_observed_mask=mask)
        self.assertEqual(tuple(out.prediction_outputs.shape), (1, config.patch_size, 2))
