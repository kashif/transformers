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
from transformers.testing_utils import require_torch, torch_device


if is_torch_available():
    from transformers import Toto2ForPrediction, Toto2Model


def _tiny_config(**overrides):
    defaults = dict(
        patch_size=16,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        layer_group_size=4,
        num_variate_layers_per_group=1,
        variate_layer_first=False,
        qk_norm=False,
        per_dim_scale=True,
        use_xpos=True,
        context_length=256,
    )
    defaults.update(overrides)
    return Toto2Config(**defaults)


@require_torch
class Toto2ScaffoldTest(unittest.TestCase):
    def test_build_and_forward(self):
        config = _tiny_config()
        model = Toto2ForPrediction(config).to(torch_device).eval()

        batch, num_var, time = 2, 3, 128
        past_values = torch.randn(batch, num_var, time, device=torch_device)
        with torch.no_grad():
            out = model(past_values=past_values, horizon_len=config.patch_size)

        num_q = len(config.quantiles)
        self.assertEqual(tuple(out.quantiles.shape), (num_q, batch, num_var, config.patch_size))
        self.assertEqual(tuple(out.mean_predictions.shape), (batch, num_var, config.patch_size))
        # Quantiles must be sorted along the quantile axis.
        diffs = out.quantiles.diff(dim=0)
        self.assertTrue((diffs >= -1e-5).all(), f"quantiles should be monotone; min diff {diffs.min().item():.2e}")

    def test_alternating_stack_layer_types(self):
        config = _tiny_config()
        model = Toto2Model(config).eval()
        # With num_variate_layers_per_group=1 and variate_layer_first=False, the last layer of each group is variate.
        expected = [False, False, False, True]  # single group of 4 layers
        actual = [layer.is_variate for layer in model.layers]
        self.assertEqual(actual, expected)

    def test_past_values_mask_passed_through(self):
        config = _tiny_config()
        model = Toto2ForPrediction(config).eval()

        past_values = torch.randn(1, 2, 64)
        mask = torch.ones_like(past_values, dtype=torch.bool)
        mask[..., :16] = False  # mask the first patch

        with torch.no_grad():
            out = model(past_values=past_values, past_values_mask=mask)
        self.assertEqual(tuple(out.mean_predictions.shape), (1, 2, config.patch_size))
