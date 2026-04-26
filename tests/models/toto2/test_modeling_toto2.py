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

"""Tests for the Toto2 scaffold.

The model wires into `ModelTesterMixin` so that the repo-wide attention-backend
equivalence suite (`test_eager_matches_sdpa_inference` × 24 variants,
`test_flash_attn_2_inference_equivalence` × padding sides,
`test_flash_attn_kernels_inference_equivalence` for
`kernels-community/flash-attn3`, `test_flash_attn_kernels_mps_inference_equivalence`
for `kernels-community/metal-flash-sdpa`, and the FA3/FA4 variants) all run
against Toto2. Conversion of the official `Datadog/Toto-2.0-4m` weights is
TODO; a slow integration test with the real checkpoint will land alongside that.
"""

import random
import unittest

import torch
from parameterized import parameterized

from transformers import Toto2Config, is_torch_available
from transformers.testing_utils import require_torch, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION,
    ModelTesterMixin,
    floats_tensor,
)


if is_torch_available():
    from transformers import Toto2ForPrediction, Toto2Model


class Toto2ModelTester:
    def __init__(
        self,
        parent,
        batch_size: int = 2,
        num_input_channels: int = 3,
        patch_size: int = 16,
        context_length: int = 128,
        hidden_size: int = 64,
        intermediate_size: int = 128,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 4,
        num_key_value_heads: int = 4,
        head_dim: int = 16,
        layer_group_size: int = 2,
        num_variate_layers_per_group: int = 1,
        variate_layer_first: bool = False,
        qk_norm: bool = False,
        per_dim_scale: bool = True,
        use_xpos: bool = True,
        is_training: bool = True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_input_channels = num_input_channels
        self.patch_size = patch_size
        self.context_length = context_length
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
        self.is_training = is_training

        # `Toto2ForPrediction.forward` runs the stack over `[context | pred_zero_patches]`, so the
        # hidden-state / attention tensors checked by `ModelTesterMixin` cover `context_patches + 1`
        # (the default single-block decode emits one output patch).
        self.seq_length = context_length // patch_size + 1

    def get_config(self):
        return Toto2Config(
            patch_size=self.patch_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            layer_group_size=self.layer_group_size,
            num_variate_layers_per_group=self.num_variate_layers_per_group,
            variate_layer_first=self.variate_layer_first,
            qk_norm=self.qk_norm,
            per_dim_scale=self.per_dim_scale,
            use_xpos=self.use_xpos,
            context_length=self.context_length,
        )

    def get_pipeline_config(self):
        return self.get_config()

    def prepare_config_and_inputs(self):
        past_values = torch.randn(self.batch_size, self.context_length, self.num_input_channels, device=torch_device)
        return self.get_config(), past_values

    def prepare_config_and_inputs_for_common(self):
        config, past_values = self.prepare_config_and_inputs()
        inputs_dict = {"past_values": past_values}
        return config, inputs_dict


@require_torch
class Toto2ModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (Toto2ForPrediction,) if is_torch_available() else ()
    test_resize_embeddings = False
    is_encoder_decoder = False
    test_inputs_embeds = False
    test_all_params_have_gradient = False
    test_headmasking = False
    test_pruning = False
    test_missing_keys = False
    test_model_parallel = False

    def setUp(self):
        self.model_tester = Toto2ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Toto2Config, has_text_modality=False)

    # ------------------------------------------------------------------
    # Toto2-specific behaviour.
    # ------------------------------------------------------------------

    def test_create_and_run_model(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = Toto2ForPrediction(config).to(torch_device).eval()
        results = model(**inputs_dict)
        num_q = len(config.quantiles)
        expected = (
            self.model_tester.batch_size,
            config.patch_size * config.num_output_patches,
            self.model_tester.num_input_channels,
            num_q,
        )
        self.assertEqual(tuple(results.quantiles.shape), expected)
        self.assertEqual(tuple(results.prediction_outputs.shape), expected[:3])
        diffs = results.quantiles.diff(dim=-1)
        self.assertTrue((diffs >= -1e-5).all(), f"quantiles should be monotone; min diff {diffs.min().item():.2e}")

    def test_alternating_stack_layer_types(self):
        config = self.model_tester.get_config()
        model = Toto2Model(config).eval()
        expected = [False, True]  # one group of 2 layers, variate-last
        actual = [layer.is_variate for layer in model.layers]
        self.assertEqual(actual, expected)

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)
        if return_labels:
            rng = random.Random(42)
            inputs_dict["future_values"] = floats_tensor(
                [
                    self.model_tester.batch_size,
                    self.model_tester.patch_size,  # one output patch
                    self.model_tester.num_input_channels,
                ],
                rng=rng,
            )
        return inputs_dict

    # ------------------------------------------------------------------
    # Overrides for common tests that do not apply to a 3-D multivariate forecaster.
    # ------------------------------------------------------------------

    @unittest.skip(reason="Toto2 has no input embedding table (projects patched values directly).")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="Toto2's attention mask is built internally per layer-type (causal time / open variate).")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    def test_eager_matches_sdpa_inference(
        self, name, dtype, padding_side, use_attention_mask, output_attentions, enable_kernels
    ):
        """The generic parameterization injects external attention masks and mutates RMSNorm eps, which
        is not compatible with Toto2's internally-built causal mask. We verify eager↔SDPA equivalence on the
        native `(B, T, N)` forward instead."""
        if not self.all_model_classes[0]._supports_sdpa:
            self.skipTest("Model does not support SDPA")
        torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
        tolerance = {torch.float32: 1e-4, torch.bfloat16: 5e-3, torch.float16: 5e-3}[torch_dtype]

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        eager = Toto2ForPrediction._from_config(config, attn_implementation="eager")
        eager.to(dtype=torch_dtype, device=torch_device).eval()
        sdpa = Toto2ForPrediction._from_config(config, attn_implementation="sdpa")
        sdpa.load_state_dict(eager.state_dict())
        sdpa.to(dtype=torch_dtype, device=torch_device).eval()

        past_values = inputs_dict["past_values"].to(dtype=torch_dtype, device=torch_device)
        with torch.no_grad():
            out_e = eager(past_values=past_values)
            out_s = sdpa(past_values=past_values)
        diff = (out_e.prediction_outputs - out_s.prediction_outputs).abs().max().item()
        self.assertLess(diff, tolerance, f"eager vs sdpa max diff {diff:.2e}")

    def test_past_observed_mask_passed_through(self):
        config = self.model_tester.get_config()
        model = Toto2ForPrediction(config).eval()
        past_values = torch.randn(1, 64, 2)
        mask = torch.ones_like(past_values, dtype=torch.bool)
        mask[:, :16, :] = False
        with torch.no_grad():
            out = model(past_values=past_values, past_observed_mask=mask)
        self.assertEqual(tuple(out.prediction_outputs.shape), (1, config.patch_size, 2))
