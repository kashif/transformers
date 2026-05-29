# Copyright 2025 The HuggingFace Team. All rights reserved.
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
"""
Tests for ``PreTrainedTokenizerBase.get_renderer`` and ``render_conversation`` — the optional,
non-Jinja rendering path backed by the ``renderers`` package.

The multi-turn ``bridge_to_next_turn`` case exercises the property that the Jinja
``apply_chat_template`` path cannot guarantee: a per-family renderer extends a sampled token
stream verbatim instead of re-encoding it. Adapted from
https://gist.github.com/mikasenghaas/e336d15761cf49af5e4eb662356a5d78.
"""

import tempfile
import unittest

import transformers.tokenization_utils_base as tokenization_utils_base
from transformers import AutoRenderer, AutoTokenizer
from transformers.testing_utils import require_renderers
from transformers.tokenization_utils_base import RenderedTokens, _DefaultJinjaRenderer


CALC_TOOL = {
    "type": "function",
    "function": {
        "name": "calc",
        "description": "Compute an arithmetic expression.",
        "parameters": {
            "type": "object",
            "properties": {"expr": {"type": "string"}},
            "required": ["expr"],
        },
    },
}

TOOL_CALL_MESSAGES = [
    {"role": "user", "content": "What's 2+2?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": "calc", "arguments": {"expr": "2+2"}}}],
    },
]


class RendererFallbackTest(unittest.TestCase):
    """Behaviour that does not depend on the renderers package being importable."""

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    def test_passes_through_an_existing_renderer_object(self):
        sentinel = object()
        self.assertIs(self.tokenizer.get_renderer(sentinel), sentinel)

    def test_falls_back_to_builtin_when_renderers_unavailable(self):
        original = tokenization_utils_base.is_renderers_available
        tokenization_utils_base.is_renderers_available = lambda: False
        try:
            renderer = self.tokenizer.get_renderer()
            self.assertIsInstance(renderer, _DefaultJinjaRenderer)
            # The fallback cannot prove a byte-for-byte extension, so it declines the bridge.
            self.assertIsNone(renderer.bridge_to_next_turn([1], [2], [{"role": "tool", "content": "4"}]))
            self.assertEqual(renderer.get_stop_token_ids(), [self.tokenizer.eos_token_id])
            # strict refuses the non-bridging fallback instead of silently returning it.
            with self.assertRaises(ValueError):
                self.tokenizer.get_renderer(strict=True)
        finally:
            tokenization_utils_base.is_renderers_available = original

    def test_fallback_render_conversation_returns_aligned_message_indices(self):
        original = tokenization_utils_base.is_renderers_available
        tokenization_utils_base.is_renderers_available = lambda: False
        try:
            messages = [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "again"},
            ]
            rendered = self.tokenizer.render_conversation(messages, add_generation_prompt=True)
            self.assertIsInstance(rendered, RenderedTokens)
            # One message index per token; the generation prompt is structural scaffolding (-1).
            self.assertEqual(len(rendered.token_ids), len(rendered.message_indices))
            self.assertEqual(set(rendered.message_indices), {0, 1, 2, -1})
            # The token ids match apply_chat_template exactly (the fallback wraps it).
            expected = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_dict=False
            )
            self.assertEqual(rendered.token_ids, list(expected))
        finally:
            tokenization_utils_base.is_renderers_available = original


@require_renderers
class RendererTest(unittest.TestCase):
    """Behaviour when the optional renderers package is installed."""

    def test_resolves_per_family_renderer(self):
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        renderer = tokenizer.get_renderer()
        # Qwen3 has a hand-coded renderer; it is not the generic apply_chat_template fallback.
        self.assertEqual(type(renderer).__name__, "Qwen3Renderer")

    def test_unregistered_model_uses_renderers_default(self):
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        renderer = tokenizer.get_renderer()
        self.assertEqual(type(renderer).__name__, "DefaultRenderer")

    def test_strict_rejects_generic_fallback_but_a_declaration_satisfies_it(self):
        # A model that auto-resolves to the generic DefaultRenderer fails strict resolution: silently
        # accepting it is how multi-turn token corruption slips into RL training unnoticed.
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        with self.assertRaises(ValueError):
            tokenizer.get_renderer(strict=True)
        # A per-family renderer (here, the model's own) satisfies strict resolution.
        tokenizer3 = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        self.assertEqual(type(tokenizer3.get_renderer(strict=True)).__name__, "Qwen3Renderer")

    def test_render_conversation_matches_apply_chat_template_for_default_renderer(self):
        # The renderers DefaultRenderer wraps apply_chat_template, so token ids must agree.
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        rendered = tokenizer.render_conversation(messages)
        expected = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False)
        self.assertEqual(list(rendered.token_ids), list(expected))
        self.assertEqual(len(rendered.token_ids), len(rendered.message_indices))

    def test_bridge_extends_sampled_stream_byte_for_byte(self):
        # The core renderer contract a Jinja template cannot offer: a per-family bridge extends
        # previous_prompt_ids + previous_completion_ids verbatim, so a sampled stream is never
        # re-encoded. Qwen3 has such a bridge; the generic fallback returns None instead.
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        renderer = tokenizer.get_renderer()

        prompt_ids = renderer.render_ids(TOOL_CALL_MESSAGES[:1], tools=[CALC_TOOL], add_generation_prompt=True)
        full = renderer.render_ids(TOOL_CALL_MESSAGES, tools=[CALC_TOOL])
        completion_ids = full[len(prompt_ids) :]

        bridged = renderer.bridge_to_next_turn(
            previous_prompt_ids=prompt_ids,
            previous_completion_ids=completion_ids,
            new_messages=[{"role": "tool", "name": "calc", "content": "4"}],
            tools=[CALC_TOOL],
        )
        self.assertIsNotNone(bridged, "Qwen3 renderer should be able to bridge a tool turn")
        bridged_ids = list(bridged.token_ids)
        prev = list(prompt_ids) + list(completion_ids)
        self.assertEqual(bridged_ids[: len(prev)], prev, "bridge must extend the sampled stream verbatim")

    def test_full_token_in_token_out_loop_is_reachable(self):
        # The whole renderer protocol — not just render — is usable from the tokenizer entry point:
        # render_ids -> get_stop_token_ids -> parse_response -> bridge_to_next_turn, the complete
        # TITO loop a multi-turn RL trainer drives.
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        renderer = tokenizer.get_renderer()

        prompt_ids = renderer.render_ids([{"role": "user", "content": "hi"}], add_generation_prompt=True)
        self.assertGreater(len(prompt_ids), 0)
        self.assertIn(tokenizer.eos_token_id, renderer.get_stop_token_ids())

        # Treat the rendered assistant turn as a "sampled" completion and round-trip it.
        full = renderer.render_ids([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}])
        completion_ids = full[len(prompt_ids) :]
        parsed = renderer.parse_response(completion_ids)
        self.assertEqual(parsed.content, "hello")

    def test_render_conversation_exposes_rich_rendered_tokens(self):
        # With renderers installed, render_conversation returns the renderer's own RenderedTokens,
        # so the richer per-token signals RL training needs (e.g. sampled_mask for loss/length
        # penalties, message_roles) are available through the tokenizer entry point, not only token ids.
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        rendered = tokenizer.render_conversation(
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        )
        for field in ("token_ids", "message_indices", "sampled_mask", "message_roles"):
            self.assertTrue(hasattr(rendered, field), f"RenderedTokens should expose {field}")

    def test_renderer_declaration_round_trips_and_overrides_auto_resolution(self):
        # A model author can declare a renderer in tokenizer_config.json. It must survive a
        # save/load round-trip and take precedence over name-based auto-resolution.
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        self.assertEqual(type(tokenizer.get_renderer()).__name__, "Qwen3Renderer")  # auto-resolved

        tokenizer._renderer = "default"  # author opts into the generic renderer instead
        with tempfile.TemporaryDirectory() as tmp_dir:
            tokenizer.save_pretrained(tmp_dir)
            reloaded = AutoTokenizer.from_pretrained(tmp_dir)
        self.assertEqual(reloaded._renderer, "default")
        self.assertEqual(type(reloaded.get_renderer()).__name__, "DefaultRenderer")


class AutoRendererTest(unittest.TestCase):
    def test_cannot_be_instantiated_directly(self):
        with self.assertRaises(OSError):
            AutoRenderer()

    @require_renderers
    def test_from_pretrained_resolves_per_family(self):
        self.assertEqual(type(AutoRenderer.from_pretrained("Qwen/Qwen3-0.6B")).__name__, "Qwen3Renderer")
        self.assertEqual(
            type(AutoRenderer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")).__name__, "DefaultRenderer"
        )


if __name__ == "__main__":
    unittest.main()
