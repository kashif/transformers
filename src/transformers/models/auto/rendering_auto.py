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
"""Auto Renderer class."""

from .tokenization_auto import AutoTokenizer


__all__ = ["AutoRenderer"]


class AutoRenderer:
    """
    This is a generic renderer class that is instantiated as one of the renderer classes of the library when created
    with the [`~AutoRenderer.from_pretrained`] class method.

    A *renderer* is the opt-in, non-Jinja counterpart to a chat template: a Python object that renders messages to
    token ids, parses sampled token ids back to structured messages, and extends a multi-turn rollout *without*
    re-encoding model-sampled history (Token-In, Token-Out). It is aimed at multi-turn RL training and serving, where
    re-applying the chat template every turn drifts the tokens the policy actually sampled and loses the per-turn
    boundaries needed for loss masking. Renderers are provided by the optional
    [`renderers`](https://github.com/PrimeIntellect-ai/renderers) package; without it, a built-in fallback wrapping
    `apply_chat_template` is returned.

    This class cannot be instantiated directly using `__init__()` (it raises an error).
    """

    def __init__(self, *args, **kwargs):
        raise OSError(
            "AutoRenderer is designed to be instantiated "
            "using the `AutoRenderer.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, *inputs, renderer=None, strict=False, trust_remote_code=False, **kwargs
    ):
        r"""
        Instantiate the renderer for a model from a pretrained model vocabulary, resolved the same way as the model's
        tokenizer (in-library renderers registry via a `"renderer"` declaration in `tokenizer_config.json`, custom Hub
        code via `auto_map["AutoRenderer"]`, or name-based auto-detection — see
        [`~PreTrainedTokenizerBase.get_renderer`]).

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                A model id of a model hosted on the Hub, or a path to a directory containing tokenizer files.
            renderer (`str`, renderer config, or renderer object, *optional*):
                Override renderer resolution with a renderer name or config to construct, or an already-built object.
            strict (`bool`, *optional*, defaults to `False`):
                Require a per-family renderer and raise if only the generic `apply_chat_template` fallback is
                available. Recommended for multi-turn RL training. See [`~PreTrainedTokenizerBase.get_renderer`].
            trust_remote_code (`bool`, *optional*, defaults to `False`):
                Whether to allow loading the tokenizer and/or a renderer defined in custom code on the Hub.
            inputs, kwargs:
                Forwarded to [`AutoTokenizer.from_pretrained`].

        Examples:

        ```python
        >>> from transformers import AutoRenderer

        >>> renderer = AutoRenderer.from_pretrained("Qwen/Qwen3-8B")  # doctest: +SKIP
        >>> prompt_ids = renderer.render_ids(  # doctest: +SKIP
        ...     [{"role": "user", "content": "hi"}], add_generation_prompt=True
        ... )
        ```
        """
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, *inputs, trust_remote_code=trust_remote_code, **kwargs
        )
        return tokenizer.get_renderer(renderer, strict=strict, trust_remote_code=trust_remote_code, **kwargs)
