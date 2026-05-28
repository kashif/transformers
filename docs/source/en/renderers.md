<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Renderers

A chat template is a *string* contract: it maps a list of messages to text, which is then tokenized. That is exactly
what you want for single-turn inference. It quietly breaks down for **multi-turn reinforcement learning**, where the
loop samples a turn, runs a tool or environment step, and continues — repeatedly — and you must train on the tokens the
policy *actually sampled*.

A *renderer* is the opt-in, non-Jinja counterpart to a chat template. It is a Python object that:

- renders messages to **token ids** (not a string),
- parses sampled token ids back into a structured assistant message,
- and extends a multi-turn rollout *without re-encoding* the model's sampled tokens (Token-In, Token-Out).

Renderers are provided by the optional [`renderers`](https://github.com/PrimeIntellect-ai/renderers) package. When it
is not installed, Transformers returns a built-in fallback that wraps [`~PreTrainedTokenizerBase.apply_chat_template`],
so the API always exists.

## Why a chat template is not enough for RL

The natural RL loop keeps the conversation as a list of messages and re-renders it with `apply_chat_template` every
turn. This is correct for one turn and subtly wrong across turns:

- **Re-tokenization drift.** Decoding then re-encoding is not the identity (byte-pair merges are not stable across
  boundaries, JSON whitespace and argument ordering are negotiable). Re-rendering the history can yield slightly
  different ids than the ones the model sampled, so you backpropagate on tokens the policy never produced.
- **Lost per-turn boundaries.** After rendering the whole conversation you no longer know which tokens were sampled by
  the assistant and which are template scaffolding — exactly what you need for the loss mask.
- **Turn-separator boundaries.** A real sampler stops *at* the end-of-turn token; the template adds trailing
  scaffolding (e.g. ChatML emits `<|im_end|>\n`). Naively stitching the next turn drops or duplicates that boundary
  token. On GLM, the engine stops on `<|observation|>` and the next-turn bridge prepends it again — doubling it.
- **Thinking lifecycle.** Many models strip `<think>...</think>` from past turns. A strictly growing token buffer
  trains "preserve thinking", which then diverges from how the model is served.
- **Truncation without an end-of-turn token.** If a turn hits `max_tokens`, there is no stop token to stitch from, and
  how to continue is model-specific.

These failures are silent — a template-level check does not catch them, and the only fully reliable check is to run a
training. A renderer makes the messages-to-tokens boundary an inspectable, unit-tested Python object instead, with a
per-family bridge that extends the sampled stream verbatim (or declines, so the caller falls back to a full render).

## Loading a renderer

[`AutoRenderer`] resolves the renderer for a model the same way its tokenizer is resolved:

```python
>>> from transformers import AutoRenderer

>>> renderer = AutoRenderer.from_pretrained("Qwen/Qwen3-8B")  # doctest: +SKIP
>>> prompt_ids = renderer.render_ids(  # doctest: +SKIP
...     [{"role": "user", "content": "What is 2+2?"}], add_generation_prompt=True
... )
>>> # Feed prompt_ids to a Token-In, Token-Out endpoint; it returns the sampled completion_ids.
>>> parsed = renderer.parse_response(completion_ids)  # doctest: +SKIP
```

Equivalently, from a tokenizer you already have:

```python
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")  # doctest: +SKIP
>>> renderer = tokenizer.get_renderer()  # doctest: +SKIP
```

For multi-turn RL training, pass `strict=True`. Resolution can silently land on the generic
`apply_chat_template` fallback — which cannot guarantee a safe `bridge_to_next_turn` — when no per-family renderer is
installed or declared for the model. That silent degradation is exactly how token-level corruption slips into training
unnoticed. `strict=True` raises instead, naming the fix (install `renderers`, or declare a `"renderer"` for the model):

```python
>>> renderer = AutoRenderer.from_pretrained("zai-org/GLM-4.5", strict=True)  # doctest: +SKIP
```

For the next turn, extend the previous sampled stream instead of re-rendering the history:

```python
>>> next_prompt_ids = renderer.bridge_to_next_turn(  # doctest: +SKIP
...     previous_prompt_ids=prompt_ids,
...     previous_completion_ids=completion_ids,
...     new_messages=[{"role": "tool", "content": "4"}],
... )
```

`bridge_to_next_turn` guarantees its result starts with `previous_prompt_ids + previous_completion_ids` byte-for-byte.
If a per-family renderer cannot prove that (or none is installed), it returns `None` and you fall back to a full render.

## Rendering with per-token attribution

[`~PreTrainedTokenizerBase.render_conversation`] returns token ids together with a `message_indices` array — one entry
per token, giving the index of the source message (`-1` for structural scaffolding). This is enough to build a
per-token loss mask in a single render, without `{% generation %}` markers in the template.

```python
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")  # doctest: +SKIP
>>> rendered = tokenizer.render_conversation(  # doctest: +SKIP
...     [
...         {"role": "user", "content": "What is 2+2?"},
...         {"role": "assistant", "content": "It is 4."},
...     ]
... )
>>> rendered.token_ids  # doctest: +SKIP
>>> rendered.message_indices  # token i belongs to message rendered.message_indices[i]  # doctest: +SKIP
```

## Declaring a renderer on the Hub

A model author can ship a renderer with the model so it is selected automatically, just like the model's configuration
and tokenizer. The simplest form is a declaration in `tokenizer_config.json` resolved against the installed
`renderers` package:

```json
{ "renderer": "qwen3" }
```

A model with custom code on the Hub can instead point at its own renderer class, loaded with `trust_remote_code=True`:

```json
{ "auto_map": { "AutoRenderer": "rendering_mymodel.MyModelRenderer" } }
```

A declaration takes precedence over name-based auto-detection, and survives a `save_pretrained` / `from_pretrained`
round-trip.

## Installing renderers

```bash
pip install transformers[renderers]
```

Without the extra, [`AutoRenderer`] and [`~PreTrainedTokenizerBase.get_renderer`] still work and return a fallback that
wraps `apply_chat_template` — it renders messages and recovers `message_indices`, but has no per-family parsing and
declines `bridge_to_next_turn`. Install `renderers` for the hand-coded per-family renderers (`qwen3`, `glm-5`,
`deepseek-v3`, `gpt-oss`, and others) that handle the multi-turn cases above.

## API

[[autodoc]] AutoRenderer
    - from_pretrained
