<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2026-04-09 and added to Hugging Face Transformers on 2026-04-19.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Toto 2

## Overview

Toto 2 (Time series Optimized Transformer for Observability, version 2) is Datadog's second-generation multivariate
time-series foundation model. The architecture interleaves causal *time-axis* attention layers with bidirectional
*variate-axis* attention layers (1 variate layer per group of 4 in the released 4M checkpoint), uses μP/u-μP-style
initialization and attention scaling, partial rotary position embeddings with xPos length-extrapolation on the time
axis, and predicts 9 quantile levels per future patch through a residual MLP head. Context is patched (`patch_size=32`
for the 4M checkpoint) and per-patch causally standardized before embedding.

Family checkpoints (Apache 2.0):

| Checkpoint | Parameters |
|---|---|
| [`Datadog/Toto-2.0-4m`](https://huggingface.co/Datadog/Toto-2.0-4m) | 4M |
| [`Datadog/Toto-2.0-22m`](https://huggingface.co/Datadog/Toto-2.0-22m) | 22M |
| [`Datadog/Toto-2.0-313m`](https://huggingface.co/Datadog/Toto-2.0-313m) | 313M |
| [`Datadog/Toto-2.0-1B`](https://huggingface.co/Datadog/Toto-2.0-1B) | 1B |
| [`Datadog/Toto-2.0-2.5B`](https://huggingface.co/Datadog/Toto-2.0-2.5B) | 2.5B |

The original code is at [github.com/DataDog/toto](https://github.com/DataDog/toto).

This model was contributed by [kashif](https://huggingface.co/kashif).

> [!IMPORTANT]
> The current Transformers integration is an **architectural scaffold**: configuration + model classes + forward
> math are in place and produce sensible shapes, but the weight-conversion script and integration tests against
> the released Datadog checkpoints are pending. Outputs from the scaffold with random weights should not be used
> for forecasting until conversion lands.

## Usage example

```python
import torch
from transformers import Toto2Config, Toto2ForPrediction

config = Toto2Config()                       # defaults match Toto-2.0-4m
model = Toto2ForPrediction(config).eval()

# Multivariate context: (batch, num_variates, time). `time` must be a multiple of config.patch_size.
past_values = torch.randn(1, 4, 512)
with torch.no_grad():
    outputs = model(past_values=past_values, horizon_len=config.patch_size)

# Quantile forecasts have shape (num_quantiles, batch, num_variates, horizon).
quantile_forecast = outputs.quantiles
median_forecast = outputs.mean_predictions   # (batch, num_variates, horizon)
```

## Toto2Config

[[autodoc]] Toto2Config

## Toto2Model

[[autodoc]] Toto2Model
    - forward

## Toto2ForPrediction

[[autodoc]] Toto2ForPrediction
    - forward
