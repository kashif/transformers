# coding=utf-8
# Copyright 2025 Google LLC and HuggingFace Inc. team.
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
"""PyTorch TimesFM model."""

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ..llama.modeling_llama import eager_attention_forward
from ..t5.modeling_t5 import T5LayerNorm
from .configuration_timesfm import TimesFmConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "google/timesfm-2.0-500m-pytorch"
_CONFIG_FOR_DOC = "TimesFmConfig"


@dataclass
class TimesFmOutput(BaseModelOutput):
    """
    Args:
        loc (`torch.Tensor` of shape `(batch_size, )`):
            The mean of the time series inputs.
        scale (`torch.Tensor` of shape `(batch_size,)`):
            The scale of the time series inputs.
        past_key_values (`List[Cache]`, *optional*):
            Contains the precomputed key and value hidden states of the attention blocks used for
            faster decoding. Can be used as a cache for future predictions.
    """

    loc: Optional[torch.Tensor] = None
    scale: Optional[torch.Tensor] = None
    past_key_values: Optional[List[Cache]] = None


@dataclass
class TimesFmOutputForPrediction(BaseModelOutput):
    """
    Args:
        mean_predictions (`torch.Tensor` of shape `(batch_size, sequence_length)`):
            The mean predictions of the time series.
        full_predictions (`torch.Tensor` of shape `(batch_size, sequence_length)`):
            The full predictions of the time series including the mean and the quantiles.
        loss (`torch.Tensor` of shape `(1,)`, *optional*, returned when `future_target` is provided):
            The loss of the TimesFM model.
        past_key_values (`List[Cache]`, *optional*):
            Contains the precomputed key and value hidden states of the attention blocks used for
            faster decoding. Can be used as a cache for future predictions.
    """

    mean_predictions: Optional[torch.Tensor] = None
    full_predictions: Optional[torch.Tensor] = None
    loss: Optional[Union[torch.Tensor, float]] = None
    past_key_values: Optional[List[Cache]] = None


class TimesFmMLP(nn.Module):
    """Pax MLP in pytorch."""

    def __init__(self, config: TimesFmConfig):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm = nn.LayerNorm(normalized_shape=hidden_size, eps=1e-6)

    def forward(self, x, paddings=None):
        gate_inp = self.layer_norm(x)
        gate = self.gate_proj(gate_inp)
        gate = F.relu(gate)
        outputs = self.down_proj(gate)
        if paddings is not None:
            outputs = outputs * (1.0 - paddings[:, :, None])
        return outputs + x


class TimesFmResidualBlock(nn.Module):
    """TimesFM residual block."""

    def __init__(self, input_dims, hidden_dims, output_dims):
        super().__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims

        # Hidden Layer
        self.input_layer = nn.Linear(input_dims, hidden_dims)
        self.activation = nn.SiLU()
        # Output Layer
        self.output_layer = nn.Linear(hidden_dims, output_dims)
        # Residual Layer
        self.residual_layer = nn.Linear(input_dims, output_dims)

    def forward(self, x):
        hidden = self.input_layer(x)
        hidden = self.activation(hidden)
        output = self.output_layer(hidden)
        residual = self.residual_layer(x)
        return output + residual


class TimesFmRMSNorm(T5LayerNorm):
    pass


class TimesFmPositionalEmbedding(nn.Module):
    """Generates position embedding for a given 1-d sequence."""

    def __init__(self, config: TimesFmConfig):
        super().__init__()
        self.min_timescale = config.min_timescale
        self.max_timescale = config.max_timescale
        self.embedding_dims = config.hidden_size

    def forward(self, seq_length=None, position=None):
        """Generates a Tensor of sinusoids with different frequencies.

        Args:
            seq_length: an optional Python int defining the output sequence length.
              if the `position` argument is specified.
            position: [B, seq_length], optional position for each token in the
              sequence, only required when the sequence is packed.

        Returns:
            [B, seqlen, D] if `position` is specified, else [1, seqlen, D]
        """
        if position is None:
            assert seq_length is not None
            # [1, seqlen]
            position = torch.arange(seq_length, dtype=torch.float32).unsqueeze(0)
        else:
            assert position.ndim == 2, position.shape

        num_timescales = self.embedding_dims // 2
        log_timescale_increment = math.log(float(self.max_timescale) / float(self.min_timescale)) / max(
            num_timescales - 1, 1
        )
        inv_timescales = self.min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) * -log_timescale_increment
        )
        scaled_time = position.unsqueeze(2) * inv_timescales.unsqueeze(0).unsqueeze(0)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=2)
        # Padding to ensure correct embedding dimension
        signal = F.pad(signal, (0, 0, 0, self.embedding_dims % 2))
        return signal


class TimesFmAttention(nn.Module):
    """Implements the attention used in TimesFM. One key difference is that there is _per_dim_scaling of the query."""

    def __init__(self, config: TimesFmConfig, layer_idx: int):
        super().__init__()
        self.attn_implementation = config._attn_implementation
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.layer_idx = layer_idx

        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.num_key_value_groups = 1

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_heads * self.head_dim
        self.scaling = nn.Parameter(torch.empty((self.head_dim,)))

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size)

    def _scale_query(self, query: torch.Tensor) -> torch.Tensor:
        scale = F.softplus(self.scaling).mul(1.442695041 / math.sqrt(self.head_dim))
        return query * scale[None, None, None, :]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, input_len = hidden_states.shape[:2]
        hidden_shape = (batch_size, -1, self.num_heads, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states = self._scale_query(query_states)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Write new kv cache if past_key_value is provided
        if past_key_value is not None and cache_position is not None:
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, {"cache_position": cache_position}
            )

        attention_interface: Callable = eager_attention_forward
        if self.attn_implementation != "eager":
            if self.attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=1.0,
            **kwargs,
        )
        attn_output = attn_output.reshape(batch_size, input_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class TimesFmDecoderLayer(nn.Module):
    """Transformer layer."""

    def __init__(self, config: TimesFmConfig, layer_idx: int):
        super().__init__()

        self.self_attn = TimesFmAttention(config, layer_idx=layer_idx)
        self.mlp = TimesFmMLP(config)
        self.input_layernorm = TimesFmRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        paddings: torch.Tensor,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, scores = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        # MLP
        hidden_states = self.mlp(hidden_states, paddings=paddings)

        return scores, hidden_states


def timesfm_get_large_negative_number(dtype: torch.dtype) -> torch.Tensor:
    """Returns a large negative value for the given dtype."""
    if dtype.is_floating_point:
        dtype_max = torch.finfo(dtype).max
    else:
        dtype_max = torch.iinfo(dtype).max
    return torch.tensor(-0.7 * dtype_max, dtype=dtype)


def _prepare_4d_attention_mask(
    attention_mask: Optional[torch.Tensor],
    sequence_length: int,
    dtype: torch.dtype,
    device: torch.device,
    is_causal: bool = True,
) -> Optional[torch.Tensor]:
    """
    Creates 4D attention mask and combines causal and padding masks if needed.

    Args:
        attention_mask: Optional tensor of shape (batch_size, seq_length) containing padding mask
        sequence_length: Length of the sequence
        dtype: Data type of the mask
        device: Device of the mask
        is_causal: Whether to apply causal masking

    Returns:
        4D attention mask of shape (batch_size, 1, seq_length, seq_length)
    """
    # Handle padding mask
    if attention_mask is not None:
        # Convert 2D padding mask to 4D attention mask
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = attention_mask * timesfm_get_large_negative_number(dtype)

    # Create causal mask if needed
    if is_causal:
        causal_mask = torch.triu(
            torch.ones((sequence_length, sequence_length), dtype=dtype, device=device)
            * timesfm_get_large_negative_number(dtype),
            diagonal=1,
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        # Combine with padding mask if it exists
        if attention_mask is not None:
            attention_mask = torch.minimum(attention_mask, causal_mask)
        else:
            attention_mask = causal_mask

    return attention_mask


def timesfm_masked_mean_std(inputs: torch.Tensor, padding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculates mean and standard deviation of `inputs` across axis 1.

    It excludes values where `padding` is 1.

    Args:
        inputs: A PyTorch tensor of shape [b, n, p].
        padding: A PyTorch tensor of shape [b, n, p] with values 0 or 1.

    Returns:
        A tuple containing the mean and standard deviation.
        We return the statistics of the first patch with more than three non-padded values.
    """

    # Selecting the first patch with more than 3 unpadded values.
    def _get_patch_index(arr: torch.Tensor):
        indices = torch.argmax((arr >= 3).to(torch.int32), dim=1)
        row_sum = (arr >= 3).to(torch.int32).sum(dim=1)
        return torch.where(row_sum == 0, arr.shape[1] - 1, indices)

    pad_sum = torch.sum(1 - padding, dim=2)
    patch_indices = _get_patch_index(pad_sum)
    bidxs = torch.arange(inputs.shape[0])

    arr = inputs[bidxs, patch_indices, :]
    pad = padding[bidxs, patch_indices, :]

    # Create a mask where padding is 0
    mask = 1 - pad

    # Calculate the number of valid elements
    num_valid_elements = torch.sum(mask, dim=1)
    num_valid_elements = torch.where(
        num_valid_elements == 0,
        torch.tensor(1, dtype=num_valid_elements.dtype, device=num_valid_elements.device),
        num_valid_elements,
    )

    # Calculate the masked sum and squared sum
    masked_sum = torch.sum(arr * mask, dim=1)
    masked_squared_sum = torch.sum((arr * mask) ** 2, dim=1)

    # Calculate the masked mean and standard deviation
    masked_mean = masked_sum / num_valid_elements
    masked_var = masked_squared_sum / num_valid_elements - masked_mean**2
    masked_var = torch.where(
        masked_var < 0.0,
        torch.tensor(0.0, dtype=masked_var.dtype, device=masked_var.device),
        masked_var,
    )
    masked_std = torch.sqrt(masked_var)

    return masked_mean, masked_std


def timesfm_shift_padded_seq(mask: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
    """Shifts rows of seq based on the first 0 in each row of the mask.

    Args:
        mask: mask tensor of shape [B, N]
        seq: seq tensor of shape [B, N, P]

    Returns:
        The shifted sequence.
    """
    batch_size, num_seq, feature_dim = seq.shape

    new_mask: torch.BoolTensor = mask == 0

    # Use argmax to find the first True value in each row
    indices = new_mask.to(torch.int32).argmax(dim=1)

    # Handle rows with all zeros
    indices[~new_mask.any(dim=1)] = -1

    # Create index ranges for each sequence in the batch
    idx_range = torch.arange(num_seq).to(seq.device).unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, feature_dim)

    # Calculate shifted indices for each element in each sequence
    shifted_idx = (idx_range - indices[:, None, None]) % num_seq

    # Gather values from seq using shifted indices
    shifted_seq = seq.gather(1, shifted_idx)

    return shifted_seq


def timesfm_moving_average(arr: torch.Tensor, window_size: int) -> list[torch.Tensor]:
    """Calculates the moving average using PyTorch's convolution function."""
    # Pad with zeros to handle initial window positions
    arr_padded = F.pad(arr, (window_size - 1, 0), "constant", 0)
    # Create a convolution kernel
    kernel = torch.ones(window_size, dtype=arr.dtype, device=arr.device) / window_size
    # Apply convolution to calculate the moving average
    smoothed_arr = F.conv1d(arr_padded.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0)).squeeze()
    return [smoothed_arr, arr - smoothed_arr]


TIMESFM_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`TimesFmConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare TimesFM Model outputting raw hidden-states without any specific head on top.",
    TIMESFM_START_DOCSTRING,
)
class TimesFmPreTrainedModel(PreTrainedModel):
    """handles the loading for all models."""

    config_class = TimesFmConfig
    base_model_prefix = "timesfm"
    _no_split_modules = ["TimesFmDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    main_input_name = "inputs"
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = True

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0, std=self.config.initializer_range)

        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

        elif isinstance(module, TimesFmRMSNorm):
            nn.init.zeros_(module.weight)

        elif isinstance(module, TimesFmMLP):
            # Initialize gate projection
            module.gate_proj.weight.data.normal_(mean=0, std=self.config.initializer_range)
            if module.gate_proj.bias is not None:
                nn.init.zeros_(module.gate_proj.bias)

            # Initialize down projection
            module.down_proj.weight.data.normal_(mean=0, std=self.config.initializer_range)
            if module.down_proj.bias is not None:
                nn.init.zeros_(module.down_proj.bias)

            # Initialize layer norm
            nn.init.ones_(module.layer_norm.weight)
            nn.init.zeros_(module.layer_norm.bias)

        elif isinstance(module, TimesFmAttention):
            # Initialize qkv projection
            module.q_proj.weight.data.normal_(mean=0, std=self.config.initializer_range)
            module.k_proj.weight.data.normal_(mean=0, std=self.config.initializer_range)
            module.v_proj.weight.data.normal_(mean=0, std=self.config.initializer_range)
            if module.q_proj.bias is not None:
                nn.init.zeros_(module.q_proj.bias)
            if module.k_proj.bias is not None:
                nn.init.zeros_(module.k_proj.bias)
            if module.v_proj.bias is not None:
                nn.init.zeros_(module.v_proj.bias)

            # Initialize output projection
            module.o_proj.weight.data.normal_(mean=0, std=self.config.initializer_range)
            if module.o_proj.bias is not None:
                nn.init.zeros_(module.o_proj.bias)

            # Initialize scaling parameter
            nn.init.ones_(module.scaling)

        elif isinstance(module, TimesFmResidualBlock):
            # Initialize hidden layer
            module.input_layer.weight.data.normal_(mean=0, std=self.config.initializer_range)
            if module.input_layer.bias is not None:
                nn.init.zeros_(module.input_layer.bias)

            # Initialize output layer
            module.output_layer.weight.data.normal_(mean=0, std=self.config.initializer_range)
            if module.output_layer.bias is not None:
                nn.init.zeros_(module.output_layer.bias)

            # Initialize residual layer
            module.residual_layer.weight.data.normal_(mean=0, std=self.config.initializer_range)
            if module.residual_layer.bias is not None:
                nn.init.zeros_(module.residual_layer.bias)

        elif isinstance(module, TimesFmPositionalEmbedding):
            pass

    def generate(self, *args, **kwargs):
        """
        This method is disabled for TimesFM models. TimesFM models are designed for time series forecasting and should be used
        with the forward() method instead. For forecasting, use:

        ```python
        # For basic forecasting:
        outputs = model(input_ts=your_time_series, input_padding=your_padding, freq=your_frequency)

        # For prediction with quantiles:
        outputs = model.forward(
            inputs=your_time_series_list,
            freq=your_frequencies,
            window_size=optional_window_size,
            future_target=optional_target,
            forecast_context_len=optional_context_length
        )
        ```

        See the model's documentation for more details on the forward method parameters.
        """
        raise NotImplementedError(
            "The generate() method is not implemented for TimesFM models as they are designed for time series "
            "forecasting. Please use the forward() method instead. See the docstring of this method for usage examples."
        )


TIMESFM_INPUTS_DOCSTRING = r"""
    Args:
        inputs: list of time series forecast contexts. Each context time series
            should be a torch Tensor of potentially different context lengths.
        freq: frequency of each context time series in the inputs. 0 for high frequency
            (default), 1 for medium, and 2 for low.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail. tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare TimesFM Model outputting raw hidden-states without any specific head on top.",
    TIMESFM_START_DOCSTRING,
)
class TimesFmModel(TimesFmPreTrainedModel):
    """Patched time-series decoder without any specific output layer."""

    def __init__(self, config: TimesFmConfig):
        super().__init__(config)

        self.config = config
        self.input_ff_layer = TimesFmResidualBlock(
            input_dims=2 * config.patch_length,
            output_dims=config.hidden_size,
            hidden_dims=config.intermediate_size,
        )
        self.freq_emb = nn.Embedding(num_embeddings=config.freq_size, embedding_dim=config.hidden_size)
        self.layers = nn.ModuleList(
            [TimesFmDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        if self.config.use_positional_embedding:
            self.position_emb = TimesFmPositionalEmbedding(config=config)

        # Initialize weights and apply final processing
        self.post_init()

    def _forward_transform(
        self, inputs: torch.Tensor, patched_pads: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Input is of shape [B, N, P]."""
        mu, sigma = timesfm_masked_mean_std(inputs, patched_pads)
        sigma = torch.where(
            sigma < self.config.tolerance,
            torch.tensor(1.0, dtype=sigma.dtype, device=sigma.device),
            sigma,
        )

        # Normalize each patch
        outputs = (inputs - mu[:, None, None]) / sigma[:, None, None]
        outputs = torch.where(
            torch.abs(inputs - self.config.pad_val) < self.config.tolerance,
            torch.tensor(self.config.pad_val, dtype=outputs.dtype, device=outputs.device),
            outputs,
        )
        return outputs, (mu, sigma)

    def _preprocess_input(
        self, input_ts: torch.Tensor, input_padding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Preprocess input for stacked transformer."""
        # Reshape into patches (using view for efficiency)
        bsize = input_ts.shape[0]
        patched_inputs = input_ts.view(bsize, -1, self.config.patch_length)
        patched_pads = input_padding.view(bsize, -1, self.config.patch_length)

        patched_inputs = torch.where(
            torch.abs(patched_pads - 1.0) < self.config.tolerance,
            torch.tensor(0.0, dtype=patched_inputs.dtype, device=patched_inputs.device),
            patched_inputs,
        )
        patched_pads = torch.where(
            torch.abs(patched_inputs - self.config.pad_val) < self.config.tolerance,
            torch.tensor(1.0, dtype=patched_pads.dtype, device=patched_pads.device),
            patched_pads,
        )
        patched_inputs, stats = self._forward_transform(patched_inputs, patched_pads)

        # B x N x D
        patched_inputs = patched_inputs * (1.0 - patched_pads)
        concat_inputs = torch.cat([patched_inputs, patched_pads], dim=-1)
        model_input = self.input_ff_layer(concat_inputs)

        # A patch should not be padded even if there is at least one zero.
        patched_padding = torch.min(patched_pads, dim=-1)[0]  # Get the values from the min result
        if self.config.use_positional_embedding:
            pos_emb = self.position_emb(model_input.shape[1]).to(model_input.device)
            pos_emb = torch.concat([pos_emb] * model_input.shape[0], dim=0)
            pos_emb = timesfm_shift_padded_seq(patched_padding, pos_emb)
            model_input += pos_emb

        return model_input, patched_padding, stats, patched_inputs

    @add_start_docstrings_to_model_forward(TIMESFM_INPUTS_DOCSTRING)
    def forward(
        self,
        inputs: torch.Tensor,
        input_padding: torch.LongTensor,
        freq: torch.Tensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[TimesFmOutput, tuple[torch.Tensor, ...]]:
        """
        input_padding (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The padding indicator of the time series.
        """

        model_input, patched_padding, stats, _ = self._preprocess_input(
            input_ts=inputs,
            input_padding=input_padding,
        )
        f_emb = self.freq_emb(freq)  # B x 1 x D
        model_input += f_emb

        # Convert paddings to attention mask and combine with causal mask
        hidden_states = model_input
        attention_mask = _prepare_4d_attention_mask(
            attention_mask=patched_padding,
            sequence_length=hidden_states.shape[1],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
            is_causal=True,
        )

        all_attentions = []
        all_hidden_states = []

        for layer in self.layers[: self.config.num_hidden_layers]:
            scores, hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                paddings=patched_padding,
                past_key_value=past_key_values,
                cache_position=cache_position,
                output_attentions=output_attentions,
            )
            if output_attentions:
                all_attentions.append(scores)
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

        if output_hidden_states:
            all_hidden_states = [model_input] + all_hidden_states
        else:
            all_hidden_states = None

        if return_dict:
            return TimesFmOutput(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states,
                attentions=all_attentions if output_attentions else None,
                loc=stats[0],
                scale=stats[1],
                past_key_values=past_key_values,
            )
        else:
            return (
                hidden_states,
                all_hidden_states,
                all_attentions if output_attentions else None,
                stats[0],
                stats[1],
                past_key_values,
            )


class TimesFmModelForPrediction(TimesFmPreTrainedModel):
    """TimesFM model for quantile and mean prediction."""

    def __init__(self, config: TimesFmConfig):
        super().__init__(config)

        self.config = config
        self.context_len = config.context_length
        self.horizon_len = config.horizon_length

        self.decoder = TimesFmModel(config)

        # quantile and mean output
        self.horizon_ff_layer = TimesFmResidualBlock(
            input_dims=config.hidden_size,
            output_dims=config.horizon_length * (1 + len(config.quantiles)),
            hidden_dims=config.intermediate_size,
        )

        # Initialize weights and apply final processing
        self.post_init()

    def _preprocess(
        self, inputs: Sequence[torch.Tensor], freq: Sequence[int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Formats and pads raw inputs to feed into the model.

        This function both pads each time series to match the context length, and
        pads the inputs to meet the SPMD shape requirement.

        Args:
          inputs: A list of 1d Tensors. Each Tensor is the context time series of
            a single forecast task.
          freq: list of frequencies

        Returns:
        A tuple of:
        - the padded input time series to meet the model required context.
        - the padding indicator.
        - the number of padded examples for SPMD so that each core has the same
            number (a multiple of `batch_size`) of examples.
        """
        input_ts, input_padding, inp_freq = [], [], []

        for i, ts in enumerate(inputs):
            input_len = ts.shape[0]
            padding = torch.zeros(input_len + self.horizon_len, dtype=ts.dtype, device=ts.device)
            if input_len < self.context_len:
                num_front_pad = self.context_len - input_len
                ts = torch.cat([torch.zeros(num_front_pad, dtype=ts.dtype, device=ts.device), ts], dim=0)
                padding = torch.cat([torch.ones(num_front_pad, dtype=ts.dtype, device=padding.device), padding], dim=0)
            elif input_len > self.context_len:
                ts = ts[-self.context_len :]
                padding = padding[-(self.context_len + self.horizon_len) :]

            input_ts.append(ts)
            input_padding.append(padding)
            inp_freq.append(freq[i])

        return (
            torch.stack(input_ts, dim=0),
            torch.stack(input_padding, dim=0),
            torch.tensor(inp_freq, dtype=torch.int32).reshape(-1, 1),
        )

    def _postprocess_output(
        self, model_output: torch.Tensor, stats: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Postprocess output of stacked transformer."""

        # B x N x (H.Q)
        output_ts = self.horizon_ff_layer(model_output)

        # Reshape using view
        b, n, _ = output_ts.shape
        output_ts = output_ts.view(b, n, self.config.horizon_length, len(self.config.quantiles) + 1)

        return self._reverse_transform(output_ts, stats)

    def _reverse_transform(self, outputs: torch.Tensor, stats: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Output is of shape [B, N, P, Q]."""
        mu, sigma = stats
        return outputs * sigma[:, None, None, None] + mu[:, None, None, None]

    def _quantile_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        losses = []
        for i, q in enumerate(self.config.quantiles):
            errors = targets - predictions[..., i]
            loss = torch.max((q - 1) * errors, q * errors)
            losses.append(loss.mean())
        return torch.stack(losses).mean()

    @add_start_docstrings_to_model_forward(TIMESFM_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TimesFmOutputForPrediction, config_class=_CONFIG_FOR_DOC)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TimesFmOutputForPrediction,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        inputs: Sequence[torch.Tensor],
        freq: Optional[Sequence[Union[torch.Tensor, int]]] = None,
        window_size: Optional[int] = None,
        future_target: Optional[torch.Tensor] = None,
        forecast_context_len: Optional[int] = None,
        return_forecast_on_context: bool = False,
        truncate_negative: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[TimesFmOutputForPrediction, tuple[torch.Tensor, ...]]:
        r"""
        window_size (`int`, *optional*):
            Window size of trend + residual decomposition. If None then we do not do decomposition.
        future_target (`torch.Tensor`, *optional*):
            Optional future target time series to be used for loss computation.
        forecast_context_len (`int`, *optional*):
            Optional max context length.
        return_forecast_on_context (`bool`, *optional*):
            True to return the forecast on the context when available, i.e. after the first input patch.
        truncate_negative (`bool`, *optional*):
            Truncate to only non-negative values if any of the contexts have non-negative values,
            otherwise do nothing.

        Returns:
            A TimesFmOutputForPrediction object or a tuple containing:
                - the mean forecast of size (# inputs, # forecast horizon),
                - the full forecast (mean + quantiles) of size
                    (# inputs,  # forecast horizon, 1 + # quantiles).
                - loss: the mean squared error loss + quantile loss if future_target is provided.
        """
        if return_dict is None:
            return_dict = self.config.use_return_dict

        if forecast_context_len is None:
            fcontext_len = self.context_len
        else:
            fcontext_len = forecast_context_len

        # Get device from first input tensor
        device = inputs[0].device

        # Truncate inputs to forecast_context_len
        inputs = [ts[-fcontext_len:] for ts in inputs]
        inp_min = torch.min(torch.stack([torch.min(ts) for ts in inputs]))

        if window_size is not None:
            new_inputs = []
            if freq is not None:
                new_freqs = []
            for i, ts in enumerate(inputs):
                new_inputs.extend(timesfm_moving_average(ts, window_size))
                if freq is not None:
                    new_freqs.extend([freq[i]] * 2)
            inputs = new_inputs
            if freq is not None:
                freq = new_freqs

        if freq is None:
            logger.info("No frequency provided via `freq`. Default to high (0).")
            freq = [0] * len(inputs)

        if output_attentions is None:
            output_attentions = self.config.output_attentions
        if output_hidden_states is None:
            output_hidden_states = self.config.output_hidden_states

        input_ts, input_padding, inp_freq = self._preprocess(inputs, freq)

        # Move tensors to the same device as input
        input_ts = input_ts.to(device)
        input_padding = input_padding.to(device)
        inp_freq = inp_freq.to(device)

        final_out = input_ts
        context_len = final_out.shape[1]
        full_outputs = []

        if input_padding.shape[1] != final_out.shape[1] + self.horizon_len:
            raise ValueError(
                "Length of paddings must match length of input + horizon_len:"
                f" {input_padding.shape[1]} != {final_out.shape[1]} + {self.horizon_len}"
            )
        output_patch_len = self.config.horizon_length

        num_decode_patches = (self.horizon_len + output_patch_len - 1) // output_patch_len
        for step_index in range(num_decode_patches):
            current_padding = input_padding[:, 0 : final_out.shape[1]]
            input_ts = final_out[:, -fcontext_len:]
            input_padding = current_padding[:, -fcontext_len:]
            decoder_output = self.decoder(
                input_ts,
                input_padding,
                inp_freq,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            fprop_outputs = self._postprocess_output(
                decoder_output.last_hidden_state,
                (decoder_output.loc, decoder_output.scale),
            )

            if return_forecast_on_context and step_index == 0:
                # For the first decodings step, collect the model forecast on the
                # context except the unavailable first input batch forecast.
                new_full_ts = fprop_outputs[:, :-1, : self.config.patch_length, :]
                # We have to use reshape and not view for non-contiguous memory
                new_full_ts = new_full_ts.reshape(new_full_ts.size(0), -1, new_full_ts.size(3))

                full_outputs.append(new_full_ts)

            # (full batch, last patch, output_patch_len, index of mean forecast = 0)
            new_ts = fprop_outputs[:, -1, :output_patch_len, 0]
            new_full_ts = fprop_outputs[:, -1, :output_patch_len, :]
            # (full batch, last patch, output_patch_len, all output indices)
            full_outputs.append(new_full_ts)
            final_out = torch.concatenate([final_out, new_ts], axis=-1)

        if return_forecast_on_context:
            # `full_outputs` indexing starts at after the first input patch.
            full_outputs = torch.concatenate(full_outputs, axis=1)[
                :, : (context_len - self.config.patch_length + self.horizon_len), :
            ]
        else:
            # `full_outputs` indexing starts at the forecast horizon.
            full_outputs = torch.concatenate(full_outputs, axis=1)[:, 0 : self.horizon_len, :]

        mean_outputs = full_outputs[:, :, 0]
        if window_size is not None:
            mean_outputs = mean_outputs[0::2, ...] + mean_outputs[1::2, ...]
            full_outputs = full_outputs[0::2, ...] + full_outputs[1::2, ...]
        if inp_min >= 0 and truncate_negative:
            mean_outputs = torch.maximum(mean_outputs, 0.0)
            full_outputs = torch.maximum(full_outputs, 0.0)

        loss = None
        if future_target is not None:
            mse_loss = F.mse_loss(mean_outputs, future_target)
            quantile_loss = self._quantile_loss(full_outputs[:, :, 1:], future_target)
            loss = mse_loss + quantile_loss

        if return_dict:
            return TimesFmOutputForPrediction(
                last_hidden_state=decoder_output.last_hidden_state,
                attentions=decoder_output.attentions if output_attentions else None,
                hidden_states=decoder_output.hidden_states if output_hidden_states else None,
                mean_predictions=mean_outputs,
                full_predictions=full_outputs,
                loss=loss,
                past_key_values=decoder_output.past_key_values,
            )
        else:
            return_tuple = [decoder_output.last_hidden_state]
            if output_hidden_states:
                return_tuple.append(decoder_output.hidden_states)
            if output_attentions:
                return_tuple.append(decoder_output.attentions)
            return_tuple += [mean_outputs, full_outputs, loss]
            return_tuple += [decoder_output.past_key_values]
            return tuple(return_tuple)


__all__ = ["TimesFmModelForPrediction", "TimesFmPreTrainedModel", "TimesFmModel"]
