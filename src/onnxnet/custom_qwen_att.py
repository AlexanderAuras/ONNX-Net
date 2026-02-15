from typing import TYPE_CHECKING, override

from torch import Tensor
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2.modeling_qwen2 import eager_attention_forward
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modular_qwen3 import Qwen3Attention
from transformers.processing_utils import Unpack


if TYPE_CHECKING:
    from collections.abc import Callable


class CustomQWenAttention(Qwen3Attention):
    def __init__(self, config: Qwen3Config, layer_idx: int) -> None:
        super().__init__(config, layer_idx)

    @override
    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: Tensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[Tensor, Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)  # TODO Replace

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable[..., tuple[Tensor, Tensor | None]] = eager_attention_forward
        if self.config._attn_implementation != "eager":  # noqa: SLF001  # pyright: ignore [reportPrivateUsage]
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]  # noqa: SLF001  # pyright: ignore [reportPrivateUsage]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama  # pyright: ignore [reportCallIssue]
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
