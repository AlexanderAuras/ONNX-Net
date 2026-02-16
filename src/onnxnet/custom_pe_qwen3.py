from typing import override

import torch
from torch import Tensor
from transformers import Qwen3ForSequenceClassification
from transformers.modeling_outputs import BaseModelOutputWithPast, SequenceClassifierOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs


class CustomPEQwen3(Qwen3ForSequenceClassification):
    @override
    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        token_pes: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> SequenceClassifierOutputWithPast:
        print(type(input_ids), str(input_ids).replace("\n", "")[:100])
        print(type(attention_mask), str(attention_mask).replace("\n", "")[:100])
        print(type(labels), str(labels).replace("\n", "")[:100])
        print(type(token_pes), str(token_pes).replace("\n", "")[:100])
        inputs_embeds = self.model.embed_tokens(input_ids)
        inputs_embeds += token_pes
        del kwargs["token_pes"]
        transformer_outputs: BaseModelOutputWithPast = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=None,
            **kwargs,
        )
        hidden_states = transformer_outputs.last_hidden_state
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            msg = "Cannot handle batch sizes > 1 if no padding token is defined."
            raise ValueError(msg)
        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
            non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config)

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
