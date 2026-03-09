from typing import override

import torch
from torch import Tensor
from transformers import Qwen3ForSequenceClassification
from transformers.loss.loss_utils import ForSequenceClassificationLoss
from transformers.modeling_outputs import BaseModelOutputWithPast, SequenceClassifierOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs


class CustomPEQwen3(Qwen3ForSequenceClassification):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss_function = ForSequenceClassificationLoss

    @override
    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        token_pes: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> SequenceClassifierOutputWithPast:
        base_model = getattr(self, self.base_model_prefix)
        # inputs_embeds_raw = base_model.embed_tokens(input_ids)
        # inputs_embeds = inputs_embeds_raw + cast("Tensor", token_pes).to(inputs_embeds_raw.dtype)
        inputs_embeds = None
        transformer_outputs: BaseModelOutputWithPast = base_model(
            input_ids=input_ids,  # aasdhisdfh siduhfisdf
            attention_mask=attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=None,
            **kwargs,
        )
        hidden_states = transformer_outputs.last_hidden_state
        hidden_states = hidden_states.to(self.score.weight.dtype)  # HACK
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
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                pooled_logits=pooled_logits,
                config=self.config,
            )

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
