from typing import List, Optional, Union
import torch
from transformers import AutoModelForCausalLM, PreTrainedModel, PretrainedConfig, GenerationConfig
from transformers.generation.utils import GenerateOutput
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast

from .token_lm import TokenLM
from ..utils.calculation_utils import calc_nll


def compute_loss(logits, labels, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs):
    logits = logits.float()
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    shift_logits = shift_logits.view(-1, logits.size(-1))
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = torch.nn.functional.cross_entropy(
        shift_logits, shift_labels, reduction=reduction, ignore_index=ignore_index)
    if reduction == "sum":
        loss = loss / num_items_in_batch
    return loss


class UnitLMConfig(PretrainedConfig):
    model_type = "speech_language_model"
    keys_to_ignore_at_inference = ["past_key_values"]
    is_composition = True

    def __init__(self, base_model_name: str = "facebook/opt-350M",
                 base_config: Optional[Union[PretrainedConfig, dict]] = None,
                 vocab_size: int = 502,
                 twist_init: bool = True,
                 use_cache: bool = True,
                 pad_token_id: int = 0,
                 bos_token_id: int = 1,
                 eos_token_id: int = 1,
                 trust_remote_code: Optional[bool] = None,
                 use_safetensors: Optional[bool] = None,
                 **kwargs):
        """
        base_model_name: str - the name of the base model to use. could use a huggingface model name or a path to a model. should be a causal language model
        base_config: Optional[Union[PretrainedConfig,dict]] - the config of the base model. if None, will use the config of the base model
        num_units: int - the number of units in the model
        twist_init: bool - whether to use twist initialization for the weights
        use_cache: bool - whether to use cache in the model
        """
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.vocab_size = vocab_size
        self.twist_init = twist_init
        self.use_cache = use_cache
        self.trust_remote_code = trust_remote_code
        self.use_safetensors = use_safetensors
        kwargs["torch_dtype"] = self.torch_dtype
        if base_config is None:
            self.base_config = AutoConfig.from_pretrained(base_model_name,
                                                          pad_token_id=pad_token_id,
                                                          bos_token_id=bos_token_id,
                                                          eos_token_id=eos_token_id, trust_remote_code=self.trust_remote_code, **kwargs)
        elif isinstance(base_config, dict):
            base_config["torch_dtype"] = None
            self.base_config = AutoConfig.from_pretrained(
                base_model_name, trust_remote_code=self.trust_remote_code, **base_config, attn_implementation=kwargs.get("attn_implementation", None))
        else:
            self.base_config = base_config

        self.max_position_embeddings = getattr(self.base_config, "max_position_embeddings", None)

        # it is possible that the base model has a default value for tie_word_embeddings, so we need to make sure they are the same
        # if they are different it causes problems in from_pretrained
        self.tie_word_embeddings = self.base_config.tie_word_embeddings


class UnitLM(PreTrainedModel, TokenLM):
    """
    A unit language model that operates over a single stream of audio semantic units, such as HuBERT.
    """
    config_class = UnitLMConfig
    base_model_prefix = "lm"
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True

    def __init__(self, config: UnitLMConfig, from_pretrained: bool = False):
        super(UnitLM, self).__init__(config)

        self.lm = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=config.base_model_name,
                                                       config=config.base_config,
                                                       trust_remote_code=config.trust_remote_code,
                                                       use_safetensors=config.use_safetensors,
                                                       torch_dtype=config.base_config.torch_dtype) \
            if (not from_pretrained and config.twist_init) else AutoModelForCausalLM.from_config(config.base_config,
                                                                                                trust_remote_code=config.trust_remote_code, 
                                                                                                torch_dtype=config.base_config.torch_dtype)
        self.lm.resize_token_embeddings(config.vocab_size)
        self.config = config

        self.lm.generation_config.pad_token_id = config.base_config.pad_token_id
        self.lm.generation_config.bos_token_id = config.base_config.bos_token_id
        self.lm.generation_config.eos_token_id = config.base_config.eos_token_id

        # Hack to prevent the model from reinitializing the output embeddings (after the resize the flag is lost, causing post_init to reinit the weights of the output embeddings)
        # remove when fixed in transformers https://github.com/huggingface/transformers/issues/35141
        self.lm.get_output_embeddings()._is_hf_initialized = True
        self.post_init()

    def init_weights(self):
        self.lm.init_weights()

    def get_input_embeddings(self):
        return self.lm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.lm.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.lm.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.lm.set_decoder(decoder)

    def get_decoder(self):
        return self.lm.get_decoder()

    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                **loss_kwargs):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs["logits"] if return_dict else outputs[0]

        loss = None
        if labels is not None:
            loss = compute_loss(logits, labels, **loss_kwargs)

        if not return_dict:
            return ((loss,) + outputs) if loss is not None else outputs

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.inference_mode()
    def log_likelihood(self, tokens: torch.Tensor, mean_nll: bool, ignore_tokens:Optional[List[int]] = None) -> torch.Tensor:
        logits = self(input_ids=tokens).logits
        if ignore_tokens is not None:
            logits[:, :, ignore_tokens] = float('-inf')
        shifted_x = tokens[..., 1:]
        shifted_logits = logits[..., :-1, :]
        # Create a mask that is True where the tokens are not padding tokens
        shifted_x[shifted_x == self.config.base_config.pad_token_id] = -100
        # Calculate the likelihoods
        return -calc_nll(shifted_logits, shifted_x, shifted_x.ne(-100), mean_nll)

    def generate(self, inputs: Optional[torch.Tensor] = None, generation_config: Optional[GenerationConfig] = None,
                 **kwargs) -> Union[GenerateOutput, torch.LongTensor]:
        return self.lm.generate(inputs, generation_config=generation_config, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args, **kwargs):
        """
        Instantiate a UnitLM from a pretrained model.
        overrides the from_pretrained method to add the from_pretrained argument so it is always set to True.
        makes sure that the base model is not downloaded when UnitLM is loaded
        """
        model_args = (True,) + model_args
        return super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            **kwargs,
        )