import torch
from typing import List, Optional
from abc import ABC, abstractmethod
from omegaconf import DictConfig


class TokenLM(ABC):
    @abstractmethod
    @torch.inference_mode()
    def log_likelihood(self, tokens: torch.Tensor, mean_nll: bool) -> torch.Tensor:
        """
        Given a padded tensor of Tokens (i.e. tokenised audio samples with the appropriate tokeniser), calculate the log
        likelihood for each sample.
        :param tokens: A tensor of tokenised audio samples (as output of appropriate AudioTokeniser with padding=True)
        :param mean_nll: whether to take mean instead of sum thus cancelling length bias
        :return:
        """
        ...

    @abstractmethod
    @torch.inference_mode()
    def generate(self, inputs: Optional[torch.Tensor] = None, generation_config = None, **kwargs) -> torch.Tensor:
        """
        Given a List of Tokens (i.e. tokenised audio samples with the appropriate tokeniser), generate the continuation
        tokens
        """
        ...


def tlm_factory(cfg: DictConfig) -> TokenLM:
    if cfg.tlm_type == 'twist' or cfg.tlm_type == 'gslm':
        from .unit_lm import UnitLM, UnitLMConfig
        if cfg.pretrained_model:
            return UnitLM.from_pretrained(
                cfg.pretrained_model, 
                attn_implementation=cfg.config_args.get("attn_implementation",None),
                torch_dtype=cfg.config_args.get("torch_dtype", None),
                use_cache=cfg.config_args.get("use_cache", False)
            )
        config = UnitLMConfig(**cfg.config_args)
        return UnitLM(config)
    else:
        raise ValueError(f'Unknown slm type: {cfg.tlm_type}')
