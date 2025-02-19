import torch
from abc import ABC, abstractmethod
from omegaconf import DictConfig


class AudioVocoder(ABC, torch.nn.Module):

    @abstractmethod
    def vocode(self, tokens: torch.LongTensor, **kwargs) -> torch.Tensor:
        pass


def vocoder_factory(cfg: DictConfig) -> AudioVocoder:
    if cfg.vocoder_type == 'hifigan':
        from .hifi_gan_vocoder import HiFiGANVocoder
        return HiFiGANVocoder(cfg.dense_model_name,
                              cfg.quantizer_model_name,
                              cfg.vocab_size,
                              cfg.vocoder_suffix,
                              cfg.speaker_meta,
                              cfg.style_meta)
    elif cfg.vocoder_type == None:
        return None
    else:
        raise ValueError(f'Unknown vocoder type: {cfg.vocoder_type}')
