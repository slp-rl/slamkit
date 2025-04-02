import torch
from typing import List, Optional, Dict, Union
from abc import ABC, abstractmethod
from omegaconf import DictConfig

from ..feature_extractor.audio_feature_extractor import AudioFeatureExtractor


class AudioTokeniser(ABC, torch.nn.Module):
    text_tokeniser = None

    @abstractmethod
    def audio_represent(self, wav: torch.Tensor, lens: Optional[torch.Tensor] = None) -> List[Dict]:
        """
        Convert the audio to a dictionary representation of the tokens, for instance:
        {'units': [17, 5, ... 3], 'duration': [1, 2, ... 1]}, all functions of the tokeniser use this and potentially
        additional text information to stringify and tokenise the samples
        :param wav: a batch of audio samples
        :param lens: the length of the non-padding part of the audio samples
        :return: a List of Dicts, one for each sample in the batch
        """
        pass

    @abstractmethod
    def stringify_representation(self, reps: List[Dict], mode: str = 'test') -> List[str]:
        """
        Convert the audio representation dictionary to a string representation, such as '<Hu17><Pi13>...<Hu3>' for each
        sample.
        :param reps: a list of audio sample representations
        :param mode: the mode of the tokeniser, either 'train' or 'test'. If train it may incorporate text information
        :return: a list of string representations
        """
        pass

    @abstractmethod
    def string_tokenise(self, audio_repr: List[str], return_tensors: Optional[str] = None) -> dict:
        """
        Convert an audio to a string representation, such as '<Hu17><Pi5>...<Hu3>' into input to an HF CausalLM using
        a text tokeniser.
        :param audio_repr: a list of audio sample representations
        :param return_tensors: whether to return tensors or not
        :return: an HF text tokeniser output {'input_ids': ..., 'attention_mask': ...}
        """
        pass

    @abstractmethod
    def tokenise(self, wav: torch.Tensor, lens: Optional[torch.Tensor] = None) -> dict:
        """
        Tokenise the audio samples
        :param wav: a batch of audio samples
        :param lens: the length of the non-padding part of the audio samples
        :return: an HF text tokeniser output {'input_ids': ..., 'attention_mask': ...}
        """
        return self.string_tokenise(self.audio_stringify(wav, lens))

    @abstractmethod
    def build_prompt(self, wav: torch.Tensor, lens: Optional[torch.Tensor] = None,
                     output_modality: Optional[str] = None) -> dict:
        """
        Tokenise the audio samples as a prompt for generation including any special tokens needed
        :param wav: a batch of audio samples
        :param lens: the length of the non-padding part of the audio samples
        :param output_modality: the modality of the output, if it is a multi-modal tokeniser from SPEECH and TEXT
        :return: an HF text tokeniser output {'input_ids': ..., 'attention_mask': ...}
        """
        return self.string_tokenise(self.audio_stringify(wav, lens))

    @abstractmethod
    def prepare_sample(self, sample: dict, **tokenise_kwargs) -> torch.Tensor:
        """
        Prepare a sample from dict of string representations to a sample ready to a tokenised sample
        :param sample: a sample from a dataset
        :return: a set of tokens for the sample
        """
        pass

    @abstractmethod
    def decode_sample(self, tokens: torch.tensor, output_modality: str = "SPEECH") -> Union[torch.Tensor, str]:
        """
        Take a tokenised sample and decode it back to Feature_extractor output, thus matching the expected input to the
        vocoder
        :param tokens: a sample from a dataset
        :param output_modality: the modality of the output if it is a multi-modal tokeniser from SPEECH or TEXT
        :return: Hubert units
        """
        pass

    @abstractmethod
    def get_ignore_tokens(self, used_token_modality: str) -> List[int]:
        """
        Get the tokens to ignore in the log likelihood calculation
        :param used_token_modality: the tokens modality to use
        :return: a list of tokens to ignore based on the modality from ["TEXT", "SPEECH"]
        """
        pass



def _init_feature_extractor(fe_type: str, cfg: DictConfig) -> AudioFeatureExtractor:
    if fe_type == 'hubert':
        from ..feature_extractor.hubert_feature_extractor import HubertFeatureExtractor
        return HubertFeatureExtractor(**cfg)
    else:
        raise ValueError(f'Unknown speech tokeniser type: {fe_type}')


def tokeniser_factory(cfg:DictConfig) -> AudioTokeniser:
    # We set the vocabulary size in the tokeniser to match the feature extractor
    cfg.params.num_units = cfg.feature_extractor.num_units
    if cfg.params.load_fe:
        feature_extractor = _init_feature_extractor(cfg.feature_extractor_type, cfg.feature_extractor)
    else:
        feature_extractor = None
    if cfg.tokeniser_type == 'unit':
        from .unit_tokeniser import UnitTokeniser
        return UnitTokeniser(feature_extractor, **cfg.params)
    elif cfg.tokeniser_type == 'interleave':
        from .interleaving_tokeniser import InterleavingTokeniser
        return InterleavingTokeniser(feature_extractor, **cfg.params)
    else:
        raise ValueError(f'Unknown tokeniser type: {cfg.tokeniser_type}')
