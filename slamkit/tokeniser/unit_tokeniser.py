import re
import torch
from typing import List, Optional, Union, Dict
from itertools import groupby

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer, BatchEncoding
import json

from ..feature_extractor.audio_feature_extractor import AudioFeatureExtractor
from .audio_tokeniser import AudioTokeniser


class UnitTokeniser(AudioTokeniser):
    def __init__(self, speech_tokeniser: AudioFeatureExtractor,
                 dedup: bool = True,
                 bos_eos_token_id: int = 1,
                 pad_token_id: int = 0,
                 num_units: int = 500,
                 load_fe: bool = True):
        super().__init__()
        self.model = speech_tokeniser if load_fe else None
        self.dedup = dedup
        self.bos_token_id = bos_eos_token_id
        self.eos_token_id = bos_eos_token_id
        self.pad_token_id = pad_token_id
        self.num_units = num_units
        self.text_tokeniser = self._init_text_tokeniser()

    def _init_text_tokeniser(self) -> Union[PreTrainedTokenizerFast,PreTrainedTokenizer]:
        # We place the Unit token consecutively after the special tokens
        offset = max(self.eos_token_id, self.bos_token_id, self.pad_token_id) + 1
        vocab = {f"<Un{i}>": i + offset for i in range(self.num_units)}
        vocab.update({"<PAD>": self.pad_token_id, "<S>": self.bos_token_id})

        tokeniser = Tokenizer(WordLevel(vocab, unk_token="<UNK>"))
        tokeniser.pre_tokenizer = Split(pattern=">", behavior="merged_with_previous")
        tokeniser.add_special_tokens(["<PAD>"])
        tokeniser.enable_padding(pad_id=self.pad_token_id, pad_token="<PAD>")
        tokeniser.post_processor = TemplateProcessing(
            single="<S> $0 <S>",
            special_tokens=[("<S>", tokeniser.token_to_id("<S>"))],
        )
        return PreTrainedTokenizerFast(tokenizer_object=tokeniser)
    
    def __call__(self, sample: Union[Dict, str], **tokenise_kwargs) -> BatchEncoding:
        if isinstance(sample, dict):
            sample = self.stringify_representation([sample])[0]
        return self.text_tokeniser(sample, **tokenise_kwargs)
                 
    def audio_represent(self, wav: torch.Tensor, lens: Optional[torch.Tensor] = None) -> List[Dict]:
        toks = self.model.extract(wav, lens)
        if self.dedup:
            toks = [zip(*[(k, len(list(g))) for k, g in groupby(t.tolist())]) for t in toks]
        else:
            toks = [(t, [1] * len(t)) for t in toks]
        return [{'units': u, 'duration': d} for u, d in toks]

    def stringify_representation(self, reps: List[Dict], mode: str = 'test') -> List[str]:
        return [''.join([f'<Un{u}>' for u in cur['units']]) for cur in reps]

    def audio_stringify(self, wav: torch.Tensor, lens: Optional[torch.Tensor] = None) -> List[str]:
        return self.stringify_representation(self.audio_represent(wav, lens))

    def string_tokenise(self, audio_repr: List[str], **tokenise_kwargs) -> BatchEncoding:
        return self.text_tokeniser(audio_repr, **tokenise_kwargs)

    @torch.inference_mode()
    def tokenise(self, wav: torch.Tensor, lens: Optional[torch.Tensor] = None) -> BatchEncoding:
        return self.string_tokenise(self.audio_stringify(wav, lens), return_tensors='pt', padding=True)

    def build_prompt(self, wav: torch.Tensor, lens: Optional[torch.Tensor] = None) -> BatchEncoding:
        tokens = self.string_tokenise(self.audio_stringify(wav, lens), return_tensors='pt', padding=True)
        # remove eos and irrelevent keys
        tokens = {k: v[...,:-1] for k, v in tokens.items() if k != "token_type_ids"}
        return BatchEncoding(tokens)

    def prepare_sample(self, sample: dict, **tokenise_kwargs) -> BatchEncoding:
        return self.string_tokenise(sample['audio_repr'], **tokenise_kwargs)

    def decode_sample(self, tokens: torch.tensor) -> torch.Tensor:
        # remove special tokens
        tokens = tokens[(tokens != self.pad_token_id) & (tokens != self.bos_token_id) & (tokens != self.eos_token_id)]
        audio_repr = self.text_tokeniser.decode(tokens)
        return torch.tensor([int(re.search(r'<Un(\d+)>', tok).group(1)) for tok in audio_repr.split()])
    
    @property
    def fe_sample_rate(self) -> int:
        if self.model is None:
            raise RuntimeError("This tokeniser does not have a feature extractor, please make sure you are using the correct feature extractor")
        return self.model.sample_rate
    
    def save_pretrained(self, save_directory: str, **kwargs):
        save_dict = {
            'dedup': self.dedup,
            'bos_eos_token_id': self.bos_token_id,
            'pad_token_id': self.pad_token_id,
            'num_units': self.num_units,
            'load_fe': False
        }
        with open(f'{save_directory}/tokeniser_config.json', 'w') as f:
            json.dump(save_dict, f)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs) -> 'UnitTokeniser':
        with open(f'{pretrained_model_name_or_path}/tokeniser_config.json', 'r') as f:
            config = json.load(f)
        return cls(speech_tokeniser=None, **config, **kwargs)

    def get_ignore_tokens(self, _: Optional[str]) -> Optional[List[int]]:
        """
        Get the tokens to ignore in the log likelihood calculation
        :param used_token_modality: the tokens modality to use
        :return: a list of tokens to ignore based on the modality
        """
        return None

