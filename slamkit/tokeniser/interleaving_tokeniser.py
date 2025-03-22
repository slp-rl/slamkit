import math
import torch
import numpy as np
import re
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Union
from transformers import AutoTokenizer
from itertools import groupby
from bisect import bisect_left, bisect_right


from .audio_tokeniser import AudioTokeniser
from ..feature_extractor.audio_feature_extractor import AudioFeatureExtractor

SPEECH_TOKEN = '<speech>'
TEXT_TOKEN = '<text>'

class ContentType(Enum):
    TEXT = "TEXT"
    SPEECH = "SPEECH"

@dataclass
class GenerationInput:
    content: Union[str, os.PathLike, torch.Tensor, np.ndarray]
    content_type: ContentType

    @classmethod
    def from_tuple(cls, tup):
        content_type, content = tup
        content_type = content_type.upper()
        assert content_type in [
            "SPEECH",
            "TEXT",
        ], f"expects content_type to be one of ['SPEECH', 'TEXT'], found '{content_type}'"
        if content_type == "TEXT":
            content_type = ContentType.TEXT
        elif content_type == "SPEECH":
            content_type = ContentType.SPEECH
        return cls(content=content, content_type=content_type)

    def to(self, device):
        if self.content_type == ContentType.TEXT:
            return self
        if isinstance(self.content, np.ndarray):
            return GenerationInput(torch.from_numpy(self.content).to(device), self.content_type)
        if isinstance(self.content, str):
            return GenerationInput(self.content, self.content_type)
        return GenerationInput(self.content.to(device), self.content_type)

InterleavedInputs = List[GenerationInput]



def select_spans_poisson(array_size: int, lambda_param: int, eta: float):
    """
    Select spans from the input array following a Poisson distribution for span lengths,
    dynamically removing irrelevant indices to improve efficiency.

    Args:
        array_size (int): size for output array.
        lambda_param (int): The lambda parameter for the Poisson distribution.
        eta (float): Fraction of the array to be selected.

    Returns:
        np.ndarray: Binary mask of the same size as input_array, with 1 indicating selected spans.
    """
    total_to_select = math.ceil(array_size * eta)
    mask = np.zeros(array_size, dtype=int)

    # Initialize eligible starting indices
    eligible_indices = set(range(array_size))

    selected_count = 0
    while selected_count < total_to_select and eligible_indices:
        # Randomly pick a starting index from eligible indices, and a span length from the Poisson distribution
        start_index = np.random.choice(list(eligible_indices))
        span_length = np.random.poisson(lambda_param)
        end_index = min(start_index + span_length, array_size)

        # Ensure the span does not overlap an already selected span
        if np.any(mask[start_index:end_index] == 1):
            continue

        # Select the span
        mask[start_index:end_index] = 1
        selected_count += end_index - start_index

        # Remove affected indices from the eligible set
        for i in range(start_index, end_index):
            eligible_indices.discard(i)
        if end_index < array_size:
            eligible_indices.discard(end_index)
    return mask


class InterleavingTokeniser(AudioTokeniser):
    """A tokeniser supporting text and speech unit interleaving tokenisation with discrete audio tokens and text token,
    similar to SpiritLM - https://arxiv.org/abs/2402.05755 and others"""
    def __init__(self, speech_tokeniser: AudioFeatureExtractor,
                 dedup: bool = True,
                 pad_token_id: int = 0,
                 num_units: int = 500,
                 load_fe: bool = True,
                 text_tokeniser_path: str = 'facebook/opt-125m',
                 interleave_method: str = 'random',
                 interleave_span: Optional[int] = None,
                 interleave_prob: Optional[float] = None,
                 ):
        super().__init__()
        self.speech_fe = speech_tokeniser if load_fe else None
        self.dedup = dedup
        self.pad_token_id = pad_token_id
        self.num_units = num_units
        self.text_tokeniser = InterleavingTokeniser._init_text_tokeniser(text_tokeniser_path, pad_token_id, num_units)
        self.interleave_method = interleave_method
        self.interleave_span = interleave_span
        self.interleave_prob = interleave_prob

    @staticmethod
    def _init_text_tokeniser(text_tokeniser_path: str, pad_token_id: int, num_units: int) -> AutoTokenizer:
        tokeniser = AutoTokenizer.from_pretrained(text_tokeniser_path)
        tokeniser.pad_token_id = pad_token_id
        tokeniser.padding_side = "right"
        tokeniser.add_tokens([f"<Un{x}>" for x in range(num_units)] + [SPEECH_TOKEN, TEXT_TOKEN])
        return tokeniser

    @torch.inference_mode()
    def audio_represent(self, wav: torch.Tensor, lens: Optional[torch.Tensor] = None) -> List[Dict]:
        toks = self.speech_fe.extract(wav, lens)
        if self.dedup:
            toks = [zip(*[(k, len(list(g))) for k,g in groupby(t.tolist())]) for t in toks]
        else:
            toks = [(t, [1]*len(t)) for t in toks]
        return [{'units': u, 'duration': d} for u, d in toks]

    def _assign_interleaved_modality(self, aligned_text: List) -> List:
        modalities = []
        if self.interleave_method == 'random':
            for w, s, e in aligned_text:
                cur_modality = 'text' if torch.rand(1) < 0.5 else 'audio'
                modalities.append((w, s, e, cur_modality))
        elif self.interleave_method == 'span':
            patience = 0
            for w, s, e in aligned_text:
                mod_now = 'text' if torch.rand(1) >= self.interleave_prob else 'audio'
                if mod_now == 'audio':
                    patience = self.interleave_span
                cur_modality = 'audio' if patience > 0 else 'text'
                modalities.append((w, s, e, cur_modality))
                patience -= 1
        elif self.interleave_method == 'poisson':
            speech_words = select_spans_poisson(len(aligned_text), self.interleave_span, self.interleave_prob)
            for i, (w, s, e) in enumerate(aligned_text):
                cur_modality = 'audio' if speech_words[i] > 0 else 'text'
                modalities.append((w, s, e, cur_modality))
        return modalities

    def _create_interleaved_text(self, rep: Dict, aligned_text: List) -> str:
        out = ''
        cur = []
        unit_time = np.cumsum(rep['duration']) * self.speech_fe.get_unit_duration()
        for i in range(len(aligned_text)):
            w, s, e, m = aligned_text[i]
            if i == 0:
                out += SPEECH_TOKEN if m != 'text' else TEXT_TOKEN
            if m == 'text':
                cur.append(w)
            else:
                cur.append((s, e))
            if i == (len(aligned_text) - 1) or m != aligned_text[i + 1][3]:
                if m == 'text':
                    out += ''.join(cur)
                    if i != (len(aligned_text) - 1):
                        out += SPEECH_TOKEN
                else:
                    start, end = cur[0][0], cur[-1][1]
                    start_unit, end_unit = bisect_left(unit_time, start), bisect_right(unit_time, end)
                    out += ''.join([f'<Un{u}>' for u in rep['units'][start_unit:end_unit]])
                    if i != (len(aligned_text) - 1):
                        out += TEXT_TOKEN
                cur = []
        return out

    def _interleave_units(self, rep: Dict) -> str:
        # Decide which words are speech and which are text
        modalities = self._assign_interleaved_modality(rep['aligned_text'])
        # create the output text based on previously assigned modality of each word
        return self._create_interleaved_text(rep, modalities)

    def stringify_representation(self, reps: List[Dict], mode: str = 'test') -> List[str]:
        out = []
        for cur in reps:
            if mode == 'train':
                out.append(self._interleave_units(cur))
            elif mode == 'test':
                out.append(''.join([f'<Un{u}>' for u in cur['units']]))
        return out

    def string_tokenise(self, audio_repr: List[str], **tokenise_kwargs) -> dict:
        return self.text_tokeniser(audio_repr, add_special_tokens=True, **tokenise_kwargs)

    def _stringify_interleaved(self, inp: Union[InterleavedInputs, List[tuple]]) -> str:
        """
        Convert a single interleaved input to a string representation, by extracting Units from the audio and joining it
        with the text segments. This currently doesn't batch audio segments, instead works sequentially which could be
        slow with lots of audio segments.
        """
        if inp and isinstance(inp[0], tuple):  # Convert tuples to GenerationInput
            inp = [GenerationInput.from_tuple(i) for i in inp]

        cur_str = ""
        prev_mod = None
        for segment in inp:
            if segment.content_type.value == ContentType.SPEECH.value:
                if prev_mod != "s":
                    cur_str += f"{SPEECH_TOKEN}"
                cur_str += self.stringify_representation(self.audio_represent(segment.content.unsqueeze(0)))[0]
                prev_mod = "s"  # speech
            elif segment.content_type.value == ContentType.TEXT.value:
                if prev_mod != "t":
                    cur_str += f"{TEXT_TOKEN}"
                cur_str += segment.content
                prev_mod = "t"  # text
            else:
                raise ValueError(f"Unknown content type: {segment.content_type.value}")
        return cur_str

    def tokenise(self, inputs: Union[torch.Tensor, List[InterleavedInputs]], lens: torch.Tensor = None) -> dict:
        if isinstance(inputs, torch.Tensor):  # Input is Speech Only batch
            str_repr = self.stringify_representation(self.audio_represent(inputs, lens))
            return self.string_tokenise(str_repr, return_tensors='pt', padding=True)
        elif isinstance(inputs, list):  # Input is Interleaved batch
            str_reps = []
            for inp in inputs:
                str_reps.append(self._stringify_interleaved(inp))
            return self.string_tokenise(str_reps, return_tensors='pt', padding=True)
        else:
            raise ValueError(f"Inputs should be a list of InterleavedInputs or a torch.Tensor, got {type(inputs)}")

    def build_prompt(self, wav: torch.Tensor, lens: Optional[torch.Tensor] = None) -> dict:
        # TODO: this requires special treatment if we want to create specific modalities etc
        s_prompt = self.stringify_representation(self.audio_represent(wav, lens))
        tokens = self.string_tokenise(s_prompt, return_tensors='pt', padding=True)
        # remove eos if exists, some tokenizers don't add it
        if self.text_tokeniser.eos_token_id is not None and (tokens['input_ids'][..., -1] == self.text_tokeniser.eos_token_id).any():
            tokens = {k: v[..., :-1] for k, v in tokens.items()}
        return tokens

    def prepare_sample(self, sample: dict, **tokenise_kwargs) -> dict:
        return self.string_tokenise(sample['audio_repr'], **tokenise_kwargs)

    def decode_sample(self, tokens: torch.tensor) -> torch.Tensor:
        # currently assume tokens are only audio tokens beacuse we can use used_token_modality to ignore text tokens when generating audio
        # other options will be added later
        tokens = tokens[(tokens != self.text_tokeniser.pad_token_id) & (tokens != self.text_tokeniser.bos_token_id) & (tokens != self.text_tokeniser.eos_token_id)]
        audio_repr = self.text_tokeniser.decode(tokens)
        return torch.tensor([int(num) for num in re.findall(r'<Un(\d+)>', audio_repr)])
    
    @property
    def fe_sample_rate(self) -> int:
        if self.speech_fe is None:
            raise RuntimeError("This tokeniser does not have a feature extractor, please make sure you are using the correct feature extractor")
        return self.speech_fe.sample_rate
    
    def get_ignore_tokens(self, used_token_modality: Optional[str]) -> Optional[List[int]]:
        """
        Get the tokens to ignore in the log likelihood calculation
        :param used_token_modality: the tokens modality to use
        :return: a list of tokens to ignore based on the modality
        """
        num_text_tokens = len(self.text_tokeniser) - self.num_units - len([SPEECH_TOKEN, TEXT_TOKEN])
        special_tokens = [self.text_tokeniser.bos_token_id, self.text_tokeniser.eos_token_id]
        if used_token_modality == "speech_only":
            text_tokens = list(x for x in range(0, num_text_tokens) if x not in special_tokens)
            text_tokens += [self.text_tokeniser.encode(SPEECH_TOKEN)[0], self.text_tokeniser.encode(TEXT_TOKEN)[0]]
            return text_tokens
        if used_token_modality == "text_only":
            speech_tokens = list(x for x in range(num_text_tokens, len(self.text_tokeniser)) if x not in special_tokens + [self.text_tokeniser.encode(SPEECH_TOKEN)[0], self.text_tokeniser.encode(TEXT_TOKEN)[0]])
            return speech_tokens
        return None