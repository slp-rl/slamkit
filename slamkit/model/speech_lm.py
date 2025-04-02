import torch
from typing import List, Optional

from .token_lm import TokenLM
from ..tokeniser import AudioTokeniser


class SpeechLM:
    """
    This class wraps a trained TokenLM, as well an AudioTokeniser (and optionally a Vocoder), and provides a simple
    interface for operating over audio samples
    """

    def __init__(self, model: TokenLM, tokeniser: AudioTokeniser, vocoder=None, device='cuda'):
        self.model = model
        self.model.eval()
        self.tokeniser = tokeniser
        self.vocoder = vocoder
        self.device = device
        self.to(device)

    def log_likelihood(self, wavs: torch.Tensor, lens: Optional[torch.Tensor] = None,
                       mean_nll: bool = True, used_token_modality:Optional[str] = None) -> torch.Tensor:
        """
        Given a zero padded tensor wavs, and a tensor for the length of non-padding in each sample calculate the log
        likelihood for each sample.
        :param wavs: A tensor (B, L) of audio zero padded at the end
        :param lens: A tensor (B,) of the length of the non-padding part of the audio
        :param mean_nll: whether to take mean instead of sum thus cancelling length bias
        :param used_token_modality: the tokens modality to use
        :return:
        """
        self.tokeniser.text_tokeniser.padding_side = 'right'
        tokens = self.tokeniser.tokenise(wavs, lens)['input_ids'].to(self.device)
        ignore_tokens = self.tokeniser.get_ignore_tokens(used_token_modality)
        return self.model.log_likelihood(tokens, mean_nll, ignore_tokens)

    def generate(self, wavs: torch.Tensor, lens: Optional[torch.Tensor] = None, output_modality=None,
                 remove_prompt=False, **kwargs) -> List[torch.Tensor]:
        """
        Given a batch of wavs zero padded, generate the continuation tokens or audio if a vocoder is present
        """
        self.tokeniser.text_tokeniser.padding_side = 'left'
        tokens = self.tokeniser.build_prompt(wavs, lens, output_modality=output_modality).to(self.device)
        ignore_tokens = self.tokeniser.get_ignore_tokens(output_modality)
        if ignore_tokens is not None:
            ignore_tokens = [[tok] for tok in ignore_tokens]
        conts = self.model.generate(**tokens, **kwargs, bad_words_ids=ignore_tokens)
        if remove_prompt:
            conts = conts[..., tokens['input_ids'].size(1):]
        decoded_conts = [self.tokeniser.decode_sample(c, output_modality=output_modality) for c in conts]
        if self.vocoder is not None and output_modality.upper() == 'SPEECH':
            return [self.vocoder.vocode(cont.to(self.device)) if cont.shape[-1] > 0 else torch.as_tensor([]).to(device=self.device) for cont in decoded_conts]
        else:
            return decoded_conts

    def to(self, device):
        self.device = device
        self.model.to(device)
        self.tokeniser.to(device)
        if self.vocoder is not None:
            self.vocoder.to(device)
        return self
