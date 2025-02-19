from .audio_vocoder import AudioVocoder
import torch


class HiFiGANVocoder(AudioVocoder):
    def __init__(self,
                 dense_model_name: str,
                 quantizer_model_name: int,
                 vocab_size: bool, vocoder_suffix: str = None,
                 speaker_meta: str = None,
                 style_meta: str = None,
                 ):
        super().__init__()
        from textless.vocoders.hifigan.vocoder import CodeHiFiGANVocoder
        self.vocoder = CodeHiFiGANVocoder.by_name(dense_model_name,
                                                  quantizer_model_name,
                                                  vocab_size,
                                                  vocoder_suffix,
                                                  speaker_meta, style_meta)

    def vocode(self, tokens: torch.LongTensor, **kwargs) -> torch.Tensor:
        return self.vocoder(tokens, dur_prediction=(self.vocoder.model.dur_predictor is not None), **kwargs)
