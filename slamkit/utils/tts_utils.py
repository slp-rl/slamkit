from typing import Optional, List, Generator
import torch
import g2p_en


class FastSpeech2:
    """
    A wrapper for facebook/fastspeech2-en-ljspeech
    FastSpeech2 trained on the LJ Speech dataset
    """


    def __init__(self, cache_dir: Optional[str] = None, save_sr: int = 16000, eos_padding: int = 30):
        from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
        from fairseq.models.text_to_speech.hub_interface import TTSHubInterface

        models, cfg, self.task = load_model_ensemble_and_task_from_hf_hub(
            "facebook/fastspeech2-en-ljspeech",
            arg_overrides={"vocoder": "hifigan", "fp16": False},
            cache_dir=cache_dir,
        )

        self.sr = self.task.sr
        self.save_sr = save_sr
        self.eos_padding = eos_padding

        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.g2p = g2p_en.G2p()

        self.model = models[0]
        self.model.to(self.device)

        TTSHubInterface.update_cfg_with_data_cfg(cfg, self.task.data_cfg)
        self.generator = self.task.build_generator(models, cfg)


    def generate_wav(self, text: str, alignment: bool = False):
        from fairseq.models.text_to_speech.hub_interface import TTSHubInterface

        sample = TTSHubInterface.get_model_input(self.task, text)

        sample['net_input']['src_tokens'] = sample['net_input']['src_tokens'].to(self.device)
        sample['net_input']['src_lengths'] = sample['net_input']['src_lengths'].to(self.device)

        output = self.generator.generate(self.model, sample)

        if not alignment:
            return output
        else:
            # post process to extract the text alignment
            attn = output[0]['attn']

            current_index = 1
            alignment = []

            for i, word in enumerate(text.split()):
                phonemes = [{",": "sp", ";": "sp"}.get(p, p) for p in self.g2p(word)]
                phonemes = [c for c in phonemes if c.isalnum()]

                # getting the range of text tokens corresponding for a word
                first_token_index = current_index
                last_token_index = current_index + len(phonemes) - 1

                # obtaining the temporal range of the tokens [first_token_index, last_token_index]
                alignment_indices = (torch.tensor((first_token_index, last_token_index), dtype=torch.float32,
                                                  device=self.device).unsqueeze(1) == attn).nonzero(as_tuple=True)[1]
                new_start = alignment_indices[0].item()
                new_end = alignment_indices[-1].item()

                alignment.append((
                    ' ' + word,
                    round((new_start * 256) / self.sr, 3),
                    round((new_end * 256) / self.sr, 3)
                ))
                current_index += len(phonemes)

            return output, alignment


def kokoro(texts: List[str], voice: str = "af_heart", speed: int = 1) -> Generator['KPipeline.Result', None, None]:
    """
    A utility function for hexgrad/Kokoro-82M
    available voices can be found at - https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md
    """
    from kokoro import KPipeline

    lang_code = voice[0]
    pipeline = KPipeline(lang_code=lang_code)

    generator = pipeline(
        texts, voice=voice,
        speed=speed
    )
    return generator

