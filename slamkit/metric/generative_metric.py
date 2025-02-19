import torchaudio
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from glob import glob, iglob
import logging
from typing import Optional

from .metric_utils import get_whisper_pipeline, get_llm, get_llm_preplexity
from ..model.speech_lm import SpeechLM
from ..utils.calculation_utils import calc_auto_bleu

logger = logging.getLogger(__name__)



class PromptDataset(Dataset):
    def __init__(self, glob_path, prompt_length=None, sample_rate=16000, num_files=None):
        self.prompt_length = prompt_length
        self.sample_rate = sample_rate
        if num_files is None:
            self.data = glob(glob_path, recursive=True)
        else:
            paths = iglob(glob_path, recursive=True)
            self.data = [next(paths) for _ in range(num_files)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file = self.data[idx]
        audio, sr = torchaudio.load(file)
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
        audio = audio.squeeze(0)
        if self.prompt_length is not None:
            audio = audio[:int(self.prompt_length*self.sample_rate)]
        return audio, audio.shape[-1]

def pad_collate(batch):
    audio, l = zip(*batch)
    audio = pad_sequence(audio, batch_first=True, padding_value=0)
    return audio, torch.tensor(l)


def generate(model: SpeechLM, data_path: str, batch_size: int, used_tokens_modality:Optional[str] = None, prompt_length: Optional[int] = None, sample_rate: int = 16000,
             num_files: Optional[int] = None, num_workers: int = 8,
             pin_memory: bool = True, **generate_kwargs):
    dataset = PromptDataset(data_path, prompt_length=prompt_length,
                            sample_rate=sample_rate, num_files=num_files)
    assert len(dataset) > 0, f"no samples found for {data_path}"
    dl = DataLoader(dataset, batch_size=batch_size, collate_fn=pad_collate,
                    num_workers=num_workers, pin_memory=pin_memory)
    res = []
    prompts = []
    with torch.inference_mode():
        for audio, l in tqdm(dl):
            audio = audio.to(model.device)
            res.extend(model.generate(audio, l, used_tokens_modality, **generate_kwargs))
            prompts.extend(audio)
    return {'generate': res, "prompts": prompts}
        
def asr_perplexity(model: SpeechLM, data_path: str, batch_size: int, whisper_model: str, llm_name_or_path: str, used_tokens_modality:Optional[str] = None,
                   prompt_length: Optional[int] = None, auto_bleu_n: int = 2,
                   sample_rate: int = 16000, num_files: Optional[int] = None , num_workers: int = 8, pin_memory: bool = True, **generate_kwargs): 
    from nltk.tokenize import NLTKWordTokenizer
    nltk_word_tokenizer = NLTKWordTokenizer()

    dataset = PromptDataset(data_path, num_files=num_files, prompt_length=prompt_length, sample_rate=sample_rate)
    assert len(dataset) > 0, f"no samples found for {data_path}"
    dl = DataLoader(dataset, batch_size=batch_size, collate_fn=pad_collate,
                    num_workers=num_workers, pin_memory=pin_memory)
    whisper_pipeline = get_whisper_pipeline(whisper_model, device=model.device)
    llm, text_lm_tokeniser = get_llm(llm_name_or_path, device=model.device)
    nlls, gen, prompts, bleus = [], [], [], []
    for audio, l in tqdm(dl):
        audio = audio.to(model.device)
        gen_res = model.generate(audio, l, used_tokens_modality, **generate_kwargs)
        gen.extend(gen_res)
        prompts.extend(audio)
        gen_res_texts = whisper_pipeline([gen.cpu().numpy() for gen in gen_res], batch_size=len(gen_res))
        res_texts = [res_text["text"] if gen.shape[-1] > 0 else "" for gen, res_text in zip(gen_res, gen_res_texts)]
        bleus.extend([calc_auto_bleu(res_text, nltk_word_tokenizer, auto_bleu_n) for res_text in res_texts])
        nlls.extend(get_llm_preplexity(llm, text_lm_tokeniser, res_texts, device=model.device))
    
    return {'asr_perplexity': torch.stack(nlls).mean().exp().item(),
            f"auto-belu-{auto_bleu_n}": torch.as_tensor(bleus).mean().item(), 
            "generate": gen, "prompts": prompts}
