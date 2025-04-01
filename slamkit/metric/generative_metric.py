import json
import torchaudio
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from glob import glob, iglob
import logging
from typing import Optional
import os

from .metric_utils import get_whisper_pipeline, get_llm, get_llm_preplexity, get_judge
from ..model.speech_lm import SpeechLM
from ..utils.calculation_utils import calc_auto_bleu

logger = logging.getLogger(__name__)

def get_cut_location(alignment, prompt_length):
    """
    Find the closest word in the alignment to the prompt length.
    """
    endtimes = torch.tensor([word[2] for word in alignment])
    dist = torch.abs(endtimes - prompt_length)
    time = endtimes[dist.argmin()].item()        
    return time

def is_shorter(file, min_file_length):
    metadata = torchaudio.info(file)
    if metadata.num_frames < min_file_length * metadata.sample_rate:
        return True
    return False

class PromptDataset(Dataset):
    def __init__(self, glob_path, prompt_length=None, sample_rate=16000, num_files=None, min_file_length=None, use_alignment=False, alignment_folder=None):
        self.prompt_length = prompt_length
        self.sample_rate = sample_rate
        if num_files is None:
            self.data = glob(glob_path, recursive=True)
            if min_file_length is not None:
                self.data = list(filter(lambda x: not is_shorter(x, min_file_length), self.data))
        else:
            self.data = []
            paths = iglob(glob_path, recursive=True)
            for path in paths:
                if len(self.data) >= num_files:
                    break
                if min_file_length is not None:
                    if is_shorter(path, min_file_length):
                        continue
                self.data.append(path)
        self.use_alignment = use_alignment
        self.alignment_folder = alignment_folder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file = self.data[idx]
        audio, sr = torchaudio.load(file)
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
        if audio.ndim == 2:
            audio = audio.mean(dim=0)
        if self.prompt_length is not None and not self.use_alignment:
            audio = audio[:int(self.prompt_length*self.sample_rate)]
        elif self.prompt_length is not None and self.use_alignment:
            alignment_path = self.get_alignment_path(file)
            with open(alignment_path, 'r') as f:
                meta = json.load(f)
                alignment = meta['aligned_text']
            prompt_length = get_cut_location(alignment, self.prompt_length)
            audio = audio[:int(prompt_length*self.sample_rate)]
        return audio, audio.shape[-1]
    
    def get_alignment_path(self, file):
        if self.alignment_folder is None:
            return file.replace(".wav", ".json")
        basename = os.path.basename(file)
        alignment_path = os.path.join(self.alignment_folder, basename[:basename.find(".")] + ".json")
        return alignment_path

def pad_collate(batch):
    audio, l = zip(*batch)
    audio = pad_sequence(audio, batch_first=True, padding_value=0)
    return audio, torch.tensor(l)


def generate(model: SpeechLM, data_path: str, batch_size: int, used_tokens_modality:Optional[str] = None, prompt_length: Optional[int] = None, 
             min_file_length: Optional[int] = None, alignment_folder: Optional[str] = None, use_alignment: bool = False, sample_rate: int = 16000,
             num_files: Optional[int] = None, num_workers: int = 8,
             pin_memory: bool = True, **generate_kwargs):
    dataset = PromptDataset(data_path, prompt_length=prompt_length,
                            sample_rate=sample_rate, num_files=num_files, min_file_length=min_file_length, 
                            alignment_folder=alignment_folder, use_alignment=use_alignment)
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
                   prompt_length: Optional[int] = None, min_file_length: Optional[int] = None, alignment_folder: Optional[str] = None, use_alignment: bool = False,
                   auto_bleu_n: int = 2,
                   sample_rate: int = 16000, num_files: Optional[int] = None , num_workers: int = 8, pin_memory: bool = True, **generate_kwargs): 
    from nltk.tokenize import NLTKWordTokenizer
    nltk_word_tokenizer = NLTKWordTokenizer()

    dataset = PromptDataset(data_path, num_files=num_files, prompt_length=prompt_length, sample_rate=sample_rate, min_file_length=min_file_length, 
                            alignment_folder=alignment_folder, use_alignment=use_alignment)
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

def llm_as_judge(model: SpeechLM, data_path: str, batch_size: int, whisper_model: str, llm_name_or_path:str, instruction:str, used_tokens_modality:Optional[str] = None,
                 prompt_length: Optional[int] = None, min_file_length: Optional[int] = None, alignment_folder: Optional[str] = None, use_alignment: bool = False,
                 sample_rate: int = 16000, num_files: Optional[int] = None , num_workers: int = 8, pin_memory: bool = True, **generate_kwargs):
    
    dataset = PromptDataset(data_path, num_files=num_files, prompt_length=prompt_length, sample_rate=sample_rate, min_file_length=min_file_length, 
                            alignment_folder=alignment_folder, use_alignment=use_alignment)
    assert len(dataset) > 0, f"no samples found for {data_path}"
    assert instruction is not None, "llm_as_judge requires instruction"
    assert "[prompt_audio_transcription]" in instruction, "llm_as_judge requires [prompt_audio_transcription] in instruction"
    assert "[generated_audio_transcription]" in instruction, "llm_as_judge requires [generated_audio_transcription] in instruction"
    dl = DataLoader(dataset, batch_size=batch_size, collate_fn=pad_collate,
                    num_workers=num_workers, pin_memory=pin_memory)
    whisper_pipeline = get_whisper_pipeline(whisper_model, device=model.device)
    judge = get_judge(llm_name_or_path, device=model.device, batch_size=batch_size)
    res, gen, prompts, texts = [], [], [], []
    prompt_texts, gen_texts = [], []
    for audio, l in tqdm(dl, desc="Generating"):
        audio = audio.to(model.device)
        gen_res = model.generate(audio, l, used_tokens_modality, remove_prompt=True, **generate_kwargs)
        gen.extend(gen_res)
        prompts.extend(audio)
        prompt_audio_transcriptions = whisper_pipeline([sample.cpu().numpy() for sample in audio], batch_size=len(audio))
        generated_audio_transcriptions = whisper_pipeline([gen.cpu().numpy() for gen in gen_res], batch_size=len(gen_res))
        prompt_audio_transcriptions = [res_text["text"] if sample.shape[-1] > 0 else "" for sample, res_text in zip(audio, prompt_audio_transcriptions)]
        generated_audio_transcriptions = [res_text["text"] if gen.shape[-1] > 0 else "" for gen, res_text in zip(gen_res, generated_audio_transcriptions)]
        text = [
            instruction.replace("[prompt_audio_transcription]", prompt_audio_transcription).replace("[generated_audio_transcription]", generated_audio_transcription) for 
            prompt_audio_transcription, generated_audio_transcription in zip(prompt_audio_transcriptions, generated_audio_transcriptions)
        ]
        texts.extend(text)
        prompt_texts.extend(prompt_audio_transcriptions)
        gen_texts.extend(generated_audio_transcriptions)
    res = judge(texts)
    res = list(filter(lambda x: x is not None, res))
    text_res = [(p_text,g_text) for p_text, g_text, r in zip(prompt_texts, gen_texts, res)]
    logger.info("got response for {} out of {}".format(len(res), len(dataset)))
    return {'llm_as_judge': torch.as_tensor(res).float().mean().item(), "generate": gen, "prompts": prompts, "text_res": text_res}
