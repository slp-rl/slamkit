import logging
logger = logging.getLogger(__name__)

import torchaudio
import torch
from tqdm import tqdm
from glob import glob, iglob
from typing import List, Optional
from torch.utils.data import DataLoader, Dataset

from ..tokeniser.interleaving_tokeniser import GenerationInput


class CrossModalPromptDataset(Dataset):
    def __init__(self, glob_path, prompt_length=None, prompt_modality='TEXT', sample_rate=16000, num_files=None):
        super().__init__()
        if num_files is None:
            self.data = glob(glob_path, recursive=True)
        else:
            self.data = []
            paths = iglob(glob_path, recursive=True)
            for path in paths:  # Used to avoid StopIteration when requesting more files than available
                if len(self.data) >= num_files:
                    break
                self.data.append(path)

        self.prompt_modality = prompt_modality
        self.prompt_length = prompt_length  # only relevant for audio prompts
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _load_text(txt_file):
        with open(txt_file, 'r') as f:
            return f.read().strip()

    @staticmethod
    def _load_audio(path, sr=16000):
        wav, _sr = torchaudio.load(path)
        if sr != _sr:
            wav = torchaudio.functional.resample(wav, _sr, sr)
        return wav[0]

    def __getitem__(self, idx):
        if self.prompt_modality == 'SPEECH':
            w = CrossModalPromptDataset._load_audio(self.data[idx], self.sample_rate)
            if self.prompt_length is not None:
                w = w[:int(self.prompt_length * self.sample_rate)]
            return list([GenerationInput.from_tuple(('SPEECH', w))])
        elif self.prompt_modality == 'TEXT':
            return list([GenerationInput.from_tuple(('TEXT', CrossModalPromptDataset._load_text(self.data[idx])))])
        else:
            raise ValueError(f'Unknown prompt modality {self.prompt_modality}')


def collate_fn(batch):
    return batch  # Preserves the batch as list-of-lists structure


def _list_to_device(l: List[GenerationInput], device):
    return [t.to(device) for t in l]

def generate(model, data_path: str, batch_size: int, prompt_modality: Optional[str] = None,
             output_modality: Optional[str] = None, prompt_length: Optional[int] = None, sample_rate: int = 16000,
             num_files: Optional[int] = None, num_workers: int = 8, pin_memory: bool = True, **generate_kwargs):
    dataset = CrossModalPromptDataset(data_path, prompt_modality=prompt_modality, prompt_length=prompt_length,
                                      sample_rate=sample_rate, num_files=num_files)
    assert len(dataset) > 0, f"no samples found for {data_path}"
    dl = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    res = []
    prompts = []
    with torch.inference_mode():
        for inp in tqdm(dl):
            inp = [_list_to_device(p, model.device) for p in inp]
            res.extend(model.generate(inp, output_modality=output_modality, **generate_kwargs))
            prompts.extend(inp)
    return {'generate': res, "prompts": prompts}