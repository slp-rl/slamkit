import logging
logger = logging.getLogger(__name__)

import torchaudio
import torch
from pathlib import Path
from tqdm import tqdm
from typing import List
from torch.utils.data import DataLoader, Dataset

from ..tokeniser.interleaving_tokeniser import GenerationInput


class CrossModalMetricDataset(Dataset):
    def __init__(self, path, subfolder=True, prompt_modality='TEXT', cont_modality='SPEECH'):
        super().__init__()
        self.data = []
        if subfolder:
            for f in Path(path).iterdir():
                if f.is_dir():
                    self.data += f.glob("*_correct.wav")
        else:
            self.data += Path(path).glob("*_correct.wav")

        self.prompt_modality = prompt_modality
        self.cont_modality = cont_modality

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _load_text(txt_file):
        with open(txt_file, 'r') as f:
            return f.read().strip()

    def __getitem__(self, idx):
        base_path = str(self.data[idx]).split("_correct.wav")[0]
        pos_wav, neg_wav, prompt_wav = base_path + "_correct.wav", base_path + "_incorrect.wav", base_path + "_mutual.wav"
        pos_txt, neg_txt, prompt_txt = base_path + "_correct.txt", base_path + "_incorrect.txt", base_path + "_mutual.txt"
        prompt = CrossModalMetricDataset._load_text(prompt_txt) if self.prompt_modality == 'TEXT' else torchaudio.load(prompt_wav)[0][0]
        pos = CrossModalMetricDataset._load_text(pos_txt) if self.cont_modality == 'TEXT' else torchaudio.load(pos_wav)[0][0]
        neg = CrossModalMetricDataset._load_text(neg_txt) if self.cont_modality == 'TEXT' else torchaudio.load(neg_wav)[0][0]
        pos_sample = [(self.prompt_modality, prompt), (self.cont_modality, pos)]
        neg_sample = [(self.prompt_modality, prompt), (self.cont_modality, neg)]
        return [GenerationInput.from_tuple(t) for t in pos_sample], [GenerationInput.from_tuple(t) for t in neg_sample]


def collate_fn(batch):
    positives, negatives = zip(*batch)  # Unzip into two lists
    return list(positives), list(negatives)  # Ensure lists of lists


def _list_to_device(l: List[GenerationInput], device):
    return [t.to(device) for t in l]


def _modelling_metric(model, dataset, used_token_modality, mean_nll: bool=True,
                     batch_size: int = 1, num_workers=8, pin_memory=True):
    dl = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    res_list = []

    for sample_files in tqdm(dl):
        pos, neg = sample_files
        pos, neg = [_list_to_device(p, model.device) for p in pos], [_list_to_device(n, model.device) for n in neg]
        with torch.no_grad():
            pos_likelihood = model.log_likelihood(pos, used_token_modality=used_token_modality, mean_nll=mean_nll)
            neg_likelihood = model.log_likelihood(neg, used_token_modality=used_token_modality, mean_nll=mean_nll)
        res = torch.zeros_like(pos_likelihood)
        res[pos_likelihood > neg_likelihood] = 1
        res[pos_likelihood == neg_likelihood] = 0.5
        res[pos_likelihood < neg_likelihood] = 0

        res_list.append(res)

    res_list = torch.cat(res_list)
    return res_list.float().mean().cpu().item()


def cm_storycloze(model, data_path, prompt_modality, cont_modality, used_token_modality=None, mean_nll=True,
                  batch_size=1, num_workers=8, pin_memory=True, subfolder=False):
    dataset = CrossModalMetricDataset(data_path, prompt_modality=prompt_modality, cont_modality=cont_modality,
                                      subfolder=subfolder)
    assert len(dataset) > 0, f"no samples found for {data_path}"
    res = _modelling_metric(model, dataset, used_token_modality, mean_nll, batch_size, num_workers, pin_memory)
    logging.info(f"StoryCloze: {res:.4f}")
    return {'StoryCloze': res}
