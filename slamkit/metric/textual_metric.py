import logging
logger = logging.getLogger(__name__)

import json
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import re


class HellaSwagDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.data = []

        with open(path, 'r') as file:
            self.data = [json.loads(line) for line in file]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        positive_index = data['label']
        ctx = data["ctx_a"] + " " + data["ctx_b"].capitalize()
        query = HellaSwagDataset.preprocess(data["activity_label"] + ": " + ctx)
        endings = [HellaSwagDataset.preprocess(ending) for ending in data['endings']]
        full_sentences = [query + ending for ending in endings]

        return full_sentences[positive_index:] + full_sentences[:positive_index]

    @staticmethod
    def preprocess(text):
        text = text.strip()
        # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text



def textual_metric(model, dataset, used_token_modality, mean_nll: bool=True,
                     batch_size: int = 1, num_workers=8, pin_memory=True):
    dl = DataLoader(dataset, batch_size=batch_size ,num_workers=num_workers, pin_memory=pin_memory)
    res_list = []

    counter = 0
    for sample_files in tqdm(dl):
        counter +=1

        with torch.no_grad():
            results = [
                model.text_log_likelihood(sample, used_token_modality=used_token_modality, mean_nll=mean_nll)
                for sample in sample_files
            ]

        res = (results[0] > torch.stack(results[1:]).max(dim=0).values).int()
        res_list.append(res)

    res_list = torch.cat(res_list)
    return res_list.float().mean().cpu().item()


def hellaswag(model, data_path, used_token_modality, mean_nll=True,
               batch_size=1, num_workers=8, pin_memory=True):
    dataset = HellaSwagDataset(data_path)
    assert len(dataset) > 0, f"no samples found for {data_path}"
    res = textual_metric(model, dataset, used_token_modality, mean_nll, batch_size, num_workers, pin_memory)
    logging.info(f"HellaSwag: {res:.4f}")
    return {'HellaSwag': res}

