import logging
logger = logging.getLogger(__name__)

import torchaudio
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence


class ModellingMetricDataset(Dataset):
    def __init__(self, path, sep="_", subfolder=True):
        super().__init__()
        self.data = []
        if subfolder:
            for f in Path(path).iterdir():
                if f.is_dir():
                    self.data += sorted(list(f.glob("*.wav")), key=lambda x: int(x.name.split(sep)[0]))
        else:
            self.data += sorted(list(Path(path).glob("*.wav")), key=lambda x: int(x.name.split(sep)[0]))

    def __len__(self):
        return len(self.data) // 2

    def __getitem__(self, idx):
        pos_file, neg_file = self.data[2 * idx], self.data[2 * idx + 1]
        pos = torchaudio.load(pos_file)[0][0]
        neg = torchaudio.load(neg_file)[0][0]
        return pos, neg, pos.shape[-1], neg.shape[-1]


class SalmonDataset(Dataset):
    def __init__(self, path, part):
        self.data = []
        self.salmon_path = Path(path)
        dir_path = self.salmon_path / part
        paths = list(dir_path.glob("*.wav"))

        max_sample_index = -1
        for path in paths:
            stem = str(path.stem)
            parts = stem.split("_")
            sample_index = int(parts[1])
            if sample_index > max_sample_index:
                max_sample_index = sample_index

        self.data = [[] for _ in range(max_sample_index + 1)]

        for path in paths:
            stem = str(path.stem)
            parts = stem.split("_")
            sample_index = int(parts[1])
            self.data[sample_index].append(str(path))

        for sample_list in self.data:
            sample_list.sort()

        self.data = [lst for lst in self.data if lst]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_files = self.data[idx]
        pos = torchaudio.load(sample_files[0])[0][0]
        neg = torchaudio.load(sample_files[1])[0][0]
        return pos, neg, pos.shape[-1], neg.shape[-1]


def pad_collate(batch):
    pos, neg, l_pos, l_neg = zip(*batch)
    # pad with silence
    pos = pad_sequence(pos, batch_first=True, padding_value=0)
    neg = pad_sequence(neg, batch_first=True, padding_value=0)
    return pos, neg, torch.tensor(l_pos), torch.tensor(l_neg)


def modelling_metric(model, dataset, used_token_modality, mean_nll: bool=True,
                     batch_size: int = 1, num_workers=8, pin_memory=True):
    dl = DataLoader(dataset, batch_size=batch_size, collate_fn=pad_collate ,num_workers=num_workers, pin_memory=pin_memory)
    res_list = []

    for sample_files in tqdm(dl):
        pos, neg, l_pos, l_neg = sample_files
        pos, neg = pos.to(model.device), neg.to(model.device)
        l_pos, l_neg = l_pos.to(model.device), l_neg.to(model.device)
        with torch.no_grad():
            pos_likelihood = model.log_likelihood(pos, l_pos, used_token_modality=used_token_modality, mean_nll=mean_nll)
            neg_likelihood = model.log_likelihood(neg, l_neg, used_token_modality=used_token_modality, mean_nll=mean_nll)
        res = torch.zeros_like(pos_likelihood)
        res[pos_likelihood > neg_likelihood] = 1
        res[pos_likelihood == neg_likelihood] = 0.5
        res[pos_likelihood < neg_likelihood] = 0

        res_list.append(res)

    res_list = torch.cat(res_list)
    return res_list.float().mean().cpu().item()


def salmon(model, salmon_path, used_token_modality, mean_nll, parts, batch_size, num_workers=8, pin_memory=True):
    if parts[0] == "all":
        parts = ['bg_alignment/', 'bg_all_consistency/', 'bg_domain_consistency/', 'gender_consistency/',
                 'rir_consistency/', 'sentiment_alignment/', 'sentiment_consistency/', 'speaker_consistency/']

    out = dict()
    for part in parts:
        dataset = SalmonDataset(salmon_path, part)
        assert len(dataset) > 0, f"no samples found for {part}"
        cur_res = modelling_metric(model, dataset, used_token_modality, mean_nll, batch_size, num_workers, pin_memory)
        logging.info(f"SALMon - {part}: {cur_res:.4f}")
        out[part] = cur_res

    return out


def swuggy(model, data_path, used_token_modality, mean_nll=True,
           batch_size=1, num_workers=8, pin_memory=True, subfolder=False):
    dataset = ModellingMetricDataset(data_path, sep='_', subfolder=subfolder)
    assert len(dataset) > 0, f"no samples found for {data_path}"
    res = modelling_metric(model, dataset, used_token_modality, mean_nll, batch_size, num_workers, pin_memory)
    logging.info(f"sWUGGY: {res:.4f}")
    return {'sWUGGY': res}


def sblimp(model, data_path, used_token_modality,  mean_nll=True,
           batch_size=1, num_workers=8, pin_memory=True, subfolder=False):
    dataset = ModellingMetricDataset(data_path, sep='+', subfolder=subfolder)
    assert len(dataset) > 0, f"no samples found for {data_path}"
    res = modelling_metric(model, dataset, used_token_modality, mean_nll, batch_size, num_workers, pin_memory)
    logging.info(f"sBLIMP: {res:.4f}")
    return {'sBLIMP': res}

def storycloze(model, data_path, used_token_modality, mean_nll=True,
               batch_size=1, num_workers=8, pin_memory=True, subfolder=False):
    dataset = ModellingMetricDataset(data_path, sep='_', subfolder=subfolder)
    assert len(dataset) > 0, f"no samples found for {data_path}"
    res = modelling_metric(model, dataset, used_token_modality, mean_nll, batch_size, num_workers, pin_memory)
    logging.info(f"StoryCloze: {res:.4f}")
    return {'StoryCloze': res}
