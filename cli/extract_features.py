import logging
logger = logging.getLogger(__name__)

import os
import json
import torchaudio
import pickle
from glob import iglob
from tqdm import tqdm
from multiprocessing import Pool
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import hydra
from omegaconf import DictConfig
from typing import Optional, Tuple
from functools import partial

from slamkit.tokeniser import tokeniser_factory


class WavDataset(Dataset):
    def __init__(self, data_path: str, ext: str = 'flac', cache_path: Optional[str] = None, sample_rate: int = 16000, torchaudio_backend: Optional[str] = None):
        self.torchaudio_backend = torchaudio_backend
        self.sample_rate = sample_rate
        save_path = None
        if cache_path is not None:
            os.makedirs(cache_path + '/data/', exist_ok=True)
            save_path = f'{cache_path}/data/{data_path.split("/")[-2]}.pkl'  # TODO: make this more robust
            if os.path.exists(save_path):
                with open(save_path, 'rb') as f:
                    self.files = pickle.load(f)
                return
        files = iglob(os.path.join(data_path,f'**/*.{ext}'), recursive=True)
        with Pool() as p:
            self.files = list(tqdm(p.imap(partial(WavDataset._load_sample_meta,backend=self.torchaudio_backend), files)))
        self.files = sorted(self.files, key=lambda x: x[1], reverse=True)  # sort by duration to minimize padding

        if save_path:
            with open(save_path, 'wb') as f:
                pickle.dump(self.files, f)

    @staticmethod
    def _load_sample_meta(f_path, backend=None):
        return f_path, torchaudio.info(f_path, backend=backend).num_frames

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f_name, _ = self.files[idx]
        data,sr = torchaudio.load(f_name, backend=self.torchaudio_backend)
        if sr != self.sample_rate:
            data = torchaudio.functional.resample(data, sr , self.sample_rate)
        if data.dim() == 2:
            data = data.mean(dim=0)
        return f_name, data, len(data)

    def skip(self, skip: int):
        self.files = self.files[skip:]

    def take(self, take: int):
        self.files = self.files[:take]


def pad_wav_collate(batch) -> Tuple[list, torch.Tensor, list, torch.Tensor]:
    f_names, wavs, lens = zip(*batch)
    return f_names, pad_sequence(wavs, batch_first=True, padding_value=0), torch.tensor(lens)


@hydra.main(config_name='extract_features', config_path='../config', version_base="1.3")
def extract_features(cfg: DictConfig):
    """
    This function extracts features, such as discrete HuBERT clusters and durations, from a dataset of audio files. It
    then and saves this dictionary format to jsonl file.
    Note! that this function works with a full tokeniser object for completeness and guarnteed consistency, however it
    is truly dependent on the feature extractor only. Therefore, the output can be used for different tokenisers over
    the same features (such as SpiritLM or Twist over Hubert-25) without re-extraction.
    """
    tokeniser = tokeniser_factory(cfg.tokeniser).to(cfg.device)
    ds = WavDataset(cfg.data_path, cfg.ext, cfg.cache_path, cfg.sample_rate,cfg.torchaudio_backend)
    if cfg.data_skip is not None:
        ds.skip(cfg.data_skip)
    if cfg.data_take is not None:
        ds.take(cfg.data_take)
    dl = DataLoader(ds, collate_fn=pad_wav_collate, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    if os.path.exists(cfg.out_path):
        logging.warning(f'{cfg.out_path} already exists. Appending to it.')
    os.makedirs(os.path.dirname(cfg.out_path), exist_ok=True)
    out_file = open(cfg.out_path, 'a+')
    for f, w, l in tqdm(dl):
        out = []
        speech_repr = tokeniser.audio_represent(w.to(cfg.device), l.to(cfg.device))
        for cur_f, cur_repr in zip(f, speech_repr):
            cur_repr['file_name'] = cur_f
            out.append(json.dumps(cur_repr) + '\n')
        out_file.writelines(out)

    out_file.close()


if __name__ == '__main__':
    extract_features()
