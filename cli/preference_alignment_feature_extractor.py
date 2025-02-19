import logging
from typing import Optional
logger = logging.getLogger(__name__)

import json
import torchaudio
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import hydra
from omegaconf import DictConfig
import json

from slamkit.tokeniser import tokeniser_factory


class PreferenceAlignmentDataset(Dataset):
    def __init__(self, data_path: str, sample_rate: int = 16000, torchaudio_backend: Optional[str] = None):
        self.torchaudio_backend = torchaudio_backend
        self.sample_rate = sample_rate
        self.preference_data = []
        with open(data_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                self.preference_data.append(data)
        
    def __len__(self):
        return len(self.preference_data)
    
    def load_audio(self, path: str):
        waveform, sample_rate = torchaudio.load(path, backend=self.torchaudio_backend)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        return waveform.squeeze(0)

    def __getitem__(self, idx):
        data = self.preference_data[idx]
        prompt_path, chosen_path, rejected_path = data['prompt_path'], data['chosen_path'], data['rejected_path']
        prompt_waveform = self.load_audio(prompt_path)
        chosen_waveform = self.load_audio(chosen_path)
        rejected_waveform = self.load_audio(rejected_path)
        return data, prompt_waveform, prompt_waveform.shape[-1], chosen_waveform, chosen_waveform.shape[-1], rejected_waveform, rejected_waveform.shape[-1]
        
    def subsample_data(self, skip: Optional[int], take: Optional[int]):
        if skip is not None:
            self.preference_data = self.preference_data[skip:]
        if take is not None:
            self.preference_data = self.preference_data[:take]

def pad_collate_fn(batch):
    data, prompt_waveforms, prompt_l, chosen_waveforms, chosen_l, rejected_waveforms, rejected_l = zip(*batch)
    # zip(*batch) returns tuples so '+' concatenates the tuples, this means wav is a tensor of shape (3*len(data), max_len)
    wavs = pad_sequence(prompt_waveforms + chosen_waveforms + rejected_waveforms, batch_first=True)
    return data, wavs, torch.as_tensor(prompt_l+chosen_l+rejected_l)

@hydra.main(config_name='preference_alignment_feature_extractor', config_path='../config', version_base="1.3")
def extract_features(cfg: DictConfig):
    """
    This function extracts features, such as discrete HuBERT clusters and durations, from a preference alignment dataset of audio files. It
    accepts a jsonl file with the following format:
    {"prompt_path" : "path/to/prompt.wav", "chosen_path" : "path/to/chosen.wav", "rejected_path" : "path/to/other.wav"}
    returns jsonl file with the following format:
    {"prompt_path" : "path/to/prompt.wav", "chosen_path" : "path/to/chosen.wav", "rejected_path" : "path/to/other.wav", "prompt" : {"audio_repr": <audio_repr>}, "chosen" : {"audio_repr": <audio_repr>}, "rejected" : {"audio_repr": <audio_repr>}}
    """
    tokeniser = tokeniser_factory(cfg.tokeniser).to(cfg.device)
    dataset = PreferenceAlignmentDataset(cfg.data_path, cfg.sample_rate, cfg.torchaudio_backend)
    dataset.subsample_data(cfg.skip, cfg.take)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, collate_fn=pad_collate_fn)
    with open(cfg.out_path, 'w') as f:
        for data, wavs, lens in tqdm(dataloader):
            wavs, lens = wavs.to(cfg.device), lens.to(cfg.device)
            data_len = len(data)
            tokenised = tokeniser.audio_represent(wavs, lens)
            prompt_tokenised = tokenised[:data_len]
            chosen_tokenised = tokenised[data_len:2*data_len]
            rejected_tokenised = tokenised[2*data_len:]
            for i, data_point in enumerate(data):
                data_point['prompt'] = prompt_tokenised[i]
                data_point['chosen'] = chosen_tokenised[i]
                data_point['rejected'] = rejected_tokenised[i]
                f.write(json.dumps(data_point) + '\n')

if __name__ == '__main__':
    extract_features()
