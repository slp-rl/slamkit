import os
import torch
import joblib
import warnings
import math
import numpy as np
import torch.nn.functional as F
from typing import Optional
from torch.hub import download_url_to_file

from transformers import HubertModel, HubertConfig

from .audio_feature_extractor import AudioFeatureExtractor


class HubertFeatureExtractor(AudioFeatureExtractor):
    def __init__(self, pretrained_model: str = 'facebook/hubert-base-ls960',
                 kmeans_path: str = "https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960_L9_km500.bin",
                 layer: int = 9, num_units: int = 500, compile: bool = False, cache_path: Optional[str] = None, load_config_only: bool = False):
        super().__init__()
        if cache_path is None:
            cache_path = os.environ.get('SLAMKIT_CACHE', os.path.expanduser('~/.cache/slamkit'))
        os.makedirs(cache_path, exist_ok=True)
        
        self.layer = layer
        self.num_units = num_units

        if load_config_only:
            self.config_model = HubertConfig.from_pretrained(pretrained_model)
            return
        if not os.path.exists(f'{cache_path}/kmeans_model.bin'):
            download_url_to_file(kmeans_path, f'{cache_path}/kmeans_model.bin')
        self.model = HubertModel.from_pretrained(pretrained_model)
        if compile:
            self.model = torch.compile(self.model, dynamic=True)
        self.config_model = self.model.config
        self.clustering = load_kmeans_model(f'{cache_path}/kmeans_model.bin')


    @torch.inference_mode()
    def extract(self, wav, lens=None):
        cont = self.model(F.pad(wav, (40, 40)), output_hidden_states=True).hidden_states[self.layer].cpu().numpy()
        toks = batch_cluster(self.clustering, cont)
        if lens is not None:
            # relative length of the non-padding part
            rel_l = ((lens.float() / wav.shape[1]) * toks.shape[1]).ceil().int()
        else:
            rel_l = [toks.shape[-1]] * len(toks)

        return [t[:l] for t, l in zip(toks, rel_l)]

    def get_unit_duration(self) -> float:
        return math.prod(self.config_model.conv_stride) / self.sample_rate
    
    @property
    def sample_rate(self) -> int:
        return 16_000

def load_kmeans_model(checkpoint_path: str):
    with open(checkpoint_path, "rb") as fd:
        with warnings.catch_warnings():
            # produces lots of version warnings which can be annoying when we have many workers
            warnings.simplefilter("ignore")
            kmeans_model = joblib.load(fd)
            # some of the GSLM checkpoints (CPC) were saved under a different scikit version
            if not hasattr(kmeans_model, "_n_threads"):
                kmeans_model._n_threads = 40

    kmeans_model.verbose = False
    return kmeans_model


def batch_cluster(model, data: np.array) -> np.array:
    """Cluster a batch of contiuous sequences into a batch of sequences of tokens, while handling shapes.
    Args:
        model: The clustering model.
        data: [B, T, C] A batch of continuous sequences.
        Returns: [B, T] A batch of token sequences.
    """
    B, T, C = data.shape
    return model.predict(data.reshape(B * T, C)).reshape(B, T)
