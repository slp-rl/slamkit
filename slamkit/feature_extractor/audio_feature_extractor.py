import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, List


class AudioFeatureExtractor(ABC, torch.nn.Module):
    @abstractmethod
    @torch.inference_mode()
    def extract(self, wav: torch.Tensor, lens: Optional[torch.Tensor] = None) -> List[np.array]:
        """
        Extract features from the audio samples
        :param wav: a batch of audio samples
        :param lens: the length of the non-padding part of the audio samples
        :return: a batch of features
        """
        pass

    @abstractmethod
    def get_unit_duration(self) -> float:
        """
        Get the duration of each unit in seconds (this is used for interleaving, and is the sample_rate divided by the
        down sampling factor)
        :return: time in seconds
        """
        pass

    @property
    def sample_rate(self) -> int:
        raise NotImplementedError("This feature extractor does not have a sample rate")