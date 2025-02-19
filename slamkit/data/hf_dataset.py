import logging
logger = logging.getLogger(__name__)

import os
from itertools import chain
from omegaconf import DictConfig, ListConfig
from typing import Tuple, Optional
from datasets import load_dataset, Dataset, DatasetDict, interleave_datasets, concatenate_datasets
from transformers import DataCollatorForLanguageModeling, DataCollatorWithFlattening
from glob import glob

from ..utils.calculation_utils import calc_auto_bleu
from ..tokeniser import AudioTokeniser


def split_into_chunks(lst, chunk_size):
    # Split a list into chunks of a given size
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def chunk_texts(examples, chunk_size: int):
    """
    Chunk a pre-tokenised Huggingface datasets into smaller fixed size chunks, while keeping the remainder and not
    adding special tokens individually thus keeping global consistency
    """
    return {k: list(chain.from_iterable(split_into_chunks(l, chunk_size) for l in v)) for k, v in examples.items()}


def init_dataset(cfg: DictConfig, tokeniser: AudioTokeniser) -> Tuple[Dataset, DataCollatorForLanguageModeling]:
    if cfg.data.get('saved_ds_path', False) and os.path.isdir(cfg.data.saved_ds_path):
        logger.info(f'Loading dataset from {cfg.data.saved_ds_path}')
        dataset = DatasetDict.load_from_disk(cfg.data.saved_ds_path)
    elif type(cfg.data.train_path) == list or type(cfg.data.train_path) == ListConfig:
        assert len(cfg.data.train_path) == len(
            cfg.data.train_ratios), 'Number of train paths should match number of train ratios'
        if type(cfg.data.val_path) == str:
            cfg.data.val_path = [cfg.data.val_path]
        assert len(cfg.data.train_path) >= len(
            cfg.data.val_path), 'Number of train paths should be more or equal than number of val paths'
        if len(cfg.data.train_path) > len(cfg.data.val_path):
            cfg.data.val_path = cfg.data.val_path + [None] * (len(cfg.data.train_path) - len(cfg.data.val_path))
        out = []
        for i in range(len(cfg.data.train_path)):
            logger.info(f'Parsing datasets {cfg.data.train_path[i]} and {cfg.data.val_path[i]}')
            ds = parse_single_dataset(cfg, tokeniser, cfg.data.train_path[i], cfg.data.val_path[i])
            if cfg.data.get('repetitions', None):  # Used for creating multi-epochs in interleaving
                ds['train'] = concatenate_datasets([ds['train']] * cfg.data.repetitions[i])
            out.append(ds)
        ds_train = interleave_datasets([d['train'] for d in out], probabilities=cfg.data.train_ratios,
                                       stopping_strategy=cfg.data.stopping_strategy)
        ds_val = concatenate_datasets([d['validation'] for d in out if 'validation' in d])
        dataset = DatasetDict({'train': ds_train, 'validation': ds_val})
    else:
        dataset = parse_single_dataset(cfg, tokeniser, cfg.data.train_path, cfg.data.val_path)

    # save dataset to disk
    if cfg.data.get('saved_ds_path', False) and not os.path.isdir(cfg.data.saved_ds_path):
        logger.info(f'Saving dataset to {cfg.data.saved_ds_path}')
        dataset.save_to_disk(cfg.data.saved_ds_path)

    if cfg.data.packing:
        collator = DataCollatorWithFlattening(return_tensors='pt')
    else:
        collator = DataCollatorForLanguageModeling(tokeniser.text_tokeniser, mlm=False, return_tensors='pt')

    return dataset, collator


def get_filter_fn(sample_units_min_length=None, sample_units_max_length=None):
    assert sample_units_min_length is not None or sample_units_max_length is not None, 'At least one of sample_units_min_length or sample_units_max_length should be non None'
    if sample_units_min_length is None:

        def filter_fn(x):
            return len(x['input_ids']) <= sample_units_max_length

        return filter_fn

    if sample_units_max_length is None:

        def filter_fn(x):
            return len(x['input_ids']) >= sample_units_min_length

        return filter_fn

    def filter_fn(x):
        return sample_units_min_length <= len(x['input_ids']) <= sample_units_max_length

    return filter_fn


def parse_single_dataset(cfg: DictConfig, tokeniser: AudioTokeniser, train_path: str, val_path: Optional[str] = None,
                         verbose: bool = False):
    data_dct = {'train': glob(train_path)}
    if val_path is not None:
        data_dct['validation'] = glob(val_path)
    dataset = load_dataset('json', data_files=data_dct, num_proc=cfg.data.num_proc)
    dataset = dataset.map(lambda x: tokeniser.prepare_sample(x), num_proc=cfg.data.num_proc)
    cols_to_remove = list(set(dataset['train'].column_names) - {'input_ids', 'attention_mask', 'labels'})
    cols_to_remove.sort()  # needed because remove columns creates different fingerprint based on order. keep it sorted for caching mechnism
    dataset = dataset.remove_columns(cols_to_remove)

    if cfg.data.get("sample_units_max_length", None):
        dataset["train"] = dataset["train"].filter(get_filter_fn(
            sample_units_max_length=cfg.data.sample_units_max_length), num_proc=cfg.data.num_proc)

    # Split tokens into chunks of up to model context size
    if cfg.model.context_len is not None:
        dataset = dataset.map(chunk_texts, batched=True, num_proc=cfg.data.num_proc,
                              fn_kwargs={'chunk_size': cfg.model.context_len})

    if cfg.data.get("chunk_units_min_length", None):
        dataset["train"] = dataset["train"].filter(get_filter_fn(
            sample_units_min_length=cfg.data.chunk_units_min_length), num_proc=cfg.data.num_proc)

    if verbose:
        get_token_stats(dataset['train'])

    return dataset


def get_token_stats(ds: Dataset):
    import numpy as np
    ds = ds.map(lambda x: {'num_tokens': len(x['input_ids'])}, num_proc=32)
    logger.info(f"Statistics over tokens: sum: {sum(ds['num_tokens'])}, len_ds: {len(ds)}, n_tokens (mean, var): "
                f"{np.mean(ds['num_tokens'])}, {np.var(ds['num_tokens'])}")

def get_repetition_filter_fn(auto_bleu_n, max_auto_bleu):
    from nltk.tokenize import NLTKWordTokenizer
    nltk_word_tokenizer = NLTKWordTokenizer()

    def filter_fn(x):
        text = x['prompt_text'] + " " + x['chosen_text']
        return calc_auto_bleu(text, nltk_word_tokenizer, auto_bleu_n) < max_auto_bleu
    
    return filter_fn


def init_preference_optimization_dataset(cfg: DictConfig):
    data_dct = {'train': glob(cfg.train_path)}
    if cfg.val_path is not None:
        data_dct['validation'] = glob(cfg.val_path)
    dataset = load_dataset('json', data_files=data_dct, num_proc=cfg.num_proc)
    if cfg.get("repetition_filter", False):
        dataset = dataset.filter(get_repetition_filter_fn(cfg.auto_bleu_n, cfg.max_auto_bleu), num_proc=cfg.num_proc)
    cols_to_remove = list(set(dataset['train'].column_names) - {'prompt', 'chosen', 'rejected'})
    cols_to_remove.sort()  # needed because remove columns creates different fingerprint based on order. keep it sorted for caching mechnism
    dataset = dataset.remove_columns(cols_to_remove)  # maybe we want to do more preprocessing here e.g. context len?
    return dataset
