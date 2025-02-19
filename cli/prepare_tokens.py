from functools import partial
from multiprocessing.pool import ThreadPool
from omegaconf import DictConfig
import hydra
from tqdm import tqdm
from pathlib import Path
import json
import os
import logging
logger = logging.getLogger(__name__)

from slamkit.tokeniser import tokeniser_factory

def process_jsonl(line, tokeniser, requires_meta, meta_path):
    try:
        cur = json.loads(line)
        if requires_meta:
            meta_file = f'{meta_path}/{Path(cur["file_name"]).stem}' if meta_path else \
                os.path.splitext(cur['file_name'])[0]
            if not os.path.exists(meta_file + '.json'):
                logging.warning(f'{meta_file} does not exist. Skipping')
                return
            with open(meta_file + '.json', 'r') as f:
                meta = json.load(f)
            cur.update(meta)
        cur['audio_repr'] = tokeniser.stringify_representation([cur], mode='train')[0]
        cur.pop('units', None)
        cur.pop('duration', None)
        cur.pop('text', None)
        cur.pop('aligned_text', None)
        cur.pop('split_sentence', None)
        return json.dumps(cur)
    except Exception as e:
        logging.warning(f'Failed to process {line}. Error: {e}, skipping')
        return


@hydra.main(config_name='prepare_tokens', config_path='../config', version_base="1.3")
def prepare_tokens(cfg: DictConfig):
    tokeniser = tokeniser_factory(cfg.tokeniser)

    os.makedirs(cfg.out_path, exist_ok=True)
    out_path = f'{cfg.out_path}/{cfg.data_path.split("/")[-1]}'
    if os.path.exists(out_path):
        logging.warning(f'{out_path} already exists. Deleting it!')
        os.remove(out_path)

    logging.info(f'Starting to prepare tokens')
    with open(cfg.data_path, 'r') as f_in, open(out_path, 'a+') as f_out:
        with ThreadPool(cfg.n_threads) as p:
            for jsonl in tqdm(p.imap(partial(process_jsonl, tokeniser=tokeniser, requires_meta=cfg.tokeniser.get("requires_meta", False), meta_path=cfg.meta_path), f_in)):
                if jsonl:
                        f_out.write(jsonl + '\n')


if __name__ == '__main__':
    prepare_tokens()
