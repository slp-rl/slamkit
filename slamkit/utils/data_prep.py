import os
import json
import random
from pathlib import Path
from glob import iglob


def parse_ls_text(data_path, ext='trans.txt'):
    """A simple function to parse the LibriSpeech dataset transcriptions into a single dictionary with the utterance ID
    as the key and the transcription as the value."""
    files = iglob(data_path + f'**/*.{ext}', recursive=True)
    out = dict()
    for file in files:
        with open(file, 'r') as f:
            for l in f:
                parts = l.split()
                out[parts[0]] = ' '.join(parts[1:]).lower()
    return out

def parse_transcriptions(data_path, out_path=None):
    with open(data_path, 'r') as f_in:
        data = json.load(f_in)

    for k, v in data.items():
        meta_file = f'{out_path}/{Path(k).stem}' if out_path else os.path.splitext(k)[0]
        meta_file += '.json'
        out = {'file_name': k, 'aligned_text': [tuple(w.values()) for w in v], 'text': ''.join([w["word"] for w in v])}
        with open(meta_file, 'w') as f_out:
            json.dump(out, f_out)

def gopher_rules_pass(sample) -> bool:
    """ function returns True if the sample complies with Gopher rules """
    signals = json.loads(sample["quality_signals"])

    # rule 1: number of words between 50 and 10'000
    word_count = signals["rps_doc_word_count"][0][2]
    if word_count < 50 or word_count > 100_000:
        return False

    # rule 2: mean word length between 3 and 10
    mean_word_length = signals["rps_doc_mean_word_length"][0][2]
    if mean_word_length < 3 or mean_word_length > 10:
        return False

    # rule 2: symbol to word ratio below 0.1
    symbol_word_ratio = signals["rps_doc_symbol_to_word_ratio"][0][2]
    if symbol_word_ratio > 0.1:
        return False

    # rule 3: 90% of lines need to start without a bullet point
    n_lines = signals["ccnet_nlines"][0][2]
    n_lines_bulletpoint_start = sum(map(lambda ln: ln[2], signals["rps_lines_start_with_bulletpoint"]))
    if n_lines_bulletpoint_start / n_lines > 0.9:
        return False

    # rule 4: the ratio between characters in the most frequent 2-gram and the total number
    # of characters must be below 0.2
    top_2_gram_frac = signals["rps_doc_frac_chars_top_2gram"][0][2]
    if top_2_gram_frac > 0.2:
        return False

    # rule 5: ...

    return True


def parse_red_pajama(out_dir, snapshot='2023-14'):
    print('Parsing Red Pajama dataset')
    from tqdm import tqdm
    from datasets import load_dataset
    ds_iterator = load_dataset(
        "togethercomputer/RedPajama-Data-V2",
        snapshots=[snapshot],
        languages=["en"],
        name="default",
        streaming=True,
        trust_remote_code=True,
    )

    os.makedirs(out_dir, exist_ok=True)
    out_path = f'{out_dir}/{snapshot}-en.jsonl'
    f_out = open(out_path, 'a+')

    for sample in tqdm(ds_iterator["train"]):
        if not gopher_rules_pass(sample):
            continue

        out = {'file_name': sample["doc_id"], 'audio_repr': sample["raw_content"]}
        f_out.write(json.dumps(out) + '\n')


def split_repr_file(repr_path, val_path):
    with open(val_path, 'r') as f_val:
        val_data = f_val.readlines()
    val_data = {json.loads(l)['file_name'].split('librilight-vad')[-1] for l in val_data}

    out_val = open(repr_path.replace('.json', '_val.json'), 'w')
    out_train = open(repr_path.replace('.json', '_train.json'), 'w')

    with open(repr_path, 'r') as f_in:
        for l in f_in:
            data = json.loads(l)
            if data['file_name'].split('librilight-vad')[-1] in val_data:
                out_val.write(l)
            else:
                out_train.write(l)

def train_val_split(data_path, val_size=0.01, seed=None):
    """A simple function to split a json file into a train and validation set, according to an approximate  ratio
    without loading the data into memory."""
    out_val = open(data_path.replace('.json', '_val.json'), 'w')
    out_train = open(data_path.replace('.json', '_train.json'), 'w')

    if seed:
        random.seed(seed)
    with open(data_path, 'r') as f_in:
        for l in f_in:
            if random.random() < val_size:
                out_val.write(l)
            else:
                out_train.write(l)


def create_spoken_swag(hf_name: str, out_path: str, num_samples=None, split='validation'):
    """A simple function to create a spoken version of the SWAG dataset from the Hugging Face datasets library."""
    from datasets import load_dataset
    ds = load_dataset(hf_name, split=split)

    # Filter only gold samples
    ds = ds.filter(lambda x: x['gold-source'] == 'gold')

    # Add TTS speaker
    speakers = ['af_heart', 'am_fenrir', 'bf_emma', 'bm_george']
    ds = ds.map(lambda x: {'speaker': random.choice(speakers), **x})

    def select_pos_neg(sample):
        pos_label = sample['label']
        neg_label = random.choice(list((set(range(4)) - {pos_label})))
        pos = sample['sent2'] + ' ' + sample[f'ending{pos_label}']
        neg = sample['sent2'] + ' ' + sample[f'ending{neg_label}']
        base = f'{out_path}/audio/' + sample['video-id'] + "_" + sample['fold-ind'] + "_" + sample['speaker']
        return {'prompt_text': sample['sent1'], 'chosen_text': pos, 'rejected_text': neg,
                'prompt_path': f'{base}_prompt.wav', 'chosen_path': f'{base}_chosen.wav', 'rejected_path': f'{base}_rejected.wav'}

    ds = ds.map(select_pos_neg)
    ds = ds.remove_columns(['video-id', 'fold-ind', 'sent1', 'sent2', 'ending0', 'ending1', 'ending2', 'ending3', 'label', 'gold-source', 'startphrase'])

    # take only subset
    if num_samples:
        ds = ds.select(range(num_samples))

    # write metadata file
    os.makedirs(out_path, exist_ok=True)
    with open(f'{out_path}/spoken_swag_{split}.jsonl', 'w') as out:
        for sample in ds:
            out.write(json.dumps(sample) + '\n')

    # Synthesise audio
    from tts_utils import kokoro
    import soundfile as sf
    os.makedirs(f'{out_path}/audio', exist_ok=True)

    for s in speakers:
        print(f'Synthesising speaker {s}')
        cur_ds = ds.filter(lambda x: x['speaker'] == s)

        for sub in ['prompt', 'chosen', 'rejected']:
            print(f'Synthesising {sub}')
            text = cur_ds[sub + '_text']
            paths = cur_ds[sub + '_path']
            generator = kokoro(texts=text, voice=s)

            for i, (_, _, audio) in enumerate(generator):
                sf.write(paths[i], audio, 24000)


def create_spoken_hellaswag(hf_name: str, out_path: str, num_samples=None, split='validation'):
    """A simple function to create a spoken version of the SWAG dataset from the Hugging Face datasets library."""
    from datasets import load_dataset
    ds = load_dataset(hf_name, split=split)

    # Filter weird samples with non-text symbols like [header], [substeps] etc
    ds = ds.filter(lambda x: not any([t in x['ctx'] for t in ['[', ']', "/", "http", "\\"]]))

    # Add TTS speaker
    speakers = ['af_heart', 'am_fenrir', 'bf_emma', 'bm_george']
    ds = ds.map(lambda x: {'speaker': random.choice(speakers), **x})

    def select_pos_neg(sample):
        pos_label = int(sample['label'])
        neg_label = random.choice(list((set(range(4)) - {pos_label})))
        pos = sample['ctx_b'] + ' ' + sample['endings'][pos_label]
        neg = sample['ctx_b'] + ' ' + sample['endings'][neg_label]
        base = f'{out_path}/audio/{sample["source_id"]}_{sample["ind"]}'
        return {'prompt_text': sample['ctx_a'], 'chosen_text': pos, 'rejected_text': neg,
                'prompt_path': f'{base}_prompt.wav', 'chosen_path': f'{base}_chosen.wav', 'rejected_path': f'{base}_rejected.wav'}

    ds = ds.map(select_pos_neg)
    ds = ds.remove_columns(['ind', 'activity_label', 'ctx_a', 'ctx_b', 'ctx', 'endings', 'source_id', 'split', 'split_type' ,'label'])

    # take only subset
    if num_samples:
        ds = ds.select(range(num_samples))

    # write metadata file
    os.makedirs(out_path, exist_ok=True)
    with open(f'{out_path}/spoken_swag_{split}.jsonl', 'w') as out:
        for sample in ds:
            out.write(json.dumps(sample) + '\n')

    # Synthesise audio
    from tts_utils import kokoro
    import soundfile as sf
    os.makedirs(f'{out_path}/audio', exist_ok=True)

    for s in speakers:
        print(f'Synthesising speaker {s}')
        cur_ds = ds.filter(lambda x: x['speaker'] == s)

        for sub in ['prompt', 'chosen', 'rejected']:
            print(f'Synthesising {sub}')
            text = cur_ds[sub + '_text']
            paths = cur_ds[sub + '_path']
            print(len(text), len(paths))
            generator = kokoro(texts=text, voice=s)

            for i, (_, _, audio) in enumerate(generator):
                sf.write(paths[i], audio, 24000)