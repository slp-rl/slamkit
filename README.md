# Slam
The official code for ["_Slamming_: Training a Speech Language Model on One GPU in a Day"]().

<p align="center">
    🌐 <a href="https://pages.cs.huji.ac.il/adiyoss-lab/slamming/" target="_blank">Project</a> | 📃 <a href="https://arxiv.org/abs/" target="_blank">Paper - soon!</a> | 🤗 <a href="https://huggingface.co/collections/slprl/slam-67b58a61b57083505c8876b2" target="_blank">Models & Datasets</a><br>
</p>


![https://pages.cs.huji.ac.il/adiyoss-lab/slamming/](media/slam_web.png)

💻 We also plan to expand this repository to include more features, if you would like
to develop this open source and contribute see [Contributing]().

## Installation
The code was tested with the following `requirements.txt` and `python=3.12`, but should 
also work with other python versions.
```
pip install -r requirements.txt
```
Note that these are minimal requirements, and some features may require additional installation.
For instance, the Vocoder is based on textlesslib [`fairseq`](https://github.com/facebookresearch/textlesslib) so see
their installation guide for more details if you plan to use it.

## Usage
❗If you are only interested in evaluating or generating with a pre-trained SpeechLM, you can skip
straight to the [Evaluation](#evaluation) section.

Our pacakge is built with four main scripts, corresponding to four main stages: [extract_features.py](https://github.com/gallilmaimon/slm_eval/blob/main/slm_eval/extract_features.py), [prepare_tokens.py](https://github.com/gallilmaimon/slm_eval/blob/main/slm_eval/prepare_tokens.py), [train.py](https://github.com/gallilmaimon/slm_eval/blob/main/slm_eval/train.py), [eval.py](https://github.com/gallilmaimon/slm_eval/blob/main/slm_eval/eval.py). 
The core idea is to separate certain logics to share pre-computed representations as much as possible thus save time. 
We explain about each part more below. 

Our codebase uses [Hydra](https://hydra.cc) to manage configurations, we suggest that you read a bit about it if you are unfamiliar with it.

### Extract features
This script takes audio files and outputs a file with discrete token data, using a pre-trained speech tokeniser. 
The representation operate at Tokeniser level to guarantee consistency across different steps, however, this truly only 
depends on the feature extractor thus can be shared between different tokenisers over the same features 
(such as text-interleaving or units only over the same Hubert). 

The output of this script can be directly used for the next phase, which is tokeniser specific.
An example of running it:
```
python slm_eval/extract_features.py data_path=../data/librispeech/audio/dev-clean/ ext=flac out_path=../../data/librispeech/repr/hubert_25hz_dedup/dev_clean.json batch_size=16 tokeniser=unit_hubert_25 tokeniser.feature_extractor.compile=true num_workers=4 data_skip=10 data_take=500"
```
We define the `data_path`, the `ext` (extension for audio), the `tokeniser` which is a config file in this case [config/tokeniser/unit_hubert_25.yaml](https://github.com/gallilmaimon/slm_eval/blob/main/config/tokeniser/unit_hubert_25.yaml).

⚠️ Note! The tokenise script operates over entire audio files without splitting them, which can consume a lot of memory for very large audio files (30 minutes) thus we recommend to run Voice activity detection (VAD) on the files before in order to split them.

❗The audio samples are sorted by length in decreasing order to minimise padding, and fail early for out of memory. You are able to subset the dataset and taking only part of the files with `data_skip=10 data_take=500`

❗using `tokeniser.feature_extractor.compile=true` runs `torch.compile` on the tokeniser which can improve runtime but incur latency in initialisng the model so probably best not to use when debugging.

### Prepare tokens
This script takes the output of `extract_features.py` and prepares the tokens as a string representation for 
training. This script is already dependent on the tokeniser and the tokeniser specific features, such as text.


### Pre-Train
This script takes pre-tokenised data (as output from `prepare_tokens.py`) and trains a speech language model over the tokens.

An example of running it:
```bash
python slm_eval/train.py data.train_path=`../../data/librispeech/tokens/hubert_25hz_dedup/*.json` data.val_path=../../data/librispeech/tokens/hubert_25hz_dedup/dev_clean.json tokeniser=unit_hubert_25 training_args.num_train_epochs=1 training_args.per_device_train_batch_size=16 training_args.gradient_accumulation_steps=4 data.num_proc=32 logger=wandb logger.entity=<Entity> logger.project=<Project> training_args.output_dir=../outputs/baseline
```

❗Note that the `train_path` and `val_path` can be a specific file or a glob path for several files like in the example above. Use this to merge shards or datasets!

❗Note that `training_args` is arguments to a HuggingFace🤗 model so you can pass any argument [it expects](https://huggingface.co/docs/transformers/v4.46.3/en/main_classes/trainer#transformers.TrainingArguments), for instance add `+dataloader_num_workers=8` to use 8 workers!

🧪 We give an example of logging results to Weights & Biases but you could remove this and results will be printed locally


### Preference Alignment
...


### Eval
This script is used to evaluate metrics of your awesomely trained SpeechLM. 
We currently support modelling metrics: sWUGGY, sBLIMP, Spoken StoryCloze and [SALMon🍣](https://github.com/slp-rl/salmon).
We also support generating continuations, and generative metrics: GenPPL and Auto-BLEU.

An example of running it:
```bash
slm_eval/eval.py tokeniser=unit_hubert_25 metric=storycloze metric.data_path=<TSC_PATH> batch_size=32 model.pretrained_model=<TRAINED_MODEL_PATH>
```

❗Note that the `model.pretrained_model` needs to point to a folder of a specific step within the output folder `training_args.output_dir` from training.

❗Note that we currently only support running one metric with each run of `eval.py`

## Results
We provide some results for our pre-trained models, compared to other SLMs.

| Model                                     | GPUs    | Params | Num Tokens    | sBLIMP ↑  | sStoryCloze ↑ | tStoryCloze ↑ | GenPPL ↓ | Auto-BLEU ↓ |
|-------------------------------------------|---------|--------|---------------|-----------|---------------|---------------|----------|-------------|
| **Speech only pre-training**              |         |        |               |           |               |               |          |             |
| GSLM                                      | 8×V100  | 100M   | 1B            | 54.2      | 53.3          | 66.6          | —        | —           |
| SyllableLM                                | 4×A40   | 300M   | 16B           | 63.7      | —             | 75.4          | —        | —           |
| TWIST-350M                                | 8×V100  | 305M   | 10.8B         | 56.2      | —             | —             | 137.3    | 3.46        |
| TWIST-1.3B                                | 32×V100 | 1B     | 10.8B         | 57.0      | 52.4          | 70.6          | 131.8    | 3.20        |
| TWIST-7B                                  | 32×V100 | 7B     | 36B           | 59.0      | 55.3          | 74.1          | 93.74    | 3.06        |
| TWIST-13B                                 | 32×V100 | 13B    | 36B           | 59.2      | 55.4          | 76.4          | —        | —           |
| Scaled Optimal                            | —       | 823M   | 82B           | **61.3**  | 56.7          | 78.0          | —        | —           |
| Moshi                                     | ?×H100  | 7B     | ?             | 58.9      | **58.7**      | **81.8**      | —        | —           |
| SpiritLM                                  | 64×A100 | 7B     | 100B          | 58.0      | 54.8          | 72.9          | —        | —           |
| **With text / preference optimization**   |         |        |               |           |               |               |          |             |
| Scaling Interleaving                      | —       | 9B     | ~1T           | —         | **62.4**      | 82.9          | —        | —           |
| Moshi                                     | ?×H100  | 7B     | ~720B         | 58.8      | 60.8          | 83.0          | —        | —           |
| SpiritLM                                  | 64×A100 | 7B     | 100B          | 58.3      | 61.0          | 82.9          | —        | —           |
| AlignSLM-1.3B                             | 64×A100 | 1B     | 10.8B + ~158B | 59.8      | 55.0          | 80.0          | —        | —           |
| AlignSLM-7B                               | 64×A100 | 7B     | 36B + ~158B   | **62.3**  | 61.1          | **86.8**      | —        | —           |
| **Ours (_Slam_)**                         |         |        |               |           |               |               |          |             |
| _Slam_ (-DPO)                             | 2×A100  | 358M   | 16.7B         | 58.53     | 58.15         | 80.71         | 67.3     | 3.25        |
| _Slam_                                    | 1×A5000 | 358M   | 1.4B + 5M     | 58.86     | 58.04         | 82.04         | 62.8     | 3.88        |
| _Slam_ (scaled)                           | 2×A100  | 358M   | 16.7B + 9M    | **61.11** | **61.30**     | **84.18**     | **46.6** | 3.75        |


## Contributing
We welcome contributions to this repository. Want to add support for new tokenisers? 
Add support for even more efficient implementations? If you are interested in building this open source 
project - open an Issue, and one of the maintainers will guide you in opening a PR!

## Citation
If you found this repository useful, please cite our work:
```bibtex
@article{maimon2024slamming,
          Soon!
         }
```
