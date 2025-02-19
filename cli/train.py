import math
from omegaconf import OmegaConf, DictConfig
import hydra
import os

from slamkit.tokeniser import tokeniser_factory
from slamkit.data import init_dataset
from slamkit.model import tlm_factory
from slamkit.trainer import SLAMTrainer, SLAMTrainingArguments, RunTimeStopperCallback, MaxTokensStopperCallback
from slamkit.utils.init_utils import init_wandb, init_compile

import logging
logger = logging.getLogger(__name__)


@hydra.main(config_name='train', config_path='../config', version_base="1.3")
def main(cfg: DictConfig):
    if cfg.tokeniser.tokeniser_type == 'interleave':  # For interleaved data tokeniser must match model
        if cfg.tokeniser.params.text_tokeniser_path != cfg.model.config_args.base_model_name:
            logger.warning(f"Text tokeniser {cfg.tokeniser.params.text_tokeniser_path}, doesn't match model changing it"
                           f" to: {cfg.model.config_args.base_model_name}")
            cfg.tokeniser.params.text_tokeniser_path = cfg.model.config_args.base_model_name

    # Update num_epochs based on stopping tokens
    if cfg.get('train_max_tokens', None) is not None and cfg.get('ds_token_size', None) > 0:
        EPS = 0.01
        cfg.training_args.num_train_epochs = (cfg.train_max_tokens / cfg.ds_token_size) * (1 + EPS)
        logger.info(f'Updated num_train_epochs to {cfg.training_args.num_train_epochs} based on train_max_tokens and ds_token_size')

    tokeniser = tokeniser_factory(cfg.tokeniser)
    logger.info('tokeniser inited')

    ds, collator = init_dataset(cfg, tokeniser)
    logger.info('datasets loaded')
    if cfg.training_args.torch_compile:
        init_compile()
        logger.info('torch compile inited')

    if cfg.model.config_args.vocab_size == -1:
        logger.info('Model vocab_size is -1, thus setting it to tokeniser vocab size')
        cfg.model.config_args.vocab_size = len(tokeniser.text_tokeniser)
    model = tlm_factory(cfg.model)
    if cfg.data.packing and model.config._attn_implementation != 'flash_attention_2':
        raise ValueError(
            'Packing is only supported with flash_attention_2 model')
    logger.info('model inited')

    if cfg.training_args.get('warmup_steps', 0) > 0 and cfg.training_args.get('warmup_ratio', .0) > 0:
        logger.warning('Both warmup_steps and warmup_ratio are set, setting to maximum of the two!')
        # this calculation is somewhat approximate, but should be good enough
        bs = cfg.training_args.per_device_train_batch_size * cfg.training_args.gradient_accumulation_steps * int(os.environ.get("WORLD_SIZE",1))
        n_steps = math.ceil(len(ds['train']) / bs) * cfg.training_args.num_train_epochs
        if n_steps * cfg.training_args.warmup_ratio > cfg.training_args.warmup_steps:
            cfg.training_args.warmup_steps = 0

    train_args = SLAMTrainingArguments(**OmegaConf.to_container(cfg.training_args))

    if cfg.logger.report_to == 'wandb':
        name = os.path.basename(os.path.normpath(cfg.training_args.output_dir))

        if int(os.environ.get('RANK', 0)) == 0:
            init_wandb(cfg, name)

        train_args.run_name = name
        train_args.report_to = ['wandb']
        logger.info('wandb inited')
    else:
        train_args.report_to = []

    callbacks = None
    if cfg.get("run_time", None) is not None:
        callbacks = [RunTimeStopperCallback(cfg.run_time)]
    if cfg.get("train_max_tokens", None) is not None:
        callback = MaxTokensStopperCallback(cfg.train_max_tokens)
        if callbacks is None:
            callbacks = [callback]
        else:
            callbacks.append(callback)

    trainer = SLAMTrainer(
        model=model,
        args=train_args,
        data_collator=collator,
        train_dataset=ds['train'],
        eval_dataset=ds['validation'],
        callbacks=callbacks
    )

    trainer.train(resume_from_checkpoint=cfg.cont_training)


if __name__ == '__main__':
    main()
