from slamkit.tokeniser import tokeniser_factory
from slamkit.data import init_preference_optimization_dataset
from slamkit.model import tlm_factory
from trl import DPOConfig
from slamkit.trainer import SLAMDPOTrainer
from omegaconf import DictConfig, OmegaConf
import hydra
from slamkit.trainer.callbacks import RunTimeStopperCallback
from slamkit.utils.init_utils import init_wandb, init_compile

import os
import logging
logger = logging.getLogger(__name__)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


@hydra.main(config_name='preference_alignment_train', config_path='../config', version_base="1.3")
def main(cfg: DictConfig):
    if cfg.tokeniser.tokeniser_type == 'interleave':
        raise ValueError('Interleave tokeniser not supported for Preference Alignment yet')

    tokeniser = tokeniser_factory(cfg.tokeniser)
    logger.info('tokeniser inited')

    ds = init_preference_optimization_dataset(cfg.data)
    logger.info('datasets loaded')

    if cfg.training_args.torch_compile:
        init_compile()
        logger.info('torch compile inited')

    if cfg.model.config_args.vocab_size == -1:
        logger.info('Model vocab_size is -1, thus setting it to tokeniser vocab size')
        cfg.model.config_args.vocab_size = len(tokeniser.text_tokeniser)
    model = tlm_factory(cfg.model)
    logger.info('model inited')

    train_args = DPOConfig(**OmegaConf.to_container(cfg.training_args))

    if cfg.logger.report_to == 'wandb':
        name = os.path.basename(os.path.normpath(cfg.training_args.output_dir))

        if int(os.environ.get('RANK', 0)) == 0:
            init_wandb(cfg, name)

        train_args.run_name = name
        train_args.report_to = ['wandb']
    else:
        train_args.report_to = []

    callbacks = None
    if cfg.get("run_time", None) is not None:
        callbacks = [RunTimeStopperCallback(cfg.run_time)]

    trainer = SLAMDPOTrainer(
        model=model,
        tokenizer=tokeniser,
        args=train_args,
        train_dataset=ds['train'],
        eval_dataset=ds['validation'],
        callbacks=callbacks
    )

    trainer.train(resume_from_checkpoint=cfg.get('cont_training', None))


if __name__ == '__main__':
    main()
