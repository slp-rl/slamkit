import os
from omegaconf import DictConfig, OmegaConf

def init_wandb(cfg: DictConfig, name: str):
    import wandb
    run = wandb.init(project=cfg.logger.project,
                     entity=cfg.logger.entity,
                     name=name,
                     group=cfg.logger.group,
                     resume=cfg.logger.resume,
                     config=OmegaConf.to_container(cfg))
    hydra_config_path = os.path.join(
        run.dir, f'{run.entity}-{run.name}-config.yaml')
    OmegaConf.save(cfg, hydra_config_path)
    run.save(hydra_config_path, policy='now')


def init_compile():
    os.environ["ACCELERATE_DYNAMO_USE_DYNAMIC"] = "1"