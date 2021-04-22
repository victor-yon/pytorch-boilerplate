import wandb

from utils.logger import logger
from utils.output import get_run_dir
from utils.settings import settings


def init_wandb():
    wandb.init(project=settings.project_name,
               name=settings.run_name,
               dir=get_run_dir(),
               reinit=True,
               config=settings.get_public_dict(),
               allow_val_change=True)
    logger.debug('Weights & Biases initialized.')


def update_settings():
    wandb.config.update(settings.get_public_dict())
