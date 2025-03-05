import pyrallis

from src.latent_nerf.configs.train_config import TrainConfig
from src.latent_nerf.training.trainer import Trainer

# Yaniv ---------------------------------------------------------------------------------------------------------------
from utils.general_utils import safe_state
import torch
import random
import numpy as np
from src.latent_nerf.training.trainer import Trainer_gaus


@pyrallis.wrap()
def main(cfg: TrainConfig):
    # Yaniv ------------------------------------------------------------------------------------------------------------
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
    # torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # trainer = Trainer(cfg)
    trainer = Trainer_gaus(cfg)
    if cfg.log.eval_only:
        trainer.full_eval()
    else:
        trainer.train()


if __name__ == '__main__':
    main()
