import pyrallis

from src.latent_nerf.configs.train_config import TrainConfig
from src.latent_nerf.training.trainer import Trainer

# Yaniv ---------------------------------------------------------------------------------------------------------------
from utils.general_utils import safe_state
import torch
import random
import numpy as np
from src.latent_nerf.training.trainer import Trainer_gaus
import itertools
from pathlib import Path
import shutil
import gc


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

# @pyrallis.wrap()
# def main(cfg: TrainConfig):
#     # Yaniv ------------------------------------------------------------------------------------------------------------
#     random.seed(0)
#     np.random.seed(0)
#     torch.manual_seed(0)
#     torch.cuda.set_device(torch.device("cuda:0"))
#     # torch.autograd.set_detect_anomaly(args.detect_anomaly)
#
#     my_path = 'E:/deep-final/experiments/'
#     position_lr = [0.16, 0.016, 0.0016]
#     feature_lr = [0.5, 0.05, 0.005]
#     opacity_lr = [0.5, 0.05, 0.005]
#     scaling_lr = [0.5, 0.05, 0.005, 0.0005]
#     exposure_lr = [0.1, 0.01, 0.001]
#     percent_dense = [0.2, 0.1, 0.01, 0.05]
#     # op_reset = [30000, 30, 100, 150]
#     # trainer = Trainer(cfg)
#     parent_folder = Path(r"E:\deep-final\experiments")
#     src = Path(r"C:\Users\Yaniv\deep\final\other_projects\gaus\gaussian-nerf\experiments\lego_tractor\vis\train")
#
#     for pos_lr, feat_lr, opac_lr, scale_lr, expos_lr, dens in itertools.product(
#             position_lr, feature_lr, opacity_lr, scaling_lr, exposure_lr, percent_dense):
#     # def __init__(self, cfg: TrainConfig)
#     #     if cfg.log.eval_only:
#     #         trainer.full_eval()
#     #     else:
#         trainer = Trainer_gaus(cfg)
#         trainer.train(pos_lr, feat_lr, opac_lr, scale_lr, expos_lr, dens)
#         folder_name = f"{pos_lr:.4f}-{feat_lr:.4f}-{opac_lr:.4f}-{scale_lr:.4f}-{expos_lr:.4f}-{dens:.4f}"
#         folder_path = parent_folder / folder_name
#         folder_path.mkdir(exist_ok=True)
#         shutil.copytree(src, folder_path, dirs_exist_ok=True)
#         del trainer
#         gc.collect()
#         torch.cuda.empty_cache()
#
#
#
# if __name__ == '__main__':
#     main()

# @pyrallis.wrap()
# def main(cfg: TrainConfig):
#     # Yaniv ------------------------------------------------------------------------------------------------------------
#     random.seed(0)
#     np.random.seed(0)
#     torch.manual_seed(0)
#     torch.cuda.set_device(torch.device("cuda:0"))
#     # torch.autograd.set_detect_anomaly(args.detect_anomaly)
#
#     my_path = 'E:/deep-final/experiments/'
#     position_lr = [0.16, 0.016, 0.0016]
#     scaling_lr = [0.5, 0.05, 0.005, 0.0005]
#     exposure_lr = [0.1, 0.01, 0.001]
#     percent_dense = [0.2, 0.1, 0.01, 0.05]
#     # op_reset = [30000, 30, 100, 150]
#     # trainer = Trainer(cfg)
#     parent_folder = Path(r"E:\deep-final\experiments2")
#     src = Path(r"\Users\Yaniv\deep\final\other_projects\gaus\gaussian-nerf\experiments\lego_tractor\vis\train")
#
#     # for pos_lr, feat_lr, opac_lr, scale_lr, expos_lr, dens in itertools.product(
#     #         position_lr, feature_lr, opacity_lr, scaling_lr, exposure_lr, percent_dense):
#     # def __init__(self, cfg: TrainConfig)
#     #     if cfg.log.eval_only:
#     #         trainer.full_eval()
#     #     else:
#
#     start_iter = 1161
#
#     # Convert the iterator to a list and slice from the starting point
#     all_combinations = list(itertools.product(
#         position_lr, scaling_lr, exposure_lr, percent_dense
#     ))
#
#     iteration = 0
#     # Iterate from the 627th combination
#     for pos_lr, scale_lr, expos_lr, dens in itertools.product(position_lr, scaling_lr, exposure_lr, percent_dense):
#         trainer = Trainer_gaus(cfg)
#         trainer.train(pos_lr, scale_lr, expos_lr, dens)
#         folder_name = f"{iteration:.4f}-{pos_lr:.4f}-{scale_lr:.4f}-{expos_lr:.4f}-{dens:.4f}"
#         folder_path = parent_folder / folder_name
#         folder_path.mkdir(exist_ok=True)
#         shutil.copytree(src, folder_path, dirs_exist_ok=True)
#         del trainer
#         gc.collect()
#         torch.cuda.empty_cache()
#         iteration += 1
#
#
#
# if __name__ == '__main__':
#     main()