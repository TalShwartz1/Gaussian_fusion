import sys
from pathlib import Path
from typing import Tuple, Any, Dict, Callable, Union, List

import imageio
import numpy as np
import pyrallis
import torch
from PIL import Image
from loguru import logger
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import utils
from src.latent_nerf.configs.train_config import TrainConfig
from src.latent_nerf.models.renderer import NeRFRenderer
from src.latent_nerf.training.nerf_dataset import NeRFDataset, GaussianDataset
from src.stable_diffusion import StableDiffusion
from src.utils import make_path, tensor2numpy

# Yaniv ----------------------------------------------------------------------------------------------------------------
import os
from argparse import Namespace
import uuid
from scene import Scene, GaussianModel
from arguments import ModelParams, OptimizationParams, PipelineParams
from utils.general_utils import get_expon_lr_func
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
from PIL import Image

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam

    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

try:
    from fused_ssim import fused_ssim

    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.train_step = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        utils.seed_everything(self.cfg.optim.seed)

        # Make dirs
        self.exp_path = make_path(self.cfg.log.exp_dir)
        self.ckpt_path = make_path(self.exp_path / 'checkpoints')
        self.train_renders_path = make_path(self.exp_path / 'vis' / 'train')
        self.eval_renders_path = make_path(self.exp_path / 'vis' / 'eval')
        self.final_renders_path = make_path(self.exp_path / 'results')

        self.init_logger()
        pyrallis.dump(self.cfg, (self.exp_path / 'config.yaml').open('w'))
        # Yaniv --------------------------------------------------------------------------------------------------------
        # self.gaussians = GaussianModel(cfg.g_model.sh_degree, cfg.g_optim.optimizer_type)
        # lp = ModelParams(cfg.g_model)
        #
        # # args = Namespace(
        # #     sh_degree=3,
        # #     _source_path="",
        # #     _model_path="",
        # #     _images="images",
        # #     _depths="",
        # #     _resolution=-1,
        # #     _white_background=False,
        # #     train_test_exp=False,
        # #     data_device="cuda",
        # #     eval=False
        # # )
        #
        # # dataset = lp.extract(args)
        #
        # dataset = lp.extract_default()
        # self.scene = Scene(dataset, self.gaussians)
        #
        # op = OptimizationParams(cfg.g_optim)
        # opt = op.extract_default()
        #
        # pp = PipelineParams(cfg.g_pipline)
        # pipe = pp.extract_default()
        #
        # self.gaussians.training_setup(opt)
        # self.scene.gaussians.training_setup(opt)
        #
        # self.bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        # self.background = torch.tensor(self.bg_color, dtype=torch.float32, device="cuda")
        # self.use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
        # self.depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final,
        #                                          max_steps=opt.iterations)
        #
        # self.viewpoint_stack = self.scene.getTrainCameras().copy()
        # self.viewpoint_indices = list(range(len(self.viewpoint_stack)))
        # self.ema_loss_for_log = 0.0
        # self.ema_Ll1depth_for_log = 0.0
        # self.first_iter = 0

        self.nerf = self.init_nerf()
        self.diffusion = self.init_diffusion()
        self.text_z = self.calc_text_embeddings()
        # print("text_z = ", self.text_z.shape)
        # print("text_z shape = ", (len(self.text_z),))
        # print(pred_rgb.shape)
        self.losses = self.init_losses()
        self.optimizer, self.scaler = self.init_optimizer()
        self.dataloaders = self.init_dataloaders()

        # self.opt = opt
        # self.dataset = dataset
        # self.pipe = pipe

        self.past_checkpoints = []
        if self.cfg.optim.resume:
            self.load_checkpoint(model_only=False)
        if self.cfg.optim.ckpt is not None:
            self.load_checkpoint(self.cfg.optim.ckpt, model_only=True)

        logger.info(f'Successfully initialized {self.cfg.log.exp_name}')

    def init_nerf(self) -> NeRFRenderer:
        if self.cfg.render.backbone == 'grid':
            from src.latent_nerf.models.network_grid import NeRFNetwork
        else:
            raise ValueError(f'{self.cfg.render.backbone} is not a valid backbone name')

        model = NeRFNetwork(self.cfg.render).to(self.device)
        logger.info(
            f'Loaded {self.cfg.render.backbone} NeRF, #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')
        logger.info(model)
        return model

    def init_diffusion(self) -> StableDiffusion:
        diffusion_model = StableDiffusion(self.device, model_name=self.cfg.guide.diffusion_name,
                                          concept_name=self.cfg.guide.concept_name,
                                          latent_mode=self.nerf.latent_mode)
        for p in diffusion_model.parameters():
            p.requires_grad = False
        return diffusion_model

    def calc_text_embeddings(self) -> Union[torch.Tensor, List[torch.Tensor]]:
        ref_text = self.cfg.guide.text
        if not self.cfg.guide.append_direction:
            text_z = self.diffusion.get_text_embeds([ref_text])
        else:
            text_z = []
            for d in ['front', 'side', 'back', 'side', 'overhead', 'bottom']:
                text = f"{ref_text}, {d} view"
                text_z.append(self.diffusion.get_text_embeds([text]))
        return text_z

    def init_optimizer(self) -> Tuple[Optimizer, Any]:
        optimizer = torch.optim.Adam(self.nerf.get_params(self.cfg.optim.lr), betas=(0.9, 0.99), eps=1e-15)
        scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.optim.fp16)
        return optimizer, scaler

    def init_dataloaders(self) -> Dict[str, DataLoader]:
        train_dataloader = NeRFDataset(self.cfg.render, device=self.device, type='train', H=self.cfg.render.train_h,
                                       W=self.cfg.render.train_w, size=100).dataloader()
        val_loader = NeRFDataset(self.cfg.render, device=self.device, type='val', H=self.cfg.render.eval_h,
                                 W=self.cfg.render.eval_w,
                                 size=self.cfg.log.eval_size).dataloader()
        # Will be used for creating the final video
        val_large_loader = NeRFDataset(self.cfg.render, device=self.device, type='val', H=self.cfg.render.eval_h,
                                       W=self.cfg.render.eval_w,
                                       size=self.cfg.log.full_eval_size).dataloader()
        dataloaders = {'train': train_dataloader, 'val': val_loader, 'val_large': val_large_loader}
        return dataloaders

    def init_losses(self) -> Dict[str, Callable]:
        losses = {}
        if self.cfg.optim.lambda_shape > 0 and self.cfg.guide.shape_path is not None:
            from src.latent_nerf.training.losses.shape_loss import ShapeLoss
            losses['shape_loss'] = ShapeLoss(self.cfg.guide)
        if self.cfg.optim.lambda_sparsity > 0:
            from src.latent_nerf.training.losses.sparsity_loss import sparsity_loss
            losses['sparsity_loss'] = sparsity_loss
        return losses

    def init_logger(self):
        logger.remove()  # Remove default logger
        log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{message}</level>"
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, format=log_format)
        logger.add(self.exp_path / 'log.txt', colorize=False, format=log_format)

    def train(self):
        logger.info('Starting training ^_^')
        # Evaluate the initialization
        self.evaluate(self.dataloaders['val'], self.eval_renders_path)
        self.nerf.train()

        pbar = tqdm(total=self.cfg.optim.iters, initial=self.train_step,
                    bar_format='{desc}: {percentage:3.0f}% training step {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        while self.train_step < self.cfg.optim.iters:
            # Keep going over dataloader until finished the required number of iterations
            for data in self.dataloaders['train']:
                if self.nerf.cuda_ray and self.train_step % self.cfg.render.update_extra_interval == 0:
                    with torch.cuda.amp.autocast(enabled=self.cfg.optim.fp16):
                        self.nerf.update_extra_state()

                self.train_step += 1
                pbar.update(1)

                # return here!!
                # self.optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=self.cfg.optim.fp16):
                    pred_rgbs, pred_ws, loss = self.train_render(data)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                if self.train_step % self.cfg.log.save_interval == 0:
                    self.save_checkpoint(full=True)
                    self.evaluate(self.dataloaders['val'], self.eval_renders_path)
                    self.nerf.train()

                if np.random.uniform(0, 1) < 0.05:
                    # Randomly log rendered images throughout the training
                    self.log_train_renders(pred_rgbs)
        logger.info('Finished Training ^_^')
        logger.info('Evaluating the last model...')
        self.full_eval()
        logger.info('\tDone!')

    def evaluate(self, dataloader: DataLoader, save_path: Path, save_as_video: bool = False):
        logger.info(f'Evaluating and saving model, iteration #{self.train_step}...')
        self.nerf.eval()
        save_path.mkdir(exist_ok=True)

        if save_as_video:
            all_preds = []
            all_preds_normals = []
            all_preds_depth = []

        for i, data in enumerate(dataloader):
            with torch.cuda.amp.autocast(enabled=self.cfg.optim.fp16):
                preds, preds_depth, preds_normals = self.eval_render(data)

            pred, pred_depth, pred_normals = tensor2numpy(preds[0]), tensor2numpy(preds_depth[0]), tensor2numpy(
                preds_normals[0])

            if save_as_video:
                all_preds.append(pred)
                all_preds_normals.append(pred_normals)
                all_preds_depth.append(pred_depth)
            else:
                if not self.cfg.log.skip_rgb:
                    Image.fromarray(pred).save(save_path / f"{self.train_step}_{i:04d}_rgb.png")
                Image.fromarray(pred_normals).save(save_path / f"{self.train_step}_{i:04d}_normals.png")
                Image.fromarray(pred_depth).save(save_path / f"{self.train_step}_{i:04d}_depth.png")

        if save_as_video:
            all_preds = np.stack(all_preds, axis=0)
            all_preds_normals = np.stack(all_preds_normals, axis=0)
            all_preds_depth = np.stack(all_preds_depth, axis=0)

            dump_vid = lambda video, name: imageio.mimsave(save_path / f"{self.train_step}_{name}.mp4", video, fps=25,
                                                           quality=8, macro_block_size=1)

            if not self.cfg.log.skip_rgb:
                dump_vid(all_preds, 'rgb')
            dump_vid(all_preds_normals, 'normals')
            dump_vid(all_preds_depth, 'depth')
        logger.info('Done!')

    def full_eval(self):
        self.evaluate(self.dataloaders['val_large'], self.final_renders_path, save_as_video=True)

    def train_render(self, data: Dict[str, Any]):
        rays_o, rays_d = data['rays_o'], data['rays_d']  # [B, N, 3]

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        if self.cfg.optim.start_shading_iter is None or self.train_step < self.cfg.optim.start_shading_iter:
            shading = 'albedo'
            ambient_ratio = 1.0
        else:
            shading = 'lambertian'
            ambient_ratio = 0.1

        bg_color = torch.rand((B * N, 3), device=rays_o.device)  # Will be used if bg_radius <= 0
        outputs = self.nerf.render(rays_o, rays_d, staged=False, perturb=True, bg_color=bg_color,
                                   ambient_ratio=ambient_ratio, shading=shading, force_all_rays=True)
        pred_rgb = outputs['image'].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        pred_ws = outputs['weights_sum'].reshape(B, 1, H, W)

        # text embeddings
        if self.cfg.guide.append_direction:
            dirs = data['dir']  # [B,]
            text_z = self.text_z[dirs]
        else:
            text_z = self.text_z

        # Guidance loss
        # print("text shape = ", text_z.shape)
        # print(pred_rgb.shape)
        loss_guidance = self.diffusion.train_step(text_z, pred_rgb)
        loss = loss_guidance

        # Sparsity loss
        if 'sparsity_loss' in self.losses:
            loss += self.cfg.optim.lambda_sparsity * self.losses['sparsity_loss'](pred_ws)

        # Shape loss
        # print(" is shape_loss: ", 'shape_loss' in self.losses, "\n")
        if 'shape_loss' in self.losses:
            loss += self.cfg.optim.lambda_shape * self.losses['shape_loss'](outputs['xyzs'], outputs['sigmas'])

        print("loss = ", loss, " = ", "guidance + sparsity = ", loss_guidance, " + ",
              self.cfg.optim.lambda_sparsity * self.losses['sparsity_loss'](pred_ws), "\n")

        if loss != self.cfg.optim.lambda_sparsity * self.losses['sparsity_loss'](pred_ws):
            print("FLAG!! loss != sparsity loss \n")

        return pred_rgb, pred_ws, loss

    def eval_render(self, data, bg_color=None, perturb=False):
        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        if bg_color is not None:
            bg_color = bg_color.to(rays_o.device)
        else:
            bg_color = torch.ones(3, device=rays_o.device)  # [3]

        shading = data['shading'] if 'shading' in data else 'albedo'
        ambient_ratio = data['ambient_ratio'] if 'ambient_ratio' in data else 1.0
        light_d = data['light_d'] if 'light_d' in data else None

        outputs = self.nerf.render(rays_o, rays_d, staged=True, perturb=perturb, light_d=light_d,
                                   ambient_ratio=ambient_ratio, shading=shading, force_all_rays=True, bg_color=bg_color)

        pred_depth = outputs['depth'].reshape(B, H, W)
        if self.nerf.latent_mode:
            pred_latent = outputs['image'].reshape(B, H, W, 3 + 1).permute(0, 3, 1, 2).contiguous()
            if self.cfg.log.skip_rgb:
                # When rendering in a size that is too large for decoding
                pred_rgb = torch.zeros(B, H, W, 3, device=pred_latent.device)
            else:
                pred_rgb = self.diffusion.decode_latents(pred_latent).permute(0, 2, 3, 1).contiguous()
        else:
            pred_rgb = outputs['image'].reshape(B, H, W, 3).contiguous().clamp(0, 1)

        pred_depth = pred_depth.unsqueeze(-1).repeat(1, 1, 1, 3)

        # Render again for normals
        shading = 'normal'
        outputs_normals = self.nerf.render(rays_o, rays_d, staged=True, perturb=perturb, light_d=light_d,
                                           ambient_ratio=ambient_ratio, shading=shading, force_all_rays=True,
                                           disable_background=True)
        pred_normals = outputs_normals['image'][:, :, :3].reshape(B, H, W, 3).contiguous()

        return pred_rgb, pred_depth, pred_normals

    def log_train_renders(self, pred_rgbs: torch.Tensor):
        if self.nerf.latent_mode:
            pred_rgb_vis = self.diffusion.decode_latents(pred_rgbs).permute(0, 2, 3,
                                                                            1).contiguous()  # [1, 3, H, W]
        else:
            pred_rgb_vis = pred_rgbs.permute(0, 2, 3,
                                             1).contiguous().clamp(0, 1)  #
        save_path = self.train_renders_path / f'step_{self.train_step:05d}.jpg'
        save_path.parent.mkdir(exist_ok=True)

        pred = tensor2numpy(pred_rgb_vis[0])

        Image.fromarray(pred).save(save_path)

    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(self.ckpt_path.glob('*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                logger.info(f"Latest checkpoint is {checkpoint}")
            else:
                logger.info("No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        if 'model' not in checkpoint_dict:
            self.nerf.load_state_dict(checkpoint_dict)
            logger.info("loaded model.")
            return

        missing_keys, unexpected_keys = self.nerf.load_state_dict(checkpoint_dict['model'], strict=False)
        logger.info("loaded model.")
        if len(missing_keys) > 0:
            logger.warning(f"missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            logger.warning(f"unexpected keys: {unexpected_keys}")

        if self.cfg.render.cuda_ray:
            if 'mean_count' in checkpoint_dict:
                self.nerf.mean_count = checkpoint_dict['mean_count']
            if 'mean_density' in checkpoint_dict:
                self.nerf.mean_density = checkpoint_dict['mean_density']

        if model_only:
            return

        self.past_checkpoints = checkpoint_dict['checkpoints']
        self.train_step = checkpoint_dict['train_step'] + 1
        logger.info(f"load at step {self.train_step}")

        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                logger.info("loaded optimizer.")
            except:
                logger.warning("Failed to load optimizer.")

        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                logger.info("loaded scaler.")
            except:
                logger.warning("Failed to load scaler.")

    def save_checkpoint(self, full=False):

        name = f'step_{self.train_step:06d}'

        state = {
            'train_step': self.train_step,
            'checkpoints': self.past_checkpoints,
        }

        if self.nerf.cuda_ray:
            state['mean_count'] = self.nerf.mean_count
            state['mean_density'] = self.nerf.mean_density

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['scaler'] = self.scaler.state_dict()

        state['model'] = self.nerf.state_dict()

        file_path = f"{name}.pth"

        self.past_checkpoints.append(file_path)

        if len(self.past_checkpoints) > self.cfg.log.max_keep_ckpts:
            old_ckpt = self.ckpt_path / self.past_checkpoints.pop(0)
            old_ckpt.unlink(missing_ok=True)

        torch.save(state, self.ckpt_path / file_path)


# Yaniv ----------------------------------------------------------------------------------------------------------------
class Trainer_gaus:

    def __init__(self, cfg: TrainConfig):
        if not SPARSE_ADAM_AVAILABLE and cfg.g_optim.optimizer_type == "sparse_adam":
            sys.exit(
                f"Trying to use sparse adam but it is not installed,"
                f" please install the correct rasterizer using pip install [3dgs_accel].")

        self.cfg = cfg
        self.train_step = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        utils.seed_everything(self.cfg.optim.seed)

        # Make dirs
        self.exp_path = make_path(self.cfg.log.exp_dir)
        self.ckpt_path = make_path(self.exp_path / 'checkpoints')
        self.train_renders_path = make_path(self.exp_path / 'vis' / 'train')
        self.eval_renders_path = make_path(self.exp_path / 'vis' / 'eval')
        self.final_renders_path = make_path(self.exp_path / 'results')

        self.init_logger()
        # self.tb_writer = self.prepare_output_and_logger()

        # pyrallis.dump(self.cfg, (self.exp_path / 'config.yaml').open('w'))

        # self.nerf = self.init_nerf()
        self.gaussians = GaussianModel(cfg.g_model.sh_degree, cfg.g_optim.optimizer_type)
        lp = ModelParams(cfg.g_model)

        # args = Namespace(
        #     sh_degree=3,
        #     _source_path="",
        #     _model_path="",
        #     _images="images",
        #     _depths="",
        #     _resolution=-1,
        #     _white_background=False,
        #     train_test_exp=False,
        #     data_device="cuda",
        #     eval=False
        # )

        # dataset = lp.extract(args)

        dataset = lp.extract_default()
        self.scene = Scene(dataset, self.gaussians)

        op = OptimizationParams(cfg.g_optim)
        opt = op.extract_default()

        pp = PipelineParams(cfg.g_pipline)
        pipe = pp.extract_default()

        self.gaussians.training_setup(opt)
        self.scene.gaussians.training_setup(opt)

        # In the future - add load checkpoint --------------------------------------------------------------------------

        self.bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(self.bg_color, dtype=torch.float32, device="cuda")
        self.use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
        self.depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final,
                                                 max_steps=opt.iterations)

        self.viewpoint_stack = self.scene.getTrainCameras().copy()
        self.viewpoint_indices = list(range(len(self.viewpoint_stack)))
        self.ema_loss_for_log = 0.0
        self.ema_Ll1depth_for_log = 0.0
        self.first_iter = 0

        self.diffusion = self.init_diffusion()
        self.text_z = self.calc_text_embeddings()
        # print("text_z shape = ", (len(self.text_z),))
        self.losses = self.init_losses()
        # self.optimizer, self.scaler = self.init_optimizer()
        self.dataloaders = self.init_dataloaders()

        self.opt = opt
        self.dataset = dataset
        self.pipe = pipe

    def init_diffusion(self) -> StableDiffusion:
        diffusion_model = StableDiffusion(self.device, model_name=self.cfg.guide.diffusion_name,
                                          concept_name=self.cfg.guide.concept_name,
                                          latent_mode=False)
        for p in diffusion_model.parameters():
            p.requires_grad = False
        return diffusion_model

    def calc_text_embeddings(self) -> Union[torch.Tensor, List[torch.Tensor]]:
        ref_text = self.cfg.guide.text
        if not self.cfg.guide.append_direction:
            text_z = self.diffusion.get_text_embeds([ref_text])
        else:
            text_z = []
            for d in ['front', 'side', 'back', 'side', 'overhead', 'bottom', '']:
                text = f"{ref_text}, {d} view"
                text_z.append(self.diffusion.get_text_embeds([text]))
        return text_z

    # def init_optimizer(self) -> Tuple[Optimizer, Any]:
    #     optimizer = torch.optim.Adam(self.nerf.get_params(self.cfg.optim.lr), betas=(0.9, 0.99), eps=1e-15)
    #     scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.optim.fp16)
    #     return optimizer, scaler

    def init_dataloaders(self) -> Dict[str, DataLoader]:
        # Create the training dataloader
        train_dataloader = GaussianDataset(
            scene=self.scene,  # The scene should provide access to training cameras
            device=self.device,
            type='train',
            size=100
        ).dataloader()

        # Create the validation dataloader (for regular eval)
        val_loader = GaussianDataset(
            scene=self.scene,
            device=self.device,
            type='val',
            size=self.cfg.log.eval_size
        ).dataloader()

        # Create a validation dataloader for large-scale evaluation or video generation.
        # Adjust the dataset parameters (for example, the total number of views)
        # as needed.
        val_large_loader = GaussianDataset(
            scene=self.scene,
            device=self.device,
            type='val',
            size=self.cfg.log.full_eval_size
        ).dataloader()

        dataloaders = {
            'train': train_dataloader,
            'val': val_loader,
            'val_large': val_large_loader
        }
        return dataloaders

    def init_losses(self) -> Dict[str, Callable]:
        losses = {}
        if self.cfg.optim.lambda_shape > 0 and self.cfg.guide.shape_path is not None:
            from src.latent_nerf.training.losses.shape_loss import ShapeLoss
            losses['shape_loss'] = ShapeLoss(self.cfg.guide)
        if self.cfg.optim.lambda_sparsity > 0:
            from src.latent_nerf.training.losses.sparsity_loss import sparsity_loss
            losses['sparsity_loss'] = sparsity_loss
        return losses

    def init_logger(self):
        logger.remove()  # Remove default logger
        log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{message}</level>"
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, format=log_format)
        logger.add(self.exp_path / 'log.txt', colorize=False, format=log_format)

    def prepare_output_and_logger(self):
        args = self.cfg.optim
        if not args.model_path:
            if os.getenv('OAR_JOB_ID'):
                unique_str = os.getenv('OAR_JOB_ID')
            else:
                unique_str = str(uuid.uuid4())
            args.model_path = os.path.join("./output/", unique_str[0:10])

        # Set up output folder
        print("Output folder: {}".format(args.model_path))
        os.makedirs(args.model_path, exist_ok=True)
        with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
            cfg_log_f.write(str(Namespace(**vars(args))))

        # Create Tensorboard writer
        tb_writer = None
        if TENSORBOARD_FOUND:
            tb_writer = SummaryWriter(args.model_path)
        else:
            print("Tensorboard not available: not logging progress")
        return tb_writer

    def train(self):
        logger.info('Starting training ^_^')

        # Evaluate the initialization
        # self.evaluate(self.dataloaders['val'], self.eval_renders_path)

        logger.info("Fetching train and validation viewpoints from the existing scene...")
        train_viewpoint_stack = self.scene.getTrainCameras().copy()
        train_viewpoint_indices = list(range(len(train_viewpoint_stack)))

        # In the future - add load checkpoint --------------------------------------------------------------------------
        # self.evaluate(self.eval_renders_path)

        self.gaussians.train()
        self.scene.gaussians.train()  # should i also do that??

        pbar = tqdm(total=self.cfg.optim.iters,
                    initial=self.train_step,
                    bar_format='{desc}: {percentage:3.0f}% training step {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        # Come back here later -----------------------------------------------------------------------------------------

        # self.first_iter = 0
        #
        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)

        # first_iter += 1
        iteration = 0
        print("total number of iterations: ",  self.cfg.optim.iters, '\n')
        while self.train_step < self.cfg.optim.iters:
            for data in self.dataloaders['train']:

                iter_start.record()
                self.gaussians.update_learning_rate(iteration)

                # Every 1000 its we increase the levels of SH up to a maximum degree
                if iteration % 1000 == 0:
                    self.gaussians.oneupSHdegree()

                # # Randomly shuffle train viewpoints if stack is exhausted
                # if not train_viewpoint_stack:
                #     train_viewpoint_stack = self.scene.getTrainCameras().copy()
                #     train_viewpoint_indices = list(range(len(train_viewpoint_stack)))
                #
                # # Pop a random viewpoint from the stack
                # rand_idx = np.random.randint(0, len(train_viewpoint_indices))
                # viewpoint_cam = train_viewpoint_stack.pop(rand_idx)
                # v_idx = train_viewpoint_indices.pop(rand_idx)

                # # Render
                # if (iteration - 1) == self.debug_from:
                #     self.pipe.debug = True

                self.train_step += 1
                pbar.update(1)

                # return here!!
                # self.scene.gaussians.optimizer.zero_grad(set_to_none=True)  # should it rather be just self.gaussians?
                # also do i need set_to_none=True? ---------------------------------------------------------------------

                # viewpoint_cam = data['viewpoint_cam']
                # render_pkg, loss = self.train_render(viewpoint_cam)
                render_pkg, loss = self.train_render(data)
                loss_guidance = loss
                pred_image = render_pkg["render"]
                viewspace_points = render_pkg["viewspace_points"]
                visibility_filter = render_pkg["visibility_filter"]
                radii = render_pkg["radii"]

                viewpoint_cam = data['viewpoint_cam']
                if viewpoint_cam.alpha_mask is not None:
                    alpha_mask = viewpoint_cam.alpha_mask.cuda()
                    pred_image *= alpha_mask

                gt_image = viewpoint_cam.original_image.cuda()
                Ll1 = l1_loss(pred_image, gt_image)

                if FUSED_SSIM_AVAILABLE:
                    ssim_value = fused_ssim(pred_image.unsqueeze(0), gt_image.unsqueeze(0))
                else:
                    ssim_value = ssim(pred_image, gt_image)

                loss += (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim_value)

                # Depth regularization
                Ll1depth_pure = 0.0
                if self.depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
                    invDepth = render_pkg["depth"]
                    mono_invdepth = viewpoint_cam.invdepthmap.cuda()
                    depth_mask = viewpoint_cam.depth_mask.cuda()

                    Ll1depth_pure = torch.abs((invDepth - mono_invdepth) * depth_mask).mean()
                    Ll1depth = self.depth_l1_weight(iteration) * Ll1depth_pure
                    loss += Ll1depth
                    Ll1depth = Ll1depth.item()
                else:
                    Ll1depth = 0

                loss.backward()

                # print("total loss = Guidance + Ll1 + ssim_value + Ll1depth \n")
                # print(loss,  " = ", loss_guidance,  " + ", (1.0 - self.opt.lambda_dssim) * Ll1
                #       , " + ", self.opt.lambda_dssim * (1.0 - ssim_value), "+", Ll1depth,  "\n")

                iter_end.record()

                with torch.no_grad():
                    self.ema_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_loss_for_log
                    self.ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * self.ema_Ll1depth_for_log
                    # # if self.train_step % self.cfg.log.save_interval == 0:
                    # if iteration % 10 == 0:
                    #     self.save_checkpoint(full=True)  # come back here later --------------------------------------
                    #     self.evaluate(self.eval_renders_path)
                    #     self.scene.gaussians.train()

                    # if self.train_step % 10 == 0:
                    if iteration % 10 == 0:
                        pbar.set_postfix(
                            {
                                "Loss": f"{self.ema_loss_for_log:.7f}",
                                "Depth Loss": f"{self.ema_Ll1depth_for_log:.7f}",
                            }
                        )

                    if self.train_step % self.cfg.log.save_interval == 0:
                        print("\n[ITER {}] Saving Gaussians".format(iteration))
                        self.scene.save(iteration)

                    # Densification and pruning
                    if self.train_step < self.opt.densify_until_iter:

                        self.gaussians.max_radii2D[visibility_filter] = torch.max(
                            self.gaussians.max_radii2D[visibility_filter],
                            radii[visibility_filter])
                        self.gaussians.add_densification_stats(viewspace_points, visibility_filter)

                        # print(" im here", iteration, " > ", self.opt.densify_from_iter, " = ",
                        #       iteration > self.opt.densify_from_iter
                        #       , "\n")

                        # print(" im here", iteration, " % ", self.opt.densification_interval, " = ",
                        #       iteration % self.opt.densification_interval == 0
                        #       , "\n")
                        print("densification_interval = ", self.opt.densification_interval, '\n')
                        if iteration > self.opt.densify_from_iter and iteration % self.opt.densification_interval == 0:

                            size_threshold = 20 if iteration > self.opt.opacity_reset_interval else None
                            # return here!!
                            self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, 0.005,
                                                             self.scene.cameras_extent, size_threshold, radii)

                            # num_visible_points = visibility_filter.sum().item()
                            # print(f"Iteration {iteration}: Visible points for densification = {num_visible_points}")
                            # print(
                            #     f"Iteration {iteration}: Max radius = {radii.max().item()}, Min radius = {radii.min().item()}")
                            # self.gaussians.densify_and_prune(
                            #     self.opt.densify_grad_threshold, 0.01,  # Increase from 0.005
                            #     self.scene.cameras_extent, None, radii  # Remove `size_threshold`
                            # )

                        if iteration % self.opt.opacity_reset_interval == 0 or (
                                self.dataset.white_background and iteration == self.opt.densify_from_iter):
                            self.gaussians.reset_opacity()

                        # Optimizer step
                        if iteration < self.opt.iterations:
                            self.gaussians.exposure_optimizer.step()
                            self.gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                            if self.use_sparse_adam:
                                visible = radii > 0
                                self.gaussians.optimizer.step(visible, radii.shape[0])
                                self.gaussians.optimizer.zero_grad(set_to_none=True)
                            else:
                                self.gaussians.optimizer.step()
                                self.gaussians.optimizer.zero_grad(set_to_none=True)

                    # if self.train_step % self.cfg.log.save_interval == 0:
                    #     self.evaluate(self.dataloaders['val'], self.eval_renders_path)
                    #     self.scene.gaussians.train()
                    #     print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    #     torch.save((self.gaussians.capture(), iteration),
                    #                self.scene.model_path + "/chkpnt" + str(iteration) + ".pth")

                    # return here!! ------------------------------------------------------------------------------------
                    if self.train_step % self.cfg.log.save_interval == 0:
                        self.evaluate(self.dataloaders['val'], self.eval_renders_path)
                        self.scene.gaussians.train()
                        print("\n[ITER {}] Saving Checkpoint".format(iteration))
                        # Ensure the directory exists
                        os.makedirs(self.ckpt_path, exist_ok=True)

                        # Build the full file path in a platform-independent way
                        filepath = os.path.join(self.ckpt_path, "chkpnt" + str(iteration) + ".pth")

                        # Save the checkpoint
                        torch.save((self.gaussians.capture(), iteration), filepath)


                    # return here!! ------------------------------------------------------------------------------------
                    # print(pred_image.shape)
                    if np.random.uniform(0, 1) < 0.05:
                        self.log_train_renders(pred_image)

                        self.gaussians.print_gaussian_stats()

                    # if np.random.uniform(0, 1) < 0.05:
                    #     # Randomly log rendered images throughout the training
                    #     self.log_train_renders(pred_image)

                iteration += 1
        logger.info('Finished Training ^_^')
        logger.info('Evaluating the last model...')
        self.full_eval()
        logger.info('\tDone!')

    def evaluate(self, dataloader: DataLoader, save_path: Path, save_as_video: bool = False):

        logger.info(f'Evaluating and saving model, iteration #{self.train_step}...')
        save_path.mkdir(exist_ok=True)
        self.scene.gaussians.eval()

        if save_as_video:
            all_preds = []
            all_preds_normals = []
            all_preds_depth = []

        for i, data in enumerate(dataloader):

            # Render predictions
            viewpoint_cam = data['viewpoint_cam']
            preds, preds_depth = self.eval_render(data)  # Use `viewpoint_cam` directly

            # print("preds dim = ", preds.shape)
            # print("preds_depth dim = ", preds_depth.shape)

            # For the RGB prediction:
            # 1. Remove batch dimension: preds[0] becomes [3, 543, 979]
            # 2. Permute dimensions to get [543, 979, 3]
            # 3. Convert to CPU and detach, then to a NumPy array.
            pred = preds[0].permute(1, 2, 0).cpu().detach().numpy()
            # Scale from [0, 1] to [0, 255] and convert to uint8:
            pred = (pred * 255).astype(np.uint8)

            # For the depth prediction:
            # 1. Remove batch dimension: preds_depth[0] becomes [543, 979, 3] (already in channel-last order)
            pred_depth = preds_depth[0].cpu().detach().numpy()
            # Scale and convert:
            pred_depth = (pred_depth * 255).astype(np.uint8)

            # pred = (preds.permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)
            # pred_depth = (preds_depth.permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)

            # pred, pred_depth = preds[0].cpu().detach(), preds_depth[0].cpu().detach()

            # return here !! -------------------------------------------------------------------------------------------
            # pred, pred_depth = tensor2numpy(preds[0]), tensor2numpy(preds_depth[0])

            if save_as_video:
                all_preds.append(pred)
                all_preds_depth.append(pred_depth)
            else:
                if not self.cfg.log.skip_rgb:
                    Image.fromarray(pred).save(save_path / f"{self.train_step}_{i:04d}_rgb.png")
                Image.fromarray(pred_depth).save(save_path / f"{self.train_step}_{i:04d}_depth.png")

        if save_as_video:
            all_preds = np.stack(all_preds, axis=0)
            all_preds_depth = np.stack(all_preds_depth, axis=0)

            dump_vid = lambda video, name: imageio.mimsave(save_path / f"{self.train_step}_{name}.mp4", video, fps=25,
                                                           quality=8, macro_block_size=1)

            if not self.cfg.log.skip_rgb:
                dump_vid(all_preds, 'rgb')
            dump_vid(all_preds_depth, 'depth')
        logger.info('Done!')

    # def full_eval(self):
    #     self.evaluate(self.dataloaders['val_large'], self.final_renders_path, save_as_video=True)
    def full_eval(self):
        self.evaluate(self.dataloaders['val_large'], self.final_renders_path, save_as_video=True)

    def train_render(self, data: Dict[str, Any]):

        B = 1
        H, W = data['H'], data['W']
        viewpoint_cam = data['viewpoint_cam']
        outputs = render(
            viewpoint_camera=viewpoint_cam,
            pc=self.scene.gaussians,
            pipe=self.pipe,
            bg_color=self.background,
            scaling_modifier=1.0,
            use_trained_exp=False
        )

        image = outputs['render']  # This should have shape [C, H, W]
        # C, H, W = image.shape
        # Reshape and permute to match [B, C, H, W]
        pred_rgb = image.unsqueeze(0)  # Add batch dimension, resulting in shape [1, C, H, W]
        # Since C, H, W are already aligned.
        pred_rgb = pred_rgb.contiguous()

        # print(outputs['depth'].shape)
        # pred_ws = outputs['depth'].reshape(B, 1, H, W)

        text_z = self.text_z
        # print("text = ", text_z)
        # text_z = torch.stack(self.text_z, dim=0).mean(dim=0, keepdim=True)
        # I don't believe this is ok.. ---------------------------------------------------------------------------------
        # maybe add data['dirs'] and maybe add other losses ??----------------------------------------------------------
        # print("text_z = ", print((len(text_z),)))
        # print(pred_rgb.shape)

        # Guidance loss
        loss_guidance = self.diffusion.train_step(text_z, pred_rgb)
        loss = loss_guidance

        # # Sparsity loss
        # sparsity_loss = self.cfg.optim.lambda_sparsity * self.losses['sparsity_loss'](pred_ws)
        #
        # loss += sparsity_loss

        return outputs, loss

    def eval_render(self, data: Dict[str, Any]):
        B = 1
        H, W = data['H'], data['W']
        viewpoint_cam = data['viewpoint_cam']
        outputs = render(
            viewpoint_camera=viewpoint_cam,
            pc=self.scene.gaussians,
            pipe=self.pipe,
            bg_color=self.background,
            scaling_modifier=1.0,
            use_trained_exp=False
        )

        # print(outputs['depth'].shape)
        # pred_depth = outputs['depth'].reshape(B, H, W)
        # return here !! -----------------------------------------------------------------------------------------------
        pred_depth = outputs['depth']
        pred_depth = pred_depth.unsqueeze(-1).repeat(1, 1, 1, 3)

        image = outputs['render']  # This should have shape [C, H, W]
        # C, H, W = image.shape
        # Reshape and permute to match [B, C, H, W]
        pred_rgb = image.unsqueeze(0)  # Add batch dimension, resulting in shape [1, C, H, W]
        # Since C, H, W are already aligned.
        pred_rgb = pred_rgb.contiguous().clamp(0, 1)

        return pred_rgb, pred_depth

    # def log_train_renders(self, pred_rgbs: torch.Tensor):
    #
    #     if pred_rgbs.dim() == 4:
    #         pred_rgb_vis = pred_rgbs.permute(0, 2, 3, 1).contiguous().clamp(0, 1)
    #     elif pred_rgbs.dim() == 3:
    #         pred_rgb_vis = pred_rgbs.permute(1, 2, 0).contiguous().clamp(0, 1)
    #     else:
    #         raise RuntimeError(f"Unexpected pred_rgbs shape: {pred_rgbs.shape}")
    #
    #     save_path = self.train_renders_path / f'step_{self.train_step:05d}.jpg'
    #     save_path.parent.mkdir(exist_ok=True)
    #
    #     pred = tensor2numpy(pred_rgb_vis[0])
    #
    #     Image.fromarray(pred).save(save_path)

    #     Yaniv - in the future add load_checkpoint ------------------------------------------------------------------------

    # def log_train_renders(self, pred_rgbs: torch.Tensor):
    #
    #     if pred_rgbs.dim() == 4:
    #         pred_rgb_vis = pred_rgbs.permute(0, 2, 3, 1).contiguous().clamp(0, 1)
    #     elif pred_rgbs.dim() == 3:
    #         pred_rgb_vis = pred_rgbs.permute(1, 2, 0).contiguous().clamp(0, 1)
    #     else:
    #         raise RuntimeError(f"Unexpected pred_rgbs shape: {pred_rgbs.shape}")
    #
    #     save_path = self.train_renders_path / f'step_{self.train_step:05d}.jpg'
    #     save_path.parent.mkdir(exist_ok=True)
    #
    #     # Convert tensor to numpy array with correct scaling
    #     pred = (pred_rgb_vis[0].cpu().numpy() * 255).astype(np.uint8)
    #
    #     # Ensure shape is (H, W, 3)
    #     if pred.shape[-1] == 1:  # Grayscale, convert to 3-channel
    #         pred = np.repeat(pred, 3, axis=-1)
    #
    #     print(pred.shape)
    #     Image.fromarray(pred).save(save_path)

    def log_train_renders(self, pred_rgbs: torch.Tensor):
        if pred_rgbs.dim() == 4:
            # When pred_rgbs is [B, 3, H, W]:
            pred_rgb_vis = pred_rgbs.permute(0, 2, 3, 1).contiguous().clamp(0, 1)
            # Pick the first image in the batch:
            pred = (pred_rgb_vis[0].cpu().numpy() * 255).astype(np.uint8)
        elif pred_rgbs.dim() == 3:
            # When pred_rgbs is [3, H, W]:
            # Permute from [3, H, W] to [H, W, 3]:
            pred_rgb_vis = pred_rgbs.permute(1, 2, 0).contiguous().clamp(0, 1)
            pred = (pred_rgb_vis.cpu().numpy() * 255).astype(np.uint8)
        else:
            raise RuntimeError(f"Unexpected pred_rgbs shape: {pred_rgbs.shape}")

        save_path = self.train_renders_path / f'step_{self.train_step:05d}.jpg'
        save_path.parent.mkdir(exist_ok=True)
        # print(pred.shape)  # Expect (543, 979, 3)
        Image.fromarray(pred).save(save_path)
