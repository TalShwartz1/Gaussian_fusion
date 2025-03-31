from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger

from .render_config import RenderConfig


@dataclass
class GuideConfig:
    """ Parameters defining the guidance """
    # Guiding text prompt
    text: str = 'a lego man'
    # Append direction to text prompts
    # Yaniv ------------------------------------------------------------------------------------------------------------
    # append_direction: bool = True
    append_direction: bool = True
    # A Textual-Inversion concept to use
    concept_name: Optional[str] = None
    # A huggingface diffusion model to use
    diffusion_name: str = 'CompVis/stable-diffusion-v1-4'
    # Guiding mesh
    shape_path: Optional[str] = None
    # Scale of mesh in 1x1x1 cube
    mesh_scale: float = 0.7
    # Define the proximal distance allowed
    proximal_surface: float = 0.3


@dataclass
class OptimConfig:
    """ Parameters for the optimization process """
    # Loss scale for alpha entropy
    lambda_sparsity: float = 5e-4
    # Loss scale for mesh-guidance
    lambda_shape: float = 5e-6
    # Seed for experiment
    seed: int = 0
    # Total iters - Yaniv ----------------------------------------------------------------------------------------------
    # iters: int = 5000
    iters: int = 10000
    # Learning rate
    # lr: float = 1e-3
    lr: float = 0.03e-3
    # use amp mixed precision training
    fp16: bool = True
    # Start shading at this iteration
    start_shading_iter: Optional[int] = None
    # Resume from checkpoint
    resume: bool = False
    # Load existing model
    ckpt: Optional[str] = None


@dataclass
class LogConfig:
    """ Parameters for logging and saving """
    # Experiment name
    exp_name: str = "experiment"
    # Experiment output dir
    exp_root: Path = Path('experiments/')
    # How many steps between save step
    save_interval: int = 1000
    # Run only test
    eval_only: bool = False
    # Number of angles to sample for eval during training
    eval_size: int = 10
    # Number of angles to sample for eval after training
    full_eval_size: int = 100
    # Number of past checkpoints to keep
    max_keep_ckpts: int = 2
    # Skip decoding and vis only depth and normals
    # yaniv - skip_rgb -------------------------------------------------------------------------------------------------
    skip_rgb: bool = False
    # YANIV ------------------------------------------------------------------------------------------------------------
    data_root = Path('data/my_data/')

    @property
    def exp_dir(self) -> Path:
        return self.exp_root / self.exp_name


# Yaniv ----------------------------------------------------------------------------------------------------------------
@dataclass
class ModelConfig:
    sh_degree: int = 3
    _source_path: str = Path('data/my_data/')
    _model_path: str = ""
    _images: str = "images"
    _depths: str = ""
    _resolution: int = -1
    _white_background: bool = False
    train_test_exp: bool = False
    data_device: str = "cuda"
    eval: bool = False


@dataclass
class PipelineConfig:
    convert_SHs_python: bool = False
    compute_cov3D_python: bool = False
    debug: bool = False
    antialiasing: bool = False


@dataclass
class OptimConfig_g:
    # iterations: int = 120
    # position_lr_init: float = 0.00016
    # position_lr_final: float = 0.0000016
    # position_lr_delay_mult: float = 0.01
    # position_lr_max_steps: int = 30_000
    # feature_lr: float = 0.0025
    # opacity_lr: float = 0.025
    # scaling_lr: float = 0.005
    # rotation_lr: float = 0.001
    # exposure_lr_init: float = 0.01
    # exposure_lr_final: float = 0.001
    # exposure_lr_delay_steps: int = 0
    # exposure_lr_delay_mult: float = 0.0
    # percent_dense: float = 0.01
    # lambda_dssim: float = 0.2
    # densification_interval: int = 20
    # opacity_reset_interval: int = 30
    # densify_from_iter: int = 1
    # densify_until_iter: int = 15_000
    # densify_grad_threshold: float = 0.0002
    # depth_l1_weight_init: float = 1.0
    # depth_l1_weight_final: float = 0.01
    # random_background: bool = False
    # optimizer_type: str = "default"
    # yaniv - changing lr in OptimConfig_g -----------------------------------------------------------------------------
    iterations: int = 6000
    position_lr_init: float = 0.0016
    position_lr_final: float = 0.00016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 7
    feature_lr: float = 0.05
    opacity_lr: float = 0.005
    scaling_lr: float = 0.004
    rotation_lr: float = 0.05
    exposure_lr_init: float = 0.00001
    exposure_lr_final: float = 0.000001
    exposure_lr_delay_steps: int = 0
    exposure_lr_delay_mult: float = 0.0
    percent_dense: float = 0.01
    lambda_dssim: float = 0.2
    densification_interval: int = 1000
    opacity_reset_interval: int = 100
    densify_from_iter: int = 1
    densify_until_iter: int = 5800
    densify_grad_threshold: float = 0.000002
    depth_l1_weight_init: float = 1.0
    depth_l1_weight_final: float = 0.01
    random_background: bool = False
    optimizer_type: str = "default"


@dataclass
class TrainConfig:
    """ The main configuration for the coach trainer """
    log: LogConfig = field(default_factory=LogConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    guide: GuideConfig = field(default_factory=GuideConfig)
    # Yaniv ------------------------------------------------------------------------------------------------------------
    g_model: ModelConfig = field(default_factory=ModelConfig)
    g_pipline: PipelineConfig = field(default_factory=PipelineConfig)
    g_optim: OptimConfig_g = field(default_factory=OptimConfig_g)

    def __post_init__(self):
        if self.log.eval_only and (self.optim.ckpt is None and not self.optim.resume):
            logger.warning(
                'NOTICE! log.eval_only=True, but no checkpoint was chosen -> Manually setting optim.resume to True')
            self.optim.resume = True
