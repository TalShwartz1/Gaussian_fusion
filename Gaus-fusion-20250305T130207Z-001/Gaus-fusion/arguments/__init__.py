#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os
# Yaniv ----------------------------------------------------------------------------------------------------------------
from src.latent_nerf.configs.train_config import ModelConfig, OptimConfig_g, PipelineConfig

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()

        for key in vars(self):  # Iterate over attributes of self (ModelParams)
            arg_name = key.lstrip("_")  # Remove underscore if present
            if hasattr(args, key):  # Direct match (for args like "_source_path")
                setattr(group, arg_name, getattr(args, key))
            elif hasattr(args, arg_name):  # Match non-underscore version (for "source_path")
                setattr(group, arg_name, getattr(args, arg_name))
            else:
                setattr(group, arg_name, getattr(self, key))  # Use default value

        return group

# Yaniv ----------------------------------------------------------------------------------------------------------------
class ModelParams(ParamGroup):
    def __init__(self, arg, sentinel=False):
        """
        Initialize ModelParams in two ways:
        - If `arg` is a parser: Use argparse-style initialization.
        - If `arg` is a ModelConfig instance: Use dataclass-style initialization.
        """
        if isinstance(arg, ArgumentParser):  # Case 1: Using argparse
            parser = arg
            self.sh_degree = 3
            self._source_path = ""
            self._model_path = ""
            self._images = "images"
            self._depths = ""
            self._resolution = -1
            self._white_background = False
            self.train_test_exp = False
            self.data_device = "cuda"
            self.eval = False
            super().__init__(parser, "Loading Parameters", sentinel)

        elif isinstance(arg, ModelConfig):  # Case 2: Using ModelConfig
            cfg_model = arg
            self.sh_degree = cfg_model.sh_degree
            self.source_path = os.path.abspath(cfg_model._source_path)  # Ensure absolute path
            self.model_path = cfg_model._model_path
            self.images = cfg_model._images
            self.depths = cfg_model._depths
            self.resolution = cfg_model._resolution
            self.white_background = cfg_model._white_background
            self.train_test_exp = cfg_model.train_test_exp
            self.data_device = cfg_model.data_device
            self.eval = cfg_model.eval

        else:
            raise TypeError("ModelParams must be initialized with an ArgumentParser or ModelConfig")

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

    def get_default_args(self):
        """Returns a Namespace with default arguments"""
        return Namespace(
            sh_degree=self.sh_degree,
            _source_path=self.source_path,
            _model_path=self.model_path,
            _images=self.images,
            _depths=self.depths,
            _resolution=self.resolution,
            _white_background=self.white_background,
            train_test_exp=self.train_test_exp,
            data_device=self.data_device,
            eval=self.eval
        )

    def extract_default(self):
        """Automatically extracts the default parameters without requiring a manual Namespace"""
        return self.extract(self.get_default_args())

class PipelineParams(ParamGroup):
    def __init__(self, arg):
        """
        Initialize PipelineParams in two ways:
        - If arg is a parser: Use argparse-style initialization.
        - If arg is a PipelineConfig instance: Use dataclass-style initialization.
        """
        if isinstance(arg, ArgumentParser):  # Case 1: Using argparse
            parser = arg
            self.convert_SHs_python = False
            self.compute_cov3D_python = False
            self.debug = False
            self.antialiasing = False
            super().__init__(parser, "Pipeline Parameters")

        elif isinstance(arg, PipelineConfig):  # Case 2: Using PipelineConfig
            cfg_pipe = arg
            self.convert_SHs_python = cfg_pipe.convert_SHs_python
            self.compute_cov3D_python = cfg_pipe.compute_cov3D_python
            self.debug = cfg_pipe.debug
            self.antialiasing = cfg_pipe.antialiasing

        else:
            raise TypeError("PipelineParams must be initialized with an ArgumentParser or PipelineConfig")

    def get_default_args(self):
        """Returns a Namespace with default arguments"""
        return Namespace(
            convert_SHs_python=self.convert_SHs_python,
            compute_cov3D_python=self.compute_cov3D_python,
            debug=self.debug,
            antialiasing=self.antialiasing
        )

    def extract_default(self):
        """Automatically extracts the default parameters without requiring a manual Namespace"""
        return self.extract(self.get_default_args())

class OptimizationParams(ParamGroup):
    def __init__(self, arg):
        """
        Initialize OptimizationParams in two ways:
        - If `arg` is a parser: Use argparse-style initialization.
        - If `arg` is an OptimConfig_g instance: Use dataclass-style initialization.
        """
        if isinstance(arg, ArgumentParser):  # Case 1: Using argparse
            parser = arg
            self.iterations = 2500
            # self.position_lr_init = 0.00016
            self.position_lr_init = 0.00005
            self.position_lr_final = 0.0000016
            self.position_lr_delay_mult = 0.01
            # self.position_lr_max_steps = 30_000
            self.position_lr_max_steps = 50_000
            # self.feature_lr = 0.0025
            self.feature_lr = 0.0015
            # self.opacity_lr = 0.025
            self.opacity_lr = 0.01
            # self.scaling_lr = 0.005
            self.scaling_lr = 0.0025
            # self.rotation_lr = 0.001
            self.rotation_lr = 0.0005
            self.exposure_lr_init = 0.01
            self.exposure_lr_final = 0.001
            # self.exposure_lr_delay_steps = 0
            self.exposure_lr_delay_steps = 50
            self.exposure_lr_delay_mult = 0.0
            self.percent_dense = 0.01
            self.lambda_dssim = 0.2
            # self.densification_interval = 100
            # self.opacity_reset_interval = 3000
            # yaniv ----------------------------------------------------------------------------------------------------
            # self.densify_from_iter = 500
            self.densification_interval = 100
            self.opacity_reset_interval = 3000
            self.densify_from_iter = 3
            self.densify_until_iter = 1250
            self.densify_grad_threshold = 0.0002
            self.depth_l1_weight_init = 1.0
            self.depth_l1_weight_final = 0.01
            self.random_background = False
            self.optimizer_type = "default"

            # self.iterations = 30_000
            # self.position_lr_init = 0.0016
            # self.position_lr_final = 0.000016
            # self.position_lr_delay_mult = 0.01
            # self.position_lr_max_steps = 30_000
            # self.feature_lr = 0.025
            # self.opacity_lr = 0.25
            # self.scaling_lr = 0.05
            # self.rotation_lr = 0.01
            # self.exposure_lr_init = 0.1
            # self.exposure_lr_final = 0.01
            # self.exposure_lr_delay_steps = 0
            # self.exposure_lr_delay_mult = 0.0
            # self.percent_dense = 0.01
            # self.lambda_dssim = 0.2
            #
            # # Extremely aggressive densification parameters
            # self.densification_interval = 1
            # self.opacity_reset_interval = 30
            # self.densify_from_iter = 1
            # self.densify_until_iter = 15_000  # or extend if needed
            # self.densify_grad_threshold = 0.01
            #
            # self.depth_l1_weight_init = 1.0
            # self.depth_l1_weight_final = 0.01
            # self.random_background = False
            # self.optimizer_type = "sparse_adam"


            super().__init__(parser, "Optimization Parameters")

        elif isinstance(arg, OptimConfig_g):  # Case 2: Using OptimConfig_g
            cfg = arg
            self.iterations = cfg.iterations
            self.position_lr_init = cfg.position_lr_init
            self.position_lr_final = cfg.position_lr_final
            self.position_lr_delay_mult = cfg.position_lr_delay_mult
            self.position_lr_max_steps = cfg.position_lr_max_steps
            self.feature_lr = cfg.feature_lr
            self.opacity_lr = cfg.opacity_lr
            self.scaling_lr = cfg.scaling_lr
            self.rotation_lr = cfg.rotation_lr
            self.exposure_lr_init = cfg.exposure_lr_init
            self.exposure_lr_final = cfg.exposure_lr_final
            self.exposure_lr_delay_steps = cfg.exposure_lr_delay_steps
            self.exposure_lr_delay_mult = cfg.exposure_lr_delay_mult
            self.percent_dense = cfg.percent_dense
            self.lambda_dssim = cfg.lambda_dssim
            self.densification_interval = cfg.densification_interval
            self.opacity_reset_interval = cfg.opacity_reset_interval
            self.densify_from_iter = cfg.densify_from_iter
            self.densify_until_iter = cfg.densify_until_iter
            self.densify_grad_threshold = cfg.densify_grad_threshold
            self.depth_l1_weight_init = cfg.depth_l1_weight_init
            self.depth_l1_weight_final = cfg.depth_l1_weight_final
            self.random_background = cfg.random_background
            self.optimizer_type = cfg.optimizer_type

        else:
            raise TypeError("OptimizationParams must be initialized with an ArgumentParser or OptimConfig_g")

    def get_default_args(self):
        """Returns a Namespace with default arguments"""
        return Namespace(
            iterations=self.iterations,
            position_lr_init=self.position_lr_init,
            position_lr_final=self.position_lr_final,
            position_lr_delay_mult=self.position_lr_delay_mult,
            position_lr_max_steps=self.position_lr_max_steps,
            feature_lr=self.feature_lr,
            opacity_lr=self.opacity_lr,
            scaling_lr=self.scaling_lr,
            rotation_lr=self.rotation_lr,
            exposure_lr_init=self.exposure_lr_init,
            exposure_lr_final=self.exposure_lr_final,
            exposure_lr_delay_steps=self.exposure_lr_delay_steps,
            exposure_lr_delay_mult=self.exposure_lr_delay_mult,
            percent_dense=self.percent_dense,
            lambda_dssim=self.lambda_dssim,
            densification_interval=self.densification_interval,
            opacity_reset_interval=self.opacity_reset_interval,
            densify_from_iter=self.densify_from_iter,
            densify_until_iter=self.densify_until_iter,
            densify_grad_threshold=self.densify_grad_threshold,
            depth_l1_weight_init=self.depth_l1_weight_init,
            depth_l1_weight_final=self.depth_l1_weight_final,
            random_background=self.random_background,
            optimizer_type=self.optimizer_type
        )

    def extract_default(self):
        """Automatically extracts the default parameters without requiring a manual Namespace"""
        return self.extract(self.get_default_args())

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
