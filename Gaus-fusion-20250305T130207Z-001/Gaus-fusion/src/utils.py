import random
import os
from pathlib import Path

import numpy as np
import torch
import math

def get_view_direction(thetas, phis, overhead, front):
    #                   phis [B,];          thetas: [B,]
    # front = 0         [0, front)
    # side (left) = 1   [front, 180)
    # back = 2          [180, 180+front)
    # side (right) = 3  [180+front, 360)
    # top = 4                               [0, overhead]
    # bottom = 5                            [180-overhead, 180]
    res = torch.zeros(thetas.shape[0], dtype=torch.long)
    # first determine by phis

    # res[(phis < front)] = 0
    res[(phis >= (2 * np.pi - front / 2)) & (phis < front / 2)] = 0

    # res[(phis >= front) & (phis < np.pi)] = 1
    res[(phis >= front / 2) & (phis < (np.pi - front / 2))] = 1

    # res[(phis >= np.pi) & (phis < (np.pi + front))] = 2
    res[(phis >= (np.pi - front / 2)) & (phis < (np.pi + front / 2))] = 2

    # res[(phis >= (np.pi + front))] = 3
    res[(phis >= (np.pi + front / 2)) & (phis < (2 * np.pi - front / 2))] = 3
    # override by thetas
    res[thetas <= overhead] = 4
    res[thetas >= (np.pi - overhead)] = 5
    return res


def get_view_direction_float(theta, phi, overhead, front):
    #                   phis [B,];          thetas: [B,]
    # front = 0         [0, front)
    # side (left) = 1   [front, 180)
    # back = 2          [180, 180+front)
    # side (right) = 3  [180+front, 360)
    # top = 4                               [0, overhead]
    # bottom = 5                            [180-overhead, 180]
    # Convert angles from degrees to radians for internal calculations
    theta_rad = math.radians(theta)
    phi_rad = math.radians(phi)

    # Initialize the result variable
    res = None

    # Determine the view direction based on phi
    if (phi >= (360 - front / 2)) or (phi < front / 2):
        res = 0  # front
    elif (phi >= front / 2) and (phi < (180 - front / 2)):
        res = 1  # side (left)
    elif (phi >= (180 - front / 2)) and (phi < (180 + front / 2)):
        res = 2  # back
    elif (phi >= (180 + front / 2)) and (phi < (360 - front / 2)):
        res = 3  # side (right)

    # Override by theta for top and bottom views
    if theta >= (180 - overhead):
        res = 4  # top
    elif theta <= overhead:
        res = 5  # bottom

    return res


def tensor2numpy(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().numpy()
    tensor = (tensor * 255).astype(np.uint8)
    return tensor


def make_path(path: Path) -> Path:
    path.mkdir(exist_ok=True, parents=True)
    return path


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
