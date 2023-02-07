import errno
import os
from typing import Optional

import gdown
import torch
import torch.nn as nn
from torch import Tensor
from generative.networks.nets import DiffusionModelUNet

def ddpm_2d(
    model_dir: Optional[str] = None,
    file_name: str = "ddpm_2d.pth",
    progress: bool = True,
):
    if model_dir is None:
        hub_dir = torch.hub.get_dir()
        model_dir = os.path.join(hub_dir, "checkpoints")

    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    filename = file_name
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        gdown.download(
            url="https://drive.google.com/uc?export=download&id=1j9lRK-D7enXtywIhzQwdaxPKy3AWObwd",
            output=cached_file,
            quiet=not progress,
        )

    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        num_channels=(128, 256, 256),
        attention_levels=(False, True, True),
        num_res_blocks=1,
        num_head_channels=256,
    )
    model.load_state_dict(torch.load(cached_file, map_location='cpu'))
    return model
	
def ddpm_2d_v2(
    model_dir: Optional[str] = None,
    file_name: str = "ddpm_2d.pth",
    progress: bool = True,
):
    if model_dir is None:
        hub_dir = torch.hub.get_dir()
        model_dir = os.path.join(hub_dir, "checkpoints")

    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    filename = file_name
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        gdown.download(
            url="https://drive.google.com/uc?export=download&id=1UTaeBMHIFxQDxWMoQNqSffGhbUDVRatZ",
            output=cached_file,
            quiet=not progress,
        )

    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        num_channels=(128, 256, 256),
        attention_levels=(False, True, True),
        num_res_blocks=1,
        num_head_channels=256,
    )
    model.load_state_dict(torch.load(cached_file, map_location='cpu'))
    return model