""" Render SMPL motion with Pytorch3D."""


import torch
import torch.nn as nn
import numpy as np

from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from pytorch3d.renderer import (
    look_at_view_transform,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    TexturesUV,
    TexturesVertex,
    PerspectiveCameras,
    FoVOrthographicCameras,
)

from pytorch3d.renderer.blending import hard_rgb_blend, BlendParams
