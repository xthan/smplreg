"""Shader for rendering."""

import torch.nn as nn
from pytorch3d.renderer.blending import hard_rgb_blend, BlendParams


class SimpleShader(nn.Module):
    def __init__(self, device="cpu", blend_params=None):
        super().__init__()
        self.blend_params = (
            blend_params
            if blend_params is not None
            else BlendParams(background_color=(0.0, 0.0, 0.0))
        )

    def forward(self, fragments, meshes, **kwargs):
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        images = hard_rgb_blend(texels, fragments, blend_params)
        return images  # (N, H, W, 3) RGBA image
