""" Render SMPL motion with Pytorch3D."""
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    TexturesUV,
    TexturesVertex,
    PerspectiveCameras,
    FoVPerspectiveCameras,
)

from smplreg.vis.shader import SimpleShader


class VertexMeshRenderer:
    """Render a mesh given vertex coordinates and colors."""

    def __init__(self, config):
        """
        Args:
            config: Configure of the renderer parameters.
        """
        self.device = config.device
        # Define Orthographic camera for texture map rendering.
        R = torch.eye(3).unsqueeze(0)
        T = torch.tensor([config.renderer.cam_trans])
        cameras = FoVPerspectiveCameras(
            device=self.device,
            fov=60,
            R=R,
            T=T,
        )
        # Define the settings for rasterization and shading.
        raster_settings = RasterizationSettings(
            image_size=config.renderer.texture_size,  # Texture map size.
            blur_radius=0,
            faces_per_pixel=1,
        )
        self.rasterizer = MeshRasterizer(
            cameras=cameras, raster_settings=raster_settings
        )
        self.renderer = MeshRenderer(
            rasterizer=self.rasterizer, shader=SimpleShader(device=self.device)
        )

    def __call__(self, verts_coords, faces, verts_colors):
        """
        Args:
            verts_coords: B x 3 x V vertex coordinates
            faces: B x F x 3 mesh faces
            verts_colors: B x 3 x V vertex colors

        Returns:
            rendered_image: B x texture_size x texture_size RGB rendered image.
        """
        # Create TexturesVertex and Meshes object.
        textures = TexturesVertex(verts_features=verts_colors)
        mesh = Meshes(verts=verts_coords, faces=faces, textures=textures).to(
            device=self.device
        )
        # TODO: Consider texture visibility.
        return self.renderer(mesh)[..., :3]
