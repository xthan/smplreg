""" Animate a registered SMPL model given a SMPL motion sequence by directly animating a mesh.
"""

import torch
import numpy as np
from smplreg.models.lbs import joint_transform
from .base_animator import BaseAnimator


class MeshAnimator(BaseAnimator):
    def __init__(self, config):
        super(MeshAnimator, self).__init__(config)

    def render(self, smpl_poses, smpl_trans):
        A = joint_transform(
            self.betas,
            smpl_poses.unsqueeze(0),
            self.smpl.v_template,
            self.smpl.shapedirs,
            self.smpl.J_regressor,
            self.smpl.parents,
        )

        # Get T-posed point cloud.
        T = torch.matmul(
            self.point_lbs_weights, A.view(1, self.point_lbs_weights.shape[-1], 16)
        ).view(1, -1, 4, 4)
        v_homo = torch.matmul(T, self.v_posed_homo)
        vertices = v_homo[:, :, :3, 0] + smpl_trans.unsqueeze(0)
        # Reverse z for pytorch3d rendering
        vertices[..., -1] = -vertices[..., -1]
        rendered_image = self.renderer(
            vertices, self.point_faces.unsqueeze(0), self.point_colors.unsqueeze(0)
        )
        rendered_image = (
            rendered_image[0].detach().cpu().numpy()[..., ::-1] * 255
        ).astype(np.uint8)

        return rendered_image

    def prepare_animation(self):
        """Generate T-posed point cloud and its lbs weights for animation"""
        # Get point cloud lbs weights from nearest SMPL mesh.
        point_coords = self.point_coords.unsqueeze(0)
        self.point_lbs_weights = self.get_feature_by_nearest_vertex(
            point_coords, self.vertices, self.smpl.lbs_weights.unsqueeze(0)
        )
        point_coords = (point_coords - self.transl) / self.scale
        homo_coord = torch.ones(
            [1, point_coords.shape[1], 1],
            dtype=point_coords.dtype,
            device=point_coords.device,
        )
        point_v_homo = torch.cat([point_coords, homo_coord], dim=2)
        A = joint_transform(
            self.betas,
            self.thetas,
            self.smpl.v_template,
            self.smpl.shapedirs,
            self.smpl.J_regressor,
            self.smpl.parents,
        )
        # Get T-posed point cloud.
        T = torch.matmul(
            self.point_lbs_weights, A.view(1, self.point_lbs_weights.shape[-1], 16)
        ).view(1, -1, 4, 4)
        self.v_posed_homo = torch.matmul(torch.inverse(T), point_v_homo.unsqueeze(-1))
