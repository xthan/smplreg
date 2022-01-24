""" Animate a registered SMPL model given a SMPL motion sequence by animating the SMPL mesh.
"""

import numpy as np

from .base_animator import BaseAnimator


class SMPLAnimator(BaseAnimator):
    def __init__(self, config):
        super(SMPLAnimator, self).__init__(config)

    def render(self, smpl_poses, smpl_trans):
        smpl_params = {
            "betas": self.betas,
            "global_orient": smpl_poses[:3].unsqueeze(0),
            "body_pose": smpl_poses[3:].unsqueeze(0),
            "transl": smpl_trans.unsqueeze(0),
            "detail": self.detail,
        }
        smpl_output = self.smpl(**smpl_params, pose2rot=True)
        # Reverse z for pytorch3d rendering
        smpl_output.vertices[..., -1] = -smpl_output.vertices[..., -1]
        rendered_image = self.renderer(
            smpl_output.vertices, self.faces, self.vertices_color
        )
        rendered_image = (
            rendered_image[0].detach().cpu().numpy()[..., ::-1] * 255
        ).astype(np.uint8)

        return rendered_image

    def prepare_animation(self):
        """Get SMPL vertex color from the nearest point in the point cloud."""
        self.vertices_color = self.get_feature_by_nearest_vertex(
            self.vertices,
            self.point_coords.unsqueeze(0),
            self.point_colors.unsqueeze(0),
        )
