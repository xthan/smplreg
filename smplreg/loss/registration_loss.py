""" Defines all losses used in the registration.
"""

import torch
import torch.nn as nn
import pytorch3d
from pytorch3d.loss import (
    chamfer_distance,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)


class RegistrationLoss:
    def __init__(self, loss_config):
        self.loss_weights = {k.lower(): v for k, v in dict(loss_config).items()}

    def __call__(self, smpl_output, point_cloud):
        """Calculate registration loss between the input SMPL outputs and point cloud."""
        loss_dict = {}
        if "chamfer" in self.loss_weights:
            if self.loss_weights["chamfer"] > 0:
                # Chamfer loss measures closest point2point distance.
                chamfer_loss = chamfer_distance(smpl_output.vertices, point_cloud)
                loss_dict["chamfer"], _ = self.loss_weights["chamfer"] * chamfer_loss

        if "coupling" in self.loss_weights:
            if self.loss_weights["coupling"] > 0:
                # Coupling loss prevent detail becomes too large.
                coupling_loss = nn.MSELoss()(
                    smpl_output.detail, torch.zeros_like(smpl_output.detail)
                )
                loss_dict["coupling"] = self.loss_weights["coupling"] * coupling_loss

        # TODO: mesh related optimization should exclude face and hand regions.
        if "laplacian" in self.loss_weights:
            if self.loss_weights["laplacian"] > 0:
                # Laplacian loss ensures smoothness of mesh.
                smpl_mesh = pytorch3d.structures.Meshes(
                    verts=smpl_output.vertices, faces=smpl_output.faces
                )
                laplacian_loss = mesh_laplacian_smoothing(smpl_mesh)
                loss_dict["laplacian"] = self.loss_weights["laplacian"] * laplacian_loss

        if "normal_consistency" in self.loss_weights:
            if self.loss_weights["normal_consistency"] > 0:
                # normal_consistency loss enforces consistency between neighboring faces.
                smpl_mesh = pytorch3d.structures.Meshes(
                    verts=smpl_output.vertices, faces=smpl_output.faces
                )
                normal_consistency_loss = mesh_normal_consistency(smpl_mesh)
                loss_dict["normal_consistency"] = (
                    self.loss_weights["normal_consistency"] * normal_consistency_loss
                )
        # TODO: add point-to-mesh distance
        # https://pytorch3d.readthedocs.io/en/latest/modules/loss.html
        return loss_dict
