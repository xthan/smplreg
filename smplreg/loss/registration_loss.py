""" Defines all losses used in the registration.
"""

import pytorch3d
from pytorch3d.loss import chamfer_distance


class RegistrationLoss:
    def __init__(self, loss_config):
        self.loss_weights = {k.lower(): v for k, v in dict(loss_config).items()}

    def __call__(self, smpl_output, point_cloud):
        """Calculate registration loss between the input SMPL outputs and point cloud."""
        loss_dict = {}
        if "chamfer" in self.loss_weights:
            chamfer_loss = chamfer_distance(smpl_output.vertices, point_cloud)
            loss_dict["chamfer"], _ = self.loss_weights["chamfer"] * chamfer_loss
        return loss_dict
