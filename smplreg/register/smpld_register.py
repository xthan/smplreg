""" Registration between a point cloud and a SMPL parametric model.
"""

import torch
import pytorch3d.io
import pickle as pkl

from scipy.spatial.transform import Rotation as R
from smplreg.register.smpl_register import SMPLRegister
from smplreg.loss.registration_loss import RegistrationLoss


class SMPLDRegister(SMPLRegister):
    def __init__(self, config):
        super(SMPLDRegister, self).__init__(config)

    def __call__(self, point_cloud, init_smpl_params):
        """Registration."""
        print(
            "Registering %s to SMPL given %s as an initialization ..."
            % (point_cloud, init_smpl_params)
        )

        # Read data from file. Only support batch_size = 1
        self.point_cloud = (
            pytorch3d.io.load_obj(point_cloud)[0].unsqueeze(0).to(self.device)
        )
        self.init_smpl_params = pkl.load(open(init_smpl_params, "rb"))

        # Rotate SMPL mesh as it is upside-down.
        pose = self.init_smpl_params["pose"]
        pose[:, :3] = (
            R.from_euler("xyz", [180, 0, 0], degrees=True) * R.from_rotvec(pose[:, :3])
        ).as_rotvec()

        # Setup parameters.
        self.betas = (
            torch.from_numpy(self.init_smpl_params["shape"]).float().to(self.device)
        )
        self.thetas = torch.from_numpy(pose).float().to(self.device)
        self.scale = torch.ones((1, 1)).to(self.device)
        self.translation = torch.zeros((1, 3)).to(self.device)
        self.detail = torch.zeros((1, 6890, 3)).to(self.device)
        self.betas.requires_grad = True
        self.thetas.requires_grad = True
        self.scale.requires_grad = True
        self.translation.requires_grad = True
        self.detail.requires_grad = False

        self.point_cloud.requires_grad = False

        # Setup optimizer.
        self.opt_params = [self.betas, self.thetas, self.scale, self.translation]
        self.optimizer = torch.optim.Adam(
            self.opt_params,
            lr=self.config.lr,
        )
        self.registration_loss = RegistrationLoss(self.config.smpl_loss_weights)
        self.icp_registration()

        # # Use pickle file to debug.
        # pkl.dump(
        #     {
        #         "betas": self.betas,
        #         "thetas": self.thetas,
        #         "scale": self.scale,
        #         "translation": self.translation,
        #     },
        #     open("outputs/tmp.pkl", "wb"),
        # )
        # saved_dict = pkl.load(open("outputs/tmp.pkl", "rb"))
        # self.betas, self.thetas, self.scale, self.translation = (
        #     saved_dict["betas"],
        #     saved_dict["thetas"],
        #     saved_dict["scale"],
        #     saved_dict["translation"],
        # )

        # Then only optimize detail
        self.detail.requires_grad = True
        self.betas.requires_grad = True
        self.thetas.requires_grad = True
        self.scale.requires_grad = False
        self.translation.requires_grad = False
        self.opt_params = [self.detail]
        self.optimizer = torch.optim.Adam(
            self.opt_params,
            lr=self.config.lr,
        )
        self.registration_loss = RegistrationLoss(self.config.smpld_loss_weights)
        self.registered_smpl_output = self.icp_registration()
