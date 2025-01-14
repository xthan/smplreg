""" Registration between a point cloud and a SMPL parametric model.
"""

import torch
import os
import pytorch3d.io
import pickle as pkl
import numpy as np

from scipy.spatial.transform import Rotation as R
from smplreg.models.smpl_wrapper import SMPLWrapper
from smplreg.loss.registration_loss import RegistrationLoss
from smplreg.loss.utils import rel_change


class SMPLRegister:
    def __init__(self, config):
        self.config = config
        self.obj_scale = 1
        self.device = torch.device(config.device)
        # Load SMPL model.
        smpl_cfg = {k.lower(): v for k, v in dict(config.smpl).items()}
        self.smpl = SMPLWrapper(**smpl_cfg).to(self.device)
        self.smpl_faces = (
            pytorch3d.io.load_obj(config.smpl.mesh)[1][0].to(self.device).unsqueeze(0)
        )

    def __call__(self, point_cloud, init_smpl_params):
        """Registration."""
        print(
            "Registering %s to SMPL given %s as an initialization ..."
            % (point_cloud, init_smpl_params)
        )

        # Read data from file. Only support batch_size = 1
        self.point_cloud = (
            pytorch3d.io.load_obj(point_cloud)[0].unsqueeze(0).to(self.device)
        ) / self.obj_scale
        self.init_smpl_params = pkl.load(open(init_smpl_params, "rb"))

        # Rotate SMPL mesh as it is upside-down.
        pose = np.zeros((1, 72), np.float32)
        pose[:, :3] = R.from_matrix(
            self.init_smpl_params["global_orient"][0][0]
        ).as_rotvec()
        pose[:, :3] = (
            R.from_euler("xyz", [180, 0, 0], degrees=True) * R.from_rotvec(pose[:, :3])
        ).as_rotvec()
        pose[:, 3:] = (
            R.from_matrix(self.init_smpl_params["body_pose"][0])
            .as_rotvec()
            .reshape((1, 69))
        )
        # Setup parameters.
        self.betas = (
            torch.from_numpy(self.init_smpl_params["betas"]).float().to(self.device)
        )
        self.thetas = torch.from_numpy(pose).float().to(self.device)

        # # Rotate SMPL mesh as it is upside-down.
        # pose = self.init_smpl_params["pose"]
        # pose[:, :3] = (
        #     R.from_euler("xyz", [180, 0, 0], degrees=True) * R.from_rotvec(pose[:, :3])
        # ).as_rotvec()

        # # Setup parameters.
        # self.betas = (
        #     torch.from_numpy(self.init_smpl_params["shape"]).float().to(self.device)
        # )
        # self.thetas = torch.from_numpy(pose).float().to(self.device)

        # Roughly align the scale and translation. The mean SMPL model has
        # a height of 1.7173 and a translation of 0
        point_cloud_max = torch.max(self.point_cloud, 1)[0]
        point_cloud_min = torch.min(self.point_cloud, 1)[0]
        point_cloud_mean = torch.mean(self.point_cloud, 1)
        self.init_scale = (point_cloud_max[0, 1] - point_cloud_min[0, 1]) / 1.8
        self.init_translation = point_cloud_mean - 0
        self.point_cloud[0] = (
            self.point_cloud[0] - self.init_translation
        ) / self.init_scale
        self.scale = torch.ones((1, 1)).to(self.device)
        self.translation = torch.zeros((1, 3)).to(self.device)
        self.betas.requires_grad = True
        self.thetas.requires_grad = True
        self.scale.requires_grad = True
        self.translation.requires_grad = True
        self.point_cloud.requires_grad = False
        self.detail = None  #  Do not optimize per vertex displacement in SMPL+D

        # Setup optimizer.
        self.opt_params = [self.betas, self.thetas, self.scale, self.translation]
        self.optimizer = torch.optim.Adam(
            self.opt_params,
            lr=self.config.lr,
        )
        self.registration_loss = RegistrationLoss(self.config.smpl_loss_weights)
        self.registered_smpl_output = self.icp_registration()
        # Denormalize the output

        self.registered_smpl_output.vertices = (
            self.registered_smpl_output.vertices * self.init_scale
            + self.init_translation
        )

    def icp_registration(self):
        # Optimize with ICP.
        self.prev_loss = None
        for i in range(self.config.max_iter):
            smpl_params = {
                "betas": self.betas,
                "global_orient": self.thetas[:, :3],
                "body_pose": self.thetas[:, 3:],
                "detail": self.detail,
            }
            smpl_output = self.smpl(**smpl_params, pose2rot=True)
            smpl_output.vertices = self.scale * smpl_output.vertices + self.translation
            smpl_output.faces = self.smpl_faces
            smpl_output.detail = self.detail
            smpl_output.scale = self.scale
            smpl_output.transl = self.translation
            loss_dict = self.registration_loss(smpl_output, self.point_cloud)
            loss = sum(loss_dict.values())

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Logging and step criterion.
            if i % 50 == 0:
                print(
                    "Step: %03d / %03d; Loss: %08f "
                    % (i, self.config.max_iter, loss.item())
                )
            if i > 0:
                loss_rel_change = rel_change(self.prev_loss, loss.item())
                if loss_rel_change < self.config.ftol:
                    print("Finished due to small relative loss change.")
                    # Could use better stop criteria.
                    break
            if all(
                [
                    torch.abs(var.grad.view(-1).max()).item() < self.config.gtol
                    for var in self.opt_params
                    if var.grad is not None
                ]
            ):
                print("Finished due to small absolute gradient value change.")
                break

            self.prev_loss = loss.item()

        return smpl_output

    def save_results(self, output_folder, name="registered_smpl"):
        """Dump the registered results the output_folder"""
        print(f"Writing results to {output_folder}/{name} with loss {self.prev_loss}")
        os.makedirs(output_folder, exist_ok=True)
        pytorch3d.io.save_obj(
            os.path.join(output_folder, "%s.obj" % name),
            verts=self.registered_smpl_output["vertices"][0] * self.obj_scale,
            faces=self.smpl_faces[0],
        )
        dump_smpl_output = dict(self.registered_smpl_output)
        dump_smpl_output["scale"] = self.scale
        dump_smpl_output["init_scale"] = self.init_scale
        dump_smpl_output["init_translation"] = self.init_translation
        dump_smpl_output["registration_loss"] = self.prev_loss
        for k in dump_smpl_output:
            try:
                dump_smpl_output[k] = dump_smpl_output[k].cpu().detach().numpy()
            except:
                pass
        pkl.dump(
            dump_smpl_output,
            open(os.path.join(output_folder, "%s.pkl" % name), "wb"),
        )
        if self.prev_loss > 0.0005:
            print(
                "---- Large registration loss ----\n, the registration likely fails.",
                name,
                " loss: ",
                self.prev_loss,
            )
