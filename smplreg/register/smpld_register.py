""" Registration between a point cloud and a SMPL parametric model.
"""

import torch
import os
import pytorch3d.io
import pickle as pkl

from scipy.spatial.transform import Rotation as R
from smplreg.register.smpl_register import SMPLRegister
from smplreg.loss.utils import rel_change


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

        # SMPL forward.
        betas = torch.from_numpy(self.init_smpl_params["shape"]).float().to(self.device)
        thetas = torch.from_numpy(pose).float().to(self.device)
        scale = torch.ones((1, 1)).to(self.device)
        translation = torch.zeros((1, 3)).to(self.device)
        detail = torch.zeros((1, 6890, 3)).to(self.device)
        betas.requires_grad = True
        thetas.requires_grad = True
        scale.requires_grad = True
        translation.requires_grad = True
        detail.requires_grad = False

        self.point_cloud.requires_grad = False

        # Setup optimizer.
        opt_params = [betas, thetas, scale, translation]
        optimizer = torch.optim.Adam(
            opt_params,
            lr=self.config.lr,
        )

        # Optimize with ICP.
        prev_loss = None
        for i in range(self.config.max_iter):
            smpl_params = {
                "betas": betas,
                "global_orient": thetas[:, :3],
                "body_pose": thetas[:, 3:],
            }
            smpl_output = self.smpl(**smpl_params, pose2rot=True)
            smpl_output.vertices = scale * smpl_output.vertices + translation
            smpl_output.detail = detail
            smpl_output.faces = self.smpl_faces
            loss_dict = self.registration_loss(smpl_output, self.point_cloud)
            loss = loss_dict["chamfer"]  # Only chamfer loss is enough

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging and stop criterion.
            if i % 10 == 0:
                print(
                    "Step: %03d / %03d; Loss: %08f "
                    % (i, self.config.max_iter, loss.item())
                )
                print(loss_dict)
            if i > 0:
                loss_rel_change = rel_change(prev_loss, loss.item())
                if loss_rel_change < self.config.ftol:
                    print("Finished due to small relative loss change.")
                    break
            if all(
                [
                    torch.abs(var.grad.view(-1).max()).item() < self.config.gtol
                    for var in opt_params
                    if var.grad is not None
                ]
            ):
                print("Finished due to small absolute gradient value change.")
                break

            prev_loss = loss.item()

        # # Use pickle file to debug.
        # # pkl.dump({'betas': betas, 'thetas': thetas, 'scale': scale, 'translation': translation},
        # #               open('outputs/tmp.pkl', 'wb'))
        # saved_dict = pkl.load(open('outputs/tmp.pkl', 'rb'))
        # betas, thetas,  = saved_dict['betas'], saved_dict['thetas']
        # scale, translation = saved_dict['scale'], saved_dict['translation']

        # Then optimize detail
        detail.requires_grad = True
        betas.requires_grad = True
        thetas.requires_grad = True
        scale.requires_grad = False
        translation.requires_grad = False
        # opt_params = [detail, betas, thetas]
        opt_params = [detail]
        detail_optimizer = torch.optim.Adam(
            opt_params,
            lr=self.config.lr,
        )

        prev_loss = None
        for i in range(self.config.max_iter):
            smpl_params = {
                "betas": betas,
                "global_orient": thetas[:, :3],
                "body_pose": thetas[:, 3:],
                "detail": detail,
            }
            smpl_output = self.smpl(**smpl_params, pose2rot=True)
            smpl_output.vertices = scale * smpl_output.vertices + translation
            smpl_output.detail = detail
            smpl_output.faces = self.smpl_faces
            loss_dict = self.registration_loss(smpl_output, self.point_cloud)
            loss = sum(loss_dict.values())

            # Optimize
            detail_optimizer.zero_grad()
            loss.backward()
            detail_optimizer.step()

            # Logging and step criterion.
            if i % 10 == 0:
                print(
                    "Step: %03d / %03d; Loss: %08f "
                    % (i, self.config.max_iter, loss.item())
                )
                print(loss_dict)
            if i > 0:
                loss_rel_change = rel_change(prev_loss, loss.item())
                if loss_rel_change < self.config.ftol:
                    print("Finished due to small relative loss change.")
                    break
            if all(
                [
                    torch.abs(var.grad.view(-1).max()).item() < self.config.gtol
                    for var in opt_params
                    if var.grad is not None
                ]
            ):
                print("Finished due to small absolute gradient value change.")
                break

            prev_loss = loss.item()

        self.registered_smpl_output = smpl_output
