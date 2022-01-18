""" Registration between a point cloud and a SMPL parametric model.
"""

import torch
import os
import pytorch3d.io
import pickle as pkl

from scipy.spatial.transform import Rotation as R
from smplreg.models.smpl_wrapper import SMPLWrapper
from smplreg.loss.registration_loss import RegistrationLoss
from smplreg.loss.utils import rel_change


class SMPLRegister:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda")
        # Load SMPL model.
        smpl_cfg = {k.lower(): v for k, v in dict(config.smpl).items()}
        self.smpl = SMPLWrapper(**smpl_cfg).to(self.device)
        self.registration_loss = RegistrationLoss(config.loss_weights)
        self.smpl_faces = pytorch3d.io.load_obj(config.smpl.mesh)[1][0]

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
        betas.requires_grad = True
        thetas.requires_grad = True
        scale.requires_grad = True
        translation.requires_grad = True
        self.point_cloud.requires_grad = False
        # Setup optimizer.
        opt_params = [betas, thetas, scale, translation]
        optimizer = torch.optim.Adam(
            opt_params,
            lr=self.config.lr,
            betas=(0.5, 0.999),
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
            loss_dict = self.registration_loss(smpl_output, self.point_cloud)
            loss = sum(loss_dict.values())

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging and step criterion.
            if i % 100 == 0:
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

        # Convert final results to numpy.
        self.registered_smpl_output = smpl_output

    def dump(self, output_folder):
        """Dump the registered results the output_folder"""
        print("Writing results to %s ..." % output_folder)
        os.makedirs(output_folder, exist_ok=True)
        pytorch3d.io.save_obj(
            os.path.join(output_folder, "registered_smpl.obj"),
            verts=self.registered_smpl_output["vertices"][0],
            faces=self.smpl_faces,
        )
        dump_smpl_output = dict(self.registered_smpl_output)

        for k in dump_smpl_output:
            if dump_smpl_output[k] is not None:
                dump_smpl_output[k] = dump_smpl_output[k].cpu().detach().numpy()
        pkl.dump(
            dump_smpl_output,
            open(os.path.join(output_folder, "registered_smpl.pkl"), "wb"),
        )
