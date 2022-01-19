import torch
import pickle
from typing import Optional

# Note that this is a customized SMPL with displacement (i.e., SMPLD)
from smplx import SMPL as SMPL
from smplx.lbs import vertices2joints
from smplx.utils import SMPLOutput


class SMPLWrapper(SMPL):
    def __init__(self, *args, joint_regressor_extra: Optional[str] = None, **kwargs):
        """
        Extension of the official SMPL implementation to support more joints.
        Args:
            Same as the original SMPL with a detail parameter modeling the per vertex displacement.
            joint_regressor_extra (str): Path to extra joint regressor.
        """
        super(SMPLWrapper, self).__init__(*args, **kwargs)
        smpl_to_openpose = [
            24,
            12,
            17,
            19,
            21,
            16,
            18,
            20,
            0,
            2,
            5,
            8,
            1,
            4,
            7,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
        ]
        if joint_regressor_extra is not None:
            self.register_buffer(
                "joint_regressor_extra",
                torch.tensor(
                    pickle.load(open(joint_regressor_extra, "rb"), encoding="latin1"),
                    dtype=torch.float32,
                ),
            )
        self.register_buffer(
            "joint_map", torch.tensor(smpl_to_openpose, dtype=torch.long)
        )

    def forward(self, *args, **kwargs) -> SMPLOutput:
        """
        Run forward pass. Same as SMPL and also append an extra set of joints if joint_regressor_extra is specified.
        """
        smpl_output = super(SMPLWrapper, self).forward(*args, **kwargs)
        joints = smpl_output.joints[:, self.joint_map, :]
        if hasattr(self, "joint_regressor_extra"):
            extra_joints = vertices2joints(
                self.joint_regressor_extra, smpl_output.vertices
            )
            joints = torch.cat([joints, extra_joints], dim=1)
        smpl_output.joints = joints
        return smpl_output
