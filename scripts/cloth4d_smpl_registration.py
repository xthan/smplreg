import os
import smplreg
from smplreg.register.smpl_register import SMPLRegister
from omegaconf import OmegaConf

smpl_pkl = "data/cloth4d/00_000016_smpl.pkl"  # Initial estimation of SMPL parameters."
config = OmegaConf.load("configs/smpl.yaml")
smpl_register = SMPLRegister(config)
point_cloud = "data/cloth4d/smplx/00_000016_uv.obj"
mesh_dir = os.path.dirname(point_cloud)
smpl_register(point_cloud, smpl_pkl)
smpl_register.save_results("outputs/", "cloth4d_registered_smpl")
