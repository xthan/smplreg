from argparse import ArgumentParser
from smplreg.register.smpl_register import SMPLRegister
from omegaconf import OmegaConf


parser = ArgumentParser()
parser.add_argument(
    "--config", default="configs/pifu_prohmr.yaml", help="Configure file."
)
parser.add_argument(
    "--point_cloud",
    default="data/hongyu_apose_pifu_output.obj",
    help="Point cloud obj file path.",
)
parser.add_argument(
    "--init_smpl_params",
    default="data/hongyu_apose_prohmr_output.pkl",
    help="Initial estimation of SMPL parameters.",
)
parser.add_argument(
    "--output_folder", default="outputs/", help="Registration output folder."
)
opts = parser.parse_args()


def main():
    config = OmegaConf.load(opts.config)
    smpl_register = SMPLRegister(config)
    smpl_register(opts.point_cloud, opts.init_smpl_params)
    smpl_register.dump(opts.output_folder)


if __name__ == "__main__":
    main()
