from argparse import ArgumentParser
from smplreg.animator.smpl_animator import SMPLAnimator
from omegaconf import OmegaConf

parser = ArgumentParser()
parser.add_argument(
    "--config", default="configs/pifu_prohmr.yaml", help="Configure file."
)
parser.add_argument(
    "--point_cloud",
    default="data/example/hongyu_apose_pifu_output.obj",
    help="Point cloud obj file path.",
)
parser.add_argument(
    "--registered_smpl_params",
    default="outputs/registered_smpld.pkl",
    help="SMPL output after registration.",
)
parser.add_argument(
    "--motion_sequence",
    default="/data/motions/gWA_sFM_cAll_d27_mWA4_ch19.pkl",
    help="Animation sequence of SMPL parameters (currently in AIST format).",
)
parser.add_argument(
    "--output_video",
    default="outputs/animation_video.mp4",
    help="Video of animated results.",
)
opts = parser.parse_args()


def main():
    config = OmegaConf.load(opts.config)
    smpl_animator = SMPLAnimator(config)
    smpl_animator(opts.point_cloud, opts.registered_smpl_params, opts.motion_sequence)


if __name__ == "__main__":
    main()
