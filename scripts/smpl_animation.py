from argparse import ArgumentParser
from smplreg.animator.smpl_animator import SMPLAnimator

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
    default="outputs/registered_smpl.pkl",
    help="SMPL output after registration.",
)
parser.add_argument(
    "--animation",
    default="data/animation/.pkl",
    help="Animation sequence of SMPL parameters.",
)
parser.add_argument(
    "--output_video", default="outputs/animation.mp4", help="Video of animated results."
)
opts = parser.parse_args()


def main():
    smpl_animator = SMPLAnimator(opts.config)
    smpl_animator(opts.point_cloud, opts.registered_smpl_params, opts.motion_sequence)
    smpl_animator.save_results(opts.output_video)


if __name__ == "__main__":
    main()
