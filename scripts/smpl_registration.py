from argparse import ArgumentParser
from omegaconf import OmegaConf


parser = ArgumentParser()
parser.add_argument("--config", default="configs/smpl.yaml", help="Configure file.")
parser.add_argument(
    "--point_cloud",
    default="data/example/hongyu_apose_pifu_output.obj",
    help="Point cloud obj file path.",
)
parser.add_argument(
    "--init_smpl_params",
    default="data/example/hongyu_apose_prohmr_output.pkl",
    help="Initial estimation of SMPL parameters.",
)
parser.add_argument(
    "--output_folder", default="outputs/", help="Registration output folder."
)
opts = parser.parse_args()


def main():
    config = OmegaConf.load(opts.config)
    if config.body_model == "smpld":
        from smplreg.register.smpld_register import SMPLDRegister as SMPLRegister
    else:
        from smplreg.register.smpl_register import SMPLRegister

    smpl_register = SMPLRegister(config)
    smpl_register(opts.point_cloud, opts.init_smpl_params)
    smpl_register.save_results(opts.output_folder, "registered_%s" % config.body_model)


if __name__ == "__main__":
    main()
