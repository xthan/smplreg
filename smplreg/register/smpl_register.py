class SMPLRegister:
    def __init__(self, config):
        self.config = config

    def __call__(self, point_cloud, init_smpl_params):
        """Registration."""
        print(
            "Registering %s to SMPL given %s as an initialization ..."
            % (point_cloud, init_smpl_params)
        )

        pass

    def dump(self, output_folder):
        """Dump the registered results the output_folder"""
        print("Writing results to %s ..." % output_folder)
        pass
