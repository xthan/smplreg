""" Base class for animating a registered SMPL model given a SMPL motion sequence.
"""

import cv2
import torch
import pickle as pkl
import numpy as np
from tqdm import tqdm

from smplreg.models.smpl_wrapper import SMPLWrapper
from smplreg.vis.renderer import VertexMeshRenderer
from pytorch3d.ops.knn import knn_points, knn_gather
from scipy.spatial.transform import Rotation as R


class BaseAnimator:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        # Load SMPL model.
        smpl_cfg = {k.lower(): v for k, v in dict(config.smpl).items()}
        self.smpl = SMPLWrapper(**smpl_cfg).to(self.device)
        self.renderer = VertexMeshRenderer(config)

    @staticmethod
    def get_feature_by_nearest_vertex(x, y, f):
        """Given two set of vertices, for each vertex in x find the nearest vertex in y.
        Args:
            x: B x X x 3, vertices of mesh A.
            y: B x Y x 3, vertices of mesh B.
            f: B x Y x F, features of mesh B.
        Return:
            x_nn_f: B x X x F, the features of x given its nearest vertex in y.
        """
        x_lengths = torch.full(
            (x.shape[0],),
            x.shape[1],
            dtype=torch.int64,
            device=x.device,
        )
        y_lengths = torch.full(
            (y.shape[0],),
            y.shape[1],
            dtype=torch.int64,
            device=y.device,
        )
        x_nn = knn_points(
            x,
            y,
            lengths1=x_lengths,
            lengths2=y_lengths,
            K=1,
        )
        x_nn_f = knn_gather(f, x_nn.idx, x_lengths)[..., 0, :]
        return x_nn_f

    @staticmethod
    def load_obj_with_verts_colors(obj_file):
        """Load a point cloud OBJ file with vertex coordinates and rgb colors."""
        verts_coords = []
        verts_colors = []
        faces = []
        lines = open(obj_file).read().splitlines()
        for line in lines:
            tokens = line.split()
            if line.startswith("v "):  # Line is a vertex, each line has 6 numbers.
                coord = [float(x) for x in tokens[1:4]]
                color = [float(x) for x in tokens[4:7]]
                if len(coord) != 3 or len(color) != 3:
                    msg = "Vertex %s does not have 6 values. Line: %s"
                    raise ValueError(msg % (str(coord), str(line)))
                verts_coords.append(coord)
                verts_colors.append(color)
            elif line.startswith(
                "f "
            ):  # Line is face. Now only support "f 1 2 3" format
                face = [
                    int(x) - 1 for x in tokens[1:4]
                ]  # -1 to be compatible with pytorch3d
                if len(face) != 3:
                    msg = "Face %s does not have 3 values. Line: %s"
                    raise ValueError(msg % (str(face), str(line)))
                faces.append(face)

        return (
            torch.tensor(verts_coords),
            torch.tensor(verts_colors),
            torch.tensor(faces),
        )

    def render(self, smpl_poses, smpl_trans):
        raise NotImplementedError

    def prepare_animation(self):
        """Prepare model for animation."""
        raise NotImplementedError

    def __call__(
        self,
        point_cloud,
        registered_smpl_params,
        motion_sequence,
        save_video="outputs/animation_video.mp4",
    ):
        """Registration.
        Args:
            point_cloud: N x 3 point cloud
            registered_smpl_params: dictionary of registered SMPL results (
                containing betas, body_pose, global_orient, vertices, etc.)
            motion_sequence: F x 24 x 3, SMPL motion sequence of F frames.
            save_video: the name of saved video.
        """
        print("Animating %s according to %s  ..." % (point_cloud, motion_sequence))

        # Read data from file. Only support batch_size = 1
        point_coords, point_colors, point_faces = self.load_obj_with_verts_colors(
            point_cloud
        )
        self.point_coords = point_coords.to(self.device)
        self.point_colors = point_colors.to(self.device)
        self.point_faces = point_faces.to(self.device)

        registered_smpl_params = pkl.load(open(registered_smpl_params, "rb"))
        motion_sequence = pkl.load(open(motion_sequence, "rb"))
        # dict_keys(['smpl_loss', 'smpl_poses', 'smpl_scaling', 'smpl_trans'])

        # Parsing body and motion information.
        self.betas = (
            torch.from_numpy(registered_smpl_params["betas"]).float().to(self.device)
        )
        pose = np.hstack(
            (
                registered_smpl_params["global_orient"],
                registered_smpl_params["body_pose"],
            )
        )
        self.thetas = torch.from_numpy(pose).float().to(self.device)

        self.detail = (
            torch.from_numpy(registered_smpl_params["detail"]).float().to(self.device)
        )
        self.faces = torch.from_numpy(registered_smpl_params["faces"]).to(self.device)
        self.transl = torch.from_numpy(registered_smpl_params["transl"]).to(self.device)
        self.scale = torch.from_numpy(registered_smpl_params["scale"]).to(self.device)

        # Calculate vertex color according to the point cloud and registered mesh.
        self.vertices = (
            torch.from_numpy(registered_smpl_params["vertices"]).float().to(self.device)
        )
        self.prepare_animation()
        smpl_poses = motion_sequence["smpl_poses"]
        smpl_poses[:, :3] = (
            R.from_euler("xyz", [-90, 0, 0], degrees=True)
            * R.from_rotvec(smpl_poses[:, :3])
        ).as_rotvec()
        smpl_motion_poses = torch.from_numpy(smpl_poses).float().to(self.device)
        smpl_motion_trans = (
            torch.from_numpy(motion_sequence["smpl_trans"] / 256)
            .float()
            .to(self.device)
        )
        video_writer = cv2.VideoWriter(
            save_video,
            cv2.VideoWriter_fourcc(*"mp4v"),
            30,
            (self.config.renderer.image_size, self.config.renderer.image_size),
        )

        # SMPL forward and render one frame a time (could be done in batches).
        for i in tqdm(range(smpl_motion_poses.shape[0])):
            rendered_image = self.render(smpl_motion_poses[i], smpl_motion_trans[i])
            # cv2.imwrite("outputs/image.png", animated_image)  #  Image debugging
            video_writer.write(rendered_image)

        video_writer.release()
