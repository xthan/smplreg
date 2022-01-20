""" Animate a registered SMPL model given a SMPL motion sequence.
"""

import cv2
import torch
import pickle as pkl
import numpy as np
from tqdm import tqdm

from smplreg.models.smpl_wrapper import SMPLWrapper
from smplreg.vis.renderer import VertexMeshRenderer
from pytorch3d.ops.knn import knn_points, knn_gather


class SMPLAnimator:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        # Load SMPL model.
        smpl_cfg = {k.lower(): v for k, v in dict(config.smpl).items()}
        self.smpl = SMPLWrapper(**smpl_cfg).to(self.device)
        self.renderer = VertexMeshRenderer(config)

    @staticmethod
    def get_vertice_texture_with_nearest_neighbor(vertices, point_coords, point_colors):
        """For each mesh vertex, find the nearest point in the point cloud, and use it
            as the vertex color.
        Args:
            vertices: B x N x 3, mesh vertices.
            point_coords: B x P x 3, point cloud.
            point_colors: B x P x 3, rgb color of each vertex.
        Return:
            vertices_colors: B x N x 3, rgb color of mesh vertices
        """
        vertices_lengths = torch.full(
            (vertices.shape[0],),
            vertices.shape[1],
            dtype=torch.int64,
            device=vertices.device,
        )
        point_lengths = torch.full(
            (point_coords.shape[0],),
            point_coords.shape[1],
            dtype=torch.int64,
            device=point_coords.device,
        )
        vertices_nn = knn_points(
            vertices,
            point_coords,
            lengths1=vertices_lengths,
            lengths2=point_lengths,
            K=1,
        )
        vertices_colors = knn_gather(point_colors, vertices_nn.idx, vertices_lengths)[
            ..., 0, :
        ]
        return vertices_colors

    @staticmethod
    def load_obj_with_verts_colors(obj_file):
        """Load a point cloud OBJ file with vertex coordinates and rgb colors."""
        verts_coords = []
        verts_colors = []
        lines = open(obj_file).read().splitlines()
        for line in lines:
            if line.startswith("v "):  # Line is a vertex, each line has 6 numbers.
                tokens = line.split()
                coord = [float(x) for x in tokens[1:4]]
                color = [float(x) for x in tokens[4:7]]
            if len(coord) != 3 or len(color) != 3:
                msg = "Vertex %s does not have 6 values. Line: %s"
                raise ValueError(msg % (str(coord), str(line)))
            verts_coords.append(coord)
            verts_colors.append(color)

        return torch.tensor(verts_coords), torch.tensor(verts_colors)

    def render(self, smpl_poses, smpl_trans):
        smpl_params = {
            "betas": self.betas,
            "global_orient": smpl_poses[:3].unsqueeze(0),
            "body_pose": smpl_poses[3:].unsqueeze(0),
            "transl": smpl_trans.unsqueeze(0),
            "detail": self.detail,
        }
        smpl_output = self.smpl(**smpl_params, pose2rot=True)
        # Reverse z for pytorch3d rendering
        smpl_output.vertices[..., -1] = -smpl_output.vertices[..., -1]

        rendered_image = self.renderer(
            smpl_output.vertices, self.faces, self.vertices_color
        )
        rendered_image = (
            rendered_image[0].detach().cpu().numpy()[..., ::-1] * 255
        ).astype(np.uint8)

        return rendered_image

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
        point_coords, point_colors = self.load_obj_with_verts_colors(point_cloud)
        point_coords = point_coords.to(self.device)
        point_colors = point_colors.to(self.device)
        registered_smpl_params = pkl.load(open(registered_smpl_params, "rb"))
        motion_sequence = pkl.load(open(motion_sequence, "rb"))
        # dict_keys(['smpl_loss', 'smpl_poses', 'smpl_scaling', 'smpl_trans'])

        # Parsing body and motion information.
        self.betas = (
            torch.from_numpy(registered_smpl_params["betas"]).float().to(self.device)
        )
        self.detail = (
            torch.from_numpy(registered_smpl_params["detail"]).float().to(self.device)
        )
        self.faces = torch.from_numpy(registered_smpl_params["faces"]).to(self.device)

        # Calculate vertex color according to the point cloud and registered mesh.
        vertices = (
            torch.from_numpy(registered_smpl_params["vertices"]).float().to(self.device)
        )
        self.vertices_color = self.get_vertice_texture_with_nearest_neighbor(
            vertices, point_coords.unsqueeze(0), point_colors.unsqueeze(0)
        )

        smpl_poses = (
            torch.from_numpy(motion_sequence["smpl_poses"]).float().to(self.device)
        )
        smpl_trans = (
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
        for i in tqdm(range(smpl_poses.shape[0])):
            rendered_image = self.render(smpl_poses[i], smpl_trans[i])
            # cv2.imwrite("outputs/image.png", animated_image)  #  Image debugging
            video_writer.write(rendered_image)

        video_writer.release()
