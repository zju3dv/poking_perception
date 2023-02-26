import loguru
import matplotlib.pyplot as plt
import json
from typing import Union

import os
import shutil
import warnings

import PIL.Image
import numpy as np
import torch
import trimesh.primitives
import wis3d
from matplotlib.cm import get_cmap
from scipy.spatial.transform import Rotation
from transforms3d import euler, affines
from wis3d.wis3d import tensor2ndarray, folder_names, file_exts

from puop.utils.comm import get_rank
from puop.utils.os_utils import magenta
from puop.utils.pn_utils import random_choice, to_array, clone_if_present, numel
from wis3d import Wis3D


class Vis3D(Wis3D):
    # ensure out_folder will be deleted once only when program starts.
    has_removed = []
    default_xyz_pattern = ('x', 'y', 'z')
    sequence_ids = {}
    default_out_folder = 'dbg'
    default_colors = {
        'orange': [249, 209, 22],
        'pink': [255, 63, 243],
        'green': [0, 255, 0],
        'blue': [2, 83, 255]
    }

    def __init__(self, xyz_pattern=None, out_folder='dbg',
                 sequence='sequence',
                 auto_increase=True,
                 enable: bool = True):
        assert enable in [True, False]
        self.enable = enable and get_rank() == 0
        if enable is True:
            if xyz_pattern is None:
                xyz_pattern = Vis3D.default_xyz_pattern
            if not os.path.isabs(out_folder):
                seq_out_folder = os.path.join(
                    os.getcwd(), out_folder, sequence)
            else:
                seq_out_folder = out_folder
            if os.path.exists(seq_out_folder) and seq_out_folder not in Vis3D.has_removed:
                shutil.rmtree(seq_out_folder)
                Vis3D.has_removed.append(seq_out_folder)
            super().__init__(out_folder, sequence, xyz_pattern)

            if seq_out_folder not in Vis3D.sequence_ids:
                Vis3D.sequence_ids[seq_out_folder] = 0
            else:
                Vis3D.sequence_ids[seq_out_folder] += 1
            self.auto_increase = auto_increase
            if auto_increase:
                scene_id = Vis3D.sequence_ids[seq_out_folder]
            else:
                scene_id = 0
            print(magenta(f'Set up Vis3D for {sequence}: {scene_id}'))
            # self.set_scene_id(scene_id)
            super().set_scene_id(scene_id)
            self.plane_model = None

    def add_box_by_6border(self, xmin, ymin, zmin, xmax, ymax, zmax, name=None):
        if not self.enable:
            return
        x = (xmin + xmax) / 2
        y = (ymin + ymax) / 2
        z = (zmin + zmax) / 2
        sx = xmax - xmin
        sy = ymax - ymin
        sz = zmax - zmin
        self.add_boxes(np.array([x, y, z]), np.array(
            [0, 0, 0]), np.array([sx, sy, sz]), name=name)

    def add_box_by_bounds(self, bounds):
        """

        :param bounds: 3x2 x,y,z
        :return:
        """
        if not self.enable:
            return
        xmin, ymin, zmin = bounds[:, 0]
        xmax, ymax, zmax = bounds[:, 1]
        self.add_box_by_6border(xmin, ymin, zmin, xmax, ymax, zmax)

    def add_boxes_by_dof(self, positions, rotations, scales, name=None, label=None):
        if not self.enable:
            return
        super().add_boxes(positions, rotations, scales, name, label)

    def add_point_cloud(self, points, colors=None, name=None, sample=1.0,
                        remove_plane=False, remove_plane_distance_thresh=0.005, remove_plane_cache_model=True,
                        max_z=1000.0, min_norm=0.01):
        if not self.enable:
            return
        if len(points.shape) != 2:
            points = points.reshape(-1, 3)
        if numel(points) == 0:
            return
        if sample < 1.0:
            points = torch.tensor(points)
            if points.ndim == 1:
                points = points.unsqueeze(0)
            sample_size = int(points.shape[0] * sample)
            if sample_size > 100:
                points, idxs = random_choice(points, sample_size, dim=0)
                if colors is not None:
                    colors = colors[idxs]
        elif sample > 1.0:
            points = torch.tensor(points)
            if points.ndim == 1:
                points = points.unsqueeze(0)
            sample_size = sample
            points, idxs = random_choice(points, sample_size, dim=0)
            if colors is not None:
                colors = colors[idxs]
        keep = points[:, 2] < max_z
        points = points[keep]
        points = to_array(points)
        if colors is not None:
            colors = colors[keep]
        norm = np.linalg.norm(points, axis=-1)
        keep = norm > min_norm
        points = points[keep]
        if colors is not None:
            colors = colors[keep]
        if remove_plane:
            from .utils_3d import open3d_plane_segment_api, point_plane_distance_api
            if not remove_plane_cache_model or self.plane_model is None:
                plane_model, inliers = open3d_plane_segment_api(points, remove_plane_distance_thresh)
                keep = np.ones([points.shape[0]], dtype=bool)
                keep[inliers] = 0
            else:
                dists = point_plane_distance_api(points, self.plane_model)
                keep = dists > remove_plane_distance_thresh
            points = points[keep]
            if colors is not None:
                colors = colors[keep]

        super().add_point_cloud(points, colors, name=name)

    @staticmethod
    def set_default_xyz_pattern(xyz_pattern):
        Vis3D.default_xyz_pattern = xyz_pattern

    def set_scene_id(self, id):
        if not self.enable:
            return
        if self.auto_increase:
            warnings.warn(
                "Auto-increase in ON. You should not set_scene_id manually.")
        super().set_scene_id(id)
        self.add_point_cloud(1000 * torch.ones([1, 3]), name='dummy')

    def add_camera_trajectory(self, poses: Union[np.ndarray, torch.Tensor], *, name: str = None) -> None:
        """
        Add a camera trajectory

        :param poses: transformation matrices of shape `(n, 4, 4)`

        :param name: output name of the camera trajectory
        """
        if not self.enable:
            return
        poses = tensor2ndarray(poses)

        poses = (self.three_to_world @ poses.T).T
        poses[:, :, [1, 2]] *= -1
        # r = Rotation.from_matrix(poses[:, :3, : 3])
        # eulers = r.as_euler('xyz')
        eulers = []
        positions = poses[:, :3, 3].reshape((-1, 3))
        for pose in poses:
            trans_euler = euler.mat2euler(pose[:3, :3], 'rxyz')
            # trans_euler = euler.mat2euler(pose[:3, :3])
            eulers.append(trans_euler)

        # print("eulers: ", eulers)
        # print("euler: ", eulers)

        filename = self.__get_export_file_name('camera_trajectory', name)
        with open(filename, 'w') as f:
            f.write(json.dumps(dict(eulers=eulers, positions=positions.tolist())))

    def add_image(self, image, name=None):
        if not self.enable:
            return
        if isinstance(image, (str, PIL.Image.Image)):
            super().add_image(image, name=name)
        elif isinstance(image, np.ndarray):
            if image.dtype == np.float32 or image.dtype == np.float64:
                image = (image * 255).astype(np.uint8)
            image = PIL.Image.fromarray(image)
            self.add_image(image, name)
        elif isinstance(image, torch.Tensor):
            image = to_array(image)
            if image.shape[0] == 3:
                image = image.permute(1, 2, 0)
            self.add_image(image, name)
        else:
            raise TypeError()

    def add_mesh(self, vertices, faces=None, vertex_colors=None, *, name=None):
        if not self.enable:
            return
        if vertices is None:
            return
        vertices = clone_if_present(vertices)
        faces = clone_if_present(faces)
        vertex_colors = clone_if_present(vertex_colors)
        if isinstance(vertices, trimesh.Trimesh):
            vertex_colors = vertices.visual.vertex_colors
        super().add_mesh(vertices, faces, vertex_colors, name=name)

    def add_plt(self, x, name=None, **kwargs):
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from matplotlib.figure import Figure
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.gca()

        # ax.text(0.0, 0.0, "Test", fontsize=45)
        ax.axis('off')
        fig.tight_layout(pad=0)

        # To remove the huge white borders
        ax.margins(0)
        x = to_array(x)
        ax.imshow(x, **kwargs)
        # plt.axis('off')
        canvas.draw()  # draw the canvas, cache the renderer

        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        self.add_image(image, name=name)

    def increase_scene_id(self):
        if not self.enable:
            return
        # self.set_scene_id(self.scene_id + 1)
        super().set_scene_id(self.scene_id + 1)

    def add_boxes(self, positions, eulers=None, extents=None, *, order=(0, 1, 2, 3, 4, 5, 6, 7), labels=None,
                  name=None):
        if not self.enable:
            return
        positions = tensor2ndarray(positions).copy()

        if eulers is None or extents is None:
            positions = np.asarray(positions).reshape(-1, 8, 3)
            corners = positions
            if order != (0, 1, 2, 3, 4, 5, 6, 7):
                for i, o in enumerate(order):
                    corners[:, o, :] = positions[:, i, :]

            positions = (corners[:, 0, :] + corners[:, 6, :]) / 2
            vector_xs = corners[:, 1, :] - corners[:, 0, :]
            vector_ys = corners[:, 4, :] - corners[:, 0, :]
            vector_zs = corners[:, 3, :] - corners[:, 0, :]

            extent_xs = np.linalg.norm(vector_xs, axis=1).reshape(-1, 1)
            extent_ys = np.linalg.norm(vector_ys, axis=1).reshape(-1, 1)
            extent_zs = np.linalg.norm(vector_zs, axis=1).reshape(-1, 1)
            extents = np.hstack((extent_xs, extent_ys, extent_zs))

            rot_mats = np.stack(
                (vector_xs / extent_xs, vector_ys / extent_ys, vector_zs / extent_zs), axis=2)
            Rs = Rotation.from_matrix(rot_mats)
            eulers = Rs.as_euler('XYZ')
        else:
            positions = tensor2ndarray(positions)
            eulers = tensor2ndarray(eulers)
            extents = tensor2ndarray(extents)
            positions = np.asarray(positions).reshape(-1, 3)
            extents = np.asarray(extents).reshape(-1, 3)
            eulers = np.asarray(eulers).reshape(-1, 3)

        boxes = []
        for i in range(len(positions)):
            box_def = self.three_to_world @ affines.compose(
                positions[i], euler.euler2mat(*eulers[i], 'rxyz'), extents[i])
            T, R, Z, _ = affines.decompose(box_def)
            box = dict(
                position=T.tolist(),
                euler=euler.mat2euler(R, 'rxyz'),
                extent=Z.tolist()
            )
            if labels is not None:
                if isinstance(labels, str):
                    labels = [labels]
                box.update({'label': labels[i]})

            boxes.append(box)

        filename = self.__get_export_file_name('boxes', name)
        with open(filename, 'w') as f:
            f.write(json.dumps(boxes))

    def __repr__(self):
        if not self.enable:
            return f'Vis3D:NA'
        else:
            return f'Vis3D:{self.sequence_name}:{self.scene_id}'

    def __get_export_file_name(self, file_type: str, name: str = None) -> str:
        export_dir = os.path.join(
            self.out_folder,
            self.sequence_name,
            "%05d" % self.scene_id,
            folder_names[file_type],
        )
        os.makedirs(export_dir, exist_ok=True)
        if name is None:
            name = "%05d" % self.counters[file_type]

        filename = os.path.join(export_dir, name + "." + file_exts[file_type])
        self.counters[file_type] += 1

        return filename
