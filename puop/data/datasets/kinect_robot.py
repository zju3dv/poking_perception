import os.path as osp
import pickle

import cv2
import imageio
import numpy as np
import torch
import tqdm
import trimesh
import zarr
from pytorch3d.structures import Meshes
from skimage.transform import rescale
from torch.utils.data import Dataset

from puop.utils.utils_3d import depth_to_rect
from puop.utils.vis3d_ext import Vis3D


class KinectRobot(Dataset):
    def __init__(self, cfg, data_dir, scene, skip, split, first_frame, last_frame, training_factor=1, transforms=None,
                 ds_len=-1) -> None:
        super().__init__()
        Vis3D.set_default_xyz_pattern(('x', '-y', '-z'))
        self.cfg = cfg.dataset.kinectrobot

        data_dir = osp.join(data_dir, scene, f'{self.cfg.name}{skip}')

        self.data_dir = data_dir
        self.scene = scene
        self.total_cfg = cfg
        self.transforms = transforms
        self.training_factor = training_factor
        self.first_frame = first_frame
        self.last_frame = last_frame

        K = np.loadtxt(osp.join(data_dir, 'K.txt'))
        K[0] = K[0] / float(training_factor)
        K[1] = K[1] / float(training_factor)

        if self.cfg.keep_frames == [] or split == 'test':
            self.frames = list(range(first_frame, last_frame + 1))
        else:
            self.frames = self.cfg.keep_frames

        camera_pose = np.loadtxt(osp.join(data_dir, 'camera_pose.txt'))
        camera_poses = np.repeat(camera_pose[None], len(self.frames), axis=0)
        self.camera_poses = camera_poses.astype(np.float32)
        self.camera_poses[:, :3, 3] = self.camera_poses[:, :3, 3] * self.cfg.data_scale

        with open(osp.join(data_dir, 'object_poses.pkl'), 'rb') as fp:
            ops = pickle.load(fp)

        object_names = list(ops[0].keys())
        self.object_names = object_names
        object_poses = {on: [] for on in object_names}

        for op in ops:
            for on in object_names:
                H_m2w = op[on].copy()
                H_m2w[:3, 3] = H_m2w[:3, 3] * self.cfg.data_scale
                object_poses[on].append(H_m2w)

        if self.cfg.keep_only > 0:
            new_object_names = []
            for vi in range(len(self.object_names)):
                if self.cfg.keep_only < 0 or vi + 1 == self.cfg.keep_only:
                    new_object_names.append(self.object_names[vi])
            self.object_names = new_object_names

        meshv, meshf = [], []
        meshes = {}
        for on in self.object_names:
            mesh: trimesh.Trimesh = trimesh.load_mesh(osp.join(data_dir, f'{on}.ply'))
            mesh.apply_scale(self.cfg.data_scale)
            meshv.append(torch.from_numpy(mesh.vertices).float())
            meshf.append(torch.from_numpy(mesh.faces).long())
            meshes[on] = mesh
        self.meshes = meshes
        self.meshespt3d = Meshes(meshv, meshf)

        H, W, _ = imageio.imread(osp.join(self.data_dir, 'color/000000.png')).shape
        H = int(H // training_factor)
        W = int(W // training_factor)

        self.object_poses = object_poses

        self.K = K
        self.H = H
        self.W = W

        self.near = self.cfg.near
        self.far = self.cfg.far

        rgbs = {}
        rgb_paths = {}
        for index in tqdm.tqdm(self.frames, leave=False, desc='loading rgb'):
            if split != 'test':
                rgb_path = osp.join(self.data_dir, 'color', f'{index:06d}.png')
                rgb_paths[index] = rgb_path
                rgbs[index] = np.array(imageio.imread(rgb_path).astype(np.float32) / 255.0)
                if training_factor != 1:
                    rgbs[index] = rescale(rgbs[index], 1. / training_factor, anti_aliasing=False, channel_axis=2)
            else:
                rgbs[index] = np.zeros([H, W, 3])
                rgb_paths[index] = ''

        self.rgbs = rgbs
        self.rgb_paths = rgb_paths
        if self.cfg.load_mask:
            masks = {}
            for index in tqdm.tqdm(self.frames, leave=False, desc='loading mask'):
                if split != 'test':
                    mask = cv2.imread(osp.join(self.data_dir, 'instance_segmaps', f'{index:06d}.png'))[:, :, 0]
                    mask[mask > len(object_names)] = 0
                    if self.cfg.keep_only > 0:
                        mask = mask == self.cfg.keep_only
                    masks[index] = mask.astype(np.uint8)
                    if training_factor != 1:
                        masks[index] = masks[index].astype(np.float32)
                        masks[index] = rescale(masks[index], 1. / training_factor, anti_aliasing=False)
                        masks[index] = masks[index].astype(np.uint8)  # check
                else:
                    masks[index] = np.zeros([H, W], dtype=int)

            self.masks = masks

        if split != 'test':
            init_mask = torch.from_numpy(
                np.load(osp.join(self.data_dir, 'init_masks.npy')))
            init_mask = init_mask[self.frames]
        else:
            init_mask = torch.ones([len(self.frames), H, W], dtype=torch.long)
        tids = init_mask.unique().int().tolist()
        if 0 in tids:
            tids.remove(0)
        smasks = []
        for tid in tids:
            nmask = ((init_mask == tid).sum(1).sum(1) > 0).sum()
            if nmask >= 8 or nmask > len(self.frames) / self.cfg.load_segany_min_divider:
                new_mask = torch.zeros_like(init_mask)
                new_mask[init_mask == tid] = 1
                if training_factor != 1:
                    new_mask = torch.from_numpy(
                        np.stack([rescale(nm.numpy().astype(np.float32), 1. / training_factor,
                                          anti_aliasing=False).astype(np.uint8) for nm in new_mask])
                    )
                if tid == 1 and self.cfg.gripper_mask_dilate > 0:
                    gmd = self.cfg.gripper_mask_dilate
                    new_mask = torch.from_numpy(
                        np.stack([cv2.dilate(nm.numpy().astype(np.uint8), np.ones([gmd, gmd])).astype(bool) for nm in
                                  new_mask])
                    )
                smasks.append(new_mask)
        self.smasks = torch.stack(smasks, dim=1)
        if self.cfg.keep_only > 0:
            for vi in range(self.smasks.shape[1]):
                if self.cfg.keep_only_samask < 0:
                    ko = self.cfg.keep_only
                else:
                    ko = self.cfg.keep_only_samask
                if vi + 1 != ko:
                    self.smasks[:, vi] = self.smasks[:, vi] * 0

        depths = {}
        for index in tqdm.tqdm(self.frames, leave=False, desc='loading depth'):
            depth_path = osp.join(self.data_dir, 'depth', f'{index:06d}.png')
            if osp.exists(depth_path):
                depth_map = cv2.imread(depth_path, 2).astype(np.float32) / 1000.0
                depth_map = depth_map * self.cfg.data_scale
                depth_map[depth_map > self.cfg.clip_far] = 0
                if training_factor != 1:
                    depth_map = rescale(depth_map, 1. / training_factor, anti_aliasing=False)
            else:
                depth_map = np.zeros([H, W], dtype=np.float32)
            depths[index] = depth_map
        self.depths = depths
        self.mdepth = np.stack(list(depths.values())).max(0)
        if self.cfg.mask_rgb_by_0mdepth:
            for frame in self.frames:
                self.rgbs[frame] = self.rgbs[frame] * (self.mdepth > 0)[:, :, None]

        if self.cfg.remove_bg:
            self.fg_masks = {}
            for fi, frame in enumerate(self.frames):
                if self.cfg.load_motion_mask and self.cfg.mask_as_motion_mask:
                    mask = self.motion_masks[frame]
                else:
                    mask = (self.smasks[fi].sum(0) > 0).numpy()
                self.rgbs[frame] = self.rgbs[frame] * (mask[:, :, None] != 0) + (1 - (mask[:, :, None] != 0))
                self.depths[frame] = self.depths[frame] * (mask != 0)
                self.fg_masks[frame] = mask
        # cache hwl
        hwl = []
        for oi, v in enumerate(self.meshespt3d.verts_list()):
            hwl.append(v.max(0).values - v.min(0).values)
        hwl = torch.stack(hwl)
        self.hwl = hwl

        if self.cfg.keep_only < 0:
            sums = self.smasks.sum(-1).sum(-1) > 0
            sa_init_poses = []
            for vi in range(sums.shape[1]):
                ni = sums[:, vi].nonzero()[0, 0]
                dep = self.depths[self.frames[ni]]
                dep = dep * self.smasks[ni, vi].numpy()
                pts_rect = depth_to_rect(self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2], dep)
                ip = np.eye(4)
                ip[:3, 3] = pts_rect[(pts_rect != 0).any(-1)].mean(0)
                ip = self.camera_poses[0] @ ip
                sa_init_poses.append(ip)
            sa_init_poses = np.stack(sa_init_poses)
            self.sa_init_poses = sa_init_poses
            if self.cfg.sa_init_pose != []:
                self.sa_init_poses[1:] = np.array(self.cfg.sa_init_pose)
            self.all_sa_valids = sums
        if self.cfg.load_raft_flow:
            raft_flows = {}
            for index in tqdm.tqdm(self.frames, leave=False, desc='loading raft_flow'):
                if split != 'test':
                    flow_path = osp.join(self.data_dir, 'raft_flow', f'{index}+1.zarr')
                    raft_flows[index] = zarr.load(flow_path)
                    if training_factor != 1:
                        raft_flows[index] = rescale(raft_flows[index], 1. / training_factor, anti_aliasing=False,
                                                    channel_axis=2)
                else:
                    raft_flows[index] = np.zeros([H, W, 2])
            self.raft_flows = raft_flows
            raft_flows_skips = {}
            for interval in self.cfg.load_raft_flow_intervals:
                rfs = {}
                for index in tqdm.tqdm(self.frames, leave=False, desc='loading raft_flow'):
                    if index + interval < max(self.frames):
                        path = osp.join(self.data_dir, 'raft_flow', f'{index}+{interval}.zarr')
                        assert osp.exists(path)
                        rf = zarr.load(path)
                    else:
                        rf = np.zeros([H, W, 2])
                    if training_factor != 1:
                        rf = rescale(rf, 1. / training_factor, anti_aliasing=False, channel_axis=2)
                    rfs[index] = rf
                raft_flows_skips[interval] = rfs
            self.raft_flows_skips = raft_flows_skips

    def real_getitem(self, idx, frame_index):
        rgb = self.rgbs[frame_index]
        rgb_path = self.rgb_paths[frame_index]
        data_dict = {'image': rgb, 'image_path': rgb_path}
        if self.cfg.load_mask: data_dict['mask'] = self.masks[frame_index]
        if self.cfg.remove_bg: data_dict['fg_mask'] = self.fg_masks[frame_index]
        if self.cfg.load_raft_flow: data_dict['raft_flow'] = self.raft_flows[frame_index]
        for skip in self.cfg.load_raft_flow_intervals:
            data_dict[f'raft_flow_{skip}'] = self.raft_flows_skips[skip][frame_index]
        data_dict['segany_mask'] = self.smasks[idx]
        if self.cfg.keep_only < 0:
            data_dict['sa_init_poses'] = self.sa_init_poses
            data_dict['all_sa_valids'] = self.all_sa_valids
        data_dict['depth'] = self.depths[frame_index]
        data_dict['mdepth'] = self.mdepth

        ops = []
        for oi in range(len(self.object_names)):
            on = self.object_names[oi]
            object_pose = self.object_poses[on][frame_index]
            ops.append(object_pose)
        ops = torch.from_numpy(np.stack(ops)).float()

        allops = []
        for oi in range(len(self.object_names)):
            ops_per_obj = []
            for fidx in self.frames:
                on = self.object_names[oi]
                object_pose = self.object_poses[on][fidx]
                ops_per_obj.append(object_pose)
            allops.append(ops_per_obj)
        allops = torch.from_numpy(np.array(allops)).float()

        camera_pose = self.camera_poses[idx]
        xyz = ops[..., :3, 3]
        box3d = torch.cat([xyz, self.hwl, ops[..., :3, :3].reshape(-1, 9)], dim=1).reshape(
            len(self.object_names) if self.cfg.keep_only < 0 else 1, -1)
        data_dict.update({
            'pose': camera_pose,
            'pose_requires_neg': False,
            'K': self.K,
            'H': self.H,
            'W': self.W,
            'focal': self.K[0, 0],
            'max_input_objects': 1,
            'object_pose': ops,
            'all_ops': allops,
            'mesh_verts': self.meshespt3d.verts_padded(),
            'mesh_faces': self.meshespt3d.faces_padded(),
            'box3d': box3d,
            'tracking_id': torch.tensor([0], dtype=torch.long),
            'cls_id': torch.tensor([0], dtype=torch.long),
            'near': torch.tensor(self.near).float(),
            'far': torch.tensor(self.far).float(),
            'frame_id': frame_index,
            'ndc': False,

        })
        if self.transforms is not None:
            data_dict = self.transforms(data_dict)
        data_dict['is_last_frame'] = frame_index == self.frames[-1]
        return data_dict

    def __getitem__(self, idx):
        frame_id = self.frames[idx]
        data_dict = self.real_getitem(idx, frame_id)
        return data_dict

    def __len__(self):
        return len(self.frames)
