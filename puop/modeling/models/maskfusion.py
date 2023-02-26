import torch
from pytorch3d.ops import corresponding_points_alignment
from torch import nn

from puop.utils.cam_utils import matrix_to_cam_fufvcucv
from puop.utils.utils_3d import backproject_flow3d_torch, transform_points, matrix_3x4_to_4x4, open3d_icp_api, \
    depth_to_rect, open3d_tsdf_fusion_api


class MaskFusion(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.total_cfg = cfg
        assert self.total_cfg.tf == 1
        self.cfg = cfg.model.maskfusion
        self.dummy_params = nn.Linear(10, 10)

    def forward(self, dps):
        assert not self.training
        T = len(dps)
        valids = dps[0]['all_sa_valids'][0].T
        object_poses = dps[0]['pose'].inverse() @ dps[0]['sa_init_poses'].float()
        nobjs = object_poses.shape[1]
        campose = dps[0]['pose'][0]
        meshes = []
        all_final_poses = []
        all_masks = []
        start_frames = [v.nonzero()[:, 0].min().item() for v in valids]
        for vi in range(nobjs):
            final_poses = torch.eye(4)[None].repeat(T, 1, 1)
            t = start_frames[vi]
            curr_pose = object_poses[:, vi]
            final_poses[t] = curr_pose
            while t < T - 1:
                next_t = t + 1
                while not valids[vi, next_t] and next_t < T - 1:
                    next_t += 1
                if not valids[vi, next_t]:
                    next_pose = curr_pose
                else:
                    if next_t - t == 1:
                        flow = dps[t]['raft_flow'][0]
                    else:
                        flow = dps[t][f'raft_flow_{next_t - t}'][0]
                    curr_mask = dps[t]['segany_mask'][0, vi].bool()
                    curr_depth = dps[t]['depth'][0]
                    next_depth = dps[next_t]['depth'][0]

                    tmp_campose = torch.eye(4, device='cuda', dtype=torch.float)
                    flow3d, pts0, pts1 = backproject_flow3d_torch(flow, curr_depth, next_depth,
                                                                  dps[0]['K'][0], tmp_campose, tmp_campose)
                    pts0 = pts0.reshape(-1, 3)[curr_mask.reshape(-1)]
                    pts1 = pts1.reshape(-1, 3)[curr_mask.reshape(-1)]
                    valid = self.filter_scene_flow(pts0, pts1, campose)

                    pts0, pts1 = pts0[valid], pts1[valid]
                    if pts0.shape[0] > 0 and pts1.shape[0] > 0 and vi > 0:
                        cpa_res = corresponding_points_alignment(pts0[None].cuda(), pts1[None].cuda())
                        Rt = torch.cat([cpa_res.R[0].T, cpa_res.T[0][:, None]], dim=1)
                        pose = matrix_3x4_to_4x4(Rt)
                        if self.cfg.icp:
                            pose_icp = open3d_icp_api(pts0.cpu().numpy(), pts1.cpu().numpy(), self.cfg.icp_thresh,
                                                      pose.cpu().numpy())
                            pose = torch.from_numpy(pose_icp.copy()).cuda().float()
                    else:
                        pose = torch.eye(4).cuda().float()
                    pose = pose.cpu()
                    next_pose = pose @ curr_pose
                final_poses[next_t] = next_pose
                t = next_t
                curr_pose = next_pose
            # reconstruction
            masks = torch.stack([d['segany_mask'][0, vi] for d in dps])
            all_masks.append(masks * (1 + vi))
            depths = torch.stack([d['depth'][0] for d in dps])
            # if vi == 0 and self.cfg.ignore0:
            #     mesh = trimesh.primitives.Box()
            # else:
            mesh = self.recon(final_poses, masks, depths, dps[0]['K'][0], dps[0]['pose'][0])
            meshes.append(mesh)
            all_final_poses.append(final_poses)

        loss_dict = {}
        output = {'object_poses': torch.stack(all_final_poses, dim=1)}
        output.update({'mesh_verts': [torch.from_numpy(mesh.vertices).float().cuda() for mesh in meshes],
                       'mesh_faces': [torch.from_numpy(mesh.faces).long().cuda() for mesh in meshes]})
        output['mask'] = torch.stack(all_masks).sum(0)
        output['valid'] = valids
        return output, loss_dict

    def recon(self, object_poses, masks, depths, K, camera_pose):
        if self.cfg.remove_plane:
            masks = self.remove_plane(depths, masks, K, camera_pose)
        # if self.cfg.fusion_use_open3d:
        mesh = open3d_tsdf_fusion_api((depths * masks.float()).cpu().numpy(),
                                      object_poses.cpu().numpy(),
                                      K, self.cfg.voxel_length)
        return mesh

    def remove_plane(self, depths, masks, K, camera_pose):
        n = depths.shape[0]
        fu, fv, cu, cv = matrix_to_cam_fufvcucv(K)
        for i in range(n):
            pts_cam = depth_to_rect(fu, fv, cu, cv, depths[i])
            pts_world = transform_points(pts_cam, camera_pose)
            keep = pts_world[:, 2] > self.cfg.remove_plane_thresh
            masks[i][keep.reshape(masks[i].shape) == 0] = 0
        return masks

    def filter_scene_flow(self, p0, p1, camera_pose):
        valid1 = (p0 - p1).norm(dim=-1) < self.cfg.scene_flow_filter.max_value
        valid2 = (p0 - p1).norm(dim=-1) > self.cfg.scene_flow_filter.min_value
        valid = valid1 & valid2
        p0_world = transform_points(p0, camera_pose)
        p1_world = transform_points(p1, camera_pose)
        world_z_max = self.cfg.scene_flow_filter.world_z_max
        world_z_min = self.cfg.scene_flow_filter.world_z_min
        valid = (p0_world[:, 2] > world_z_min) & (p0_world[:, 2] < world_z_max) \
                & (p1_world[:, 2] > world_z_min) & (p1_world[:, 2] < world_z_max) & valid
        return valid
