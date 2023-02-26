import cv2
import os

import kornia.morphology
import loguru
import numpy as np
import pytorch3d.transforms
import skimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import trimesh
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes
from puop.utils.utils_3d import se3_exp_map
from tensorboardX import SummaryWriter
from tqdm import trange
from transforms3d.euler import mat2euler
from trimesh.graph import split

from puop.utils.cam_utils import matrix_to_cam_fufvcucv
from puop.utils.comm import get_world_size, get_rank
from puop.utils.os_utils import deterministic
from puop.utils.utils_3d import transform_points, depth_to_rect, pose_distance
from puop.utils.vis3d_ext import Vis3D
from puop.utils.plt_utils import show
from . import rend_util
from .nerf import NeRF
from .volsdf import VolSDF


class PokingRecon(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.model.pokingrecon
        self.total_cfg = cfg

        if self.cfg.static_on:
            output_ch = 4
            skips = [4]
            self.model_bg = NeRF(D=self.cfg.netdepth, W=self.cfg.netwidth,
                                 input_ch=3, output_ch=output_ch, skips=skips,
                                 input_ch_views=3, use_viewdirs=self.cfg.use_viewdirs,
                                 multires=self.cfg.multires, multires_views=self.cfg.multires_views)
        if self.cfg.frame_ids == []:
            self.frame_ids = list(range(self.cfg.trim_start, self.cfg.trim_start + self.cfg.num_frames))
        else:
            self.frame_ids = self.cfg.frame_ids

        if self.cfg.dynamic_on:
            for vi in range(self.cfg.nobjs):
                setattr(self, f'model_fg{vi}', VolSDF(cfg.model.pokingrecon.volsdf))
            pose_init = torch.tensor(self.cfg.pose_init).float()
            if (pose_init == 0).all():
                pose_init = torch.zeros([self.cfg.nobjs, len(self.frame_ids), 6]).float()
            if len(self.frame_ids) != 0:
                assert list(pose_init.shape) == [self.cfg.nobjs, len(self.frame_ids),
                                                 6], f"{pose_init.shape}!={[self.cfg.nobjs, len(self.frame_ids), 6]}"
            self.pose_init = pose_init
            for vi in range(self.cfg.nobjs):
                for i in range(len(self.frame_ids)):
                    if i == 0:
                        requires_grad = False
                    else:
                        requires_grad = self.cfg.optim_pose
                    param_name = 'objpose' + str(vi) + "_" + str(self.frame_ids[i])
                    if (pose_init[vi, i] == 0).all():
                        requires_grad = False
                    if self.cfg.fix_objpose_optim_frame_ids_all_objs == []:
                        if self.frame_ids[i] in self.cfg.fix_objpose_optim_frame_ids:
                            requires_grad = False
                    else:
                        if self.frame_ids[i] in self.cfg.fix_objpose_optim_frame_ids_all_objs[vi]:
                            requires_grad = False
                    setattr(self, param_name, nn.Parameter(pose_init[vi, i].cuda(), requires_grad=requires_grad))
            for vi in range(self.cfg.nobjs):
                if len(self.cfg.pretrained_volsdf) > 0 and self.cfg.pretrained_volsdf[vi] != '':
                    ckpt = torch.load(self.cfg.pretrained_volsdf[vi], 'cpu')['model']
                    ckpt = {k: v for k, v in ckpt.items() if not k.startswith('model_bg')}
                    ckpt = {k.replace('objpose0', f'objpose{vi}').replace('model_fg0', f'model_fg{vi}'): v for k, v in
                            ckpt.items()}
                    if len(self.cfg.override_poses_all_objs) == 0:
                        if not self.cfg.override_poses:
                            ckpt = {k: v for k, v in ckpt.items() if 'objpose' not in k}
                    else:
                        if not self.cfg.override_poses_all_objs[vi]:
                            ckpt = {k: v for k, v in ckpt.items() if 'objpose' not in k}
                    self.load_state_dict(ckpt, strict=False)
        self.dbg = cfg.dbg
        self.tb_writer_iter = 0
        self._point_cloud = None

    def get_object_pose(self, objid, frameid):
        """
        :param objid: from 0.
        :param frameid:
        :return:
        """
        return getattr(self, f"objpose{objid}_{frameid}")

    def get_all_object_poses(self, frame_id=None):
        ops = []
        for objid in range(self.cfg.nobjs):
            if frame_id is None:
                ops.append(
                    torch.stack([getattr(self, 'objpose' + str(objid) + "_" + str(frameid)) for frameid in
                                 self.frame_ids]))
            else:
                ops.append(getattr(self, 'objpose' + str(objid) + "_" + str(frame_id)))
        if frame_id is None:
            ops = torch.stack(ops, 1)
        else:
            ops = torch.stack(ops, 0)
        return ops

    def init_pose_at_frame(self, frames):
        value = getattr(self, "objpose" + str(frames[0] - 1))
        for frame in frames:
            getattr(self, "objpose" + str(frame)).data = value.clone()

    def forward(self, dps):
        if self.total_cfg.solver.print_it:
            print('\nit', dps['global_step'], '\n')
        vis3d = Vis3D(sequence="pokingrecon_forward", enable=self.dbg)
        H, W, K = dps['H'][0].item(), dps['W'][0].item(), dps['K'][0]
        poses, images = dps['pose'], dps['image']
        batch_size = images.shape[0]
        rays_o_all, rays_d_all = get_rays_batch(H, W, K, poses)
        if self.dbg and self.cfg.dynamic_on:
            box3ds = dps['box3d'][0][:self.cfg.nobjs]
            if 'mesh_verts' in dps and 'mesh_faces' in dps:
                mesh_verts = dps['mesh_verts'][0]
                mesh_faces = dps['mesh_faces'][0]
                for tmpi, (box3d, mv, mf, op) in enumerate(zip(box3ds, mesh_verts, mesh_faces, dps['object_pose'][0])):
                    o = box3d.tolist()
                    x, y, z, dx, dy, dz, = o[0:6]
                    R = np.array(o[6:15]).reshape(3, 3)
                    vis3d.add_boxes(torch.tensor([x, y, z]), mat2euler(R, 'rxyz'), torch.tensor([dx, dy, dz]),
                                    name=f'boxgt_{tmpi}')
                    vis3d.add_mesh(transform_points(mv[mv.sum(1) != 0], op), mf[mf.sum(-1) > 0], name=f'meshgt_{tmpi}')
                    pred_obj_pose = self.get_object_pose(tmpi, dps['frame_id'][0].item())
                    pred_obj_pose = se3_exp_map(pred_obj_pose[None]).permute(0, 2, 1)[0]
                    vis3d.add_mesh(transform_points(mv[mv.sum(1) != 0], pred_obj_pose), mf[mf.sum(-1) > 0],
                                   name=f'mesh_pred_{tmpi}')
            vis3d.add_camera_trajectory(poses, name='camera')
            vis3d.add_image(images[0].cpu().numpy(), name='rgb')
        coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H),
                                            torch.linspace(0, W - 1, W), indexing='ij'),
                             -1)  # (H, W, 2)
        coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
        mtarget_depth_s = None
        if self.training:
            N_rand_per_img = self.cfg.N_rand // batch_size
            if self.cfg.rand_method == 'samask_random':
                if 'PYCHARM_HOSTED' in os.environ: np.random.seed(0)
                all_select_coords = []
                select_inds2 = np.random.choice(coords.shape[0],
                                                size=[N_rand_per_img - int(
                                                    N_rand_per_img * self.cfg.rand_mask_ratio)],
                                                replace=False)  # (N_rand,)
                select_coords2 = coords[select_inds2].long()  # (N_rand, 2)
                mmasks = dps['segany_mask'].sum(1)
                for bi in range(batch_size):
                    mmask = mmasks[bi]
                    fg_coords = ((mmask > 0) & (mmask < 100)).nonzero()
                    if self.dbg: np.random.seed(0)
                    size = int(N_rand_per_img * self.cfg.rand_mask_ratio)
                    if fg_coords.shape[0] > 0:
                        select_inds1 = np.random.choice(fg_coords.shape[0], size=[size], replace=True)  # (N_rand,)
                        select_coords1 = fg_coords[select_inds1].long()  # (N_rand, 2)
                    else:
                        select_inds1 = np.random.choice(coords.shape[0], size=[size], replace=True)  # (N_rand,)
                        select_coords1 = coords[select_inds1].long()  # (N_rand, 2)
                    select_coords = torch.cat([select_coords1.cpu(), select_coords2.cpu()])
                    all_select_coords.append(select_coords)
            elif self.cfg.rand_method == 'samask_inv0_random':
                if 'PYCHARM_HOSTED' in os.environ: np.random.seed(0)
                all_select_coords = []
                mmasks = dps['segany_mask']
                size = int(N_rand_per_img * self.cfg.rand_mask_ratio)
                bg_size = N_rand_per_img - size
                for bi in range(batch_size):
                    mmask = mmasks[bi]
                    gmask = mmask[0]
                    mmask = mmask[1:].sum(0) > 0
                    mmask = mmask & (~gmask).bool()
                    bg_coords = (gmask == 0).nonzero()
                    fg_coords = mmask.nonzero()
                    if self.dbg: np.random.seed(0)
                    select_inds1 = np.random.choice(fg_coords.shape[0], size=[size], replace=True)  # (N_rand,)
                    select_coords1 = fg_coords[select_inds1]  # (N_rand, 2)

                    select_inds2 = np.random.choice(bg_coords.shape[0], size=[bg_size], replace=True)  # (N_rand,)
                    select_coords2 = bg_coords[select_inds2]  # (N_rand, 2)

                    select_coords = torch.cat([select_coords1, select_coords2])
                    all_select_coords.append(select_coords.cpu())
            elif self.cfg.rand_method == 'inv_segany_mmask_random':
                if 'PYCHARM_HOSTED' in os.environ: np.random.seed(0)
                smask = dps['segany_mask']
                all_select_coords = []
                for bi in range(batch_size):
                    sec2_coords = ((smask[bi].sum(0) > 0) == 0).nonzero()
                    if self.dbg: np.random.seed(0)
                    select_inds = np.random.choice(sec2_coords.shape[0], size=[N_rand_per_img],
                                                   replace=False)  # (N_rand,)
                    select_coords = sec2_coords[select_inds].long()  # (N_rand, 2)
                    all_select_coords.append(select_coords)
            else:
                raise NotImplementedError()
            rays_o = [r[s[:, 0], s[:, 1]] for s, r in zip(all_select_coords, rays_o_all)]  # (N_rand, 3)
            rays_o = torch.cat(rays_o, 0)
            rays_d = [r[s[:, 0], s[:, 1]] for s, r in zip(all_select_coords, rays_d_all)]  # (N_rand, 3)
            rays_d = torch.cat(rays_d, 0)
            target_s = [image[s[:, 0], s[:, 1]] for image, s in zip(images, all_select_coords)]  # (N_rand, 3)
            target_s = torch.cat(target_s, 0)

            if self.cfg.mdepth_loss_on or self.cfg.mdepth_loss_bg_on or self.cfg.mdepth_fill or self.cfg.mdepth_fill_bg:
                mpts_rect = depth_to_rect(*matrix_to_cam_fufvcucv(K), dps['mdepth'][0])

            if self.cfg.depth_loss_on:
                pts_rect = [depth_to_rect(*matrix_to_cam_fufvcucv(K), d) for d in dps['depth']]
                if self.training and dps['global_step'] > self.cfg.depth_loss_miniter or not self.training:
                    selected_pts_rect = [p.reshape(H, W, 3)[s[:, 0], s[:, 1]] for p, s in
                                         zip(pts_rect, all_select_coords)]
                    selected_pts_rect = torch.cat(selected_pts_rect, 0)
                    target_depth_s = torch.norm(selected_pts_rect, dim=-1)
            if self.cfg.mdepth_loss_on or self.cfg.mdepth_loss_bg_on or self.cfg.mdepth_fill or self.cfg.mdepth_fill_bg:
                mselected_pts_rect = [mpts_rect.reshape(H, W, 3)[s[:, 0], s[:, 1]] for s in all_select_coords]
                mselected_pts_rect = torch.cat(mselected_pts_rect, 0)
                mtarget_depth_s = torch.norm(mselected_pts_rect, dim=-1)
                dps['mtarget_depth_s'] = mtarget_depth_s
            if self.cfg.motion_mask_loss:
                if 'segany_mask' in dps:
                    motion_masks = (dps['segany_mask'] * (torch.arange(dps['segany_mask'].shape[1]) + 1)[None, :, None,
                                                         None].cuda()).sum(1)
                else:
                    motion_masks = dps['motion_mask']
                target_motion_s = [motion_mask[s[:, 0], s[:, 1]] for motion_mask, s in
                                   zip(motion_masks, all_select_coords)]
                target_motion_s = torch.cat(target_motion_s)
                dps['target_motion_s'] = target_motion_s
            if self.cfg.depth_loss_outside_mask:
                fg_masks = dps['fg_mask']
                target_fg_mask = [fm[s[:, 0], s[:, 1]] for fm, s in zip(fg_masks, all_select_coords)]
                target_fg_mask = torch.cat(target_fg_mask)
        else:
            assert batch_size == 1
            N_rand_per_img = H * W
            target_s = images[0]
            rays_o = rays_o_all[0]
            rays_d = rays_d_all[0]
            select_coords = coords.long()
            if self.cfg.depth_loss_on:
                pts_rect = depth_to_rect(*matrix_to_cam_fufvcucv(K), dps['depth'][0])
                target_depth_s = torch.norm(pts_rect, dim=-1).reshape(H, W)
            if self.cfg.mdepth_loss_on or self.cfg.mdepth_loss_bg_on or self.cfg.mdepth_fill or self.cfg.mdepth_fill_bg:
                mpts_rect = depth_to_rect(*matrix_to_cam_fufvcucv(K), dps['mdepth'][0])
                mtarget_depth_s = torch.norm(mpts_rect, dim=-1).reshape(-1)
            if self.cfg.motion_mask_loss:
                if 'segany_mask' not in dps:
                    target_motion_s = dps['motion_mask'][0]
                else:
                    target_motion_s = dps['segany_mask'][0].sum(0) > 0
            if self.cfg.depth_loss_outside_mask:
                target_fg_mask = dps['fg_mask'][0]
        batch_rays = torch.stack([rays_o, rays_d], 0)  # 2,N_rand,3
        if self.cfg.recon:
            outputs = {}
            if dps['global_step'] == 0 and get_rank() == 0:
                meshes = self.extract_mesh(dps, volume_size=self.cfg.recon_volume_size, N=self.cfg.recon_N)
                outputs = {'meshes': meshes}
            return outputs, {}
        frame_ids = dps['frame_id'][:, None].repeat(1, N_rand_per_img).reshape(-1)
        rgb, depth, extras = self.render(H, W, K, rays=batch_rays,
                                         ndc=dps['ndc'][0].item(), near=dps['near'][0].item(), far=dps['far'][0].item(),
                                         frame_id=frame_ids, dps=dps, mtarget_depth_s=mtarget_depth_s)
        if not self.training:
            rgb8 = to8b(rgb).reshape(H, W, 3)
            pred_mask = extras['rgb_map_mask']
            pred_mask = pred_mask.reshape(H, W, 3)[:, :, 0] > 0.5
        img_loss = img2mse(rgb, target_s)
        loss = img_loss
        loss_dict = {}
        if self.cfg.depth_loss_on:
            if self.training and dps['global_step'] > self.cfg.depth_loss_miniter or not self.training:
                sup_mask = target_depth_s > 0
                dlossfunc = img2mse if self.cfg.depth_loss_func == 'mse' else F.l1_loss
                if self.cfg.depth_loss_all:
                    depth_loss = dlossfunc(depth, target_depth_s)
                else:
                    if self.cfg.depth_loss_outside_mask:
                        sup_mask = sup_mask | (~target_fg_mask)
                    depth_loss = dlossfunc(depth[sup_mask], target_depth_s[sup_mask])
                depth_loss = depth_loss * self.cfg.depth_loss_lambda
                loss_dict['depth_loss'] = depth_loss
        if self.cfg.mdepth_loss_on and self.cfg.retraw:
            mdloss_sigma_fg = extras['mdloss_sigma_fg']
            mdloss_sup_mask = extras['mdloss_sup_mask_fg']
            if mdloss_sup_mask.sum() == 0:
                mdepth_loss = mdloss_sigma_fg.sum() * 0.0
            else:
                mdepth_loss = mdloss_sigma_fg[mdloss_sup_mask].abs().mean()
            mdepth_loss = mdepth_loss * self.cfg.mdepth_loss_weight
            loss_dict['mdepth_loss'] = mdepth_loss
        if self.cfg.mdepth_loss_bg_on and self.cfg.retraw:
            mdloss_sigma_bg = extras['mdloss_sigma_bg']
            mdloss_sup_mask = extras['mdloss_sup_mask_bg']
            if mdloss_sup_mask.sum() == 0:
                mdepth_bg_loss = mdloss_sigma_fg.sum() * 0.0
            else:
                mdepth_bg_loss = mdloss_sigma_bg[mdloss_sup_mask].abs().mean()
            mdepth_bg_loss = mdepth_bg_loss * self.cfg.mdepth_loss_bg_weight
            loss_dict['mdepth_bg_loss'] = mdepth_bg_loss
        psnr = mse2psnr(img_loss)
        if self.cfg.sparse_loss and self.cfg.retraw and self.training:
            if self.cfg.sparse_loss_weighted:
                zvals = extras['z_vals']
                x = mtarget_depth_s[:, None] - zvals
                weight = torch.exp(self.cfg.sparse_loss_decay * x.clamp(min=0))[:, None, :]
                sparse_loss = (weight * exp_density(extras['sigma_fgs'])).sum()
                sparse_loss = sparse_loss * self.cfg.sparse_loss_weight
                loss_dict['sparse_loss'] = sparse_loss
            else:
                if self.cfg.sparse_loss_exp:
                    loss_dict['sparse_loss'] = exp_density(extras['sigma_fgs']).mean() * self.cfg.sparse_loss_weight
                else:
                    loss_dict['sparse_loss'] = extras['sigma_fgs'].mean() * self.cfg.sparse_loss_weight
            if 'sigma_fg0' in extras:
                if self.cfg.sparse_loss_exp:
                    loss_dict['sparse_loss0'] = exp_density(extras['sigma_fg0']).mean() * self.cfg.sparse_loss_weight
                else:
                    loss_dict['sparse_loss0'] = extras['sigma_fg0'].mean() * self.cfg.sparse_loss_weight
        if self.cfg.motion_mask_loss and dps[
            'global_step'] >= self.cfg.motion_mask_loss_min_iter and self.cfg.retraw and self.training:
            motion_mask_loss_exp = self.cfg.motion_mask_loss_exp
            mmloss = 0
            mml_func = torch.sum if self.cfg.motion_mask_loss_func == 'sum' else torch.mean
            if not motion_mask_loss_exp:
                for vi in range(self.cfg.nobjs):
                    mml = extras['sigma_fgs'][..., vi, :] * (target_motion_s != vi + 1).float()[..., None]
                    mmloss = mmloss + mml_func(mml)
            else:
                for vi in range(self.cfg.nobjs):
                    mml = exp_density(
                        extras['sigma_fgs'][..., vi, :] * ((target_motion_s != vi + 1).float())[..., None])
                    mmloss = mmloss + mml_func(mml)
            if self.cfg.motion_mask_loss_all_area:
                if not motion_mask_loss_exp:
                    mmloss = (extras['sigma_bg'] * (target_motion_s != 0).float()[..., None]).mean() + mmloss
                else:
                    mmloss = exp_density(
                        (extras['sigma_bg'] * (target_motion_s != 0).float()[..., None])).mean() + mmloss
            loss_dict['mm_loss'] = mmloss * self.cfg.motion_mask_loss_weight
            if 'sigma_fg0' in extras:
                if not motion_mask_loss_exp:
                    mmloss0 = (extras['sigma_fg0'] * (1 - target_motion_s.float())[..., None]).mean()
                else:
                    mmloss0 = exp_density((extras['sigma_fg0'] * (1 - target_motion_s.float())[..., None])).mean()
                if self.cfg.motion_mask_loss_all_area:
                    if not motion_mask_loss_exp:
                        mmloss0 = (extras['sigma_bg0'] * target_motion_s.float()[..., None]).mean() + mmloss0
                    else:
                        mmloss0 = exp_density(
                            (extras['sigma_bg0'] * target_motion_s.float()[..., None])).mean() + mmloss0
                loss_dict['mm_loss0'] = mmloss0 * self.cfg.motion_mask_loss_weight
        metrics = {'psnr': psnr}

        if self.cfg.dynamic_on:
            for vi in range(self.cfg.nobjs):
                alpha, beta = getattr(self, f'model_fg{vi}').forward_ab()
                metrics.update({f'alpha{vi}': alpha, f'beta{vi}': beta, })
            for vi in range(self.cfg.nobjs):
                pred_pose = torch.stack(
                    [self.get_object_pose(vi, fid.item()) for fid in dps['frame_id']])
                valid = (pred_pose != 0).any(-1)
                pred_pose = se3_exp_map(pred_pose).permute(0, 2, 1)
                if self.cfg.ignore_gripper:
                    gt_pose = dps['object_pose'][:, vi + self.cfg.vi_shift + 1]
                else:
                    gt_pose = dps['object_pose'][:, vi + self.cfg.vi_shift]

                rot_err, trans_err = pose_distance(pred_pose[valid], gt_pose[valid],
                                                   align=self.cfg.align_pose_eval)
                if rot_err.numel() > 0 and trans_err.numel() > 0:
                    trans_err_mean, trans_err_max = trans_err.mean(), trans_err.max()
                    rot_err_mean, rot_err_max = rot_err.mean() / np.pi * 180, rot_err.max() / np.pi * 180
                else:
                    rot_err_mean = rot_err_max = trans_err_mean = trans_err_max = torch.tensor([0]).cuda()
                metrics.update({f'trans_err{vi}': trans_err_mean,
                                f'rot_err{vi}': rot_err_mean,
                                f'max_trans_err{vi}': trans_err_max,
                                f'max_rot_err{vi}': rot_err_max
                                })
        if self.cfg.smooth_loss_on and self.cfg.dynamic_on:
            for vi in range(self.cfg.nobjs):
                all_poses = [self.get_object_pose(vi, fid) for fid in self.frame_ids]
                all_poses = torch.stack(all_poses)
                valid = (all_poses != 0).any(-1)
                all_poses = all_poses[valid]
                smooth_loss = 0
                if self.cfg.smooth_loss1_on:
                    mean = (all_poses[:-2] + all_poses[2:]) / 2
                    smooth_loss = smooth_loss + F.l1_loss(all_poses[1:-1], mean)
                if self.cfg.smooth_loss2_on:
                    smooth_loss = smooth_loss + F.l1_loss(all_poses[1:], all_poses[:-1])
                smooth_loss = smooth_loss * self.cfg.smooth_loss_weight
                loss_dict[f'smooth_loss{vi}'] = smooth_loss
        output = {'metrics': metrics}
        if not self.training:
            output['rgb8'] = rgb8
            output['depth'] = depth
            output['rgb_gt'] = images[0]
            if self.cfg.retraw and self.training: output['sigma_fgs'] = extras['sigma_fgs'].sum(-1)
            if self.cfg.mdepth_postprocess:
                invalid = extras['depth_fg'] > mtarget_depth_s
                pred_mask[invalid] = 0
            output['pred_mask'] = pred_mask

            if self.cfg.dynamic_on:
                obj_pose = torch.stack(
                    [self.get_object_pose(vi, dps['frame_id'].item()) for vi in range(self.cfg.nobjs)])
                obj_pose = se3_exp_map(obj_pose).permute(0, 2, 1)
                output['obj_pose'] = obj_pose
            if self.dbg:
                vis3d.add_image(images[0].cpu().numpy(), name='gt')
                vis3d.add_image(rgb8.cpu().numpy(), name='recon')
                vis3d.add_point_cloud(transform_points(
                    pts_rect, dps['pose'][0]), colors=target_s.reshape(-1, 3),
                    name='gt_pts', sample=1.0)
                vis3d.add_point_cloud(
                    transform_points(depth_to_rect(*matrix_to_cam_fufvcucv(K), depth, ray_mode=True), dps['pose'][0]),
                    sample=1, colors=rgb8.reshape(-1, 3),
                    name='pred_pts')

                vis3d.add_point_cloud(
                    transform_points(
                        depth_to_rect(*matrix_to_cam_fufvcucv(K), extras['static_depth_map'], ray_mode=True),
                        dps['pose'][0]),
                    sample=1, colors=rgb8.reshape(-1, 3),
                    name='pred_pts_static')
            dynamic_pred_depths = extras['dynamic_depth_maps']
            for tmpj in range(dynamic_pred_depths.shape[2]):
                vis3d.add_point_cloud(
                    transform_points(depth_to_rect(*matrix_to_cam_fufvcucv(K), extras['dynamic_depth_maps'][:, :, tmpj],
                                                   ray_mode=True), dps['pose'][0]),
                    sample=1, colors=rgb8.reshape(-1, 3),
                    name=f'pred_pts_dynamic_{tmpj}')
            vis3d.add_image(extras['static_rgb_map'].cpu().numpy(), name='pred_rgb_static')
            dynamic_rgb_maps = extras['dynamic_rgb_maps']
            for tmpj in range(dynamic_rgb_maps.shape[-2]):
                drm = dynamic_rgb_maps[:, :, tmpj, :]
                vis3d.add_image(drm.cpu().numpy(), name=f'pred_rgb_dynamic{tmpj}')
            if self.cfg.retraw and self.training:
                sigma_fgs_map = extras['sigma_fgs'].sum(-1)
                for tmpj in range(sigma_fgs_map.shape[-1]):
                    sf = sigma_fgs_map[:, :, tmpj]
                    vis3d.add_plt(sf, cmap='jet', name=f'sigma_fg_map{tmpj}')
            errmap = (rgb - images[0]).mean(-1)
            output['error_map'] = errmap

            tw: SummaryWriter = dps.get('tb_writer', None)
            if tw is not None:
                tw.add_image('recon', rgb8.cpu(), global_step=self.tb_writer_iter, dataformats='HWC')
                tw.add_image('gt', images[0].cpu(), global_step=self.tb_writer_iter, dataformats='HWC')
                self.tb_writer_iter += 1
        loss_dict['rgb_loss'] = F.l1_loss(rgb, target_s)

        if self.cfg.dynamic_on and self.cfg.retraw and self.training:
            for vi in range(self.cfg.nobjs):
                nablas: torch.Tensor = extras[f'nablas{vi}'][None]
                _, _ind = extras[f'tau{vi}'][None][..., :nablas.shape[-2]].max(dim=-1)
                nablas = torch.gather(nablas, dim=-2,
                                      index=_ind[..., None, None].repeat([*(len(nablas.shape) - 1) * [1], 3]))

                eik_bounding_box = self.cfg.volsdf.obj_bounding_radius
                eikonal_points = torch.empty_like(nablas).uniform_(-eik_bounding_box, eik_bounding_box).cuda()
                _, nablas_eik, _ = getattr(self, f'model_fg{vi}').implicit_surface.forward_with_nablas(
                    eikonal_points)
                nablas = torch.cat([nablas, nablas_eik], dim=-2)

                # [B, N_rays, N_pts]
                nablas_norm = torch.norm(nablas, dim=-1)
                loss_dict[f'loss_eikonal{vi}'] = self.cfg.volsdf.w_eikonal * F.mse_loss(nablas_norm,
                                                                                        nablas_norm.new_ones(
                                                                                            nablas_norm.shape),
                                                                                        reduction='mean')
        if not self.training:
            output['loss_dict'] = loss_dict
        return output, loss_dict

    def render(self, H, W, K, rays=None, ndc=True, near=0., far=1., frame_id=None, dps=None, mtarget_depth_s=None):
        rays_o, rays_d = rays
        viewdirs = F.normalize(rays_d, dim=-1).reshape(-1, 3).float()
        sh = rays_d.shape  # [..., 3]
        if ndc:
            # for forward facing scenes
            rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)
            near = 0.0
            far = 1.0
        # Create ray batch
        rays_o = torch.reshape(rays_o, [-1, 3]).float()
        rays_d = torch.reshape(rays_d, [-1, 3]).float()

        rays = torch.cat([rays_o, rays_d], -1)
        # Render and reshape
        all_ret = self.batchify_rays(rays, viewdirs, near, far, frame_id, dps,
                                     mtarget_depth_s=mtarget_depth_s)
        for k in all_ret:
            k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
            try:
                all_ret[k] = torch.reshape(all_ret[k], k_sh)
            except RuntimeError:
                pass

        k_extract = ['rgb_map', 'depth_map']
        ret_list = [all_ret[k] for k in k_extract]
        ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
        return ret_list + [ret_dict]

    def batchify_rays(self, rays_flat, view_dirs_flat, near, far, frame_id, dps,
                      mtarget_depth_s):
        """Render rays in smaller minibatches to avoid OOM.
        """
        all_ret = {}
        chunk = self.cfg.chunk if self.training else self.cfg.val_chunk
        for i in trange(0, rays_flat.shape[0], chunk,
                        disable=self.training or 'PYCHARM_HOSTED' in os.environ or (
                                get_world_size() > 1 and get_rank() > 0),
                        leave=False):
            if mtarget_depth_s is not None:
                mds = mtarget_depth_s[i:i + chunk]
            else:
                mds = None
            ret = self.render_rays(rays_flat[i:i + chunk], view_dirs_flat[i:i + chunk], near, far,
                                   frame_id[i:i + chunk], dps, mds)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
        return all_ret

    def render_rays(self, ray_batch, viewdirs, near, far, frame_ids, dps, mtarget_depth_s):
        N_samples = self.cfg.N_samples
        # N_samples_fg = self.cfg.N_samples_fg
        netchunk = self.cfg.netchunk
        retraw = self.cfg.retraw and self.training
        lindisp = self.cfg.lindisp
        perturb = self.cfg.perturb if self.training else 0
        N_rays = ray_batch.shape[0]
        # [N_rays, 3] each
        rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]
        rays_d = F.normalize(rays_d, dim=-1)

        t_vals = torch.linspace(0., 1., steps=N_samples).cuda().float()
        if not lindisp:
            z_vals = near * (1. - t_vals) + far * t_vals
        else:
            z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)

        z_vals = z_vals.expand([N_rays, N_samples])

        if perturb > 0. and not self.dbg:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).cuda().float()

            z_vals = lower + (upper - lower) * t_rand

        pts_world = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

        if self.cfg.static_on:
            raw_bg = self.run_network(self.model_bg, pts_world, viewdirs, netchunk=netchunk)
        else:
            raw_bg = torch.zeros([N_rays, N_samples, 4], device='cuda')
        ret = {}
        if self.cfg.dynamic_on:
            # transform pts and viewdirs into object frame
            if self.training:
                assert self.total_cfg.input.shuffle is False
                obj_poses = self.get_all_object_poses()[:, None, :, :].repeat(
                    1, frame_ids.shape[0] // self.cfg.num_frames, 1, 1).reshape(-1, self.cfg.nobjs, 6)
            else:
                assert self.total_cfg.test.batch_size == 1
                obj_poses = self.get_all_object_poses(frame_ids[0].item())[None].repeat(frame_ids.shape[0], 1,
                                                                                        1)
            valid_mask = ~(obj_poses == 0).all(-1)
            obj_poses = se3_exp_map(obj_poses.reshape(-1, 6)).permute(0, 2, 1).reshape(-1, self.cfg.nobjs, 4, 4)

            R = obj_poses[..., :3, :3]
            t = obj_poses[..., :3, 3]

            pts_world = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]
            raw_fgs, fgpts_ino = [], []
            rays_d_o = (rays_d.reshape(-1, 3)[:, None, :].repeat(1, self.cfg.nobjs, 1)[
                        ..., None, :] @ R)[..., 0, :]
            for vi in range(self.cfg.nobjs):
                model_fg = getattr(self, f"model_fg{vi}")
                alpha, beta = model_fg.forward_ab()
                Rthis = R[:, vi, :, :]
                tthis = t[:, vi, :]
                pts_in_o = (pts_world - tthis[:, None, :]) @ Rthis
                fgpts_ino.append(pts_in_o)
                latent_input = None
                objpose_input = None
                d = rays_d
                radiances, sdf, nablas = self.run_network(model_fg, pts_in_o, d, latent_input, objpose_input,
                                                          netchunk=netchunk)
                sigma = sdf_to_sigma(sdf, alpha, beta)
                sigma = F.relu(sigma)
                raw_fg = torch.cat([radiances, sigma], dim=-1)
                raw_fg = raw_fg * valid_mask[:, vi][:, None, None]
                raw_fgs.append(raw_fg)
                delta_i = z_vals[..., 1:] - z_vals[..., :-1]  # NOTE: aleardy real depth
                p_i = torch.exp(-sigma[..., :-1, 0] * delta_i)
                tau_i = (1 - p_i + 1e-10) * (torch.cumprod(torch.cat(
                    [torch.ones([*p_i.shape[:-1], 1], device='cuda'), p_i], dim=-1), dim=-1)[..., :-1])
                ret.update({f'tau{vi}': tau_i, f'nablas{vi}': nablas,
                            # f'sdf{vi}': sdf,
                            f'pts_o{vi}': pts_in_o
                            })
            raw_fgs = torch.stack(raw_fgs, dim=1)
        else:
            raw_fgs = torch.zeros([N_rays, self.cfg.nobjs, N_samples, 4], device='cuda', dtype=torch.float)

        fgptsz = pts_world[..., 2][:, None].repeat(1, self.cfg.nobjs, 1)
        ret_ = self.raw2outputs(raw_bg, raw_fgs, z_vals, rays_d, dps, render_mask=False,
                                mtarget_depth_s=mtarget_depth_s, ptsz=fgptsz)
        if not self.training:
            ret_mask_ = self.raw2outputs(raw_bg, raw_fgs, z_vals, rays_d, dps, render_mask=True,
                                         mtarget_depth_s=mtarget_depth_s, ptsz=fgptsz)
            ret.update({'rgb_map_mask': ret_mask_['rgb_map']})
        ret.update({'rgb_map': ret_['rgb_map'],
                    'depth_map': ret_['depth_map'],
                    'static_depth_map': ret_['static_depth_map'],
                    'dynamic_depth_maps': ret_['dynamic_depth_maps'],
                    'static_rgb_map': ret_['static_rgb_map'],
                    'dynamic_rgb_maps': ret_['dynamic_rgb_maps'],
                    'sigma_fgs': ret_['sigma_fgs'],
                    'sigma_bg': ret_['sigma_bg'],
                    'pts_world': pts_world,
                    'z_vals': z_vals,
                    # 'tau': ret_fg['weights']
                    })
        if self.cfg.mdepth_loss_on:
            mdepth_s = mtarget_depth_s.reshape(-1)[..., None].expand(z_vals.shape)
            mdloss_sup_mask = (z_vals >= mdepth_s - self.cfg.mdepth_fill_thresh)[:, None, :].repeat(1, self.cfg.nobjs,
                                                                                                    1)
            mdloss_sigma_fg = raw_fgs[..., 3]
            ret.update({'mdloss_sup_mask_fg': mdloss_sup_mask,
                        'mdloss_sigma_fg': mdloss_sigma_fg})
        if self.cfg.mdepth_loss_bg_on:
            mdepth_s = mtarget_depth_s.reshape(-1)[..., None].expand(z_vals.shape)
            mdloss_sup_mask = (z_vals >= mdepth_s - self.cfg.mdepth_fill_thresh)
            mdloss_sigma_bg = raw_bg[..., 3]
            ret.update({'mdloss_sup_mask_bg': mdloss_sup_mask,
                        'mdloss_sigma_bg': mdloss_sigma_bg})
        if not retraw:
            ret = {k: v for k, v in ret.items() if 'rgb' in k or 'depth' in k}
        return ret

    def run_network(self, fn, inputs, viewdirs, latent=None, obj_pose=None, *, netchunk=1024 * 64):
        """Prepares inputs and applies network 'fn'.
        """
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
        if viewdirs is not None:
            input_dirs = viewdirs[:, None, :3].repeat(1, inputs.shape[-2], 1)
            viewdirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        else:
            viewdirs_flat = None
        if latent is not None:
            latent_flat = latent.reshape(-1, latent.shape[-1])
        else:
            latent_flat = None
        if obj_pose is not None:
            input_obj_poses = obj_pose[:, None, :3].repeat(1, inputs.shape[-2], 1)
            input_obj_poses_flat = input_obj_poses.reshape(-1, input_obj_poses.shape[-1])
        else:
            input_obj_poses_flat = None
        outputs_flat = batchify(fn, netchunk)(inputs_flat, viewdirs_flat, latent_flat, input_obj_poses_flat)
        if isinstance(outputs_flat, torch.Tensor):
            outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
        else:
            outputs = [torch.reshape(o, list(inputs.shape[:-1]) + [-1]) for o in outputs_flat]
        return outputs

    def raw2outputs(self, raw_bg, raw_fgs, z_vals, rays_d, dps,
                    is_bkgd_only=False, render_mask=False, mtarget_depth_s=None, ptsz=None):
        white_bkgd = self.cfg.white_bkgd
        if self.cfg.dont_render_bg:
            raw_bg[..., 3].fill_(0)
        if render_mask:
            raw_fgs[..., :3].fill_(1)
            raw_bg[..., :3].fill_(0)
        for vi in self.cfg.dont_render_fg:  # 0,1,2
            raw_fgs[:, vi, :, 3].fill_(0)
        if self.cfg.mdepth_fill:
            mds = mtarget_depth_s.reshape(-1)[..., None].repeat(1, self.cfg.N_samples)
            mdepth_fill_thresh = self.cfg.mdepth_fill_thresh
            if not self.training and self.cfg.mdepth_fill_thresh_eval > 0:
                mdepth_fill_thresh = self.cfg.mdepth_fill_thresh_eval
            raw_fgs = raw_fgs * (z_vals < mds - mdepth_fill_thresh
                                 ).float()[:, None, :, None].repeat(1, self.cfg.nobjs, 1, 1)
        if self.cfg.mdepth_fill_bg:
            mds = mtarget_depth_s.reshape(-1)[..., None].repeat(1, self.cfg.N_samples)
            mdepth_fill_thresh_bg = self.cfg.mdepth_fill_thresh_bg
            if not self.training and self.cfg.mdepth_fill_thresh_bg_eval > 0:
                mdepth_fill_thresh_bg = self.cfg.mdepth_fill_thresh_bg_eval
            raw_bg = raw_bg * (z_vals < mds - mdepth_fill_thresh_bg).float()[:, :, None]
        if self.cfg.znegfill and raw_fgs.numel() > 0:
            raw_fgs = raw_fgs * (ptsz >= self.cfg.znegfill_thresh)[..., None].float()
        Nrays = raw_fgs.shape[0]
        sigma_fgs = raw_fgs[..., 3]
        sigma_bg = raw_bg[..., 3]

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([1e10], ).cuda().expand(dists[..., :1].shape)],
                          -1)  # [N_rays, N_samples]
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
        raw = torch.cat([raw_bg[:, None], raw_fgs], dim=1)

        sum_sigmas = raw[..., -1].sum(1) + 1e-10
        alphas = 1 - torch.exp(-dists * sum_sigmas)[..., :-1]

        static_sigmas = raw[:, 0, :, -1]
        static_alphas = static_sigmas[..., :-1] / sum_sigmas[..., :-1] * alphas

        dynamic_sigmas = raw[:, 1:, :, -1]
        dynamic_alphas = dynamic_sigmas[..., :-1] / sum_sigmas[:, None, :-1] * alphas[:, None, :]

        # [N_rays, N_samples]
        Ti = torch.cumprod(
            torch.cat([torch.ones((alphas.shape[0], 1), device='cuda'), 1. - alphas], -1),
            -1)[:, :-1]

        static_weights = static_alphas * Ti
        dynamic_weights = dynamic_alphas * Ti[:, None, :]

        weights = (alphas + 1e-10) * Ti

        static_rgb_map = (static_weights[..., None] * raw[:, 0, :-1, :3]).sum(-2)
        if self.cfg.dont_render_bg:
            static_rgb_map.fill_(0)
        dynamic_rgb_maps = (dynamic_weights[..., None] * raw[:, 1:, :-1, :3]).sum(-2)
        for vi in self.cfg.dont_render_fg:
            dynamic_rgb_maps[:, vi].fill_(0)
        dynamic_rgb_map = dynamic_rgb_maps.sum(1)
        rgb_map = static_rgb_map + dynamic_rgb_map
        ###########

        depth_map = torch.sum(weights * z_vals[..., :-1], dim=-1)
        static_depth_map = torch.sum(static_weights * z_vals[..., :-1], dim=-1)
        dynamic_depth_maps = torch.sum(dynamic_weights * z_vals[..., :-1][:, None, :], dim=-1)

        ret = {}
        if white_bkgd:
            acc_map = torch.sum(weights, -1)
            color = 1 if self.training else self.cfg.white_bkgd_color_inf
            rgb_map = rgb_map + (1.0 - acc_map[..., None]) * color
            ret['acc_map'] = acc_map

        ret.update({'rgb_map': rgb_map,
                    'static_rgb_map': static_rgb_map,
                    'dynamic_rgb_maps': dynamic_rgb_maps,
                    'weights': weights,
                    'depth_map': depth_map,
                    'static_depth_map': static_depth_map,
                    'dynamic_depth_maps': dynamic_depth_maps,
                    'sigma_fgs': sigma_fgs,
                    'sigma_bg': sigma_bg,
                    })
        return ret

    def extract_mesh(self, dps, volume_size=2.0, level=0.0, N=512, filepath='./surface.ply', show_progress=True,
                     chunk=16 * 1024):
        s = volume_size
        voxel_grid_origin = [-s / 2., -s / 2., -s / 2.]
        volume_size = [s, s, s]

        overall_index = np.arange(0, N ** 3, 1).astype(np.int)
        xyz = np.zeros([N ** 3, 3])

        # transform first 3 columns
        # to be the x, y, z index
        xyz[:, 2] = overall_index % N
        xyz[:, 1] = (overall_index / N) % N
        xyz[:, 0] = ((overall_index / N) / N) % N

        # transform first 3 columns
        # to be the x, y, z coordinate
        xyz[:, 0] = (xyz[:, 0] * (s / (N - 1))) + voxel_grid_origin[2]
        xyz[:, 1] = (xyz[:, 1] * (s / (N - 1))) + voxel_grid_origin[1]
        xyz[:, 2] = (xyz[:, 2] * (s / (N - 1))) + voxel_grid_origin[0]

        def batchify(query_fn, inputs: torch.Tensor, vi, chunk=chunk, latent_vector=None):
            out = []
            for i in tqdm.tqdm(range(0, inputs.shape[0], chunk), disable=not show_progress, leave=False):
                inp = torch.from_numpy(inputs[i:i + chunk]).float().cuda()
                if latent_vector is not None:
                    l = latent_vector[None].repeat(inp.shape[0], 1)
                else:
                    l = None
                out_i = query_fn(inp, latent_vector=l)
                out.append(out_i.data.cpu().numpy())
            out = np.concatenate(out, axis=0)
            return out

        def batchify_color(query_fn, inputs: torch.Tensor, view_dirs: torch.Tensor, chunk=chunk):
            out = []
            for i in tqdm.tqdm(range(0, inputs.shape[0], chunk), disable=not show_progress, leave=False):
                inp = inputs[i:i + chunk].float().cuda()
                vd = view_dirs[i:i + chunk].float().cuda()
                out_i, _, _ = query_fn(inp, vd)
                out.append(out_i.data.cpu().numpy())
            out = np.concatenate(out, axis=0)
            return out

        vis3d = Vis3D(out_folder="dbg", sequence="poking_recon")
        all_verts, all_faces, all_colors = [], [], []
        for vi in range(self.cfg.nobjs):
            modelfg = getattr(self, f'model_fg{vi}')
            out = batchify(modelfg.implicit_surface.forward, xyz, vi)
            out = out.reshape([N, N, N])
            try:
                verts, faces, normals, values = skimage.measure.marching_cubes(out, level=level, spacing=volume_size)
                mesh = trimesh.Trimesh(verts, faces)
                mesh.remove_degenerate_faces()
                mesh.remove_infinite_values()
                verts, faces = mesh.vertices, mesh.faces
                verts = torch.from_numpy(verts.copy()).float() / N - s / 2
                faces = torch.from_numpy(faces.copy()).long()
                vis3d.add_mesh(verts, faces, name=f'reconed_mesh{vi}')
                vis3d.add_box_by_6border(-s / 2, -s / 2, -s / 2, s / 2, s / 2, s / 2)
                out_colors = []
                ops = [self.get_object_pose(vi, fid) for fid in self.frame_ids]
                for op in ops:
                    op = se3_exp_map(op[None]).permute(0, 2, 1)[0]
                    view_dirs = transform_points(verts.cuda(), op) - dps['pose'][0][:3, 3][None]
                    view_dirs = F.normalize(view_dirs, dim=-1)
                    modelfg = getattr(self, f'model_fg{vi}')
                    out_color = batchify_color(modelfg.forward, verts, view_dirs)
                    out_colors.append(out_color)
                out_color = np.stack(out_colors).mean(0)
                out_color = (out_color * 255).astype(np.uint8)
                out_color = torch.from_numpy(out_color)
                vis3d.add_mesh(verts, faces, out_color, name=f'reconed_mesh{vi}_colored')
            except Exception as e:
                print(e)
                verts = torch.empty([0, 3]).float().cuda()
                faces = torch.empty([0, 3], dtype=torch.long).cuda()
                out_color = torch.empty([0, 3])
                print('marching_cubes failed.')
            all_verts.append(verts)
            all_faces.append(faces)
            all_colors.append(out_color)
            if 'mesh_verts' in dps and 'mesh_faces' in dps:
                gt_verts = dps['mesh_verts'][0]
                gt_faces = dps['mesh_faces'][0]
                for gi, (gv, gf) in enumerate(zip(gt_verts, gt_faces)):
                    vis3d.add_mesh(gv[gv.sum(-1) != 0], gf[gf.sum(-1) > 0], name=f'gtmesh{gi}')
        meshes = Meshes(all_verts, all_faces, TexturesVertex(all_colors))
        return meshes


def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W),
                          torch.linspace(0, H - 1, H), indexing='ij')  # pytorch's meshgrid has indexing='ij'
    i = i.t().to(K.device)
    j = j.t().to(K.device)
    dirs = torch.stack(
        [(i - K[0, 2].item()) / K[0, 0].item(), (j - K[1, 2].item()) / K[1, 1].item(), torch.ones_like(i)], -1)
    # dirs = torch.stack([(i - K[0, 2]) / K[0, 0], -(j - K[1, 2]) / K[1, 1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def img2mse(x, y):
    return torch.mean((x - y) ** 2)


def mse2psnr(x):
    return -10. * torch.log(x) / torch.log(torch.Tensor([10.]).cuda())


def raw2alpha(raw, dists, act_fn=F.relu):
    return 1. - torch.exp(-act_fn(raw) * dists)


def sample_pdf(bins, weights, N_samples, det=False):
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    # (batch, len(bins))
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    # Take uniform samples
    if det:
        # u = torch.linspace(0., 1., steps=N_samples, device='cuda', dtype=torch.float)
        u = torch.linspace(0., 1., steps=N_samples).cuda().float()
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        # u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device='cuda', dtype=torch.float)
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples]).cuda().float()

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * \
         (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1. / (H / (2. * focal)) * \
         (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(*inputs):
        res = []
        for i in range(0, inputs[0].shape[0], chunk):
            args = [arg[i:i + chunk]
                    if arg is not None else None for arg in inputs]
            ri = fn(*args)
            if not isinstance(ri, tuple):
                ri = [ri]
            res.append(ri)
        collate_raw_ret = []
        dim_batchify = 0
        num_entry = 0
        for entry in zip(*res):
            if isinstance(entry[0], dict):
                tmp_dict = {}
                for list_item in entry:
                    for k, v in list_item.items():
                        if k not in tmp_dict:
                            tmp_dict[k] = []
                        tmp_dict[k].append(v)
                for k in tmp_dict.keys():
                    # [(B), N_rays*N_pts, ...] -> [(B), N_rays, N_pts, ...]
                    # tmp_dict[k] = torch.cat(tmp_dict[k], dim=dim_batchify).unflatten(dim_batchify, [_N_rays, _N_pts])
                    # NOTE: compatible with torch 1.6
                    v = torch.cat(tmp_dict[k])
                    tmp_dict[k] = v.reshape(
                        [*v.shape[:dim_batchify], -1, *v.shape[dim_batchify + 1:]])
                entry = tmp_dict
            else:
                # [(B), N_rays*N_pts, ...] -> [(B), N_rays, N_pts, ...]
                # entry = torch.cat(entry, dim=dim_batchify).unflatten(dim_batchify, [_N_rays, _N_pts])
                # NOTE: compatible with torch 1.6
                v = torch.cat(entry, dim=dim_batchify)
                entry = v.reshape(
                    [*v.shape[:dim_batchify], -1, *v.shape[dim_batchify + 1:]])
            collate_raw_ret.append(entry)
            num_entry += 1
        if num_entry == 1:
            return collate_raw_ret[0]
        else:
            return tuple(collate_raw_ret)
        # res = torch.cat(res, 0)
        # return res

    return ret


def to8b(x):
    return (255 * torch.clamp(x, 0, 1)).byte()


def ent1(x):
    xp = (1 - x).clamp(1e-8, 1 - 1e-8)
    x = x.clamp(1e-8, 1 - 1e-8)
    return -(x * x.log() + xp * xp.log())


def ent2(x, y):
    nx = x / (x + y + 1e-8)
    ny = y / (x + y + 1e-8)
    nx = nx.clamp(1e-8, 1 - 1e-8)
    ny = ny.clamp(1e-8, 1 - 1e-8)
    l = (nx * nx.log() + ny * ny.log()) * (x + y)
    return -l


def fine_sample(implicit_surface_fn, init_dvals, rays_o, rays_d,
                alpha_net, beta_net, far,
                eps=0.1, max_iter: int = 5, max_bisection: int = 10, final_N_importance: int = 64, N_up: int = 128,
                perturb=True, latent_vector=None):
    """
    @ Section 3.4 in the paper.
    Args:
        implicit_surface_fn. sdf query function.
        init_dvals: [..., N_rays, N]
        rays_o:     [..., N_rays, 3]
        rays_d:     [..., N_rays, 3]
    Return:
        final_fine_dvals:   [..., N_rays, final_N_importance]
        beta:               [..., N_rays]. beta heat map
    """
    # NOTE: this algorithm is parallelized for every ray!!!
    with torch.no_grad():
        device = init_dvals.device
        prefix = init_dvals.shape[:-1]
        d_vals = init_dvals

        def query_sdf(d_vals_, rays_o_, rays_d_, latent_vector=None):
            pts = rays_o_[..., None, :] + rays_d_[..., None, :] * d_vals_[..., :, None]
            if latent_vector is not None:
                latent_vector = latent_vector.expand(*pts.shape[:-1], latent_vector.shape[-1])
                # latent_vector = latent_vector[None, None, None, :].repeat(1, pts.shape[1], pts.shape[2], 1)
            return implicit_surface_fn(pts, latent_vector=latent_vector)

        def opacity_invert_cdf_sample(d_vals_, sdf_, alpha_, beta_, N_importance=final_N_importance, det=not perturb):
            # -------------- final: d_vals, sdf, beta_net, alpha_net
            sigma = sdf_to_sigma(sdf_, alpha_, beta_)
            # bounds = error_bound(d_vals_, sdf_, alpha_net, beta_net)
            # delta_i = (d_vals_[..., 1:] - d_vals_[..., :-1]) * rays_d_.norm(dim=-1)[..., None]
            delta_i = d_vals_[..., 1:] - d_vals_[..., :-1]  # NOTE: already real depth
            R_t = torch.cat(
                [
                    torch.zeros([*sdf_.shape[:-1], 1], device=device),
                    torch.cumsum(sigma[..., :-1] * delta_i, dim=-1)
                ], dim=-1)[..., :-1]
            # -------------- a fresh set of \hat{O}
            opacity_approx = 1 - torch.exp(-R_t)
            fine_dvals = rend_util.sample_cdf(d_vals_, opacity_approx, N_importance, det=det)
            return fine_dvals

        # final output storage.
        # being updated during the iterations of the algorithm
        final_fine_dvals = torch.zeros([*prefix, final_N_importance]).to(device)
        final_iter_usage = torch.zeros([*prefix]).to(device)

        # ----------------
        # init beta+
        # ----------------
        # [*prefix, 1]
        if not isinstance(far, torch.Tensor):
            far = far * torch.ones([*prefix, 1], device=device)
        beta = torch.sqrt((far ** 2) / (4 * (init_dvals.shape[-1] - 1) * np.log(1 + eps)))
        alpha = 1. / beta
        # alpha = alpha_net
        # [*prefix, N]

        # ----------------
        # first check of bound using network's current beta: B_{\mathcal{\tau}, \beta}
        # ----------------
        # [*prefix]
        sdf = query_sdf(d_vals, rays_o, rays_d, latent_vector=latent_vector)
        net_bounds_max = error_bound(d_vals, sdf, alpha_net, beta_net).max(dim=-1).values
        mask = net_bounds_max > eps

        # ----------------
        # first bound using beta+ : B_{\mathcal{\tau}, \beta_+}
        # [*prefix, N-1]
        bounds = error_bound(d_vals, sdf, alpha, beta)
        bounds_masked = bounds[mask]
        # NOTE: true for ANY ray that satisfy eps condition in the whole process
        final_converge_flag = torch.zeros([*prefix], device=device, dtype=torch.bool)

        # NOTE: these are the final fine sampling points for those rays that satisfy eps condition at the very beginning.
        if (~mask).sum() > 0:
            final_fine_dvals[~mask] = opacity_invert_cdf_sample(d_vals[~mask], sdf[~mask], alpha_net, beta_net)
            final_iter_usage[~mask] = 0
        final_converge_flag[~mask] = True

        cur_N = init_dvals.shape[-1]
        it_algo = 0
        # ----------------
        # start algorithm
        # ----------------
        while it_algo < max_iter:
            it_algo += 1
            # -----------------
            # the rays that not yet converged
            if mask.sum() > 0:
                # ----------------
                # upsample the samples: \mathcal{\tau} <- upsample
                # ----------------
                # [Masked, N_up]
                # NOTE: det = True should be more robust, forcing sampling points to be proportional with error bounds.
                # upsampled_d_vals_masked = rend_util.sample_pdf(d_vals[mask], bounds_masked, N_up, det=True)
                # NOTE: when using det=True, the head and the tail d_vals will always be appended, hence removed using [..., 1:-1]
                upsampled_d_vals_masked = rend_util.sample_pdf(d_vals[mask], bounds_masked, N_up + 2, det=True)[...,
                                          1:-1]

                d_vals = torch.cat([d_vals, torch.zeros([*prefix, N_up]).to(device)], dim=-1)
                sdf = torch.cat([sdf, torch.zeros([*prefix, N_up]).to(device)], dim=-1)
                # NOTE. concat and sort. work with any kind of dims of mask.
                d_vals_masked = d_vals[mask]
                sdf_masked = sdf[mask]
                d_vals_masked[..., cur_N:cur_N + N_up] = upsampled_d_vals_masked
                d_vals_masked, sort_indices_masked = torch.sort(d_vals_masked, dim=-1)
                sdf_masked[..., cur_N:cur_N + N_up] = query_sdf(upsampled_d_vals_masked, rays_o[mask], rays_d[mask],
                                                                latent_vector=latent_vector)
                sdf_masked = torch.gather(sdf_masked, dim=-1, index=sort_indices_masked)
                d_vals[mask] = d_vals_masked
                sdf[mask] = sdf_masked
                cur_N += N_up

                # ----------------
                # after upsample, check the bound using network's current beta: B_{\mathcal{\tau}, \beta}
                # ----------------
                # NOTE: for the same iteration, the number of points of input rays are the same, (= cur_N), so they can be handled parallelized.
                net_bounds_max[mask] = error_bound(d_vals[mask], sdf[mask], alpha_net, beta_net).max(dim=-1).values
                # NOTE: mask for those rays that still remains > eps after upsampling.
                sub_mask_of_mask = net_bounds_max[mask] > eps
                # mask-the-mask approach. below 3 lines: final_converge_flag[mask][~sub_mask_of_mask] = True (this won't work in python)
                converged_mask = mask.clone()
                converged_mask[mask] = ~sub_mask_of_mask

                # NOTE: these are the final fine sampling points for those rays that >eps originally but <eps after upsampling.
                if converged_mask.sum() > 0:
                    final_converge_flag[converged_mask] = True
                    final_fine_dvals[converged_mask] = opacity_invert_cdf_sample(d_vals[converged_mask],
                                                                                 sdf[converged_mask], alpha_net,
                                                                                 beta_net)
                    final_iter_usage[converged_mask] = it_algo
                # ----------------
                # using bisection method to find the new beta+ s.t. B_{\mathcal{\tau}, \beta+}==eps
                # ----------------
                if (sub_mask_of_mask).sum() > 0:
                    # mask-the-mask approach
                    new_mask = mask.clone()
                    new_mask[mask] = sub_mask_of_mask
                    # [Submasked, 1]
                    beta_right = beta[new_mask]
                    beta_left = beta_net * torch.ones_like(beta_right, device=device)
                    d_vals_tmp = d_vals[new_mask]
                    sdf_tmp = sdf[new_mask]
                    # ----------------
                    # Bisection iterations
                    for _ in range(max_bisection):
                        beta_tmp = 0.5 * (beta_left + beta_right)
                        alpha_tmp = 1. / beta_tmp
                        # alpha_tmp = alpha_net
                        # [Submasked]
                        bounds_tmp_max = error_bound(d_vals_tmp, sdf_tmp, alpha_tmp, beta_tmp).max(dim=-1).values
                        beta_right[bounds_tmp_max <= eps] = beta_tmp[bounds_tmp_max <= eps]
                        beta_left[bounds_tmp_max > eps] = beta_tmp[bounds_tmp_max > eps]
                    beta[new_mask] = beta_right
                    alpha[new_mask] = 1. / beta[new_mask]

                    # ----------------
                    # after upsample, the remained rays that not yet converged.
                    # ----------------
                    bounds_masked = error_bound(d_vals_tmp, sdf_tmp, alpha[new_mask], beta[new_mask])
                    # bounds_masked = error_bound(d_vals_tmp, rays_d_tmp, sdf_tmp, alpha_net, beta[new_mask])
                    bounds_masked = torch.clamp(bounds_masked, 0, 1e5)  # NOTE: prevent INF caused NANs

                    # mask = net_bounds_max > eps   # NOTE: the same as the following
                    mask = new_mask
                else:
                    break
            else:
                break

        # ----------------
        # for rays that still not yet converged after max_iter, use the last beta+
        # ----------------
        if (~final_converge_flag).sum() > 0:
            beta_plus = beta[~final_converge_flag]
            alpha_plus = 1. / beta_plus
            # alpha_plus = alpha_net
            # NOTE: these are the final fine sampling points for those rays that still remains >eps in the end.
            final_fine_dvals[~final_converge_flag] = opacity_invert_cdf_sample(d_vals[~final_converge_flag],
                                                                               sdf[~final_converge_flag], alpha_plus,
                                                                               beta_plus)
            final_iter_usage[~final_converge_flag] = -1
        beta[final_converge_flag] = beta_net
        return final_fine_dvals, beta, final_iter_usage


def sdf_to_sigma(sdf: torch.Tensor, alpha, beta):
    # sdf *= -1 # NOTE: this will cause inplace opt.
    # sdf = -sdf
    # mask = sdf <= 0
    # cond1 = 0.5 * torch.exp(sdf / beta * mask.float())  # NOTE: torch.where will introduce 0*inf = nan
    # cond2 = 1 - 0.5 * torch.exp(-sdf / beta * (1-mask.float()))
    # # psi = torch.where(sdf <= 0, 0.5 * expsbeta, 1 - 0.5 / expsbeta)   # NOTE: exploding gradient
    # psi = torch.where(mask, cond1, cond2)
    """
    @ Section 3.1 in the paper. From sdf:d_{\Omega} to nerf's density:\sigma.
    work with arbitrary shape prefixes.
        sdf:    [...]

    """
    # -sdf when sdf > 0, sdf when sdf < 0
    exp = 0.5 * torch.exp(-torch.abs(sdf) / beta)
    psi = torch.where(sdf >= 0, exp, 1 - exp)

    return alpha * psi


def error_bound(d_vals, sdf, alpha, beta):
    """
    @ Section 3.3 in the paper. The error bound of a specific sampling.
    work with arbitrary shape prefixes.
    [..., N_pts] forms [..., N_pts-1] intervals, hence producing [..., N_pts-1] error bounds.
    Args:
        d_vals: [..., N_pts]
        sdf:    [..., N_pts]
    Return:
        bounds: [..., N_pts-1]
    """
    device = sdf.device
    sigma = sdf_to_sigma(sdf, alpha, beta)
    # [..., N_pts]
    sdf_abs_i = torch.abs(sdf)
    # [..., N_pts-1]
    # delta_i = (d_vals[..., 1:] - d_vals[..., :-1]) * rays_d.norm(dim=-1)[..., None]
    delta_i = d_vals[..., 1:] - d_vals[..., :-1]  # NOTE: already real depth
    # [..., N_pts-1]. R(t_k) of the starting point of the interval.
    R_t = torch.cat(
        [
            torch.zeros([*sdf.shape[:-1], 1], device=device),
            torch.cumsum(sigma[..., :-1] * delta_i, dim=-1)
        ], dim=-1)[..., :-1]
    # [..., N_pts-1]
    d_i_star = torch.clamp_min(0.5 * (sdf_abs_i[..., :-1] + sdf_abs_i[..., 1:] - delta_i), 0.)
    # [..., N_pts-1]
    errors = alpha / (4 * beta) * (delta_i ** 2) * torch.exp(-d_i_star / beta)
    # [..., N_pts-1]. E(t_{k+1}) of the ending point of the interval.
    errors_t = torch.cumsum(errors, dim=-1)
    # [..., N_pts-1]
    bounds = torch.exp(-R_t) * (torch.exp(errors_t) - 1.)
    # TODO: better solution
    #     # NOTE: nan comes from 0 * inf
    #     # NOTE: every situation where nan appears will also appears c * inf = "true" inf, so below solution is acceptable
    bounds[torch.isnan(bounds)] = np.inf
    return bounds


def latentReg(z, reg):
    return sum([1 / reg * torch.norm(latent_i) for latent_i in z])


def extract_neighbor(dps, suffix):
    """

    :param dps:
    :param suffix: +1 or -1
    :return:
    """
    dps = {k[:-2]: v for k, v in dps.items() if k.endswith(suffix)}
    return dps


def exp_density(sigma, lambda_=1.0):
    # |1 - exp(-sigma)|
    return (1 - (-lambda_ * sigma).exp()).abs()


def get_rays_batch(H, W, K, c2w):
    cam_loc = c2w[..., :3, 3]
    p = c2w
    prefix = p.shape[:-2]
    device = c2w.device
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H), indexing='ij')
    i = i.t().to(device).reshape([*[1] * len(prefix), H * W]).expand([*prefix, H * W])
    j = j.t().to(device).reshape([*[1] * len(prefix), H * W]).expand([*prefix, H * W])
    dirs = torch.stack(
        [(i - K[0, 2].item()) / K[0, 0].item(), (j - K[1, 2].item()) / K[1, 1].item(), torch.ones_like(i),
         torch.ones_like(i)], -1)
    # pixel_points_cam = torch.cat([dirs, torch.ones_like(dirs[...,-1])])
    pixel_points_cam = dirs.transpose(-1, -2)
    if len(prefix) > 0:
        world_coords = torch.bmm(p, pixel_points_cam).transpose(-1, -2)[..., :3]
    else:
        world_coords = torch.mm(p, pixel_points_cam).transpose(-1, -2)[..., :3]
    rays_d = world_coords - cam_loc[..., None, :]

    rays_o = cam_loc[..., None, :].expand_as(rays_d)
    rays_o = rays_o.reshape(*prefix, H, W, 3)
    rays_d = rays_d.reshape(*prefix, H, W, 3)
    return rays_o, rays_d
