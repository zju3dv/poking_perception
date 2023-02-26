import pytorch3d.structures
import trimesh
import math

import loguru
import numpy as np
import torch
from dl_ext.primitive import safe_zip

from puop.utils.deepsdf_eval import compute_trimesh_chamfer
from puop.utils.meshrcnn_metrics import compare_meshes
from puop.utils.neucon_evaluation_utils import eval_mesh
from puop.utils.utils_3d import se3_log_map

from puop.registry import EVALUATORS
from puop.utils.utils_3d import pose_distance
from puop.utils.vis3d_ext import Vis3D


@EVALUATORS.register('maskfusion_objpose')
def maskfusion_objpose(cfg):
    def f(x, trainer):
        ds = trainer.valid_dl.dataset
        pred_object_poses = x['object_poses']
        campose = torch.from_numpy(np.stack([d['pose'] for d in ds])).float()
        pred_object_poses = campose[:, None, :, :] @ pred_object_poses
        gt_object_poses = torch.from_numpy(np.stack([d['object_pose'] for d in ds])).float()
        eval_select = trainer.cfg.model.maskfusion.eval_select
        if len(eval_select) > 0:
            gt_object_poses = gt_object_poses[:, eval_select]
        for j in range(pred_object_poses.shape[1]):
            if j == 0: continue
            loguru.logger.info(f'-----obj{j}-----')
            pred_dof6 = se3_log_map(pred_object_poses[:, j].permute(0, 2, 1), eps=1e-3, backend='opencv')
            if j < gt_object_poses.shape[1]:
                gt_dof6 = se3_log_map(gt_object_poses[:, j].permute(0, 2, 1), backend='opencv')
                gops = gt_object_poses[:, j]
            else:
                gt_dof6 = pred_dof6
                gops = pred_object_poses[:, j]
            torch.set_printoptions(sci_mode=False)
            valid = x['valid'][j]
            print('predictions, please copy them into config file.')
            preds=[]
            for g, p, va in safe_zip(gt_dof6, pred_dof6, valid):
                if va:
                    preds.append(p.numpy())
            np.set_printoptions(suppress=True)
            loguru.logger.info(np.array2string(np.stack(preds), precision=4, separator=","))
            rot_err, trans_err = pose_distance(pred_object_poses[:, j][valid], gops[valid], align=True)
            trans_err = trans_err * 100 / cfg.dataset.kinectrobot.data_scale
            rot_err_mean = math.degrees(rot_err.mean().item())
            rot_err_max = math.degrees(rot_err.max().item())
            trans_err_mean = trans_err.mean().item()
            trans_err_max = trans_err.max().item()
            loguru.logger.info(f"rot err (degree), {rot_err_mean:.4f}/{rot_err_max:.4f}")
            loguru.logger.info(f"trans err (cm), {trans_err_mean:.4f}/{trans_err_max:.4f}")

    return f


@EVALUATORS.register('maskfusion_mask')
def maskfusion_mask(cfg):
    def f(x, trainer):
        dl = trainer.valid_dl
        pred_masks = x['mask']  # T,H,W
        gt_masks = torch.cat([p['mask'] for p in dl])
        ids = list(range(1, int(pred_masks.max().item()) + 1))
        eval_select = trainer.cfg.model.maskfusion.eval_select
        for i, e in safe_zip(ids, eval_select):
            loguru.logger.info(f"#obj{i}")
            iou = ((pred_masks == i) & (gt_masks == e + 1)).sum().item() / (
                    (pred_masks == i) | (gt_masks == e + 1)).sum().item()
            loguru.logger.info(f"Mask IoU {iou:.4f}")
            inter = ((pred_masks == i) & (gt_masks == e + 1)).sum(1).sum(1)
            union = ((pred_masks == i) | (gt_masks == e + 1)).sum(1).sum(1)
            keep = inter > 0
            iou = (inter[keep].float() / union[keep].float()).mean().item()
            loguru.logger.info(f"Mask IoU (valid) {iou:.4f}")

    return f


@EVALUATORS.register('maskfusion_mesh')
def maskfusion_mesh(cfg):
    def f(x, trainer):
        vis3d = Vis3D(
            xyz_pattern=('x', 'y', 'z'),
            out_folder="dbg",
            sequence="fusionot_mesh",
            # auto_increase=,
            # enable=,
        )
        recon_verts = x['mesh_verts'][1] / 2.5
        recon_faces = x['mesh_faces'][1]
        ds = trainer.valid_dl.dataset
        d = ds[0]
        eval_select = trainer.cfg.model.maskfusion.eval_select[1]
        gt_verts = d['mesh_verts'][eval_select] / 2.5
        gt_faces = d['mesh_faces'][eval_select]
        vis3d.add_mesh(recon_verts, recon_faces, name='recon')
        vis3d.add_mesh(gt_verts, gt_faces, name='gt')
        recon_mesh = trimesh.Trimesh(recon_verts, recon_faces)
        gt_mesh = trimesh.Trimesh(gt_verts, gt_faces)

        metrics = {}

        chamfer = compute_trimesh_chamfer(gt_mesh, recon_mesh)
        metrics['chamfer'] = chamfer
        ms = eval_mesh(recon_mesh, gt_mesh, 0.01, 0.002)
        for k, v in ms.items():
            metrics['neucon ' + k] = v
        recon_meshes = pytorch3d.structures.Meshes([recon_verts], [recon_faces])
        gt_meshes = pytorch3d.structures.Meshes([gt_verts], [gt_faces])
        ms = compare_meshes(recon_meshes, gt_meshes)
        for k, v in ms.items():
            metrics['meshrcnn ' + k] = v
        metric_str = ""
        for k, v in metrics.items():
            metric_str += f"{k} {v:.6f}\n"
        loguru.logger.info(metric_str)

    return f
