import math

import loguru
import torch
import trimesh
from pytorch3d.structures import Meshes

from puop.registry import EVALUATORS
from puop.utils import comm
from puop.utils.deepsdf_eval import compute_trimesh_chamfer
from puop.utils.meshrcnn_metrics import compare_meshes
from puop.utils.neucon_evaluation_utils import eval_mesh
from puop.utils.utils_3d import pose_distance
from puop.utils.utils_3d import se3_exp_map
from puop.utils.vis3d_ext import Vis3D


@EVALUATORS.register('pokingrecon_pose_eval')
def pokingrecon_pose_eval(cfg):
    def f(x, trainer):
        metrics = {}
        if comm.get_rank() == 0 and 'obj_pose' in x[0]:
            loguru.logger.info(f"evaluating object pose on {trainer.cfg.datasets.test}")
            all_pred_obj_poses = torch.stack([a['obj_pose'] for a in x])
            all_gt_obj_poses = torch.cat([x['object_pose'] for x in trainer.valid_dl])
            nobj = all_pred_obj_poses.shape[1]
            for i in range(nobj):
                loguru.logger.info(f"eval obj#{i}")
                pred_obj_poses = all_pred_obj_poses[:, i]
                if cfg.model.pokingrecon.ignore_gripper:
                    gt_obj_poses = all_gt_obj_poses[:, i + cfg.model.pokingrecon.vi_shift + 1]
                else:
                    gt_obj_poses = all_gt_obj_poses[:, cfg.model.pokingrecon.vi_shift + i]
                # if not torch.all(torch.tensor(trainer.cfg.model.pokingrecon.pose_init) == 0):
                init_poses = torch.tensor(trainer.cfg.model.pokingrecon.pose_init)[i]
                valid = (init_poses != 0).any(-1)
                init_poses = se3_exp_map(init_poses).permute(0, 2, 1)
                rot_err, trans_err = pose_distance(init_poses[valid], gt_obj_poses[valid],
                                                   align=cfg.model.pokingrecon.align_pose_eval)
                rot_err_mean = math.degrees(rot_err.mean().item())
                rot_err_max = math.degrees(rot_err.max().item())
                trans_err_mean = trans_err.mean().item()
                trans_err_max = trans_err.max().item()
                loguru.logger.info(f"init rot err, {rot_err_mean:.4f}/{rot_err_max:.4f}")
                loguru.logger.info(f"init trans err, {trans_err_mean:.4f}/{trans_err_max:.4f}")

                rot_err, trans_err = pose_distance(pred_obj_poses[valid], gt_obj_poses[valid],
                                                   align=cfg.model.pokingrecon.align_pose_eval)
                trans_err = trans_err*100 / cfg.dataset.kinectrobot.data_scale
                rot_err_mean = math.degrees(rot_err.mean().item())
                rot_err_max = math.degrees(rot_err.max().item())
                trans_err_mean = trans_err.mean().item()
                trans_err_max = trans_err.max().item()
                loguru.logger.info(f"optimized rot err (degree), {rot_err_mean:.4f}/{rot_err_max:.4f}")
                loguru.logger.info(f"optimized trans err (cm), {trans_err_mean:.4f}/{trans_err_max:.4f}")
        return metrics

    return f


@EVALUATORS.register('pokingrecon_mask_eval')
def pokingrecon_mask_eval(cfg):
    def f(x, trainer):
        metrics = {}
        if comm.get_rank() == 0 and trainer.cfg.model.pokingrecon.dont_render_bg:
            loguru.logger.info(f"evaluating mask on {trainer.cfg.datasets.test}")
            pred_masks = torch.stack([a['pred_mask'] for a in x])
            gt_masks = torch.cat([x['mask'] for x in trainer.valid_dl])
            # ids = list(range(1, gt_masks.max().item() + 1))
            # drf = trainer.cfg.model.pokingrecon.dont_render_fg
            # if len(drf) == len(ids) - 1:  # only render 1 obj
            # i = (set((np.array(ids) - 1).tolist()) - set(drf)).pop()
            i = trainer.cfg.model.pokingrecon.eval_mask_gt_id
            loguru.logger.info(f"eval{i}")
            iou = ((pred_masks == 1) & (gt_masks == i)).sum().item() / (
                    (pred_masks == 1) | (gt_masks == i)).sum().item()
            loguru.logger.info(f"Mask IoU {iou:.4f}")
            metrics["mask_iou"] = iou
        return metrics

    return f


@EVALUATORS.register('pokingrecon_mesh_eval')
def pokingrecon_mesh_eval(cfg):
    def f(x, trainer):
        ret = {}
        if comm.get_rank() == 0 and trainer.cfg.model.pokingrecon.recon:
            vis3d = Vis3D(
                xyz_pattern=('x', 'y', 'z'),
                out_folder="dbg",
                sequence="pokingrecon_mesh_eval",
                # auto_increase=,
                enable=False,
            )
            recon_verts = x[0]['meshes'].verts_list()[0] / 2.5
            recon_faces = x[0]['meshes'].faces_list()[0]
            ds = trainer.valid_dl.dataset
            d = ds[0]
            eval_select = trainer.cfg.model.pokingrecon.eval_mask_gt_id - 1
            gt_verts = d['mesh_verts'][eval_select] / 2.5
            gt_faces = d['mesh_faces'][eval_select]
            vis3d.add_mesh(recon_verts, recon_faces, name='recon')
            vis3d.add_mesh(gt_verts, gt_faces, name='gt')
            recon_mesh = trimesh.Trimesh(recon_verts, recon_faces)
            gt_mesh = trimesh.Trimesh(gt_verts, gt_faces)
            vis3d.add_point_cloud(recon_mesh.sample(100000), name='recon')
            vis3d.add_point_cloud(gt_mesh.sample(100000), name='gt')

            metrics = {}
            chamfer = compute_trimesh_chamfer(gt_mesh, recon_mesh)
            metrics['chamfer_distance'] = chamfer
            ms = eval_mesh(recon_mesh, gt_mesh, 0.01, 0.002)
            for k, v in ms.items():
                metrics[k] = v
            recon_meshes = Meshes([recon_verts], [recon_faces])
            gt_meshes = Meshes([gt_verts], [gt_faces])
            ms = compare_meshes(recon_meshes, gt_meshes)
            for k, v in ms.items():
                metrics[k] = v
            metric_str = ""
            for k, v in metrics.items():
                metric_str += f"{k} {v:.4f}\n"
            loguru.logger.info(metric_str)
            ret.update(metrics)
        return ret

    return f
