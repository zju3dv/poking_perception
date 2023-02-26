import math

import loguru
import torch
from puop.utils.utils_3d import se3_log_map

from puop.registry import EVALUATORS
from puop.utils import comm


@EVALUATORS.register('star_pose_eval')
def build(cfg):
    def f(x, trainer):
        if comm.get_rank() == 0 and 'obj_pose' in x[0]:
            pred_obj_poses = torch.stack([a['obj_pose'] for a in x])
            gt_obj_poses = torch.cat([x['object_pose'] for x in trainer.valid_dl])
            pred_dof6 = se3_log_map(pred_obj_poses.permute(0, 2, 1))
            gt_dof6 = se3_log_map(gt_obj_poses.permute(0, 2, 1))
            init_dof6 = torch.tensor(trainer.cfg.model.star.pose_init)
            init_err = (init_dof6 - gt_dof6).abs()
            pred_err = (pred_dof6 - gt_dof6).abs()
            loguru.logger.info(f"init dof6 err,{init_err.mean().item():.4f}")
            loguru.logger.info(f"init trans err,{init_err[..., :3].mean().item():.4f}")
            loguru.logger.info(f"init trans err max,{init_err[..., :3].max().item():.4f}")
            loguru.logger.info(f"init rot err,{math.degrees(init_err[..., 3:].mean().item()):.4f}")
            loguru.logger.info(f"init rot err max,{math.degrees(init_err[..., 3:].max().item()):.4f}")

            loguru.logger.info(f"optimized dof6 err, {pred_err.mean().item():.4f}")
            loguru.logger.info(f"optimized trans err, {pred_err[..., :3].mean().item():.4f}")
            loguru.logger.info(f"optimized trans err max, {pred_err[..., :3].mean(-1).max().item():.4f}")
            loguru.logger.info(f"optimized rot err, {math.degrees(pred_err[..., 3:].mean().item()):.4f}")
            loguru.logger.info(f"optimized rot err max, {math.degrees(pred_err[..., 3:].max().item()):.4f}")

    return f


@EVALUATORS.register('print_loss')
def build(cfg):
    def f(x, ds):
        for i, d in enumerate(x):
            print('-' * 10, i, '-' * 10)
            for k, v in d.get('metrics', {}).items():
                print(k, v.item())
            for k, v in d.get('loss_dict', {}).items():
                print(k, v.item())

    return f

# def build_evaluators(cfg):
#     evaluators = []
#     for e in cfg.test.evaluators:
#         evaluators.append(EVALUATORS[e](cfg))
#     return evaluators
