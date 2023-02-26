import numpy as np
import math
import os
import os.path as osp

import imageio
import matplotlib.pyplot as plt
import torch
import trimesh
from dl_ext.primitive import safe_zip
from puop.utils.utils_3d import se3_log_map, se3_exp_map
from tqdm import tqdm

from puop.registry import VISUALIZERS
from puop.utils.plt_utils import image_grid, hover_masks_on_imgs


@VISUALIZERS.register('pokingrecon')
class PokingReconVisualizer:
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, *args, **kwargs):
        if self.cfg.model.pokingrecon.recon is True:
            self.save_mesh(*args)
        else:
            self.show_rendered_imgs(*args, **kwargs)
            if self.cfg.test.show_pose_optim:
                self.show_pose_optim(*args, **kwargs)

    def save_mesh(self, *args):
        verts_list = args[0][0]['meshes'].verts_list()
        faces_list = args[0][0]['meshes'].faces_list()
        color_list = args[0][0]['meshes'].textures.verts_features_list()
        vis_dir = osp.join(self.cfg.output_dir, 'visualization', self.cfg.datasets.test)
        if self.cfg.solver.load != "":
            vis_dir = osp.join(vis_dir, self.cfg.solver.load)
        os.makedirs(vis_dir, exist_ok=True)
        for i, (v, f, co) in enumerate(zip(verts_list, faces_list, color_list)):
            v, f, co = v.numpy(), f.numpy(), co.numpy()
            _ = trimesh.Trimesh(v, f, vertex_colors=co).export(osp.join(vis_dir, f'recon{i}.ply'))

    def show_rendered_imgs(self, *args, **kwargs):
        pa_vis_dir = osp.join(self.cfg.output_dir, 'visualization', self.cfg.datasets.test)
        if self.cfg.solver.load != "":
            pa_vis_dir = osp.join(pa_vis_dir, self.cfg.solver.load)
        os.makedirs(pa_vis_dir, exist_ok=True)
        os.makedirs(osp.join(pa_vis_dir, 'all'), exist_ok=True)
        os.makedirs(osp.join(pa_vis_dir, 'grid'), exist_ok=True)
        outputs = args[0]
        trainer = args[1]
        valid_dl = trainer.valid_dl
        preds = []
        gts = []
        error_maps = []
        vis_dir = osp.join(pa_vis_dir, 'all')
        # build outname
        outnames = ['bg'] + [f'fg{vi}' for vi in range(self.cfg.model.pokingrecon.nobjs)]
        all_keep = True
        if self.cfg.model.pokingrecon.dont_render_bg:
            outnames.remove('bg')
            all_keep = False
        for vi in self.cfg.model.pokingrecon.dont_render_fg:
            outnames.remove(f'fg{vi}')
            all_keep = False
        on = '' if all_keep else '_' + '_'.join(outnames)
        for i, o in enumerate(tqdm(outputs)):
            outname = '%05d' % i + on + '.png'
            outpath = osp.join(vis_dir, outname)
            imageio.imwrite(outpath, o['rgb8'].cpu().numpy())
            preds.append(o['rgb8'].cpu().numpy())
            gts.append(o['rgb_gt'].cpu().numpy())
            outpath = osp.join(vis_dir, f'{i:05d}_gt.png')
            imageio.imwrite(outpath, (o['rgb_gt'].cpu().numpy() * 255).astype(np.uint8))
            tmp = o['rgb_gt'].cpu().numpy().copy()
            pred_mask = o['pred_mask'].cpu().numpy()
            tmp[pred_mask] *= 0.5
            tmp[pred_mask] += 0.5
            tmp = (255 * tmp).astype(np.uint8)
            imageio.imwrite(osp.join(vis_dir, f'{i:05d}_predmask_on_gt.png'), tmp)
            if all_keep:
                outpath = osp.join(vis_dir, f'{i:05d}_errmap.png')
                errmap = (o['rgb8'] / 255.0 - o['rgb_gt']).abs().sum(-1)
                error_maps.append(errmap)
                plt.imshow(errmap, 'jet')
                plt.savefig(outpath)
        image_grid(preds, show=False)
        vis_dir = osp.join(pa_vis_dir, 'grid')
        outname = 'grid' + on
        outpath = osp.join(vis_dir, outname + '.png')
        plt.savefig(outpath)
        imageio.mimsave(osp.join(vis_dir, outname + ".mp4"), preds)
        if all_keep:
            image_grid(error_maps, show=False, cmap='jet')
            plt.savefig(osp.join(vis_dir, outname + '_errmap.png'))
            image_grid(gts, show=False)
            plt.savefig(osp.join(vis_dir, outname + '_gt.png'))
        if self.cfg.model.pokingrecon.dont_render_bg:
            pred_masks = torch.stack([o['pred_mask'] for o in outputs])
            image_grid(pred_masks, rgb=False, show=False)
            outname = 'grid_pred_masks' + on
            plt.savefig(osp.join(vis_dir, outname + '.png'))
            for pmi, pm in enumerate(pred_masks):
                imageio.imwrite(osp.join(pa_vis_dir, 'all', f'{pmi:05d}_pred_mask' + on + '.png'),
                                (pm * 255).numpy().astype(np.uint8))
            gt_masks = torch.cat([d['mask'] for d in valid_dl])
            image_grid(gt_masks, rgb=False, show=False)
            plt.savefig(osp.join(vis_dir, 'grid_gt_masks.png'))
            gm = gt_masks > 0
            outname = 'grid_masks_error' + on
            for vi in self.cfg.model.pokingrecon.dont_render_fg:
                gm[gm == vi] = 0
            image_grid(gt_masks.float() - pred_masks.float(), rgb=False, show=False, cmap='jet')
            plt.savefig(osp.join(vis_dir, outname + '.png'))
        print()

    def hist_grid(self, xs, **kwargs):
        plt.figure(figsize=(20., 20.))
        nr = int(len(xs) ** 0.5)
        nc = math.ceil(len(xs) / nr)
        for i in range(len(xs)):
            plt.subplot(nr, nc, i + 1)
            plt.hist(xs[i].reshape(-1).tolist(), **kwargs)

    def show_pose_optim(self, *args, **kwargs):
        trainer = args[1]
        pred_obj_poses = torch.stack([a['obj_pose'] for a in args[0]])
        gt_obj_poses = torch.cat([x['object_pose'] for x in trainer.valid_dl])
        for vi in range(self.cfg.model.pokingrecon.nobjs):
            pred_dof6 = se3_log_map(pred_obj_poses[:, vi].permute(0, 2, 1), backend='opencv')
            if self.cfg.model.pokingrecon.ignore_gripper:
                gt_ops = gt_obj_poses[:, vi + self.cfg.model.pokingrecon.vi_shift + 1]
            else:
                gt_ops = gt_obj_poses[:, self.cfg.model.pokingrecon.vi_shift + vi]
            gt_dof6 = se3_log_map(gt_ops.permute(0, 2, 1), backend='opencv')
            init_dof6 = torch.tensor(trainer.cfg.model.pokingrecon.pose_init)[vi]
            valids = (init_dof6 != 0).any(-1)
            ylim_min = min(pred_dof6.min(), gt_dof6.min(), init_dof6.min()).item() - 0.3
            ylim_max = max(pred_dof6.max(), gt_dof6.max(), init_dof6.max()).item() + 0.3
            from mpl_toolkits.axes_grid1 import ImageGrid
            figsize = (10., 10.) if len(pred_dof6) < 50 else (20, 20)
            fig = plt.figure(figsize=figsize)
            nr = int(len(pred_dof6) ** 0.5)
            nc = math.ceil(len(pred_dof6) / nr)
            grid = ImageGrid(fig, 111,  # similar to subplot(111)
                             nrows_ncols=(nr, nc),  # creates 2x2 grid of axes
                             axes_pad=0.1,  # pad between axes in inch.
                             )
            valid_init_dof6 = init_dof6[valids]
            valid_pred_dof6 = pred_dof6[valids]
            valid_gt_dof6 = gt_dof6[valids]
            if self.cfg.model.pokingrecon.align_pose_eval:
                valid_init_poses = se3_exp_map(valid_init_dof6).permute(0, 2, 1)
                valid_pred_poses = se3_exp_map(valid_pred_dof6).permute(0, 2, 1)
                valid_gt_poses = se3_exp_map(valid_gt_dof6).permute(0, 2, 1)

                valid_init_poses = valid_init_poses @ valid_init_poses[0].inverse()[None]
                valid_pred_poses = valid_pred_poses @ valid_pred_poses[0].inverse()[None]
                valid_gt_poses = valid_gt_poses @ valid_gt_poses[0].inverse()[None]

                valid_init_dof6 = se3_log_map(valid_init_poses.permute(0, 2, 1), backend='opencv')
                valid_pred_dof6 = se3_log_map(valid_pred_poses.permute(0, 2, 1), backend='opencv')
                valid_gt_dof6 = se3_log_map(valid_gt_poses.permute(0, 2, 1), backend='opencv')
            i = 0
            for ai, (ax, valid) in enumerate(zip(grid, valids)):
                # Iterating over the grid returns the Axes.
                ax.set_title(f'idx{ai}')
                ax.set_ylim([ylim_min, ylim_max])
                if valid:
                    gt, pred, init = valid_gt_dof6[i], valid_pred_dof6[i], valid_init_dof6[i]
                    ax.plot(torch.arange(6), gt, color='red', marker='+', label='gt')
                    ax.plot(torch.arange(6), pred, color='green', marker='o', label='pred')
                    ax.plot(torch.arange(6), init, color='gray', marker='x', label='init')
                    i += 1
            grid[0].legend(loc="upper left")
            vis_dir = osp.join(self.cfg.output_dir, 'visualization', self.cfg.datasets.test)
            if self.cfg.solver.load != "":
                vis_dir = osp.join(vis_dir, self.cfg.solver.load)
            os.makedirs(vis_dir, exist_ok=True)
            plt.savefig(osp.join(vis_dir, 'grid', f'object_pose_errmap{vi}.png'))
