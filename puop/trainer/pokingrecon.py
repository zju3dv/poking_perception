import datetime
import itertools
import os
import os.path as osp
import time
import warnings

import loguru
from dl_ext.average_meter import AverageMeter
from dl_ext.pytorch_ext.dist import *
from dl_ext.pytorch_ext.optim import OneCycleScheduler
from loguru import logger
from termcolor import colored
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from yacs.config import CfgNode

from puop.data import make_data_loader
from puop.data.samplers.ordered_distributed_sampler import OrderedDistributedSampler
from puop.modeling.build import build_model
from puop.trainer.utils import *
from puop.utils.tb_utils import get_summary_writer
from .base import BaseTrainer
from ..solver.lr_scheduler import ExponentialScheduler


class PokingReconTrainer(BaseTrainer):

    def __init__(self, cfg: CfgNode):
        self.cfg = cfg
        self.model = build_model(cfg).to(torch.device(cfg.model.device))
        print("making dataloader...", end='\r')
        self.train_dl = make_data_loader(cfg, is_train=True)
        self.valid_dl = make_data_loader(cfg, is_train=False)
        self.output_dir = cfg.output_dir
        self.num_epochs = cfg.solver.num_epochs
        self.begin_epoch = 0
        self.max_lr = cfg.solver.max_lr
        self.save_every = cfg.solver.save_every
        self.save_mode = cfg.solver.save_mode
        self.save_freq = cfg.solver.save_freq
        self.optimizer, self.optimizer_pose = self.make_optimizer()
        self.scheduler, self.scheduler_pose = self.make_lr_scheduler()
        self.epoch_time_am = AverageMeter()
        self._tb_writer = None
        self.state = TrainerState.BASE
        self.global_steps = 0
        self.best_val_loss = 100000
        self.val_loss = 100000

    def make_optimizer(self):
        lr = self.cfg.solver.max_lr
        pose_lr = self.cfg.solver.pose_lr
        model_params, pose_params = [], []
        for param in self.model.named_parameters():
            if 'objpose' in param[0]:
                pose_params.append(param[1])
            else:
                model_params.append(param[1])
        if len(model_params) > 0:
            optimizer = Adam(model_params, lr)
        else:
            optimizer = None
        if len(pose_params) > 0:
            optimizer_pose = Adam(pose_params, pose_lr)
        else:
            optimizer_pose = None
        return optimizer, optimizer_pose

    def make_lr_scheduler(self):
        if self.optimizer is not None:
            sche = ExponentialScheduler(self.optimizer, self.cfg.solver.gamma, self.cfg.solver.lrate_decay,
                                        cfg=self.cfg)
        else:
            sche = None
        if self.optimizer_pose is not None:
            sche_pose = ExponentialScheduler(self.optimizer_pose, self.cfg.solver.gamma_pose,
                                             self.cfg.solver.lrate_decay_pose, cfg=self.cfg)
        else:
            sche_pose = None
        return sche, sche_pose

    def train(self, epoch):
        loss_meter = AverageMeter()
        metric_ams = {}
        self.model.train()
        bar = tqdm(self.train_dl, leave=False) if is_main_process() else self.train_dl
        begin = time.time()
        for batch in bar:
            self.optimizer.zero_grad()
            if self.optimizer_pose is not None:
                self.optimizer_pose.zero_grad()
            batch = to_cuda(batch)
            batch['global_step'] = self.global_steps
            output, loss_dict = self.model(batch)
            loss = sum(v for k, v in loss_dict.items())
            loss.backward()
            if self.cfg.solver.do_grad_clip:
                if self.cfg.solver.grad_clip_type == 'norm':
                    clip_grad_norm_(self.model.parameters(), self.cfg.solver.grad_clip)
                else:
                    clip_grad_value_(self.model.parameters(), self.cfg.solver.grad_clip)
            self.optimizer.step()
            if self.optimizer_pose is not None:
                self.optimizer_pose.step()
            if self.scheduler is not None and isinstance(self.scheduler, (OneCycleScheduler, ExponentialScheduler)):
                self.scheduler.step()
            if self.scheduler_pose is not None and isinstance(self.scheduler_pose,
                                                              (OneCycleScheduler, ExponentialScheduler)):
                self.scheduler_pose.step()
            # record and plot loss and metrics
            reduced_loss = reduce_loss(loss)
            metrics = {}
            if 'metrics' in output:
                for k, v in output['metrics'].items():
                    reduced_s = reduce_loss(v)
                    metrics[k] = reduced_s
            if is_main_process():
                loss_meter.update(reduced_loss.item())
                lr = self.optimizer.param_groups[0]['lr']
                if self.optimizer_pose is not None:
                    lrpose = self.optimizer_pose.param_groups[0]['lr']
                self.tb_writer.add_scalar('train/loss', reduced_loss.item(), self.global_steps)
                self.tb_writer.add_scalar('train/lr', lr, self.global_steps)
                if self.optimizer_pose is not None:
                    self.tb_writer.add_scalar('train/lrpose', lrpose, self.global_steps)
                for k, v in loss_dict.items():
                    self.tb_writer.add_scalar(f'train/loss/{k}', v.item(), self.global_steps)
                bar_vals = {'epoch': epoch, 'phase': 'train', 'loss': loss_meter.avg}
                for k, v in metrics.items():
                    if k not in metric_ams.keys():
                        metric_ams[k] = AverageMeter()
                    metric_ams[k].update(v.item())
                    self.tb_writer.add_scalar(f'train/{k}', v.item(), self.global_steps)
                    bar_vals[k] = metric_ams[k].avg
                bar.set_postfix(bar_vals)
            self.global_steps += 1
            if self.global_steps % self.save_freq == 0:
                self.try_to_save(epoch, 'iteration')
        torch.cuda.synchronize()
        epoch_time = format_time(time.time() - begin)
        if is_main_process():
            metric_msgs = ['epoch %d, train, loss %.4f, time %s' % (
                epoch, loss_meter.avg, epoch_time)]
            for metric, v in metric_ams.items():
                metric_msgs.append('%s %.4f' % (metric, v.avg))
            s = ', '.join(metric_msgs)
            logger.info(s)
        if self.scheduler is not None and not isinstance(self.scheduler, (OneCycleScheduler, ExponentialScheduler)):
            self.scheduler.step()

    @torch.no_grad()
    def val(self, epoch):
        loss_meter = AverageMeter()
        metric_ams = {}
        self.model.eval()
        bar = tqdm(self.valid_dl, leave=False) if is_main_process() else self.valid_dl
        begin = time.time()
        for batch in bar:
            batch = to_cuda(batch)
            batch['global_step'] = self.global_steps
            batch['tb_writer'] = self.tb_writer
            output, loss_dict = self.model(batch)
            loss = sum(v for k, v in loss_dict.items())
            reduced_loss = reduce_loss(loss)
            metrics = {}
            if is_main_process():
                loss_meter.update(reduced_loss.item())
                bar_vals = {'epoch': epoch, 'phase': 'val', 'loss': loss_meter.avg}
                for k, v in metrics.items():
                    metric_ams[k].update(v.item())
                    bar_vals[k] = metric_ams[k].avg
                bar.set_postfix(bar_vals)
        torch.cuda.synchronize()
        epoch_time = format_time(time.time() - begin)
        if is_main_process():
            metric_msgs = ['epoch %d, val, loss %.4f, time %s' % (
                epoch, loss_meter.avg, epoch_time)]
            for metric, v in metric_ams.items():
                metric_msgs.append('%s %.4f' % (metric, v.avg))
            s = ', '.join(metric_msgs)
            logger.info(s)
            self.tb_writer.add_scalar('val/loss', loss_meter.avg, epoch)
            for metric, s in metric_ams.items():
                self.tb_writer.add_scalar(f'val/{metric}', s.avg, epoch)
            return loss_meter.avg

    def fit(self):
        os.makedirs(self.output_dir, exist_ok=True)
        if self.cfg.model.pokingrecon.fix_bg is True and self.cfg.model.pokingrecon.static_on is True:
            for param in self.model.model_bg.parameters():
                param.requires_grad = False
        num_epochs = self.num_epochs
        begin = time.time()
        for epoch in range(self.begin_epoch, num_epochs):
            epoch_begin = time.time()
            self.train(epoch)
            synchronize()
            if self.save_every is False and epoch % self.cfg.solver.val_freq == 0:
                self.val_loss = self.val(epoch)
                synchronize()
            if is_main_process():
                self.epoch_time_am.update(time.time() - epoch_begin)
                eta = (num_epochs - epoch - 1) * self.epoch_time_am.avg
                finish_time = datetime.datetime.now() + datetime.timedelta(seconds=int(eta))
                loguru.logger.info(
                    f"ETA: {format_time(eta)}, finish time: {finish_time.strftime('%m-%d %H:%M')}")
            if (1 + epoch) % self.cfg.solver.save_freq == 0 and epoch != self.begin_epoch:
                self.try_to_save(epoch, 'epoch')

            synchronize()
        if is_main_process():
            logger.info('Training finished. Total time %s' % (format_time(time.time() - begin)))

    @torch.no_grad()
    def get_preds(self):
        prediction_path = osp.join(self.cfg.output_dir, 'inference', self.cfg.datasets.test, 'predictions')
        outnames = ['bg'] + [f'fg{vi}' for vi in range(self.cfg.model.pokingrecon.nobjs)]
        all_keep = True
        if self.cfg.model.pokingrecon.dont_render_bg:
            outnames.remove('bg')
            all_keep = False
        for vi in self.cfg.model.pokingrecon.dont_render_fg:
            outnames.remove(f'fg{vi}')
            all_keep = False
        on = '' if all_keep else '_' + '_'.join(outnames)
        prediction_path = prediction_path + on + ".pth"
        if not self.cfg.test.force_recompute and osp.exists(prediction_path):
            logger.info(colored(f'predictions found at {prediction_path}, skip recomputing.', 'red'))
            outputs = torch.load(prediction_path)
        else:
            if get_world_size() > 1:
                outputs = self.get_preds_dist()
            else:
                self.model.eval()
                ordered_valid_dl = DataLoader(self.valid_dl.dataset, self.valid_dl.batch_size, shuffle=False,
                                              sampler=None, num_workers=self.valid_dl.num_workers,
                                              collate_fn=self.valid_dl.collate_fn, pin_memory=self.valid_dl.pin_memory,
                                              timeout=self.valid_dl.timeout,
                                              worker_init_fn=self.valid_dl.worker_init_fn)
                bar = tqdm(ordered_valid_dl)
                outputs = []
                for i, batch in enumerate(bar):
                    batch = to_cuda(batch)
                    batch['global_step'] = i
                    output, loss_dict = self.model(batch)
                    output = to_cpu(output)
                    outputs.append(output)
            os.makedirs(osp.dirname(prediction_path), exist_ok=True)
            if self.cfg.test.save_predictions and get_rank() == 0:
                torch.save(outputs, prediction_path)
        return outputs

    @torch.no_grad()
    def get_preds_dist(self):
        self.model.eval()
        valid_sampler = OrderedDistributedSampler(self.valid_dl.dataset, get_world_size(), rank=get_rank())
        ordered_dist_valid_dl = DataLoader(self.valid_dl.dataset, self.valid_dl.batch_size, shuffle=False,
                                           sampler=valid_sampler, num_workers=self.valid_dl.num_workers,
                                           collate_fn=self.valid_dl.collate_fn, pin_memory=self.valid_dl.pin_memory,
                                           timeout=self.valid_dl.timeout,
                                           worker_init_fn=self.valid_dl.worker_init_fn)
        bar = tqdm(ordered_dist_valid_dl) if is_main_process() else ordered_dist_valid_dl
        outputs = []
        for i, batch in enumerate(bar):
            batch = to_cuda(batch)
            batch['global_step'] = i
            output, loss_dict = self.model(batch)
            output = to_cpu(output)
            outputs.append(output)
        torch.cuda.empty_cache()
        all_outputs = all_gather(outputs)
        if not is_main_process():
            return
        all_outputs = list(itertools.chain(*all_outputs))
        all_outputs = all_outputs[:len(self.valid_dl.dataset)]
        return all_outputs

    def save(self, epoch):
        if self.save_mode == 'epoch':
            name = os.path.join(self.output_dir, 'model_epoch_%06d.pth' % epoch)
        else:
            name = os.path.join(self.output_dir, 'model_iteration_%06d.pth' % self.global_steps)
        net_sd = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        d = {'model': net_sd,
             'optimizer': self.optimizer.state_dict(),
             'scheduler': self.scheduler.state_dict(),
             'epoch': epoch,
             'best_val_loss': self.best_val_loss,
             'global_steps': self.global_steps}
        if self.optimizer_pose is not None:
            d['optimizer_pose'] = self.optimizer_pose.state_dict()
        if self.scheduler_pose is not None:
            d['scheduler_pose'] = self.scheduler_pose.state_dict()

        torch.save(d, name)

    def load(self, name: str):
        if name.endswith('.pth'):
            name = name[:-4]
        name = os.path.join(self.output_dir, name + '.pth')
        d = torch.load(name, 'cpu')
        net_sd = d['model']
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(net_sd, strict=False)
        else:
            self.model.load_state_dict(net_sd, strict=False)
        if self.optimizer is not None:
            self.optimizer.load_state_dict(d['optimizer'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(d['scheduler'])
        if self.optimizer_pose is not None:
            self.optimizer_pose.load_state_dict(d['optimizer_pose'])
        if self.scheduler_pose is not None:
            self.scheduler_pose.load_state_dict(d['scheduler_pose'])
        self.begin_epoch = d['epoch']
        self.best_val_loss = d['best_val_loss']
        if 'global_steps' in d:  # compat
            self.global_steps = d['global_steps']

    def resume(self):
        if self.cfg.solver.load == '' and self.cfg.solver.load_model == '' and len(
                self.cfg.solver.load_model_extras) == 0:
            warnings.warn('try to resume without loading anything.')
        if self.cfg.solver.load_model != '':
            logger.info(colored('loading model from %s' % self.cfg.solver.load_model, 'red'))
            self.load_model(self.cfg.solver.load_model)
        if len(self.cfg.solver.load_model_extras) > 0:
            logger.info(colored('loading model extras', 'red'))
            for lme in self.cfg.solver.load_model_extras:
                logger.info(colored('loading model extra %s' % lme, 'red'))
                self.load_model(lme)
        if self.cfg.solver.load != '':
            if self.cfg.solver.load == 'latest':
                ckpts = list(filter(lambda x: x.endswith('.pth'), os.listdir(self.output_dir)))
                if len(ckpts) == 0:
                    logger.warning(f'No ckpt found in {self.output_dir}')
                else:
                    last_ckpt = sorted(ckpts, key=lambda x: int(x[:-4].split('_')[-1]), reverse=True)[0]
                    logger.info(f'Load the lastest checkpoint {last_ckpt}')
                    self.cfg.defrost()
                    self.cfg.solver.load = last_ckpt.rstrip(".pth")
                    self.cfg.freeze()
                    self.load(last_ckpt)
            else:
                load = self.cfg.solver.load
                if isinstance(self.cfg.solver.load, int):
                    load = f'model_{self.save_mode}_{load:06d}'
                logger.info('loading checkpoint from %s' % load, 'red')
                self.load(self.cfg.solver.load)

    def load_model(self, name):
        d = torch.load(name, 'cpu')
        if 'model' in d and 'optimizer' in d:
            d = d['model']
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(d, strict=False)
        else:
            self.model.load_state_dict(d, strict=False)

    @property
    def tb_writer(self):
        if self._tb_writer is None and is_main_process():
            self._tb_writer = get_summary_writer(self.output_dir, flush_secs=20)
        return self._tb_writer

    def try_to_save(self, epoch, flag):
        if not is_main_process():
            return
        if flag == self.save_mode:
            logger.info(colored('Save model found at epoch %d.' % epoch, 'red'))
            self.save(epoch)

    def get_real_model(self):
        if isinstance(self.model, (DistributedDataParallel)):
            return self.model.module
        else:
            return self.model
