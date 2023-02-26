import os
import os.path as osp

from dl_ext.pytorch_ext.dist import *
from loguru import logger
from termcolor import colored
from torch.utils.data import DataLoader
from tqdm import tqdm

from puop.trainer.utils import *
from .base import BaseTrainer


class MaskFusionTrainer(BaseTrainer):

    def __init__(self, cfg):
        super().__init__(cfg)

    def resume(self):
        return

    @torch.no_grad()
    def get_preds(self):
        prediction_path = osp.join(self.cfg.output_dir, 'inference', self.cfg.datasets.test, 'predictions.pth')
        if not self.cfg.test.force_recompute and osp.exists(prediction_path):
            logger.info(colored(f'predictions found at {prediction_path}, skip recomputing.', 'red'))
            outputs = torch.load(prediction_path)
        else:
            if get_world_size() > 1:
                outputs = self.get_preds_dist()
            else:
                if self.cfg.test.training_mode:
                    logger.warning("Running inference with model.train()!!")
                    self.model.train()
                else:
                    self.model.eval()
                ordered_valid_dl = DataLoader(self.valid_dl.dataset, self.valid_dl.batch_size, shuffle=False,
                                              sampler=None, num_workers=self.valid_dl.num_workers,
                                              collate_fn=self.valid_dl.collate_fn, pin_memory=self.valid_dl.pin_memory,
                                              timeout=self.valid_dl.timeout,
                                              worker_init_fn=self.valid_dl.worker_init_fn)
                bar = tqdm(ordered_valid_dl)
                batches = []
                for i, batch in enumerate(bar):
                    batches.append(batch)
                outputs, loss_dict = self.model(batches)
                outputs = to_cpu(outputs)
                try:
                    outputs = torch.cat(outputs)
                except TypeError:
                    pass
            os.makedirs(osp.dirname(prediction_path), exist_ok=True)
            if self.cfg.test.save_predictions:
                torch.save(outputs, prediction_path)
        return outputs
