from puop.registry import TRAINERS
from puop.trainer.base import BaseTrainer
from puop.trainer.pokingrecon import PokingReconTrainer
from puop.trainer.maskfusion import MaskFusionTrainer


@TRAINERS.register('pokingrecon')
def build(cfg):
    return PokingReconTrainer(cfg)


@TRAINERS.register('maskfusion')
def build(cfg):
    return MaskFusionTrainer(cfg)


def build_trainer(cfg) -> BaseTrainer:
    return TRAINERS[cfg.solver.trainer](cfg)
