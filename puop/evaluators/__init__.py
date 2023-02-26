from ..registry import EVALUATORS
from .pokingrecon import pokingrecon_mask_eval, pokingrecon_pose_eval
from .maskfusion import *
from .build import *


def build_evaluators(cfg):
    evaluators = []
    for e in cfg.test.evaluators:
        evaluators.append(EVALUATORS[e](cfg))
    return evaluators
