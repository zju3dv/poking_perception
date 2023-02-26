from ..registry import VISUALIZERS
from .pokingrecon import PokingReconVisualizer


def build_visualizer(cfg):
    return VISUALIZERS[cfg.test.visualizer](cfg)
