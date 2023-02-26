import os.path as osp
import sys

from loguru import logger


def setup_logger(save_dir, distributed_rank, filename="log.txt"):
    logger.remove()
    format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    logger.add(osp.join(save_dir, filename), format=format, level="INFO")
    logger.level("WARNING", color="<red>")
    if distributed_rank > 0:
        return logger

    logger.add(sys.stdout, format=format, level="INFO")
    return logger
