import os
import os.path as osp
import random
import sys
import warnings

import loguru
import numpy as np
import torch
import torch.multiprocessing

import puop.utils.utils_3d
from puop.config import cfg
from puop.engine.defaults import default_argument_parser
from puop.trainer.build import build_trainer
from puop.utils.comm import synchronize, get_rank
from puop.utils.logger import setup_logger
from puop.utils.os_utils import archive_runs, make_source_code_snapshot, deterministic
from puop.utils.vis3d_ext import Vis3D

warnings.filterwarnings('once')
os.environ['PYOPENGL_PLATFORM'] = 'egl'


def train(cfg, local_rank, distributed, resume):
    trainer = build_trainer(cfg)
    if resume:
        trainer.resume()
    if distributed:
        trainer.to_distributed()
    trainer.fit()


def main():
    parser = default_argument_parser()
    parser.add_argument('--init_method', default='env://', type=str)
    parser.add_argument('--no_archive', default=False, action='store_true')
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if args.distributed:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method=args.init_method,
            rank=local_rank, world_size=num_gpus)
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if cfg.output_dir == '':
        assert args.config_file.startswith('configs') and args.config_file.endswith('.yaml')
        cfg.output_dir = args.config_file[:-5].replace('configs', 'models')
    if 'PYCHARM_HOSTED' in os.environ:
        loguru.logger.warning("fix random seed!!!!")
        cfg.dataloader.num_workers = 0
        cfg.backup_src = False
        np.random.seed(0)
        random.seed(0)
        # torch.use_deterministic_algorithms(True)
        torch.random.manual_seed(0)
    if cfg.solver.load != '':
        args.no_archive = True
    cfg.mode = 'train'
    cfg.freeze()
    puop.utils.utils_3d.SE3_LOG_MAP_BACKEND = cfg.SE3_LOG_MAP_BACKEND
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)
    Vis3D.default_out_folder = osp.join(output_dir, 'dbg')

    # archive previous runs
    if not args.no_archive:
        archive_runs(output_dir)
    logger = setup_logger(output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    path = os.path.join(output_dir, "config.yaml")
    logger.info("Running with full config:\n{}".format(cfg.dump(), ".yaml"))
    with open(path, "w") as f:
        f.write(cfg.dump())
    if get_rank() == 0 and cfg.backup_src is True:
        make_source_code_snapshot(output_dir)
    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.remove(2)
        logger.info(config_str)
        format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
        logger.add(sys.stdout, format=format, level="INFO")
    logger.info("Running with config:\n{}".format(cfg))
    if cfg.deterministic is True or 'PYCHARM_HOSTED' in os.environ:
        deterministic()
    model = train(cfg, local_rank, args.distributed, cfg.solver.resume)


if __name__ == "__main__":
    main()
