import os
import os.path as osp

import zarr
from easydict import EasyDict
from tqdm import trange

from puop.utils.raft_api import raft_api


def main():
    root = 'data/kinect/2022_0801_4/moved2'
    intervals = [1, 4, 5, 8, 9, 10, 11, 13]
    nimgs = len(os.listdir(osp.join(root, 'color')))
    for iv in intervals:
        for i in trange(0, nimgs - iv):
            img1path = osp.join(root, f'color/{i:06d}.png')
            img2path = osp.join(root, f'color/{i + iv:06d}.png')
            args = EasyDict({
                'model': 'models/raft/raft-things.pth',
                'small': False,
                'iters': 12,
                'split': 'train',
                'mixed_precision': False,
            })
            flow_path = osp.join(root, 'raft_flow', f'{i}+{iv}.zarr')
            if not osp.exists(flow_path):
                inf_flow = raft_api(img1path, img2path, args)
                zarr.save(flow_path, inf_flow.cpu().numpy())


if __name__ == '__main__':
    main()
