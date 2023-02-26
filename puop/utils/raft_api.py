import os
import os.path as osp
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import zarr
from PIL import Image
from tqdm import tqdm

from puop.modeling.models.raft.raft import RAFT
from puop.utils.flow_warping import FlowWarping


class RAFTAPI:
    model = None

    @classmethod
    def getinstance(cls, args=None):
        """

        :param args:
    parser.add_argument('--model', help="restore checkpoint", default='models/raft/raft-things.pth')
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--split', default='train', type=str)
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        :return:
        """
        if RAFTAPI.model is None:
            RAFTAPI.model = setup_model(args)
        return RAFTAPI.model


def pad8(img):
    """pad image such that dimensions are divisible by 8"""
    ht, wd = img.shape[2:]  # 375 1242
    pad_ht = (((ht // 8) + 1) * 8 - ht) % 8  # 1
    pad_wd = (((wd // 8) + 1) * 8 - wd) % 8  # 6
    pad_ht1 = [pad_ht // 2, pad_ht - pad_ht // 2]  # [0, 1]
    pad_wd1 = [pad_wd // 2, pad_wd - pad_wd // 2]  # [3, 3]

    img = F.pad(img, pad_wd1 + pad_ht1, mode='replicate')
    return img, {'pad_ht1': pad_ht1, 'pad_wd1': pad_wd1}


def load_image_and_pad_dict(imfile):
    if isinstance(imfile, str):
        imfile = Image.open(imfile)
    img = np.array(imfile).astype(np.uint8)[..., :3]
    img = torch.from_numpy(img).permute(2, 0, 1).float()[None].cuda()
    return pad8(img)


def setup_model(args):
    model = RAFT(args)
    ckpt = torch.load(args.model)
    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt)

    model.cuda()
    model.eval()
    return model


@torch.no_grad()
def demo(args):
    model = setup_model(args)

    img_dir = osp.expanduser(f'~/Datasets/CATER/all_actions_cameramotion/images/{args.split}')
    for video_name in sorted(os.listdir(img_dir)):
        output_dir = osp.expanduser(f'~/Datasets/CATER/all_actions_cameramotion/raft_flow/{args.split}/{video_name}')
        vis_dir = osp.expanduser(f'~/Datasets/CATER/all_actions_cameramotion/raft_flow_vis/{args.split}/{video_name}')
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)
        nimgs = len(osp.join(img_dir, video_name))
        for imgid in range(nimgs - 1):
            image1_path = os.path.join(img_dir, video_name, f'{imgid:06}.png')
            image2_path = os.path.join(img_dir, video_name, f'{imgid + 1:06}.png')
            tqdm.write(f'Inference from {image1_path} to {image2_path}')

            assert os.path.isfile(image1_path) and os.path.isfile(image2_path)

            image1, image1_pad_dict = load_image_and_pad_dict(image1_path)
            image2, image2_pad_dict = load_image_and_pad_dict(image2_path)
            assert np.allclose(image1_pad_dict['pad_ht1'], image2_pad_dict['pad_ht1'])
            assert np.allclose(image1_pad_dict['pad_wd1'], image2_pad_dict['pad_wd1'])

            flow_predictions = model(image1, image2, iters=32, test_mode=True)

            flow = flow_predictions[-1]
            if image1_pad_dict['pad_ht1'][0] != 0 or image1_pad_dict['pad_ht1'][1] != 0:
                flow_cropped = flow[:, :, image1_pad_dict['pad_ht1'][0]: -image1_pad_dict['pad_ht1'][1],
                               image1_pad_dict['pad_wd1'][0]: -image1_pad_dict['pad_wd1'][1]]
            else:
                flow_cropped = flow
            zarr.save(os.path.join(output_dir, f'{imgid:06}_{imgid + 1:06}.zarr'),
                      flow_cropped.squeeze(0).permute(1, 2, 0).cpu().numpy())

            if imgid < 10:
                ori_image2 = np.array(Image.open(image2_path)).astype(np.uint8)[..., :3]
                ori_image2 = torch.from_numpy(ori_image2).permute(2, 0, 1).float()[None].to(DEVICE)
                warper = FlowWarping()
                warped_img, _ = warper(ori_image2, flow_cropped)
                Image.fromarray(warped_img.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)). \
                    save(os.path.join(vis_dir, f'{imgid + 1:06}_{imgid:06}_raft.png'))


@torch.no_grad()
def raft_api(img1_path, img2_path, args=None):
    model = RAFTAPI.getinstance(args)
    if osp.exists(img1_path) and osp.exists(img2_path):
        image1, image1_pad_dict = load_image_and_pad_dict(img1_path)
        image2, image2_pad_dict = load_image_and_pad_dict(img2_path)
        assert np.allclose(image1_pad_dict['pad_ht1'], image2_pad_dict['pad_ht1'])
        assert np.allclose(image1_pad_dict['pad_wd1'], image2_pad_dict['pad_wd1'])
        
        flow_predictions = model(image1, image2, iters=32, test_mode=True)

        flow = flow_predictions[-1]
        if image1_pad_dict['pad_ht1'][0] != 0 or image1_pad_dict['pad_ht1'][1] != 0:
            flow_cropped = flow[:, :, image1_pad_dict['pad_ht1'][0]: -image1_pad_dict['pad_ht1'][1],
                           image1_pad_dict['pad_wd1'][0]: -image1_pad_dict['pad_wd1'][1]]
        else:
            flow_cropped = flow
        flow_cropped = flow_cropped.squeeze(0).permute(1, 2, 0)
    else:
        warnings.warn("img1path or img2path not found, return None")
        flow_cropped = None
    return flow_cropped
