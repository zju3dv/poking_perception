import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def flow_mask(flow1):
    """
    :param flow0: BS x 2 x H x W
    :param flow1: BS x 2 x H x W
    :return: BS x 1 x H x W
    """
    bs, _, h, w = flow1.shape
    x_ = torch.arange(w).view(1, -1).expand(h, -1)
    y_ = torch.arange(h).view(-1, 1).expand(-1, w)
    flow_tensor = flow1.data
    grid = torch.stack([x_, y_], dim=0).float().cuda()
    grid = grid.unsqueeze(0).expand(bs, -1, -1, -1)  # bs x 2 x H x W
    grid = grid + flow_tensor

    upper_i = grid[:, 1].long()
    left_j = grid[:, 0].long()
    lower_i = upper_i + 1
    right_j = left_j + 1
    grid_batch = torch.arange(bs).long().view(-1, 1, 1).expand(-1, h, w)  # bs x H x W

    invalid_upper = (upper_i < 0) + (upper_i >= h) > 0
    invalid_lower = (lower_i < 0) + (lower_i >= h) > 0
    invalid_left = (left_j < 0) + (left_j >= w) > 0
    invalid_right = (right_j < 0) + (right_j >= w) > 0

    upper_i[invalid_upper] = 0
    lower_i[invalid_lower] = 0
    left_j[invalid_left] = 0
    right_j[invalid_right] = 0

    mask = torch.zeros((bs, h, w)).int().cuda()
    mask[grid_batch, upper_i, left_j] = 1
    mask[grid_batch, upper_i, right_j] = 1
    mask[grid_batch, lower_i, left_j] = 1
    mask[grid_batch, lower_i, right_j] = 1

    mask = mask.unsqueeze(1)
    mask = Variable(mask)

    return mask > 0


class FlowWarping(nn.Module):
    def __init__(self, occlusion_mask=0):
        super(FlowWarping, self).__init__()
        self.occlusion_mask = occlusion_mask

    def forward(self, x, flow, flow_backward=None):
        """
        :param x: [BS, C, H, W]
        :param flow: [BS, 2, H, W]
        :param flow_backward: [BS, 2, H, W]
        :return: [BS, C, H, W], mask: [BS, 1, H, W]
        """
        assert x.size()[-2:] == flow.size()[
                                -2:], f"inconsistent shape between flow {flow.size()[-2:]} and source image {x.size()[-2:]}"
        n, c, h, w = x.size()
        x_ = torch.arange(w).view(1, -1).expand(h, -1)  # h,w
        y_ = torch.arange(h).view(-1, 1).expand(-1, w)  # h,w
        grid = torch.stack([x_, y_], dim=0).float().to(flow.device)  # 2,h,w
        grid = grid.unsqueeze(0).expand(n, -1, -1, -1)
        # grid = Variable(grid) + flow
        grid = grid + flow.contiguous()
        grid[:, 0, :, :] = 2 * grid[:, 0, :, :] / (w - 1) - 1
        grid[:, 1, :, :] = 2 * grid[:, 1, :, :] / (h - 1) - 1
        grid = grid.permute(0, 2, 3, 1)
        mask = grid.abs() <= 1
        mask = mask.sum(dim=3)

        if self.occlusion_mask and (flow_backward is not None):
            mask_o = flow_mask(flow_backward).detach()
            mask = mask + mask_o[:, 0, :, :].type_as(mask)
            mask = mask == 3
        else:
            # TOFIX
            mask = mask == 2

        mask = mask.unsqueeze(1)
        warped_img = F.grid_sample(x, grid, padding_mode='zeros')
        return warped_img, mask


def flow_warping_image(image, flow, flow_channel_first=True):
    """
    warp image
    @param image: H,W,3
    @param flow: 2,H,W
    @return:
    """
    if isinstance(image, PIL.Image.Image):
        image = np.array(image)
    if isinstance(image, np.ndarray):
        assert image.shape[2] == 3
        if image.dtype == np.uint8:
            image = image / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
    if image.ndim == 3:
        image = image[None]
    if image.shape[-1] == 3:
        image = image.permute(0, 3, 1, 2)
    if isinstance(flow, np.ndarray):
        flow = torch.from_numpy(flow)
    if flow.ndim == 3:
        flow = flow[None]
    if not flow_channel_first:
        flow = flow.permute(0, 3, 1, 2)
    warpedimg, _ = FlowWarping()(image.float(), flow.float())
    warpedimg = warpedimg[0].permute(1, 2, 0)
    return warpedimg


def flow_warping(x, flow, flow_channel_first=True):
    """
    warp image
    @param x: H,W
    @param flow: 2,H,W
    @return:
    """
    assert x.ndim == 2
    x = torch.as_tensor(x[:, :, None]).permute(2, 0, 1).unsqueeze(0)
    assert flow.ndim == 3
    flow = torch.as_tensor(flow)[None]
    if not flow_channel_first:
        flow = flow.permute(0, 3, 1, 2)
    warpedimg, _ = FlowWarping()(x.float(), flow.float())
    warpedimg = warpedimg[0].permute(1, 2, 0)[..., 0]
    return warpedimg.numpy()


def flow_warp_flow(flow1, flow2):
    h, w, _ = flow1.shape
    if isinstance(flow1, np.ndarray):
        flow1 = torch.from_numpy(flow1).float()
    if isinstance(flow2, np.ndarray):
        flow2 = torch.from_numpy(flow2).float()

    x_ = torch.arange(w).view(1, -1).expand(h, -1)  # h,w
    y_ = torch.arange(h).view(-1, 1).expand(-1, w)  # h,w
    grid = torch.stack([x_, y_], dim=0).float()  # 2,h,w
    grid = grid + flow1.contiguous().permute(2, 0, 1)
    grid = grid[None]
    grid[:, 0, :, :] = 2 * grid[:, 0, :, :] / (w - 1) - 1
    grid[:, 1, :, :] = 2 * grid[:, 1, :, :] / (h - 1) - 1
    grid = grid.permute(0, 2, 3, 1)
    query_values = F.grid_sample(flow2.permute(2, 0, 1)[None], grid, padding_mode='zeros')
    query_values = query_values[0].permute(1, 2, 0)
    composed_flow = flow1 + query_values
    return composed_flow
