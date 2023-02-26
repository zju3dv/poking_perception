import cv2
import loguru
import numpy as np
import open3d as o3d
import pytorch3d.transforms
import torch
import torch.nn.functional as F
import tqdm
import trimesh
from dl_ext.vision_ext.datasets.kitti.structures import Calibration
from multipledispatch import dispatch
from packaging import version

from puop.utils.pn_utils import to_tensor, to_array


@dispatch(np.ndarray, np.ndarray)
def canonical_to_camera(pts, pose):
    """
    :param pts: Nx3
    :param pose: 4x4
    :param calib:Calibration
    :return:
    """
    pts = Calibration.cart_to_hom(pts)
    # pts = np.hstack((pts, np.ones((pts.shape[0], 1))))  # 4XN
    pts = pts @ pose.T  # 4xN
    pts = Calibration.hom_to_cart(pts)
    return pts


@dispatch(np.ndarray, np.ndarray)
def canonical_to_camera(pts, pose):
    """
    :param pts: Nx3
    :param pose: 4x4
    :param calib:Calibration
    :return:
    """
    pts = cart_to_hom(pts)
    pts = pts @ pose.T
    pts = hom_to_cart(pts)
    return pts


def cart_to_hom(pts):
    """
    :param pts: (N, 3 or 2)
    :return pts_hom: (N, 4 or 3)
    """
    if isinstance(pts, np.ndarray):
        pts_hom = np.concatenate((pts, np.ones([*pts.shape[:-1], 1], dtype=np.float32)), -1)
    else:
        ones = torch.ones([*pts.shape[:-1], 1], dtype=torch.float32, device=pts.device)
        pts_hom = torch.cat((pts, ones), dim=-1)
    return pts_hom


@dispatch(torch.Tensor, torch.Tensor)
def canonical_to_camera(pts, pose):
    pts = cart_to_hom(pts)
    pts = pts @ pose.transpose(-1, -2)
    pts = hom_to_cart(pts)
    return pts


def hom_to_cart(pts):
    return pts[..., :-1] / pts[..., -1:]


transform_points = canonical_to_camera


def camera_to_canonical(pts, pose):
    """
    :param pts: Nx3
    :param pose: 4x4
    :return:
    """
    if isinstance(pts, np.ndarray) and isinstance(pose, np.ndarray):
        pts = pts.T  # 3xN
        pts = np.vstack((pts, np.ones((1, pts.shape[1]))))  # 4XN
        p = np.linalg.inv(pose) @ pts  # 4xN
        p[0:3] /= p[3:]
        p = p[0:3]
        p = p.T
        return p
    else:
        pts = Calibration.cart_to_hom(pts)
        pts = pts @ torch.inverse(pose).t()
        pts = Calibration.hom_to_cart(pts)
        return pts


def xyzr_to_pose4x4(x, y, z, r):
    pose = np.eye(4)
    pose[0, 0] = np.cos(r)
    pose[0, 2] = np.sin(r)
    pose[2, 0] = -np.sin(r)
    pose[2, 2] = np.cos(r)
    pose[0, 3] = x
    pose[1, 3] = y
    pose[2, 3] = z
    return pose


def xyzr_to_pose4x4_torch(x, y, z, r):
    if isinstance(x, torch.Tensor):
        pose = torch.eye(4, device=x.device, dtype=torch.float)
        pose[0, 0] = torch.cos(r)
        pose[0, 2] = torch.sin(r)
        pose[2, 0] = -torch.sin(r)
        pose[2, 2] = torch.cos(r)
        pose[0, 3] = x
        pose[1, 3] = y
        pose[2, 3] = z
        return pose
    else:
        return torch.from_numpy(xyzr_to_pose4x4_np(x, y, z, r)).float()


from multipledispatch import dispatch


@dispatch(np.ndarray)
def pose4x4_to_xyzr(pose):
    x = pose[0, 3]
    y = pose[1, 3]
    z = pose[2, 3]
    cos = pose[0, 0]
    sin = pose[0, 2]
    angle = np.arctan2(sin, cos)
    return x, y, z, angle


@dispatch(torch.Tensor)
def pose4x4_to_xyzr(pose):
    x = pose[0, 3]
    y = pose[1, 3]
    z = pose[2, 3]
    cos = pose[0, 0]
    sin = pose[0, 2]
    angle = torch.atan2(sin, cos)
    return x, y, z, angle


def xyzr_to_pose4x4_np(x, y, z, r):
    pose = np.eye(4)
    pose[0, 0] = np.cos(r)
    pose[0, 2] = np.sin(r)
    pose[2, 0] = -np.sin(r)
    pose[2, 2] = np.cos(r)
    pose[0, 3] = x
    pose[1, 3] = y
    pose[2, 3] = z
    return pose


def rotx_np(a):
    """
    :param a: np.ndarray of (N, 1) or (N), or float, or int
              angle
    :return: np.ndarray of (N, 3, 3)
             rotation matrix
    """
    if isinstance(a, (int, float)):
        a = np.array([a])
    a = a.astype(np.float).reshape((-1, 1))
    ones = np.ones_like(a)
    zeros = np.zeros_like(a)
    c = np.cos(a)
    s = np.sin(a)
    rot = np.stack([ones, zeros, zeros,
                    zeros, c, -s,
                    zeros, s, c])
    return rot.reshape((-1, 3, 3))


def roty_np(a):
    """
    :param a: np.ndarray of (N, 1) or (N), or float, or int
              angle
    :return: np.ndarray of (N, 3, 3)
             rotation matrix
    """
    if isinstance(a, (int, float)):
        a = np.array([a])
    a = a.astype(np.float).reshape((-1, 1))
    ones = np.ones_like(a)
    zeros = np.zeros_like(a)
    c = np.cos(a)
    s = np.sin(a)
    rot = np.stack([c, zeros, s,
                    zeros, ones, zeros,
                    -s, zeros, c])
    return rot.reshape((-1, 3, 3))


def roty_torch(a):
    """
    :param a: np.ndarray of (N, 1) or (N), or float, or int
              angle
    :return: np.ndarray of (N, 3, 3)
             rotation matrix
    """
    if isinstance(a, (int, float)):
        a = torch.tensor([a])
    # if a.shape[-1] != 1:
    #     a = a[..., None]
    a = a.float()
    ones = torch.ones_like(a)
    zeros = torch.zeros_like(a)
    c = torch.cos(a)
    s = torch.sin(a)
    rot = torch.stack([c, zeros, s,
                       zeros, ones, zeros,
                       -s, zeros, c], dim=-1)
    return rot.reshape(*rot.shape[:-1], 3, 3)


def rotz_np(a):
    """
    :param a: np.ndarray of (N, 1) or (N), or float, or int
              angle
    :return: np.ndarray of (N, 3, 3)
             rotation matrix
    """
    if isinstance(a, (int, float)):
        a = np.array([a])
    a = a.astype(np.float).reshape((-1, 1))
    ones = np.ones_like(a)
    zeros = np.zeros_like(a)
    c = np.cos(a)
    s = np.sin(a)
    rot = np.stack([c, -s, zeros,
                    s, c, zeros,
                    zeros, zeros, ones])
    return rot.reshape((-1, 3, 3))


def rotz_torch(a):
    if isinstance(a, (int, float)):
        a = torch.tensor([a])
    if a.shape[-1] != 1:
        a = a[..., None]
    a = a.float()
    ones = torch.ones_like(a)
    zeros = torch.zeros_like(a)
    c = torch.cos(a)
    s = torch.sin(a)
    rot = torch.stack([c, -s, zeros,
                       s, c, zeros,
                       zeros, zeros, ones], dim=-1)
    return rot.reshape(*rot.shape[:-1], 3, 3)


def rotx(t):
    """
    Rotation along the x-axis.
    :param t: tensor of (N, 1) or (N), or float, or int
              angle
    :return: tensor of (N, 3, 3)
             rotation matrix
    """
    if isinstance(t, (int, float)):
        t = torch.tensor([t])
    if t.shape[-1] != 1:
        t = t[..., None]
    t = t.type(torch.float)
    ones = torch.ones_like(t)
    zeros = torch.zeros_like(t)
    c = torch.cos(t)
    s = torch.sin(t)
    rot = torch.stack([ones, zeros, zeros,
                       zeros, c, -s,
                       zeros, s, c], dim=-1)
    return rot.reshape(*rot.shape[:-1], 3, 3)


def matrix_3x4_to_4x4(a):
    if len(a.shape) == 2:
        assert a.shape == (3, 4)
    else:
        assert len(a.shape) == 3
        assert a.shape[1:] == (3, 4)
    if isinstance(a, np.ndarray):
        if a.ndim == 2:
            ones = np.array([[0, 0, 0, 1]])
            return np.vstack((a, ones))
        else:
            ones = np.array([[0, 0, 0, 1]])[None].repeat(a.shape[0], axis=0)
            return np.concatenate((a, ones), axis=1)
    else:
        ones = torch.tensor([[0, 0, 0, 1]]).float().to(device=a.device)
        if a.ndim == 3:
            ones = ones[None].repeat(a.shape[0], 1, 1)
            ret = torch.cat((a, ones), dim=1)
        else:
            ret = torch.cat((a, ones), dim=0)
        return ret


def matrix_3x3_to_4x4(a):
    assert a.shape == (3, 3)
    if isinstance(a, np.ndarray):
        ret = np.eye(4)
    else:
        ret = torch.eye(4).float().to(a.device)
    ret[:3, :3] = a
    return ret


def img_to_rect(fu, fv, cu, cv, u, v, depth_rect):
    """
    :param u: (N)
    :param v: (N)
    :param depth_rect: (N)
    :return: pts_rect:(N, 3)
    """
    # check_type(u)
    # check_type(v)

    if isinstance(depth_rect, np.ndarray):
        x = ((u - cu) * depth_rect) / fu
        y = ((v - cv) * depth_rect) / fv
        pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1)
    else:
        x = ((u.float() - cu) * depth_rect) / fu
        y = ((v.float() - cv) * depth_rect) / fv
        pts_rect = torch.cat((x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), dim=1)
    # x = ((u - cu) * depth_rect) / fu
    # y = ((v - cv) * depth_rect) / fv
    # pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1)
    return pts_rect


def depth_to_rect(fu, fv, cu, cv, depth_map, ray_mode=False, select_coords=None):
    """

    :param fu:
    :param fv:
    :param cu:
    :param cv:
    :param depth_map:
    :param ray_mode: whether values in depth_map are Z or norm
    :return:
    """
    if len(depth_map.shape) == 2:
        if isinstance(depth_map, np.ndarray):
            x_range = np.arange(0, depth_map.shape[1])
            y_range = np.arange(0, depth_map.shape[0])
            x_idxs, y_idxs = np.meshgrid(x_range, y_range)
        else:
            x_range = torch.arange(0, depth_map.shape[1]).to(device=depth_map.device)
            y_range = torch.arange(0, depth_map.shape[0]).to(device=depth_map.device)
            y_idxs, x_idxs = torch.meshgrid(y_range, x_range, indexing='ij')
        x_idxs, y_idxs = x_idxs.reshape(-1), y_idxs.reshape(-1)
        depth = depth_map[y_idxs, x_idxs]
    else:
        x_idxs = select_coords[:, 1].float()
        y_idxs = select_coords[:, 0].float()
        depth = depth_map
    if ray_mode is True:
        if isinstance(depth, torch.Tensor):
            depth = depth / (((x_idxs.float() - cu.float()) / fu.float()) ** 2 + (
                    (y_idxs.float() - cv.float()) / fv.float()) ** 2 + 1) ** 0.5
        else:
            depth = depth / (((x_idxs - cu) / fu) ** 2 + (
                    (y_idxs - cv) / fv) ** 2 + 1) ** 0.5
    pts_rect = img_to_rect(fu, fv, cu, cv, x_idxs, y_idxs, depth)
    return pts_rect


def rect_to_img(fu, fv, cu, cv, pts_rect):
    if isinstance(pts_rect, np.ndarray):
        K = np.array([[fu, 0, cu],
                      [0, fv, cv],
                      [0, 0, 1]])
        pts_2d_hom = pts_rect @ K.T
        pts_img = Calibration.hom_to_cart(pts_2d_hom)
    else:
        device = pts_rect.device
        P2 = torch.tensor([[fu, 0, cu],
                           [0, fv, cv],
                           [0, 0, 1]], dtype=torch.float, device=device)
        pts_2d_hom = pts_rect @ P2.t()
        pts_img = Calibration.hom_to_cart(pts_2d_hom)
    return pts_img


def backproject_flow3d_torch(flow2d, depth0, depth1, intrinsics, campose0, campose1):
    """ compute 3D flow from 2D flow + depth change """
    # raise NotImplementedError()

    ht, wd = flow2d.shape[0:2]

    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]

    y0, x0 = torch.meshgrid(
        torch.arange(ht).to(depth0.device).float(),
        torch.arange(wd).to(depth0.device).float())

    x1 = x0 + flow2d[..., 0]
    y1 = y0 + flow2d[..., 1]

    X0 = depth0 * ((x0 - cx) / fx)
    Y0 = depth0 * ((y0 - cy) / fy)
    Z0 = depth0

    # X1 = depth1 * ((x1 - cx) / fx)
    # Y1 = depth1 * ((y1 - cy) / fy)
    # Z1 = depth1

    grid = torch.stack([x1, y1], dim=-1)[None]
    grid[:, :, :, 0] = grid[:, :, :, 0] / (wd - 1)
    grid[:, :, :, 1] = grid[:, :, :, 1] / (ht - 1)
    grid = grid * 2 - 1
    depth1_interp = torch.nn.functional.grid_sample(
        depth1[None, None],
        grid,
        mode='bilinear'
    )[0, 0]

    X1 = depth1_interp * ((x1 - cx) / fx)
    Y1 = depth1_interp * ((y1 - cy) / fy)
    Z1 = depth1_interp

    pts0_cam = torch.stack([X0, Y0, Z0], dim=-1)
    pts1_cam = torch.stack([X1, Y1, Z1], dim=-1)

    flow3d = pts1_cam - pts0_cam
    return flow3d, pts0_cam, pts1_cam


def _sample_at_integer_locs(input_feats, index_tensor):
    assert input_feats.ndimension() == 5, 'input_feats should be of shape [B,F,D,H,W]'
    assert index_tensor.ndimension() == 4, 'index_tensor should be of shape [B,H,W,3]'
    # first sample pixel locations using nearest neighbour interpolation
    batch_size, num_chans, num_d, height, width = input_feats.shape
    grid_height, grid_width = index_tensor.shape[1], index_tensor.shape[2]

    xy_grid = index_tensor[..., 0:2]
    xy_grid[..., 0] = xy_grid[..., 0] - ((width - 1.0) / 2.0)
    xy_grid[..., 0] = xy_grid[..., 0] / ((width - 1.0) / 2.0)
    xy_grid[..., 1] = xy_grid[..., 1] - ((height - 1.0) / 2.0)
    xy_grid[..., 1] = xy_grid[..., 1] / ((height - 1.0) / 2.0)
    xy_grid = torch.clamp(xy_grid, min=-1.0, max=1.0)
    sampled_in_2d = F.grid_sample(input=input_feats.view(batch_size, num_chans * num_d, height, width),
                                  grid=xy_grid, mode='nearest', align_corners=False).view(batch_size, num_chans, num_d,
                                                                                          grid_height,
                                                                                          grid_width)
    z_grid = index_tensor[..., 2].view(batch_size, 1, 1, grid_height, grid_width)
    z_grid = z_grid.long().clamp(min=0, max=num_d - 1)
    z_grid = z_grid.expand(batch_size, num_chans, 1, grid_height, grid_width)
    sampled_in_3d = sampled_in_2d.gather(2, z_grid).squeeze(2)
    return sampled_in_3d


def trilinear_interpolation(input_feats, sampling_grid):
    """
    interploate value in 3D volume
    :param input_feats: [B,C,D,H,W]
    :param sampling_grid: [B,H,W,3] unscaled coordinates
    :return:
    """
    assert input_feats.ndimension() == 5, 'input_feats should be of shape [B,F,D,H,W]'
    assert sampling_grid.ndimension() == 4, 'sampling_grid should be of shape [B,H,W,3]'
    batch_size, num_chans, num_d, height, width = input_feats.shape
    grid_height, grid_width = sampling_grid.shape[1], sampling_grid.shape[2]
    # make sure sampling grid lies between -1, 1
    sampling_grid[..., 0] = 2 * sampling_grid[..., 0] / (num_d - 1) - 1
    sampling_grid[..., 1] = 2 * sampling_grid[..., 1] / (height - 1) - 1
    sampling_grid[..., 2] = 2 * sampling_grid[..., 2] / (width - 1) - 1
    sampling_grid = torch.clamp(sampling_grid, min=-1.0, max=1.0)
    # map to 0,1
    sampling_grid = (sampling_grid + 1) / 2.0
    # Scale grid to floating point pixel locations
    scaling_factor = torch.FloatTensor([width - 1.0, height - 1.0, num_d - 1.0]).to(input_feats.device).view(1, 1,
                                                                                                             1, 3)
    sampling_grid = scaling_factor * sampling_grid
    # Now sampling grid is between [0, w-1; 0,h-1; 0,d-1]
    x, y, z = torch.split(sampling_grid, split_size_or_sections=1, dim=3)
    x_0, y_0, z_0 = torch.split(sampling_grid.floor(), split_size_or_sections=1, dim=3)
    x_1, y_1, z_1 = x_0 + 1.0, y_0 + 1.0, z_0 + 1.0
    u, v, w = x - x_0, y - y_0, z - z_0
    u, v, w = map(lambda x: x.view(batch_size, 1, grid_height, grid_width).expand(
        batch_size, num_chans, grid_height, grid_width), [u, v, w])
    c_000 = _sample_at_integer_locs(input_feats, torch.cat([x_0, y_0, z_0], dim=3))
    c_001 = _sample_at_integer_locs(input_feats, torch.cat([x_0, y_0, z_1], dim=3))
    c_010 = _sample_at_integer_locs(input_feats, torch.cat([x_0, y_1, z_0], dim=3))
    c_011 = _sample_at_integer_locs(input_feats, torch.cat([x_0, y_1, z_1], dim=3))
    c_100 = _sample_at_integer_locs(input_feats, torch.cat([x_1, y_0, z_0], dim=3))
    c_101 = _sample_at_integer_locs(input_feats, torch.cat([x_1, y_0, z_1], dim=3))
    c_110 = _sample_at_integer_locs(input_feats, torch.cat([x_1, y_1, z_0], dim=3))
    c_111 = _sample_at_integer_locs(input_feats, torch.cat([x_1, y_1, z_1], dim=3))
    c_xyz = (1.0 - u) * (1.0 - v) * (1.0 - w) * c_000 + \
            (1.0 - u) * (1.0 - v) * w * c_001 + \
            (1.0 - u) * v * (1.0 - w) * c_010 + \
            (1.0 - u) * v * w * c_011 + \
            u * (1.0 - v) * (1.0 - w) * c_100 + \
            u * (1.0 - v) * w * c_101 + \
            u * v * (1.0 - w) * c_110 + \
            u * v * w * c_111
    return c_xyz


def open3d_tsdf_fusion_api(depths, obj_poses, K, voxel_length, colors=None):
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=3 * voxel_length,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    if colors is None:
        colors = np.zeros([len(depths), depths[0].shape[0], depths[0].shape[1], 3], dtype=np.uint8)
    for i in tqdm.tqdm(range(len(colors))):
        H, W, _ = colors[0].shape
        pose = np.linalg.inv(obj_poses[i])
        rgb = o3d.geometry.Image(colors[i])
        depth_pred = o3d.geometry.Image(depths[i].astype(np.float32))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb, depth_pred, depth_scale=1.0, depth_trunc=5.0, convert_rgb_to_intensity=False
        )
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width=W, height=H, fx=fx, fy=fy, cx=cx, cy=cy)
        extrinsic = np.linalg.inv(pose)
        volume.integrate(rgbd, intrinsic, extrinsic)
    mesh = volume.extract_triangle_mesh()
    mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                           vertex_colors=(np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)[:, :3])
    return mesh


def open3d_icp_api(pts0, pts1, thresh, init_Rt=np.eye(4)):
    """
    R*pts0+t=pts1
    :param pts0: nx3
    :param pts1: mx3
    :param thresh: float
    :param init_Rt: 4x4
    :return:
    """
    import open3d as o3d
    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(pts0.copy())
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pts1.copy())
    if version.parse(o3d.__version__) < version.parse('0.10.0'):
        result = o3d.registration.registration_icp(
            pcd0, pcd1, thresh, init_Rt)
    else:
        result = o3d.pipelines.registration.registration_icp(
            pcd0, pcd1, thresh, init_Rt)
    return result.transformation


def rotation_distance(R1, R2, eps=1e-7):
    # http://www.boris-belousov.net/2016/12/01/quat-dist/
    if R1.ndim == 2 and R2.ndim == 2:
        R_diff = R1[:3, :3] @ R2[:3, :3].T
        trace = R_diff[0, 0] + R_diff[1, 1] + R_diff[2, 2]
    else:
        R_diff = R1[..., :3, :3] @ R2.transpose(-2, -1)[..., :3, :3]
        trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
    angle = ((trace - 1) / 2).clamp(-1 + eps, 1 - eps).acos_()  # numerical stability near -1/+1
    return angle


def pose_distance(pred, gt, eps=1e-7, align=False):
    if pred.numel() == 0 or gt.numel() == 0:
        return torch.empty([0]), torch.empty([0])
    if pred.ndim == 2 and gt.ndim == 2:
        pred = pred[None]
        gt = gt[None]
    if align:
        gt = gt @ gt[0].inverse()[None]
        pred = pred @ pred[0].inverse()[None]
    R_error = rotation_distance(pred, gt, eps)
    t_error = (pred[..., :3, 3] - gt[..., :3, 3]).norm(dim=-1)
    return R_error, t_error


def chamfer_distance(pts0, pts1, color0=None, color1=None, use_gpu=True):
    if use_gpu:
        from puop.utils.chamfer3D import dist_chamfer_3D
        chamLoss = dist_chamfer_3D.chamfer_3DDist()
        points1 = to_tensor(pts0).cuda().float()[None]
        points2 = to_tensor(pts1).cuda().float()[None]
        # points1 = torch.rand(32, 1000, 3).cuda()
        # points2 = torch.rand(32, 2000, 3, requires_grad=True).cuda()
        dist1, dist2, idx1, idx2 = chamLoss(points1, points2)
        loss = dist1.mean() + dist2.mean()
        if color0 is not None and color1 is not None:
            color0 = to_tensor(color0).cuda()
            color1 = to_tensor(color1).cuda()
            if color0.max() > 1:
                color0 = color0.float() / 255.0
            if color1.max() > 1:
                color1 = color1.float() / 255.0
            idx1 = idx1[0]
            idx2 = idx2[0]
            l1 = ((color0 - color1[idx1.long()]) ** 2).mean()
            l2 = ((color1 - color0[idx2.long()]) ** 2).mean()
            loss = loss + l1 + l2
        return loss.item()
    else:
        pts0 = to_tensor(pts0)
        pts1 = to_tensor(pts1)

        def square_distance(src, dst):
            return torch.sum((src[:, None, :] - dst[None, :, :]) ** 2, dim=-1)

        dist_src = torch.min(square_distance(pts0, pts1), dim=-1)
        dist_ref = torch.min(square_distance(pts1, pts0), dim=-1)
        chamfer_dist = torch.mean(dist_src[0]) + torch.mean(dist_ref[0])
        if color0 is not None or color1 is not None:
            raise NotImplementedError()
        return chamfer_dist.item()


def open3d_plane_segment_api(pts, distance_threshold, ransac_n=3, num_iterations=1000):
    pts = to_array(pts)
    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(pts)
    plane_model, inliers = pcd0.segment_plane(distance_threshold,
                                              ransac_n=ransac_n,
                                              num_iterations=num_iterations)
    return plane_model, inliers


def point_plane_distance_api(pts, plane_model):
    a, b, c, d = plane_model.tolist()
    if isinstance(pts, torch.Tensor):
        dists = (a * pts[:, 0] + b * pts[:, 1] + c * pts[:, 2] + d).abs() / ((a * a + b * b + c * c) ** 0.5)
    else:
        dists = np.abs(a * pts[:, 0] + b * pts[:, 1] + c * pts[:, 2] + d) / ((a * a + b * b + c * c) ** 0.5)
    return dists


def se3_exp_map(log_transform: torch.Tensor, eps: float = 1e-4):
    return pytorch3d.transforms.se3.se3_exp_map(log_transform, eps)


def se3_log_map(transform: torch.Tensor, eps: float = 1e-4, cos_bound: float = 1e-4, backend=None):
    if backend is None:
        loguru.logger.warning("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        loguru.logger.warning("!!!!se3_log_map backend is None!!!!")
        loguru.logger.warning("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        backend = 'pytorch3d'
    if backend == 'pytorch3d':
        return pytorch3d.transforms.se3.se3_log_map(transform, eps, cos_bound)
    elif backend == 'opencv':
        from pytorch3d.transforms.se3 import _se3_V_matrix, _get_se3_V_input
        # from pytorch3d.common.compat import solve
        log_rotation = []
        for tsfm in transform:
            cv2_rot = -cv2.Rodrigues(to_array(tsfm[:3, :3]))[0]
            log_rotation.append(torch.from_numpy(cv2_rot.reshape(-1)).to(transform.device).float())
        log_rotation = torch.stack(log_rotation, dim=0)
        T = transform[:, 3, :3]
        V = _se3_V_matrix(*_get_se3_V_input(log_rotation), eps=eps)
        # log_translation = solve(V, T[:, :, None])[:, :, 0]
        log_translation = torch.linalg.solve(V, T[:, :, None])[:, :, 0]
        return torch.cat((log_translation, log_rotation), dim=1)
    else:
        raise NotImplementedError()
