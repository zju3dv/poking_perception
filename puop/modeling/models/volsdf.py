import torch.nn.functional as F
import math

import torch
import numpy as np
from torch import nn, autograd


class ImplicitSurface(nn.Module):
    def __init__(self,
                 W=256,
                 D=8,
                 skips=[4],
                 W_geo_feat=256,
                 input_ch=3,
                 radius_init=1.0,
                 obj_bounding_size=2.0,
                 geometric_init=True,
                 embed_multires=6,
                 weight_norm=True,
                 use_siren=False,
                 latent_size=0
                 ):
        """
        W_geo_feat: to set whether to use nerf-like geometry feature or IDR-like geometry feature.
            set to -1: nerf-like, the output feature is the second to last level's feature of the geometry network.
            set to >0: IDR-like ,the output feature is the last part of the geometry network's output.
        """
        super().__init__()
        # occ_net_list = [
        #     nn.Sequential(
        #         nn.Linear(input_ch, W),
        #         nn.Softplus(),
        #     )
        # ] + [
        #     nn.Sequential(
        #         nn.Linear(W, W),
        #         nn.Softplus()
        #     ) for _ in range(D-2)
        # ] + [
        #     nn.Linear(W, 1)
        # ]
        self.radius_init = radius_init
        self.register_buffer('obj_bounding_size', torch.tensor([obj_bounding_size]).float())
        self.geometric_init = geometric_init
        self.D = D
        self.W = W
        self.W_geo_feat = W_geo_feat
        if use_siren:
            assert len(skips) == 0, "do not use skips for siren"
            self.register_buffer('is_pretrained', torch.tensor([False], dtype=torch.bool))
        self.skips = skips
        self.use_siren = use_siren
        self.embed_fn, input_ch = get_embedder(embed_multires)

        surface_fc_layers = []
        # NOTE: as in IDR/NeuS, the network's has D+1 layers
        for l in range(D + 1):
            # decide out_dim
            if l == D:
                if W_geo_feat > 0:
                    out_dim = 1 + W_geo_feat
                else:
                    out_dim = 1
            elif (l + 1) in self.skips:
                out_dim = W - input_ch  # recude output dim before the skips layers, as in IDR / NeuS
            else:
                out_dim = W

            # decide in_dim
            if l == 0:
                in_dim = input_ch
                if latent_size != 0:
                    in_dim = in_dim + latent_size
            else:
                in_dim = W

            if l != D:
                if use_siren:
                    layer = SirenLayer(in_dim, out_dim, is_first=(l == 0))
                else:
                    # NOTE: beta=100 is important! Otherwise, the initial output would all be > 10, and there is not initial sphere.
                    layer = DenseLayer(in_dim, out_dim, activation=nn.Softplus(beta=100))
            else:
                layer = nn.Linear(in_dim, out_dim)

            # if true preform preform geometric initialization
            if geometric_init and not use_siren:
                # --------------
                # sphere init, as in SAL / IDR.
                # --------------
                if l == D:
                    nn.init.normal_(layer.weight, mean=np.sqrt(np.pi) / np.sqrt(in_dim), std=0.0001)
                    nn.init.constant_(layer.bias, -radius_init)
                elif embed_multires > 0 and l == 0:
                    torch.nn.init.constant_(layer.bias, 0.0)
                    torch.nn.init.constant_(layer.weight[:, 3:], 0.0)  # let the initial weights for octaves to be 0.
                    torch.nn.init.normal_(layer.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif embed_multires > 0 and l in self.skips:
                    torch.nn.init.constant_(layer.bias, 0.0)
                    torch.nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(layer.weight[:, -(input_ch - 3):],
                                            0.0)  # NOTE: this contrains the concat order to be  [h, x_embed]
                else:
                    nn.init.constant_(layer.bias, 0.0)
                    nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                layer = nn.utils.weight_norm(layer)

            surface_fc_layers.append(layer)

        self.surface_fc_layers = nn.ModuleList(surface_fc_layers)

    def pretrain_hook(self, configs={}):
        configs['target_radius'] = self.radius_init
        # TODO: more flexible, bbox-like scene bound.
        configs['obj_bounding_size'] = self.obj_bounding_size.item()
        if self.geometric_init and self.use_siren and not self.is_pretrained:
            pretrain_siren_sdf(self, **configs)
            self.is_pretrained = ~self.is_pretrained
            return True
        return False

    def forward(self, x: torch.Tensor, return_h=False, latent_vector=None):
        # todo: interpolate from tsdf volume @hx
        # todo: 1. define a optimizable shape coefficients z
        # todo: 2. volume = mu + V z
        # todo: 3. interpolate tsdf(x) from volume
        x = self.embed_fn(x)
        h = x
        if latent_vector is not None:
            h = torch.cat([h, latent_vector], -1)
        for i in range(self.D):
            if i in self.skips:
                # NOTE: concat order can not change! there are special operations taken in intialization.
                h = torch.cat([h, x], dim=-1) / np.sqrt(2)
            h = self.surface_fc_layers[i](h)

        out = self.surface_fc_layers[-1](h)

        if self.W_geo_feat > 0:
            h = out[..., 1:]
            out = out[..., :1].squeeze(-1)
        else:
            out = out.squeeze(-1)
        if return_h:
            return out, h
        else:
            return out

    def forward_with_nablas(self, x: torch.Tensor, has_grad_bypass: bool = None, latent_vector=None):
        # todo: ignore nabla and h @hx
        has_grad = torch.is_grad_enabled() if has_grad_bypass is None else has_grad_bypass
        # force enabling grad for normal calculation
        with torch.enable_grad():
            x = x.requires_grad_(True)
            implicit_surface_val, h = self.forward(x, return_h=True, latent_vector=latent_vector)
            nabla = autograd.grad(
                implicit_surface_val,
                x,
                torch.ones_like(implicit_surface_val, device=x.device),
                create_graph=has_grad,
                retain_graph=has_grad,
                only_inputs=True)[0]
        if not has_grad:
            implicit_surface_val = implicit_surface_val.detach()
            nabla = nabla.detach()
            h = h.detach()
        return implicit_surface_val, nabla, h


class RadianceNet(nn.Module):
    def __init__(self,
                 D=4,
                 W=256,
                 skips=[],
                 W_geo_feat=256,
                 embed_multires=6,
                 embed_multires_view=4,
                 use_view_dirs=True,
                 weight_norm=True,
                 use_siren=False,
                 # use_obj_props=False,
                 # multires_obj=4,
                 latent_size=256
                 ):
        super().__init__()

        input_ch_pts = 3
        input_ch_views = 3
        if use_siren:
            assert len(skips) == 0, "do not use skips for siren"
        self.skips = skips
        self.D = D
        self.W = W
        self.use_view_dirs = use_view_dirs
        # self.use_obj_props = use_obj_props
        self.embed_fn, input_ch_pts = get_embedder(embed_multires)
        if latent_size > 0:
            input_ch_pts = input_ch_pts + latent_size
        if use_view_dirs:
            self.embed_fn_view, input_ch_views = get_embedder(embed_multires_view)
            in_dim_0 = input_ch_pts + input_ch_views + 3 + W_geo_feat
        else:
            in_dim_0 = input_ch_pts + W_geo_feat

        # if use_obj_props:
        #     self.embed_fn_obj, input_ch_obj = get_embedder(multires_obj)
        #     in_dim_0 = in_dim_0 + input_ch_obj
        fc_layers = []
        # NOTE: as in IDR/NeuS, the network's has D+1 layers
        for l in range(D + 1):
            # decicde out_dim
            if l == D:
                out_dim = 3
            else:
                out_dim = W

            # decide in_dim
            if l == 0:
                in_dim = in_dim_0
                # if latent_size != 0:
                #     in_dim = in_dim + latent_size
            elif l in self.skips:
                in_dim = in_dim_0 + W
            else:
                in_dim = W

            if l != D:
                if use_siren:
                    layer = SirenLayer(in_dim, out_dim, is_first=(l == 0))
                else:
                    layer = DenseLayer(in_dim, out_dim, activation=nn.ReLU(inplace=True))
            else:
                layer = DenseLayer(in_dim, out_dim, activation=nn.Sigmoid())

            if weight_norm:
                layer = nn.utils.weight_norm(layer)

            fc_layers.append(layer)

        self.layers = nn.ModuleList(fc_layers)

    def forward(
            self,
            x: torch.Tensor,
            view_dirs: torch.Tensor,
            normals: torch.Tensor,
            geometry_feature: torch.Tensor,
            latent_vector=None,
            # obj_pose=None
    ):
        # calculate radiance field
        # if self.use_obj_props:
        #     assert latent_vector is not None
        # assert obj_pose is not None
        x = self.embed_fn(x)
        if latent_vector is not None:
            x = torch.cat([x, latent_vector], dim=-1)
        # todo: ignore normals and geometry_feature if using pca-based geometry reprensetation @hx
        if self.use_view_dirs:
            view_dirs = self.embed_fn_view(view_dirs)
            radiance_input = torch.cat([x, view_dirs, normals, geometry_feature], dim=-1)
        else:
            radiance_input = torch.cat([x, geometry_feature], dim=-1)

        # if self.use_obj_props:
        #     pose_embed = self.embed_fn_obj(obj_pose)
        #     radiance_input = torch.cat([radiance_input, pose_embed], dim=-1)

        h = radiance_input
        for i in range(self.D + 1):
            if i in self.skips:
                h = torch.cat([h, radiance_input], dim=-1)
            h = self.layers[i](h)
        return h


class VolSDF(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # self.total_cfg = cfg
        self.cfg = cfg
        surface_cfg = self.cfg.surface_cfg
        radiance_cfg = self.cfg.radiance_cfg
        self.speed_factor = self.cfg.speed_factor
        ln_beta_init = np.log(self.cfg.beta_init) / self.speed_factor
        if self.cfg.nobjs == 1:
            self.ln_beta = nn.Parameter(data=torch.Tensor([ln_beta_init]), requires_grad=True)
        else:
            self.ln_beta = nn.Parameter(data=torch.full([self.cfg.nobjs], ln_beta_init), requires_grad=True)
        # self.beta = nn.Parameter(data=torch.Tensor([beta_init]), requires_grad=True)

        self.use_sphere_bg = not self.cfg.use_nerfplusplus
        self.obj_bounding_radius = self.cfg.obj_bounding_radius
        W_geo_feat = self.cfg.W_geo_feat
        self.implicit_surface = ImplicitSurface(
            W_geo_feat=W_geo_feat, input_ch=self.cfg.input_ch,
            obj_bounding_size=self.cfg.obj_bounding_radius, **surface_cfg)

        if W_geo_feat < 0:
            W_geo_feat = self.implicit_surface.W
        self.radiance_net = RadianceNet(
            W_geo_feat=W_geo_feat, **radiance_cfg)

        # if use_nerfplusplus:
        #     self.nerf_outside = NeRF(input_ch=4, multires=10, multires_view=4, use_view_dirs=True)

    def forward_ab(self):
        beta = torch.exp(self.ln_beta * self.speed_factor)
        return 1. / beta, beta

    def forward_surface(self, x: torch.Tensor, latent_vector=None):
        sdf = self.implicit_surface.forward(x, latent_vector=latent_vector)
        if self.use_sphere_bg:
            return torch.min(sdf, self.obj_bounding_radius - x.norm(dim=-1))
        else:
            return sdf

    def forward_surface_with_nablas(self, x: torch.Tensor, latent_vector=None):
        sdf, nablas, h = self.implicit_surface.forward_with_nablas(x, latent_vector=latent_vector)
        if self.use_sphere_bg:
            d_bg = self.obj_bounding_radius - x.norm(dim=-1)
            # outside_sphere = x_norm >= 3
            outside_sphere = d_bg < sdf  # NOTE: in case the normals changed suddenly near the sphere.
            sdf[outside_sphere] = d_bg[outside_sphere]
            # nabla[outside_sphere] = normals_bg_sphere[outside_sphere] # ? NOTE: commented to ensure more eikonal constraints.
        return sdf, nablas, h

    def forward(self, x: torch.Tensor, view_dirs: torch.Tensor, latent: torch.Tensor = None,
                obj_poses: torch.Tensor = None):
        sdf, nablas, geometry_feature = self.forward_surface_with_nablas(x, latent_vector=latent)
        radiances = self.radiance_net.forward(x, view_dirs, nablas, geometry_feature, latent)
        return radiances, sdf, nablas


class Embedder(nn.Module):
    def __init__(self, input_dim, max_freq_log2, N_freqs,
                 log_sampling=True, include_input=True,
                 periodic_fns=(torch.sin, torch.cos),
                 use_barf_c2f=False, barf_c2f=[0.1, 0.5]):
        '''
        :param input_dim: dimension of input to be embedded
        :param max_freq_log2: log2 of max freq; min freq is 1 by default
        :param N_freqs: number of frequency bands
        :param log_sampling: if True, frequency bands are linerly sampled in log-space
        :param include_input: if True, raw input is included in the embedding
        :param periodic_fns: periodic functions used to embed input
        '''
        super(Embedder, self).__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.out_dim = 0
        if self.include_input:
            self.out_dim += self.input_dim

        self.out_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(
                2. ** 0., 2. ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()
        self.use_barf_c2f = use_barf_c2f
        self.barf_c2f = barf_c2f

    def forward(self, input: torch.Tensor, progress=None):
        '''
        :param input: tensor of shape [..., self.input_dim]
        :return: tensor of shape [..., self.out_dim]
        '''
        assert (input.shape[-1] == self.input_dim)

        out = []
        if self.include_input:
            out.append(input)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))
        out = torch.cat(out, dim=-1)

        assert (out.shape[-1] == self.out_dim)
        if self.use_barf_c2f:
            start, end = self.barf_c2f
            L = len(self.freq_bands)
            alpha = (progress - start) / (end - start) * L
            k = torch.arange(L, dtype=torch.float32, device=out.device)
            weight = (1 - (alpha - k).clamp_(min=0, max=1).mul_(np.pi).cos_()) / 2
            # apply weights
            shape = out.shape
            out = (out.view(-1, L) * weight).view(*shape)
        return out


def get_embedder(multires, use_barf_c2f=False, barf_c2f=[0.1, 0.5], input_dim=3):
    if multires < 0:
        return nn.Identity(), input_dim

    embed_kwargs = {
        "include_input": True,  # needs to be True for ray_bending to work properly
        "input_dim": input_dim,
        "max_freq_log2": multires - 1,
        "N_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
        'use_barf_c2f': use_barf_c2f,
        'barf_c2f': barf_c2f,
    }

    embedder_obj = Embedder(**embed_kwargs)
    return embedder_obj, embedder_obj.out_dim


class Sine(nn.Module):
    def __init__(self, w0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class SirenLayer(nn.Linear):
    def __init__(self, input_dim, out_dim, *args, is_first=False, **kwargs):
        self.is_first = is_first
        self.input_dim = input_dim
        self.w0 = 30
        self.c = 6
        super().__init__(input_dim, out_dim, *args, **kwargs)
        self.activation = Sine(self.w0)

    # override
    def reset_parameters(self) -> None:
        # NOTE: in offical SIREN, first run linear's original initialization, then run custom SIREN init.
        #       hence the bias is initalized in super()'s reset_parameters()
        super().reset_parameters()
        with torch.no_grad():
            dim = self.input_dim
            w_std = (1 / dim) if self.is_first else (math.sqrt(self.c / dim) / self.w0)
            self.weight.uniform_(-w_std, w_std)

    def forward(self, x):
        out = super().forward(x)
        out = self.activation(out)
        return out


class DenseLayer(nn.Linear):
    def __init__(self, input_dim: int, out_dim: int, *args, activation=None, **kwargs):
        super().__init__(input_dim, out_dim, *args, **kwargs)
        if activation is None:
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = activation

    def forward(self, x):
        out = super().forward(x)
        out = self.activation(out)
        return out


def pretrain_siren_sdf(
        implicit_surface: ImplicitSurface,
        num_iters=5000, lr=1.0e-4, batch_points=5000,
        target_radius=0.5, obj_bounding_size=3.0,
        logger=None):
    # --------------
    # pretrain SIREN-sdf to be a sphere, as in SIREN and Neural Lumigraph Rendering
    # --------------
    from tqdm import tqdm
    from torch import optim
    device = next(implicit_surface.parameters()).device
    optimizer = optim.Adam(implicit_surface.parameters(), lr=lr)

    with torch.enable_grad():
        for it in tqdm(range(num_iters), desc="=> pretraining SIREN..."):
            pts = torch.empty([batch_points, 3]).uniform_(-obj_bounding_size, obj_bounding_size).float().to(device)
            sdf_gt = pts.norm(dim=-1) - target_radius
            sdf_pred = implicit_surface.forward(pts)

            loss = F.l1_loss(sdf_pred, sdf_gt, reduction='mean')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if logger is not None:
                logger.add('pretrain_siren', 'loss_l1', loss.item(), it)
