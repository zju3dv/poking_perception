import torch.nn.functional as F
import torch
import torch.nn as nn


class NeRF(nn.Module):
    def __init__(self, D, W, input_ch, output_ch, skips, input_ch_views, use_viewdirs, multires, multires_views=-1):
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.embed_fn = get_embedder(multires)
        input_ch = self.embed_fn.out_dim

        if use_viewdirs:
            embeddirs_fn = get_embedder(multires_views)
            input_ch_views = embeddirs_fn.out_dim
            self.embeddirs_fn = embeddirs_fn

        self.pts_linears = nn.ModuleList()
        linear = nn.Linear(input_ch, W)
        self.tensorflow_init_weights(linear)
        self.pts_linears.append(linear)
        for i in range(D - 1):
            linear = nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W)
            self.tensorflow_init_weights(linear)
            self.pts_linears.append(linear)

        views_linears = nn.Linear(input_ch_views + W, W // 2)
        self.tensorflow_init_weights(views_linears)
        self.views_linears = nn.ModuleList([views_linears])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.tensorflow_init_weights(self.feature_linear)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
            self.tensorflow_init_weights(self.alpha_linear, 'all')
            self.tensorflow_init_weights(self.rgb_linear, 'all')
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def tensorflow_init_weights(self, linear, out=None):
        # use Xavier init instead of Kaiming init
        relu_gain = torch.nn.init.calculate_gain("relu")  # sqrt(2)
        if out == "all":
            torch.nn.init.xavier_uniform_(linear.weight)
        else:
            torch.nn.init.xavier_uniform_(linear.weight, gain=relu_gain)
        torch.nn.init.zeros_(linear.bias)

    def forward(self, x, view_dirs, latent_vector=None, obj_pose=None, progress=None):
        input_pts = self.embed_fn(x, progress=progress)
        # input_pts = self.embed_fn(x)
        input_views = self.embeddirs_fn(view_dirs, progress=progress)
        # input_views = self.embeddirs_fn(view_dirs)
        # input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts  # 65536,63
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)
        # 65536 256
        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            # alpha = F.softplus(alpha)
            alpha = F.relu(alpha)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            rgb = torch.sigmoid(rgb)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs


def get_embedder(multires, barf_c2f=[-1, -1], i=0):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
        'barf_c2f': barf_c2f
    }

    embedder = Embedder(**embed_kwargs)
    return embedder


class Freq(nn.Module):
    def __init__(self, p_fn, freq):
        super().__init__()
        self.p_fn = p_fn
        self.freq = freq

    def forward(self, x):
        return self.p_fn(x * self.freq)


# Positional encoding (section 5.1)
class Embedder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.create_embedding_fn()
        self.barf_c2f = kwargs.get('barf_c2f')

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(nn.Identity())
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                # embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                embed_fns.append(Freq(p_fn, freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def forward(self, inputs, progress=None):
        outs = [fn(inputs) for fn in self.embed_fns]
        # identity = outs[0]
        # outs = outs[1:]
        # outs = outs[::2] + outs[1::2]
        # out = torch.stack(outs, -1).reshape(inputs.shape[0], -1)
        out = torch.cat(outs, -1)
        # if self.barf_c2f is not None and self.barf_c2f[0] > 0 and self.barf_c2f[1] > 0:
        #     # set weights for different frequency bands
        #     start, end = self.barf_c2f
        #     L = self.kwargs['num_freqs']
        #     alpha = (progress - start) / (end - start) * L
        #     if isinstance(alpha, torch.Tensor):
        #         alpha = alpha.reshape(-1)[0].item()
        #     k = torch.arange(L, dtype=torch.float32, device=inputs.device)
        #     weight = (1 - (alpha - k).clamp_(min=0, max=1).mul_(np.pi).cos_()) / 2
        #     # apply weights
        #     shape = out.shape
        #     out = (out.view(-1, L) * weight).reshape(shape)
        #     out = torch.cat([identity, out], dim=-1)
        return out
