import torchsparse as ts
import torch
from torch.nn import InstanceNorm1d
from torchsparse import SparseTensor


class InstanceNorm(InstanceNorm1d):
    def forward(self, input):
        if isinstance(input, torch.Tensor):
            return super(InstanceNorm, self).forward(input)
        elif isinstance(input, ts.SparseTensor):
            coords, feats, stride = input.C, input.F, input.s

            batch_size = torch.max(coords[:, -1]).item() + 1
            if batch_size == 1:
                feats = super(InstanceNorm, self).forward(feats.transpose(0, 1)[None])
                nfeats = feats[0].transpose(0, 1)
            else:
                num_channels = feats.shape[1]

                # PyTorch's GroupNorm function expects the input to be in (N, C, *)
                # format where N is batch size, and C is number of channels. "feats"
                # is not in that format. So, we extract the feats corresponding to
                # each sample, bring it to the format expected by PyTorch's GroupNorm
                # function, and invoke it.
                nfeats = torch.zeros_like(feats)
                for k in range(batch_size):
                    indices = coords[:, -1] == k
                    bfeats = feats[indices]
                    bfeats = bfeats.transpose(0, 1).reshape(1, num_channels, -1)
                    bfeats = super().forward(bfeats)
                    bfeats = bfeats.reshape(num_channels, -1).transpose(0, 1)
                    nfeats[indices] = bfeats

            output = SparseTensor(coords=coords, feats=nfeats, stride=stride)
            output.coord_maps = input.coord_maps
            output.kernel_maps = input.kernel_maps
            # output.cmaps = input.cmaps
            # output.kmaps = input.kmaps
            return output
