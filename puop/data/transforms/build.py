from . import transforms as T


def build_transforms(cfg, is_train=True):
    ts = []
    for transform in cfg.input.transforms:
        ts.append(build_transform(transform))
    transform = T.Compose(ts)
    return transform


def build_transform(t):
    raise NotImplementedError()
