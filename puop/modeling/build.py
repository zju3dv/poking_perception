from puop.modeling.models.maskfusion import MaskFusion
from puop.modeling.models.pokingrecon import PokingRecon

_META_ARCHITECTURES = {'PokingRecon': PokingRecon,
                       'MaskFusion': MaskFusion
                       }


def build_model(cfg):
    print("building model...", end='\r')
    meta_arch = _META_ARCHITECTURES[cfg.model.meta_architecture]
    model = meta_arch(cfg)
    return model
