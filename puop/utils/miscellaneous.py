from dl_ext.pytorch_ext import is_main_process


def save_config(cfg, path):
    if is_main_process():
        with open(path, 'w') as f:
            f.write(cfg.dump())
