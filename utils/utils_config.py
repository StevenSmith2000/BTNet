import importlib
import os.path as osp


def get_config(config_file):
    assert config_file.startswith('configs/'), 'config file setting must start with configs/'
    temp_config_name = osp.basename(config_file)   # the file name.extension
    temp_module_name = osp.splitext(temp_config_name)[0]   # the file name
    config = importlib.import_module("configs.base") # relative path
    cfg = config.config   #base config
    config = importlib.import_module("configs.%s" % temp_module_name)
    job_cfg = config.config  #model-specific config
    cfg.update(job_cfg)
    if cfg.output is None:
        cfg.output = osp.join('work_dirs', temp_module_name) #default output path:work_dirs/temp_module_name
    return cfg