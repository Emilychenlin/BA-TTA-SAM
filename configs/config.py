from box import Box
from configs.base_config import base_config


config = {
    "gpu_ids": "1",
    "batch_size": 1,
    "val_batchsize":1,
    "num_workers": 4,
    "num_epochs": 100,    
    "max_nums": 40,
    "num_points": 3,
    "eval_interval": 1,
    "dataset": "ISIC",
    "prompt": "box",
    "out_dir": "output/attention_visual/ISIC",
    "name": "baseline",
    "augment": True, 
    "corrupt": None,
    "visual": False,
    "opt": {
        "learning_rate": 1e-4,
    },
    "model": {
        "type": "vit_b",
    },
}

cfg = Box(base_config)
cfg.merge_update(config)