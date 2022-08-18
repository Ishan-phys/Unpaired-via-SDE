import torch

CFGS = {
    "training":{
        "batch_size":1,
        "n_iters":1001,
        "sde":"vpsde",
        "continuous":True,
        "reduce_mean":True,
        "check_pt_freq":20,
        "ckpt_dir":"./checkpoints/",
    },
    "sampling":{
        "n_steps":10,
        "sample_dir":"./datasets/samples",
    },
    "model":{
        "in_channel":6,
        "out_channel":3,
        "inner_channel":64,
        "channel_mults":[1,2,4,8],
        "attn_res":[16],
        "num_head_channels":32,
        "res_blocks":2,
        "dropout":0.2,
        "image_size":64,
    },
    "optim":{
        "weight_decay":0,
        "optimizer":'Adam',
        "lr":1e-4,
        "beta1":0.9,
        "eps":1e-8,
        "warmup":250,
        "grad_clip":1,
        "seed":42,
    },
    
    "device": 'cuda' if torch.cuda.is_available() else 'cpu',
    
    "Cycle_loss": 10,
    
    "ema": {
        "ema_rate":0.9999,
    },
    
}