from ml_collections import ConfigDict

def get_config():
    config = {}

    config["model"] = ConfigDict({
        "resolution": 32,

        "c": 128, 
        "c_enc": 128,
        "c_mult": [1,2,2,2,2], 
        "num_classes": 1000,

        #sorted in descending order of resolutions. e.g, nlayers[0] is the # of stochastic layers at the highest (32x32) resolution.
        "nlayers": [15,16,10,8,8],
        "enc_nlayers": [11,12,9,7,6],
        "num_attention": [3,3,0,0,0],
        "num_final_blocks": 3,

        "is_superres": False,
        "superres_nlayers": None,

        "zdim": 16,
    })

    config["dataset"] = ConfigDict({
        "dataset_name": 'imagenet32',
        "data_dir": "{}/data",

        # Important: this is the *microbatch* size, (i.e. the batch size before accumulation, not after).
        # so reduce this number if you are running out of memory, but would like to keep a large effective batch size.
        "batch_size": 256,

        "sigma_aug": None,
        "flip": False,
        
        "native_res": config["model"].resolution,
        "is_labeled": config["model"].num_classes > 0,
        "lower_res": None, 
    })

    config["training"] = ConfigDict({
        "iterations": 1200000,

        # if dataset.batch_size is 32, but training.total_batch_size is 128, 
        # then we accumulate grads over 4 different batches from the dataloader before updating params.
        "total_batch_size": 256,

        "sigma_q": 0.025,
        "rate_schedule": ("shifted_exp", 
            {"scale": 1000, "shift": 10}
        ),

        "global_sr_weight": 0.1,
        "skip_threshold": 300,

        #Save checkpoints frequently for continuing after pre-emption, but keep another subdirectory for more permanent checkpoint keeping.
        "checkpoint_dirs": ["{}/checkpoints_recent", "{}/checkpoints_permanent"],  
        "num_checkpoints": [10, 999999],
        "save_freq": [2500, 50000],

        "log_dir": "{}/logs",
        "log_freq": 500
    })

    config["optimizer"] = ConfigDict({
        "opt_type": "adamax",
        "beta1": 0.9,
        "beta2": 0.999,
        "startlr": 4e-5,
        "maxlr": 4e-4,
        "decay": "cosine",
        "minlr": 2e-4,
        "warmup_steps": 50,
        "decay_steps": 800000,
        "ema_decay": 0.9999,
    })

    return ConfigDict(config)
