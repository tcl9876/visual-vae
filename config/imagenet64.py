from ml_collections import ConfigDict

def get_config():
    config = {}
    config["model"] =  ConfigDict({
        "resolution": 64,

        "c": 128, 
        "c_enc": 128,
        "c_mult": [1, 2], 
        "num_classes": 1000,

        #sorted in descending order of resolutions. e.g, nlayers[0] is the # of stochastic layers at the highest (64x64) resolution.
        #the upscaling factor in this case is 2**(len(nlayers) - 1).
        "nlayers": [20, 20], 
        "enc_nlayers": [14, 14], 
        "num_attention": [0,4],
        "num_final_blocks": 3,
        
        "is_superres": True,
        "superres_nlayers": [2,2,2],

        "zdim": 16
    })

    config["dataset"] = ConfigDict({
        "dataset_name": 'imagenet64',
        "data_dir": "{}",

        # Important: this is the *microbatch* size, (i.e. the batch size before accumulation, not after).
        # so reduce this number if you are running out of memory, but would like to keep a large effective batch size.
        "batch_size": 64,

        "sigma_aug": 0.1,
        "flip": True,
        
        "native_res": config["model"].resolution,
        "is_labeled": config["model"].num_classes > 0,
        "lower_res": config["model"].resolution//(2**(len(config["model"].nlayers) - 1)), #determines the base resolution for the model input. 
    })

    config["training"] = ConfigDict({
        "iterations": 400000,
        
        # if dataset.batch_size is 32, but training.total_batch_size is 128, 
        # then we accumulate grads over 4 different batches from the dataloader before updating params. 
        "total_batch_size": 192,

        "sigma_q": 0.025,
        "rate_schedule": ("constant_per", 
            [1, 2]
        ),

        "global_sr_weight": 0.25,
        "skip_threshold": 200,

        #Save checkpoints frequently for continuing after pre-emption, but keep another subdirectory for more permanent checkpoint keeping.
        #if you only want to save checkpoints to one path, just make sure 'num_checkpoints' and 'save_freq' are lists of length 1. More than 2 savedirs is not supported.
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
        "startlr": 3e-5,
        "maxlr": 3e-4,
        "decay": "cosine",
        "minlr": 3e-4,
        "warmup_steps": 50,
        "decay_steps": 300000,
        "ema_decay": 0.9998,
    })
    
    return ConfigDict(config)