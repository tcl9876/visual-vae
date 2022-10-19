import matplotlib.pyplot as plt
import numpy as np
from tensorflow.io import write_file, gfile, read_file
from PIL import Image
from torchvision.utils import make_grid
import torch

#KL-related utils
def get_rate_schedule(rate_schedule, nlayers, renorm_value=66.7):
    schedule_name, schedule_args = rate_schedule
    if schedule_name == "shifted_exp":
        weights = np.exp(np.linspace(0, np.log(schedule_args["scale"]), sum(nlayers))) + schedule_args["shift"]
    elif schedule_name == "constant_per":
        assert len(nlayers) == len(schedule_args), f"the provided schedule has {len(schedule_args)} arguments but the model has only {len(nlayers)} different resolutions"
        weights = []
        for i in range(len(nlayers)):
            weights += [schedule_args[i]] * nlayers[i]
        weights = np.array(weights)

    weights = (weights / np.sum(weights) * renorm_value).astype('float32')
    return weights

def get_resolutions(max_res, num_res):
    resos = []
    current_res = max_res
    for i in range(num_res):
        if current_res<4: current_res = 1
        resos.append(current_res)
        current_res //= 2
    return resos

def get_evenly_spaced_indices(N, K):
    #N=16, K=1: [8]
    #N=16, K=5: [3, 5, 8, 11, 13]
    #N=16, K=0: []   
    if K==0: return []

    insert_every_n = N / (K + 1)
    indices = [round(insert_every_n * i) for i in range(1, K+1)]
    return indices


#Saving and logging related utils
def save_images(images, save_path, nrow=6, padding=2):
    images = torch.tensor(np.array(images))
    if images.shape[-1] == 3:
        images = images.permute(0, 3, 1, 2)
    
    with torch.no_grad():
        image_grid = make_grid(images, nrow=nrow, padding=padding, pad_value=255)
        image_grid = image_grid.permute(1, 2, 0).numpy().astype('uint8')
    
    Image.fromarray(image_grid).save(save_path)

#grabs images from the GPU/TPU, and then postprocesses & saves it.
def denormalize(ims, dtype='uint8'):
    return np.clip((ims+1)*127.5, 0, 255).astype(dtype)

#prints to the console and appends to the logfile at logfile_path
def print_and_log(*args, logfile_path):
    print(*args)
    for a in args:
        with gfile.GFile(logfile_path, mode='a') as f:
            f.write(str(a))

    with gfile.GFile(logfile_path, mode='a') as f:
        f.write('\n')

#a helper function that acts like plt.savefig() except supports GCS file system
def plt_savefig(figure_path):
    if not (figure_path.startswith("gs://") or figure_path.startswith("gcs://")):
        plt.savefig(figure_path)
        return
    
    plt.savefig("./tmp_figure.png")

    write_file(
        figure_path, read_file("./tmp_figure.png")
    )


#a utility to standardize dataset examples into a tuple of (img, label, img_lr)
#converts from either a dictionary or a list/tuple of 1-3 elements
def extract_train_inputs_fn(train_inputs, is_labeled, lower_res):
    if isinstance(train_inputs, (tuple, list)):
        if len(train_inputs) == 3:
            return train_inputs
        elif len(train_inputs) == 2 and len(train_inputs[1].shape) == 4:
            return (train_inputs[0], None, train_inputs[1])
        elif len(train_inputs) == 2:
            return (train_inputs[0], train_inputs[1], None)
        else:
            return (train_inputs[0], None, None)

    img = train_inputs["x"]
    if is_labeled:
        label = train_inputs["y"] if "y" in train_inputs.keys() else None
    else:
        label = None

    if lower_res:
        img_lr = train_inputs["x_lr"] if "x_lr" in train_inputs.keys() else None
    else:
        img_lr = None
    
    return img, label, img_lr