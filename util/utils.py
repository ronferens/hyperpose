import logging
import logging.config
import PIL
import json
from os.path import join, exists, split, realpath
import time
from os import makedirs, getcwd, listdir
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
import shutil
from typing import Dict
from fnmatch import fnmatch
import re


##########################
# Logging and output utils
##########################
def get_stamp_from_log() -> str:
    """
    Get the time stamp from the log file
    :return:
    """
    return split(logging.getLogger().handlers[0].baseFilename)[-1].replace(".log", "")


def create_output_dir(name: str) -> str:
    """
    Create a new directory for outputs, if it does not already exist
    :param name: (str) the name of the directory
    :return: the path to the output directory
    """
    if not exists(name):
        makedirs(name, exist_ok=True)
    return name


def init_logger(outpath: str = None, suffix: str = None) -> str:
    """
    Initialize the logger and create a time stamp for the file
    :param outpath: The output path to save the log file
    :param suffix:
    """
    path = split(realpath(__file__))[0]

    with open(join(path, 'log_config.json')) as json_file:
        log_config_dict = json.load(json_file)
        filename = log_config_dict.get('handlers').get('file_handler').get('filename')
        filename = ''.join([filename, "_", time.strftime("%y_%m_%d_%H_%M_%S", time.localtime())])

        # Creating logs' folder is needed
        if outpath is not None:
            log_path = create_output_dir(join(outpath, filename))
        else:
            log_path = create_output_dir(join(getcwd(), 'out', filename))

        if suffix is not None:
            filename += suffix
        log_config_dict.get('handlers').get('file_handler')['filename'] = join(log_path, f'{filename}.log')
        logging.config.dictConfig(log_config_dict)

        # disable external modules' loggers (level warning and below)
        logging.getLogger(PIL.__name__).setLevel(logging.WARNING)
        return log_path


def save_code_snapshot(fileprefix: str, path: str, modelname: str) -> None:
    """
    Saves a copy of the trained/tested model into the desired output path
    """
    shutil.make_archive(join(path, f'{fileprefix}_code_snapshot'), 'zip', join(getcwd(), f'models/{modelname.lower()}'))


def get_checkpoint_list(path: str, ckpt_start_index: int = 0) -> Dict:
    ckpts_list = {'ckpts': {}}
    for name in sorted(listdir(path)):
        if fnmatch(name, "*.pth"):
            m = re.match(".+_checkpoint-(\d+).pth", name)
            if m is not None:
                indx = int(m.group(1))
                if indx >= ckpt_start_index:
                    ckpts_list['ckpts'][indx] = join(path, name)
            elif 'final' in name:
                ckpts_list['final'] = join(path, name)

    return ckpts_list


##########################
# Evaluation utils
##########################
def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """
    Converts a quaternion (qx, qy, qz, qw) into a 3x3 rotation matrix.

    :param qx: x-component of the quaternion
    :param qy: y-component of the quaternion
    :param qz: z-component of the quaternion
    :param qw: w-component (scalar) of the quaternion
    :return: 3x3 NumPy rotation matrix
    """
    # Normalize quaternion to avoid numerical errors
    norm = np.sqrt(qx ** 2 + qy ** 2 + qz ** 2 + qw ** 2)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

    # Compute rotation matrix elements
    rot_mat = np.array([
        [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)]
    ])

    return rot_mat

def pose_err(est_pose, gt_pose):
    """
    Calculate the position and orientation error given the estimated and ground truth pose(s
    :param est_pose: (torch.Tensor) a batch of estimated poses (Nx7, N is the batch size)
    :param gt_pose: (torch.Tensor) a batch of ground-truth poses (Nx7, N is the batch size)
    :return: position error(s) and orientation errors(s)
    """
    posit_err = torch.norm(est_pose[:, 0:3] - gt_pose[:, 0:3], dim=1)
    est_pose_q = F.normalize(est_pose[:, 3:], p=2, dim=1)
    gt_pose_q = F.normalize(gt_pose[:, 3:], p=2, dim=1)
    inner_prod = torch.bmm(est_pose_q.view(est_pose_q.shape[0], 1, est_pose_q.shape[1]),
                           gt_pose_q.view(gt_pose_q.shape[0], gt_pose_q.shape[1], 1))
    orient_err = 2 * torch.acos(torch.abs(inner_prod)) * 180 / np.pi
    return posit_err, orient_err


##########################
# Plotting utils
##########################
def plot_loss_func(sample_count, loss_vals, loss_fig_path):
    plt.figure()
    plt.plot(sample_count, loss_vals)
    plt.grid()
    plt.title('Camera Pose Loss')
    plt.xlabel('Number of samples')
    plt.ylabel('Loss')
    plt.savefig(loss_fig_path)


##########################
# Transforms
##########################
# Augmentations
train_transforms = {
    'baseline': transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize(256),
                                    transforms.RandomCrop(224),
                                    transforms.ColorJitter(0.5, 0.5, 0.5, 0.2),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])]),

    'robotcar': transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize(256),
                                    transforms.RandomCrop(224),
                                    transforms.ColorJitter(0.7, 0.7, 0.7, 0.5),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: np.asarray(x))])

}
test_transforms = {
    'baseline': transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])
                                    ]),

    'robotcar': transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: np.asarray(x))])
}
