import warnings
import cv2
import os

from scipy.spatial.transform import Rotation as scipy_R

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmcv.parallel import collate

from mmgs.datasets import build_dataset
from mmgs.models import build_simulator
from mmgs.datasets.utils.io import readPKL
from mmgs.datasets.utils.cameras import Camera


def init_model(config, checkpoint=None, device='cuda:0', options=None, force_forward=-1):
    """Initialize a classifier from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        options (dict): Options to override some settings in the used config.

    Returns:
        nn.Module: The constructed classifier.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if options is not None:
        config.merge_from_dict(options)
    config.model.pretrained = None
    config.model.force_forward = force_forward
    model = build_simulator(config.model)
    if checkpoint is not None:
        map_loc = 'cpu' if device == 'cpu' else None
        checkpoint = load_checkpoint(model, checkpoint, map_location=map_loc)

    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model

def inference_model(model):
    """Inference image(s) with the classifier.

    Args:
        model (nn.Module): The loaded classifier.
        img (str/ndarray): The image filename or loaded image.

    Returns:
        result (dict): The classification results that contains
            `class_name`, `pred_label` and `pred_score`.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    data, active_mask = data_template(cfg)
    active_mask = active_mask[data['inputs']['mov_mask']].reshape(-1, 1)
    data = data_collate([data], samples_per_gpu=1)
    assert next(model.parameters()).is_cuda

    return data, active_mask, device

def update_simulation(model, data, prev_state, cur_state, cur_cov=None, pred_frame_idx=1, zero_init=True):
    with torch.no_grad():
        result = model(
            return_loss=False, **data,
            prev_state=prev_state, cur_state=cur_state, cur_cov=cur_cov, pred_frame_idx=pred_frame_idx, zero_init=zero_init)
        img = result['pred_img_list'][0]
        img = np.array(img)
        img = img.transpose(1, 2, 0)
        img = np.clip((img * 255), 0, 255).astype(np.uint8)
        pred_frame_idx += 1
        prev_state = torch.from_numpy(np.array(result['cur_state']))
        cur_state = torch.from_numpy(np.array(result['pred_pos']))
    return img, prev_state, cur_state, pred_frame_idx

def data_template(cfg):
    dataset = build_dataset(cfg.data.test, default_args=dict(test_mode=True))
    data = dataset[0]
    # Only use one cam
    assert len(data['inputs']['cam']) == 1
    scene_name_list = cfg.data.test.env_cfg.scene_list
    assert len(scene_name_list) == 1
    active_mask = select_area(cfg.data.test.env_cfg.data_dir, scene_name_list[0])
    return data, active_mask

def select_area(data_dir, scene_name, active_mask_name='active_mask.pkl'):
    path = os.path.join(data_dir, scene_name, active_mask_name)
    mask = readPKL(path)
    return mask

def data_collate(batch, samples_per_gpu=1):
    batch_cam = batch[0]['inputs'].pop('cam')
    scene_name = batch[0]['meta'].pop('scene_name')
    batched_data = collate(batch, samples_per_gpu=samples_per_gpu)
    batched_data['inputs']['cam'] = batch_cam
    batched_data['meta']['scene_name'] = scene_name
    return batched_data

def update_cam(cur_cam, rotation, translation):
    '''
        rotation: [x, y, z]
            x: > 0, turn down
            y: > 0, turn left
            z: > 0, vertically anti clockwise
        translation: [x, y, z]
            x: > 0, move right
            y: > 0, move down
            z: > 0, move closer
    '''
    rot = scipy_R.from_euler('xyz', rotation, degrees=True).as_matrix()
    new_R = np.dot(rot, cur_cam.R.transpose(-1, -2)).transpose(-1,-2)
    new_tranl = cur_cam.trans + np.array(translation, dtype=np.float32)
    new_cam = Camera(
        new_R, cur_cam.T,
        cur_cam.FoVx, cur_cam.FoVy,
        cur_cam.img_path, cur_cam.static_img_path,
        new_tranl, cur_cam.scale, 
        cur_cam.data_device, (cur_cam.image_height, cur_cam.image_width), cur_cam.time_stamp)
    return new_cam

def update_force(pin_mask, active_mask, force, cur_state=None, dt=1/30):
    '''
    mouse move is velocity, force is velocity
    '''
    cur_state_updated = cur_state
    if cur_state is not None:
        p_mask = pin_mask[0]
        cur_state_updated = cur_state + (1-p_mask) * active_mask * (force * dt)

    return cur_state_updated
