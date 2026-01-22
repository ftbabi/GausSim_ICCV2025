import os.path as osp
import pickle
import shutil
import tempfile
import time
import numpy as np
import os
import cv2

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info
from mmgs.datasets.utils import writePKL, to_numpy_detach
from mmgs.datasets.utils.cameras import get_camera_trajectory
from scipy.spatial.transform import Rotation as scipy_R


def single_gpu_rollout_interact(model, data_loader, show=False, out_dir=None, **show_kwargs):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    duration = 30 # Pudding

    for i, data in enumerate(data_loader):
        # Show target predictions only
        seq_num = data['meta']['seq_idx']
        scene_name = data['meta']['scene_name']
        frame_num = data['gt_label'][0].shape[1] - 1 # Temporarily use this; 
        prev_state = None
        cur_state = None
        cur_cov = None

        if 'cam_00000' not in data['inputs']['cam'][0].img_path:
            continue

        # Customized forces for puddings
        pright_vec = np.array([-1,0,0])
        pup_vec = np.array([1,0,0])
        pleftup_vec = np.array([-1,0,-1])
        assert scene_name == 'pudding'
        scene_quant = dataset.env_cfg['rot_est'][scene_name]
        scene_rot = scipy_R.from_quat(scene_quant)
        pup_vec = scene_rot.inv().apply(pup_vec).astype(np.float32).reshape(1, 3) * 0.2
        pleftup_vec = scene_rot.inv().apply(pleftup_vec).astype(np.float32).reshape(1, 3) * 0.2
        pright_vec = scene_rot.inv().apply(pright_vec).astype(np.float32).reshape(1, 3) * 0.2
        force_template_data = [
            dict(type='custom', query_idx=4008, radius=0.25, decay_radius=0.10, data=pright_vec),
            dict(type='custom', query_idx=4008, radius=0.25, decay_radius=0.10, data=pup_vec),
            dict(type='custom', query_idx=4008, radius=0.25, decay_radius=0.10, data=pleftup_vec),
        ]
        inject_frame_idx_list = [
            10, 40, 70]
        interact_range = 100

        # Change Cameras
        in_cam = data['inputs']['cam']
        up_vec = np.array([0,0,-1]) # Pudding
        scene_quant = dataset.env_cfg['rot_est'][scene_name]
        scene_rot = scipy_R.from_quat(scene_quant)
        up_vec = scene_rot.inv().apply(up_vec).astype(np.float32)
        assert len(in_cam) == 1
        # Pudding
        cam_dict = dict(radius=2.0, focus_point=np.array([-0.66, 1.35, 2.13]), up=up_vec)
        tra_cam = get_camera_trajectory(in_cam[0], duration*2, camera_cfg=cam_dict)
        force_pointer = 0
        for frame_idx in range(interact_range):
            # Adjust the cam
            data['inputs']['cam'] = [tra_cam[frame_idx%(2*duration)]]
            zero_init = False
            if frame_idx+1 in inject_frame_idx_list and frame_idx not in inject_frame_idx_list:
                zero_init = True
            elif frame_idx == 0:
                zero_init = True
            elif frame_idx < 2:
                zero_init = True
            elif frame_idx in inject_frame_idx_list:
                # Inject
                assert prev_state is not None
                scalar = 1.0
                force_data = force_template_data[force_pointer]
                if force_data['type'] == 'prior':
                    prev_state = prev_state - scalar*force_data['data'].to(cur_state)
                else:
                    cur_state = dataset.inject_external_forces_tocur(cur_state, cur_state[force_data['query_idx']].reshape(1,3), force_data['radius'], force_data['decay_radius'], force_data['data'])
                force_pointer += 1
            with torch.no_grad():
                result = model(
                    return_loss=False, **data,
                    prev_state=prev_state, cur_state=cur_state, cur_cov=cur_cov, pred_frame_idx=min(frame_idx+1, frame_num-1), zero_init=zero_init)

            if show or out_dir:
                # Can use model.show_result() to show the image result
                rendered_img = result['pred_img_list']
                # Save image
                for img, cam in zip(rendered_img, data['inputs']['cam']):
                    img = np.array(img)
                    img = img.transpose(1, 2, 0)
                    cv2_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    cam_path = cam.img_path
                    seq_dir = osp.abspath(osp.join(out_dir, scene_name, f"seq_{str(seq_num.item()).zfill(5)}"))
                    mmcv.mkdir_or_exist(seq_dir)
                    save_path = os.path.join(seq_dir, f"pred_{str(frame_idx+1).zfill(5)}.png")
                    print(f"saving to {save_path}")
                    cv2.imwrite(
                        save_path,
                        255 * cv2_img,
                    )
        
            prev_state = torch.from_numpy(np.array(result['cur_state']))
            cur_state = torch.from_numpy(np.array(result['pred_pos']))

        batch_size = data['inputs']['img'][0].shape[0]
        for _ in range(batch_size):
            prog_bar.update()
    return results

def single_gpu_rollout(model, data_loader, show=False, out_dir=None, **show_kwargs):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    for i, data in enumerate(data_loader):
        # Show target predictions only
        seq_num = data['meta']['seq_idx']
        scene_name = data['meta']['scene_name']
        eval_start_frame = data['meta']['eval_start_frame']
        frame_num = data['gt_label'][0].shape[1] - 1 # Temporarily use this; 
        assert frame_num > 0 # -1 cuz 0 is for static input, not predict
        assert eval_start_frame < frame_num
        prev_state = None
        cur_state = None
        cur_cov = None

        for frame_idx in range(frame_num):
        # for frame_idx in range(3):
            with torch.no_grad():
                result = model(
                    return_loss=False, **data,
                    prev_state=prev_state, cur_state=cur_state, cur_cov=cur_cov, pred_frame_idx=frame_idx+1)
            
            if frame_idx >= eval_start_frame:
                results.append({
                    'rollout_idx': to_numpy_detach(seq_num)[0],
                    'acc': result['acc'],
                    'scene_name': data['meta']['scene_name'],
                })

            if show or out_dir:
                rendered_img = result['pred_img_list']
                if 'gt_label' in data.keys():
                    data_gt_label = data['gt_label']
                else:
                    data_gt_label = [None for _ in range(len(rendered_img))]
                # Save image
                for img, cam, gt_img_list in zip(rendered_img, data['inputs']['cam'], data_gt_label):
                    img = np.array(img)
                    img = img.transpose(1, 2, 0)
                    cv2_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    cam_path = cam.img_path
                    cam_id = os.path.splitext(os.path.basename(cam_path))[0].split('_')[-1]
                    seq_dir = osp.abspath(osp.join(out_dir, scene_name, f"seq_{str(seq_num.item()).zfill(5)}_cam_{cam_id}"))
                    mmcv.mkdir_or_exist(seq_dir)
                    save_path = os.path.join(seq_dir, f"pred_{str(frame_idx+1).zfill(5)}.png")
                    
                    cv2.imwrite(
                        save_path,
                        255 * cv2_img,
                    )

                    # Save gt
                    if gt_img_list is not None:
                        gt_img = np.array(gt_img_list[0][frame_idx+1])
                        gt_img = gt_img.transpose(1, 2, 0)
                        cv2_gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
                        gt_save_path = os.path.join(seq_dir, f"gt_{str(frame_idx+1).zfill(5)}.png")
                        cv2.imwrite(
                            gt_save_path,
                            255 * cv2_gt_img,
                        )
        
            prev_state = torch.from_numpy(np.array(result['cur_state']))
            cur_state = torch.from_numpy(np.array(result['pred_pos']))

        batch_size = data['inputs']['img'][0].shape[0]
        for _ in range(batch_size):
            prog_bar.update()

    return results

def single_gpu_test(model, data_loader, show=False, out_dir=None, **show_kwargs):
    if data_loader.dataset.env_cfg.get('interact', False):
        return single_gpu_rollout_interact(model, data_loader, show=show, out_dir=out_dir, **show_kwargs)
    # Must use rollout
    return single_gpu_rollout(model, data_loader, show=show, out_dir=out_dir, **show_kwargs)

def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        # Check if tmpdir is valid for cpu_collect
        if (not gpu_collect) and (tmpdir is not None and osp.exists(tmpdir)):
            raise OSError((f'The tmpdir {tmpdir} already exists.',
                           ' Since tmpdir will be deleted after testing,',
                           ' please make sure you specify an empty one.'))
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)
    dist.barrier()
    for i, data in enumerate(data_loader):
        seq_num = data['meta']['seq_idx']

        with torch.no_grad():
            result = model(return_loss=False, **data)
            
        results.append({
            'rollout_idx': to_numpy_detach(seq_num)[0],
            'acc': result['acc'],
            'scene_name': data['meta']['scene_name'],
        })

        if rank == 0:
            batch_size = data['inputs']['img'][0].shape[0]

            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_result = mmcv.load(part_file)
            part_list.append(part_result)
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_result = pickle.loads(recv[:shape[0]].cpu().numpy().tobytes())
            part_list.append(part_result)
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results