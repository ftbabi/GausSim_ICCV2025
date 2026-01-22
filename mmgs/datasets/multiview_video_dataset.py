import os
import sys
from collections import defaultdict
import copy
import plotly.express as px

from scipy.spatial.transform import Rotation as scipy_R
import point_cloud_utils as pcu
import numpy as np
import torch
from torch.utils.data import Dataset
from mmgs.datasets.utils.cameras import Camera, focal2fov, fov2focal
from mmgs.datasets.utils.io import readPKL, writePKL, readJSON, read_video_image_cv2, read_video_image_rgba_cv2
from mmgs.utils.colmap_utils import (
    qvec2rotmat,
    read_extrinsics_binary,
    read_intrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_text,
    read_points3D_binary,
    read_points3D_text,
)
from mmgs.utils.colmap_utils import Image as colmap_Image
from mmgs.utils.physdreamer_utils import find_far_points
import json
from PIL import Image
from .builder import DATASETS
from mmcv.parallel import collate
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import NearestCentroid
from sklearn.metrics.pairwise import euclidean_distances

import sys
sys.path.append('gaussian-splatting')
from scene.gaussian_model import GaussianModel


def read_uint8_rgba(img_path, img_hw=None):
    if not (img_path.endswith(".png") or img_path.upper().endswith(".JPG")):
        img_path = img_path + ".png"

    with Image.open(img_path) as image:
        if img_hw is not None:
            image = image.resize((img_hw[1], img_hw[0]), Image.BILINEAR)
        im_data = np.array(image.convert("RGBA"))
    return im_data

def extract_masters(cam_meta):
    unique_mapping = cam_meta['twin']
    slave_dict = defaultdict(set)
    for t_list in unique_mapping:
        slave_dict[t_list[0]].add(t_list[1])
    return slave_dict

def expand_cameras(cam_meta, cam_extrinsics):
    master_dict = extract_masters(cam_meta)

    if len(master_dict) <= 0:
        return cam_extrinsics, None
    invalid_names = cam_meta['invalid'] + cam_meta['force_invalid']
    exp_cam_extrinsics = dict()
    exp_real_image_path = dict()
    extra_pointer = max(cam_extrinsics.keys()) + 1
    for idx, key in enumerate(cam_extrinsics):
        exp_cam_extrinsics[key] = cam_extrinsics[key] # COPY
        extr = cam_extrinsics[key]
        image_name = extr.name
        image_base_name = os.path.basename(image_name)
        assert image_base_name == image_name
        seq_name = os.path.splitext(image_base_name)[0]
        seq_idx = int(seq_name.split('_')[1])
        cam_idx = int(seq_name.split('_')[3])
        if seq_idx not in master_dict.keys():
            continue
        for slave_seq_idx in master_dict[seq_idx]:
            slave_seq_name = f"seq_{str(slave_seq_idx).zfill(5)}_cam_{str(cam_idx).zfill(5)}"
            if slave_seq_name in invalid_names:
                continue
            # Expand the slaves
            ## Change the image name
            assert len(os.path.splitext(image_base_name)) == 2
            slave_image_name = slave_seq_name + os.path.splitext(image_base_name)[1]
            exp_real_image_path[extra_pointer] = image_name
            slave_extr_tmp = copy.deepcopy(extr)
            slave_extr = colmap_Image(
                    id=slave_extr_tmp.id, qvec=slave_extr_tmp.qvec, tvec=slave_extr_tmp.tvec,
                    camera_id=slave_extr_tmp.camera_id, name=slave_image_name,
                    xys=slave_extr_tmp.xys, point3D_ids=slave_extr_tmp.point3D_ids)
            exp_cam_extrinsics[extra_pointer] = slave_extr
            extra_pointer += 1
    
    return exp_cam_extrinsics, exp_real_image_path

def readColmapCameras(
    cam_extrinsics, cam_intrinsics, images_folder, videos_folder, img_hw=None, scale_x_angle=1.0, suffix_replace=None, max_cam_total=-1, max_seq=-1, exp_real_image_path=None, invalid_name=None, split=None):

    camera_dict = defaultdict(list)
    seq_info_dict = defaultdict(list)
    
    print(f"Try to read camera number with at most {len(cam_extrinsics)}")
    for idx, key in enumerate(cam_extrinsics):
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        image_base_name = os.path.basename(extr.name)
        image_path = os.path.join(images_folder, image_base_name)
        if suffix_replace is not None:
            assert len(suffix_replace) == 2
            image_path = image_path.replace(suffix_replace[0], suffix_replace[1])
        # Filter out splits
        assert image_base_name.startswith("seq")
        # seq_xxx_frame_xxx.png
        seq_name = '_'.join(image_base_name.split('_')[0:2])
        seq_idx = int(image_base_name.split('_')[1])
        seq_dir = os.path.splitext(image_base_name)[0]
        video_path = os.path.join(videos_folder, seq_dir)
        if split is not None and seq_dir not in split:
            continue
        if invalid_name is not None and seq_dir in invalid_name:
            continue
        if not os.path.exists(video_path):
            # Including some extra camera for reconstruction
            print(f"Warning: omit invalid camera: {video_path}")
            continue
        if max_seq > 0 and seq_idx >= max_seq:
            # assert False
            continue
        existing_cam_num = np.sum([len(v) for v in seq_info_dict.values()])
        if max_cam_total > 0 and existing_cam_num >= max_cam_total: # Here, num camera
            assert False
            continue

        height = intr.height
        width = intr.width
        if img_hw is not None:
            # keep FovX not changed, change aspect ratio accrodingly
            height = int(img_hw[0] / img_hw[1] * width)
        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            if scale_x_angle != 1.0:
                focal_length_x = focal_length_x * scale_x_angle
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            if scale_x_angle != 1.0:
                focal_length_x = focal_length_x * scale_x_angle
                focal_length_y = focal_length_y * scale_x_angle
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert (
                False
            ), "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        if img_hw is not None:
            height, width = img_hw

        static_img_path = image_path
        real_image_path = image_path
        if exp_real_image_path is not None and not os.path.exists(image_path):
            assert key in exp_real_image_path.keys()
            replace_image = exp_real_image_path[key]
            if suffix_replace is not None:
                assert len(suffix_replace) == 2
                replace_image = replace_image.replace(suffix_replace[0], suffix_replace[1])
            static_img_path = os.path.join(images_folder, replace_image)
        seq_info_dict[seq_name].append(real_image_path)
        # >>>>>>
        camera_dict[seq_name].append(
                Camera(
                R=R,
                T=T,
                FoVy=FovY,
                FoVx=FovX,
                img_path=real_image_path,
                static_img_path=static_img_path,
                img_hw=(height, width),
                timestamp=None,
                data_device="cuda",
            )
        )
        
    # sys.stdout.write("\n")
    return camera_dict, seq_info_dict


@DATASETS.register_module()
class MultiviewVideoDataset(Dataset):
    def __init__(
        self,
        env_cfg, phase, **kwargs) -> None:
        super(MultiviewVideoDataset, self).__init__()

        self.env_cfg = env_cfg
        self.phase = phase

        self.gravity = np.array(env_cfg['gravity'], dtype=np.float32).reshape(1, 3)
        self.real_dt = env_cfg.get('real_dt', None)
        self.comp_dt = env_cfg.get('dt', 1/30)
        # Parse attributes
        self.data_dir = env_cfg['data_dir']
        self.sub_video_dir = env_cfg['sub_video_dir']
        self.sub_img_dir = env_cfg['sub_img_dir']
        self.max_cam_num = env_cfg['max_cam_num']
        self.max_cam_total = env_cfg['max_cam_total']
        self.max_seq = env_cfg.get('max_seq', -1)
        self.llffhold = env_cfg.get('llffhold', 0)
        self.use_random_background = env_cfg.get('use_random_background', None)
        self.const_white_bg = env_cfg.get('const_white_bg', None)
        
        self.scene_list = env_cfg.get('scene_list', None) # if None, use all scene
        if self.scene_list is None:
            self.scene_list = []
            for scene_dir in os.listdir(self.data_dir):
                if os.path.isdir(os.path.join(self.data_dir, scene_dir)):
                    self.scene_list.append(scene_dir)
        else:
            assert isinstance(self.scene_list, list)
        # Parse attributes
        assert len(self.scene_list) > 0
        self.cam_transform_fn = env_cfg.get('cam_transform_fn', f'transforms_{phase}.json')
        self.resolution = env_cfg['resolution']
        self.scale_x_angle = env_cfg['scale_x_angle']
        self.load_imgs = env_cfg['load_imgs']
        self.load_pcs = env_cfg.get('load_pcs', True)
        self.meta_dict_sclist, self.camera_list_sclist, self.np_uint8_rgba_list_sclist = self._parse_dataset(self.data_dir, self.scene_list, self.sub_img_dir, self.sub_video_dir, self.cam_transform_fn, load_imgs=self.load_imgs, max_cam_total=self.max_cam_total, max_seq=self.max_seq, llffhold=self.llffhold)
        
        self._num_frames = sum(meta_dict['num_frames'] for meta_dict in self.meta_dict_sclist)
        self.num_cameras = sum(meta_dict["num_cameras"] for meta_dict in self.meta_dict_sclist)
        assert self._num_frames == self.num_cameras
        self.idx_mapping = self._parse_idx(self.camera_list_sclist, self.max_cam_num, padding=env_cfg['pad_cam'], phase=self.phase)
        self.dataset_len = len(self.idx_mapping)
        
        # Pre load the point clouds or Put placeholder for further cached
        self.pcmask_sclist, self.pinmask_sclist, self.cln_pcmask_sclist = self._load_pointcloud_mask(self.data_dir, self.meta_dict_sclist, pcmask_name=env_cfg.get('pcmask_name', 'pc_mask.pkl'), cln_pcmask_name=env_cfg.get('cln_pcmask_name', 'cln_pc_mask.pkl'), pinmask_name=env_cfg.get('pinmask_name', 'pin_mask.json'), dup=env_cfg.get('dup', None))
        cluster_type = env_cfg.get('cluster_type', '')
        self.clustermask_sclist = self._load_cluster_mask(self.data_dir, self.meta_dict_sclist, env_cfg['cluster_cfg'], cluster_type=cluster_type, dup=env_cfg.get('dup', None))


    def _parse_idx(self, camera_list_sclist, max_cam_num, padding=True, phase='train'):
        # The dataset length is the number of camera
        idx_mapping = []
        for sc_idx, camera_list in enumerate(camera_list_sclist):
            for seq_idx, cam_list in camera_list.items():
                cam_idx_list = [_ for _ in range(len(cam_list))]
                cam_idx_list = np.array(cam_idx_list)
                if phase == 'train' or phase == 'all':
                    # Randomize
                    cam_idx_list = np.random.choice(cam_idx_list, size=cam_idx_list.shape[0], replace=False)
                for i in range(0, cam_idx_list.shape[0], max_cam_num):
                    cur_batch_idx = [i+_ for _ in range(max_cam_num) if i+_ < len(cam_idx_list)]
                    cur_batch = [cam_idx_list[i] for i in cur_batch_idx]
                    idx_mapping.append([sc_idx, seq_idx, cur_batch])
        return idx_mapping

    def __len__(self):
        return self.dataset_len
    
    def _parse_dataset(self, data_dir, scene_list, image_dir, video_dir, cam_transform_fn, load_imgs=False, max_cam_total=-1, max_seq=-1, llffhold=0):
        meta_dict_sclist, camera_dict_sclist, np_uint8_rgba_dict_sclist = [], [], []
        for scene_n in scene_list:
            scene_data_dir = os.path.join(data_dir, scene_n)
            assert os.path.exists(scene_data_dir)
            
            if cam_transform_fn is not None:
                camera_transform_file = os.path.join(scene_data_dir, cam_transform_fn)

            if cam_transform_fn is not None and os.path.exists(camera_transform_file):
                print(f"=> loading {scene_n} camera from blender format {cam_transform_fn}")
                camera_dict, meta_dict = self._read_camera_transforms(
                    scene_data_dir, camera_transform_file, img_hw=self.resolution, max_cam_total=max_cam_total, max_seq=max_seq,
                )
            else:
                assert os.path.exists(
                    os.path.join(scene_data_dir, "sparse/0")
                ), "colmap sparse folder not found!"
                print(f"=> loading {scene_n} camera from colmap format")
                cam_meta = readJSON(os.path.join(scene_data_dir, 'meta.json'))
                camera_dict, meta_dict = self._read_camera_transforms_colmap(
                    scene_data_dir, cam_meta, image_dir=image_dir, video_dir=video_dir, img_hw=self.resolution, max_cam_total=max_cam_total, max_seq=max_seq, eval=False, llffhold=llffhold,
                )
        
            assert "num_frames" in meta_dict

            np_uint8_rgba_dict = self._load_imgs_info(camera_dict)
            if load_imgs:
                data_np_uint8_rgba_dict = dict()
                for seq_name, rgba_list in np_uint8_rgba_dict.items():
                    rgba_data_list = self._load_imgs(rgba_list)
                    data_np_uint8_rgba_dict[seq_name] = rgba_data_list
                np_uint8_rgba_dict = data_np_uint8_rgba_dict

            if 'scene_name' not in meta_dict.keys():
                meta_dict['scene_name'] = scene_n
            meta_dict_sclist.append(meta_dict)
            camera_dict_sclist.append(camera_dict)
            np_uint8_rgba_dict_sclist.append(np_uint8_rgba_dict)

        return meta_dict_sclist, camera_dict_sclist, np_uint8_rgba_dict_sclist
        
    def _load_imgs_info(self, camera_dict):
        np_uint8_rgba_path_dict = dict()
        print("Loading images' information...")
        for seq_name, cam_list in camera_dict.items():
            rgba_path_list = [cam.img_path for cam in cam_list]
            np_uint8_rgba_path_dict[seq_name] = rgba_path_list
        return np_uint8_rgba_path_dict

    def __len__(self):
        return self.dataset_len

    @property
    def num_frames(self):
        return self._num_frames

    def _load_cluster_mask(self, data_dir, meta_dict_sclist, cluster_cfg, clustermask_name='cluster_mask.pkl', sc_pc_name='point_cloud.ply', cluster_type='', dup=None):
        clustermask_sclist = []
        aux_pc_dict = dict()
        for scene_idx, scene_meta in enumerate(meta_dict_sclist):
            scene_cluster = []
            scene_n = scene_meta['scene_name']
            if isinstance(cluster_cfg, dict):
                assert scene_n in cluster_cfg.keys()
                scene_cluster_cfg = cluster_cfg[scene_n]
            else:
                scene_cluster_cfg = cluster_cfg
            assert isinstance(scene_cluster_cfg, list)
            num_level = len(scene_cluster_cfg)
            unique_id = [f"nl_{num_level}"]
            # Get unique key
            for i in range(num_level):
                cluster_idx = i
                cluster_meta = scene_cluster_cfg[cluster_idx]
                path_key = ''
                if 'downsample_rate' in cluster_meta.keys():
                    digit_info = str(cluster_meta['downsample_rate']).replace('.', 'd')
                    path_key = f'dr_{digit_info}'
                else:
                    assert 'num_cluster' in cluster_meta.keys()
                    digit_info = str(cluster_meta['num_cluster']).replace('.', 'd')
                    path_key = f'nc_{digit_info}'
                unique_id.append(path_key)
            unique_id = '_'.join(unique_id)
            mask_path = os.path.join(data_dir, scene_n, f'{cluster_type}{unique_id}_{clustermask_name}')
            if not os.path.exists(mask_path):
                print(f"Generating cluster mapping to {mask_path}")
                prev_clustered_pcs = None
                cluster_mask_list = []
                for i in range(num_level):
                    cluster_idx = i
                    cluster_meta = scene_cluster_cfg[cluster_idx]
                    # path_key = ''
                    # Generate here
                    if i == 0:
                        if scene_n not in aux_pc_dict.keys():
                            mov_pcs = self.pcmask_sclist[scene_idx]
                            sc_pcs_path = os.path.join(data_dir, scene_n, sc_pc_name)
                            scene_gaussian = GaussianModel(3)
                            print(f"Initializing gaussian scene: {sc_pcs_path}")
                            scene_gaussian.load_ply(sc_pcs_path)
                            aux_pc_dict[scene_n] = scene_gaussian.get_xyz[mov_pcs].detach().cpu().numpy()
                        cur_pcs = aux_pc_dict[scene_n]
                    else:
                        cur_pcs = prev_clustered_pcs
                    # Cluster
                    if cluster_type == '':
                        if 'downsample_rate' in cluster_meta.keys():
                            num_cluster = int(cluster_meta['downsample_rate'] * cur_pcs.shape[0])
                        else:
                            num_cluster = cluster_meta['num_cluster']
                        num_cluster = max(num_cluster, 1)
                        kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(cur_pcs)
                        new_pcs = kmeans.cluster_centers_
                        pc_labels = kmeans.labels_
                    elif cluster_type == 'dis':
                        assert 'downsample_rate' in cluster_meta.keys(), "downsample_rate now serve as maximum distance"
                        downsample_dis = cluster_meta['downsample_rate']
                        cluster_handler = AgglomerativeClustering(n_clusters=None, linkage='complete', distance_threshold=downsample_dis).fit(cur_pcs)
                        pc_labels = cluster_handler.labels_
                        clf = NearestCentroid()
                        clf.fit(cur_pcs, pc_labels)
                        new_pcs = clf.centroids_
                    else:
                        assert False, "Invalid cluster type"
                    cluster_mask_list.append(pc_labels)
                    prev_clustered_pcs = new_pcs
                # Add one more for the cluster to closest pinned verts
                pinned_state = aux_pc_dict[scene_n][self.pinmask_sclist[scene_idx][:, 0]]
                dist = euclidean_distances(prev_clustered_pcs, pinned_state)
                p2pin_mapping = np.argmin(dist, axis=-1)
                cluster_mask_list.append(p2pin_mapping)
                print(f"Saving cluster mapping list to {mask_path}")
                writePKL(mask_path, dict(p2c=cluster_mask_list))
            scene_cluster = readPKL(mask_path)['p2c']
            clustermask_sclist.append(scene_cluster)
        
        if dup is not None:
            dup_clustermask_sclist = []
            for scene_cluster in clustermask_sclist:
                dup_scene_cluster = []
                to_repeat = int(dup)
                for hie_lv in range(len(scene_cluster)):
                    base_cluster = scene_cluster[hie_lv]
                    offset = np.max(base_cluster) + 1
                    if hie_lv == len(scene_cluster) - 1:
                        # Dont need to move offset
                        offset = 0
                    dup_list = [base_cluster + offset * (i+1) for i in range(to_repeat)]
                    dup_base_cluster = np.concatenate([base_cluster]+dup_list)
                    dup_scene_cluster.append(dup_base_cluster)
                dup_clustermask_sclist.append(dup_scene_cluster)
            clustermask_sclist = dup_clustermask_sclist
        print(f"Total kernels: {clustermask_sclist[0][0].shape[0]}\tTotal CMS: {clustermask_sclist[0][-1].shape[0]}")
        return clustermask_sclist


    def _load_pointcloud_mask(self, data_dir, meta_dict_sclist, pcmask_name='pc_mask.pkl', sc_pc_name='point_cloud.ply', mov_sc_pc_name='moving_part_points.ply', pinmask_name='pin_mask.json', cln_pcmask_name='cln_pc_mask.pkl', cln_sc_pc_name='clean_object_points.ply', dup=None):
        pcmask_sclist = []
        pinmask_sclist = []
        cln_pcmask_sclist = []
        for scene_meta in meta_dict_sclist:
            scene_n = scene_meta['scene_name']
            mask_path = os.path.join(data_dir, scene_n, pcmask_name)
            if not os.path.exists(mask_path):
                # Generate separately
                sc_pcs_path = os.path.join(data_dir, scene_n, sc_pc_name)
                mov_sc_pcs_path = os.path.join(data_dir, scene_n, mov_sc_pc_name)
                sc_pcs = pcu.load_mesh_v(sc_pcs_path)
                mv_raw_pcs = pcu.load_mesh_v(mov_sc_pcs_path)
                # Get the moving point mask
                not_sim_maks = find_far_points(torch.from_numpy(sc_pcs).float(), torch.from_numpy(mv_raw_pcs).float(), thres=0.01).bool()
                sim_mask_in_raw_gaussian = torch.logical_not(not_sim_maks)
                writePKL(mask_path, sim_mask_in_raw_gaussian)
            pcmask = readPKL(mask_path)
            pcmask_sclist.append(pcmask)
            # TODO: modify the pinmask
            pin_mask_idx_path = os.path.join(data_dir, scene_n, pinmask_name)
            assert os.path.exists(pin_mask_idx_path)
            pin_mask_idx = readJSON(pin_mask_idx_path)
            pin_mask = torch.zeros((torch.sum(pcmask), 1)).bool()
            for pidx in pin_mask_idx:
                pin_mask[pidx, 0] = True
            pinmask_sclist.append(pin_mask)
            cln_mask_path = os.path.join(data_dir, scene_n, cln_pcmask_name)
            if not os.path.exists(cln_mask_path):
                # Generate separately
                sc_pcs_path = os.path.join(data_dir, scene_n, sc_pc_name)
                cln_sc_pcs_path = os.path.join(data_dir, scene_n, cln_sc_pc_name)
                sc_pcs = pcu.load_mesh_v(sc_pcs_path)
                cln_raw_pcs = pcu.load_mesh_v(cln_sc_pcs_path)
                # Get the moving point mask
                not_cln_maks = find_far_points(torch.from_numpy(sc_pcs).float(), torch.from_numpy(cln_raw_pcs).float(), thres=0.01).bool()
                cln_mask_in_raw_gaussian = torch.logical_not(not_cln_maks)
                select_mask = torch.logical_or(cln_mask_in_raw_gaussian, pcmask)
                cln_mask_in_mov_gaussian = pcmask[select_mask]
                assert cln_mask_in_raw_gaussian.shape[0] == pcmask.shape[0]
                assert cln_mask_in_mov_gaussian.sum() == pcmask.sum()
                writePKL(cln_mask_path, select_mask) # Save select mask, need to filter out mov during run time
            cln_pcmask = readPKL(cln_mask_path)
            cln_pcmask_sclist.append(cln_pcmask)

        if dup is not None:
            dup_pcmask_sclist, dup_pinmask_sclist, dup_cln_pcmask_sclist = [], [], []
            for pcmask, pinmask, cln_pcmask in zip(pcmask_sclist, pinmask_sclist, cln_pcmask_sclist):
                total_num = pcmask.shape[0]
                to_repeat = int(dup)
                dup_pc_mask = pcmask
                dup_pinmask = pinmask
                dup_cln_pcmask = cln_pcmask
                if to_repeat > 0:
                    dup_pc_mask = torch.cat([dup_pc_mask]+[pcmask for _ in range(to_repeat)])
                    dup_cln_pcmask = torch.cat([dup_cln_pcmask]+[cln_pcmask for _ in range(to_repeat)])
                    dup_pinmask = torch.cat([dup_pinmask]+[torch.zeros_like(pinmask).to(pinmask) for _ in range(to_repeat)])
                dup_pcmask_sclist.append(dup_pc_mask)
                dup_pinmask_sclist.append(dup_pinmask)
                dup_cln_pcmask_sclist.append(dup_cln_pcmask)
            pcmask_sclist = dup_pcmask_sclist
            pinmask_sclist = dup_pinmask_sclist
            cln_pcmask_sclist = dup_cln_pcmask_sclist

        return pcmask_sclist, pinmask_sclist, cln_pcmask_sclist

    def __getitem__(self, idx):
        scene_idx, seq_name, cam_list_idx = self.idx_mapping[idx]
        seq_idx = int(seq_name.replace('seq_', '')) # This one start from 0, the seq_name start from 0
        assert seq_idx >= 0
        # Dynamic padding
        if self.env_cfg['pad_cam']:
            extra_cam = len(cam_list_idx) % self.max_cam_num
            num_pad = 0
            if extra_cam > 0:
                num_pad = self.max_cam_num - extra_cam
            if num_pad > 0:
                if self.phase == 'train' or self.phase == 'all':
                    replace = False
                    if num_pad > len(cam_list_idx):
                        replace = True
                    pad_candidate = np.random.choice(cam_list_idx, size=num_pad, replace=replace).tolist()
                else:
                    pad_candidate = [cam_list_idx[-1] for _ in range(num_pad)]
                for pcandidate in pad_candidate:
                    cam_list_idx.append(pcandidate)
        cam_list = [self.camera_list_sclist[scene_idx][seq_name][i] for i in cam_list_idx]
        rgba_list = [self.np_uint8_rgba_list_sclist[scene_idx][seq_name][i] for i in cam_list_idx]

        img_list, video_list = [], []
        white_bg_list = []
        for cam, rgba in zip(cam_list, rgba_list):
            if '_mov' in self.sub_img_dir and self.sub_img_dir not in cam.img_path:
                cam.img_path = cam.img_path.replace('images', self.sub_img_dir)
            if '_mov' in self.sub_img_dir and self.sub_img_dir not in cam.static_img_path:
                cam.static_img_path = cam.static_img_path.replace('images', self.sub_img_dir)
            img_path = cam.img_path
            static_img_path = cam.static_img_path
            # Load rest states
            if not self.load_imgs:
                # Lazy load images here
                rgba = read_uint8_rgba(static_img_path, self.resolution)
                if self.resolution is None:
                    self.resolution = rgba.shape[:2]
            assert self.sub_img_dir in img_path
            video_path = img_path.replace(self.sub_img_dir, self.sub_video_dir)

            scene_use_random_background = self.use_random_background[self.meta_dict_sclist[scene_idx]['scene_name']] if self.use_random_background is not None else False
            scene_const_white_bg = self.const_white_bg[self.meta_dict_sclist[scene_idx]['scene_name']] if self.const_white_bg is not None else False
            assert (not scene_use_random_background and not scene_const_white_bg) or (scene_use_random_background ^ scene_const_white_bg)
            white_bg = np.random.rand() > 0.5 and scene_use_random_background
            if scene_const_white_bg:
                white_bg = True
            white_bg_list.append(white_bg)
            # [nf, 3, H, W], RGB, [0, 255]
            video_dir = os.path.splitext(video_path)[0]
            start_frame = self.env_cfg.get('start_frame', 0)
            assert start_frame >= 0
            video_reader = read_video_image_cv2
            if '_mov' in self.sub_img_dir:
                video_reader = read_video_image_rgba_cv2
            video_range = [0] + [_ for _ in range(start_frame+1, start_frame+self.env_cfg.get('max_frame', 2))]
            video_clip = torch.from_numpy(
                video_reader(video_dir, video_range, white_bg=white_bg, use_pil=self.meta_dict_sclist[scene_idx]['scene_name']=='bunny')
                ).permute(0, 3, 1, 2)

            video_clip = video_clip / 255.0
            norm_data = rgba / 255.0
            img = norm_data[:, :, :3] * norm_data[:, :, 3:4]
            if white_bg:
                img += np.ones_like(img) * (1-norm_data[:, :, 3:4])
                img = np.clip(img, 0, 1.0)

            # shape convert from HWC to CHW
            img = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)

            img_list.append(img)
            video_list.append(video_clip)
        
        gravity = self.gravity
        if self.real_dt is not None:
            if isinstance(self.real_dt, dict):
                c_real_dt = self.real_dt[self.meta_dict_sclist[scene_idx]['scene_name']]
            else:
                c_real_dt = self.real_dt
            time_ratio = c_real_dt / self.comp_dt
            gravity = self.gravity * time_ratio**2
        rot_est = self.env_cfg.get("rot_est", None)
        if rot_est is not None:
            assert isinstance(rot_est, dict)
            scene_quant = rot_est[self.meta_dict_sclist[scene_idx]['scene_name']]
            scene_rot = scipy_R.from_quat(scene_quant)
            gravity = scene_rot.inv().apply(gravity).astype(np.float32)
        # add read flow, depth later.
        volume_scalar = self.env_cfg['volume_scalar']
        if isinstance(volume_scalar, dict):
            volume_scalar = self.env_cfg['volume_scalar'][self.meta_dict_sclist[scene_idx]['scene_name']]
        ret_dict = {
            "inputs": {
                "img": img_list,  # [n_cam, H, W, 3] value in [0, 1]
                "mov_mask": self.pcmask_sclist[scene_idx], # [n_points, 1]
                "cln_mask": self.cln_pcmask_sclist[scene_idx],
                "pin_mask": self.pinmask_sclist[scene_idx].float(), # [n_moving_points, 1]
                "p2c_mapping": self.clustermask_sclist[scene_idx],
                "cam": cam_list, # [n_cam, 1]
                "volume_scalar": np.array([volume_scalar], dtype=np.float32).reshape(1, 1),
                "external": gravity,
            },
            "gt_label": video_list,  # [n_cam, nf, 3, H, W] value in [0, 1]
            "meta": {
                "scene_idx": torch.tensor([scene_idx]).long(),
                "scene_name": self.meta_dict_sclist[scene_idx]['scene_name'],
                "idx": torch.tensor([idx]).long(),
                "seq_idx": torch.tensor([seq_idx]).long(),
                "eval_start_frame": torch.tensor([self.env_cfg.get("eval_start_frame", 0)]).long(),
            },
        }
        if self.use_random_background:
            ret_dict["inputs"]["white_bg"] = white_bg_list
        opacity_scalar = self.env_cfg.get("opacity_scalar", None)
        if opacity_scalar is not None:
            ret_dict["inputs"]["opacity_scalar"] = opacity_scalar

        return ret_dict

    def _read_camera_transforms(
        self,
        scene_data_dir,
        camera_transform_file,
        img_hw=None,
        max_cam_total=-1,
        max_seq=-1,
    ):
        with open(camera_transform_file, "r") as f:
            camera_transforms = json.load(f)

        camera_dict = defaultdict(list)
        seq_info_dict = defaultdict(list)

        frames = camera_transforms["frames"]
        fovx = camera_transforms["camera_angle_x"]

        if self.scale_x_angle != 1.0:
            fovx = fovx * self.scale_x_angle

        min_cam_idx = 100000
        for frame in frames:
            img_fn = os.path.basename(frame["file_path"])
            seq_cam_name = img_fn.replace('.png', '')
            assert img_fn.startswith("seq")
            # seq_xxx_frame_xxx.png
            ## Here frame-id actually is camera-id
            seq_name = '_'.join(img_fn.split('_')[0:2])
            seq_idx = int(img_fn.split('_')[1])
            if max_seq > 0 and seq_idx >= max_seq:
                continue

            cam_idx = int(os.path.splitext(frame["file_path"])[0][-5:])
            if cam_idx < min_cam_idx:
                min_cam_idx = cam_idx
            existing_cam_num = np.sum([len(v) for v in seq_info_dict.values()])
            if max_cam_total > 0 and existing_cam_num >= max_cam_total: # Here, num camera
                continue
                
            img_path = os.path.join(scene_data_dir, frame["file_path"])
            if not (
                img_path.endswith(".png")
                or img_path.endswith(".jpg")
                or img_path.endswith(".JPG")
                or img_path.endswith(".jpeg")
            ):
                img_path += ".png"
            if img_hw is None:
                image = Image.open(img_path)
                img_hw = image.size[::-1]

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(
                w2c[:3, :3]
            )  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            height, width = img_hw
            fovy = focal2fov(fov2focal(fovx, width), height)
            FovY = fovy
            FovX = fovx

            seq_info_dict[seq_name].append(img_path)
            camera_dict[seq_name].append(
                Camera(
                    R=R,
                    T=T,
                    FoVy=FovY,
                    FoVx=FovX,
                    img_path=img_path,
                    img_hw=img_hw,
                    timestamp=None,
                    data_device="cuda",
                )
            )

        num_frames = sum([len(camera_list) for camera_list in camera_dict.values()])
        meta_dict = {
            "num_frames": num_frames,
            "num_cameras": num_frames,
            "img_hw": img_hw,
            "seq_info":seq_info_dict,
        }

        return camera_dict, meta_dict

    def _read_camera_transforms_colmap( self,
        scene_data_dir,
        cam_meta,
        image_dir="images",
        video_dir="video_images",
        img_hw=None,
        eval=False,
        llffhold=5,
        max_cam_total=-1, max_seq=-1,
    ):
        try:
            cameras_extrinsic_file = os.path.join(scene_data_dir, "sparse/0", "images.bin")
            cameras_intrinsic_file = os.path.join(scene_data_dir, "sparse/0", "cameras.bin")
            cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        except:
            cameras_extrinsic_file = os.path.join(scene_data_dir, "sparse/0", "images.txt")
            cameras_intrinsic_file = os.path.join(scene_data_dir, "sparse/0", "cameras.txt")
            cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

        exp_cam_extrinsics, exp_real_image_path = expand_cameras(cam_meta, cam_extrinsics)
        reading_dir = "images" if image_dir == None else image_dir
        reading_video_dir = "video_images" if video_dir == None else video_dir
        split = None
        if self.phase == 'train':
            split = cam_meta.get('train', None)
        elif self.phase == 'all':
            if 'train' in cam_meta.keys():
                split = cam_meta['train']
            if 'test' in cam_meta.keys():
                if split is None:
                    split = cam_meta['test']
                else:
                    split += cam_meta['test']
        else:
            split = cam_meta.get('test', None)
        camera_dict, seq_info_dict = readColmapCameras(
            cam_extrinsics=exp_cam_extrinsics,
            cam_intrinsics=cam_intrinsics,
            images_folder=os.path.join(scene_data_dir, reading_dir),
            videos_folder=os.path.join(scene_data_dir, reading_video_dir),
            img_hw=img_hw,
            scale_x_angle=self.scale_x_angle,
            suffix_replace=['.jpg', '.png'] if "_mov" in reading_dir else None,
            max_cam_total=max_cam_total, max_seq=max_seq,
            exp_real_image_path=exp_real_image_path,
            invalid_name=cam_meta.get("invalid", None),
            split=split,
        )

        # Split the cameras
        ret_cam_infos = defaultdict(list)
        for seq_n, cam_list in camera_dict.items():
            for cam_idx in range(len(cam_list)):
                # TODO: How to define the llffhold here
                if llffhold > 0:
                    is_training = (cam_idx+1) % llffhold != 0
                    if self.phase == 'train' and is_training:
                        ret_cam_infos[seq_n].append(cam_list[cam_idx])
                    elif self.phase != 'train' and not is_training:
                        ret_cam_infos[seq_n].append(cam_list[cam_idx])
                else:
                    ret_cam_infos[seq_n].append(cam_list[cam_idx])

        if img_hw is None:
            img_hw = list(ret_cam_infos.values())[0].img_hw

        num_frames = sum([len(camera_list) for camera_list in ret_cam_infos.values()])
        meta_dict = {
            "num_frames": num_frames,
            "num_cameras": num_frames,
            "img_hw": img_hw,
            "seq_info":seq_info_dict,
        }

        return ret_cam_infos, meta_dict

    def filter_camera_with_renderd_frames(self, cam_list, rendered_dir):
        rendered_img_names = [_ for _ in os.listdir(rendered_dir) if _.endswith(".png")]
        rendered_img_names = [_.split(".")[0] for _ in rendered_img_names]

        for cam in cam_list:
            img_name = os.path.basename(cam.img_path).split(".")[0]
            if img_name not in rendered_img_names:
                cam_list.remove(cam)
        return cam_list

    def save_camera_list(self, cam_list, save_path):
        camera_list = []
        assert save_path.endswith(".json"), "save_path should be a json file"

        for cam in cam_list:
            cam_dict = {
                "R": cam.R.tolist(),
                "T": cam.T.tolist(),
                "FoVy": cam.FoVy,
                "FoVx": cam.FoVx,
                "img_path": cam.img_path,
                "image_height": cam.image_height,
                "image_width": cam.image_width,
            }
            camera_list.append(cam_dict)

        with open(save_path, "w") as f:
            json.dump(camera_list, f, indent=4)

    def interpolate_camera(self, filename1, filename2, num_frames):
        for cam in self.camera_list:
            img_name = os.path.basename(cam.img_path).split(".")[0]
            if filename1.startswith(img_name):
                cam1 = cam
            if filename2.startswith(img_name):
                cam2 = cam

        interpolated_cameras = cam1.interpolate(cam2, num_frames - 1)
        return interpolated_cameras

    def collate(self, batch, samples_per_gpu=1):
        assert len(batch) == 1, "Batch size could only be 1 for now"
        batch_cam = batch[0]['inputs'].pop('cam')
        scene_name = batch[0]['meta'].pop('scene_name')
        batched_data = collate(batch, samples_per_gpu=samples_per_gpu)
        batched_data['inputs']['cam'] = batch_cam
        batched_data['meta']['scene_name'] = scene_name
        return batched_data
        
    def evaluate_frame(self,
                 results,
                 metric=None,
                 logger=None,
                 **kwargs):
        eval_results = defaultdict(list)
        for rst in results:
            acc = rst['acc']
            for key, val in acc.items():
                eval_results[key].append(val)
        
        # Mean results
        collate_results = dict()
        for g_name, g_list in eval_results.items():
            collate_results[g_name] = dict(
                mean=np.mean(g_list),
                std=np.std(g_list),
            )

        return collate_results
    
    def evaluate_rollout(self,
                 results,
                 metric=None,
                 metric_options=None,
                 logger=None,
                 **kwargs):

        def collate_rollout(data_dict, prefix='rollout'):
            collate_dict = defaultdict(list)
            for rollout_idx, val in data_dict.items():
                cur_rst = defaultdict(list)
                for acc_entry in val:
                    for acc_key, acc_val in acc_entry.items():
                        cur_rst[acc_key].append(acc_val)
                for acc_key, val in cur_rst.items():
                    collate_dict[acc_key].append(np.mean(val))
            rst = dict()
            for key, val in collate_dict.items():
                rst[key] = {
                    f'{prefix}_mean': np.mean(val),
                    f'{prefix}_std': np.std(val)
                }
            return rst
        
        collate_results = dict()

        scene_wise_results = dict() # If only one scene, then this is equal to the following one
        rollout_wise_results = defaultdict(list)
        # Collect results
        for rst in results:
            rollout_idx = rst['rollout_idx'][0]
            scene_name = rst['scene_name']
            rollout_wise_results[rollout_idx].append(rst['acc'])
            if scene_name not in scene_wise_results.keys():
                scene_wise_results[scene_name] = defaultdict(list)
            scene_wise_results[scene_name][rollout_idx].append(rst['acc'])
            
        # Calculate rollout level results
        eval_rollout = dict()
        eval_rollout['whole'] = collate_rollout(rollout_wise_results, prefix='whole')
        for key, val in scene_wise_results.items():
            eval_rollout[key] = collate_rollout(val, prefix=key)

        return eval_rollout

    def evaluate(self,
                 results,
                 metric=None,
                 metric_options=None,
                 logger=None,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is `accuracy`.
            metric_options (dict): Options for calculating metrics. Allowed
                keys are 'topk', 'thrs' and 'average_mode'.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
        Returns:
            dict: evaluation results
        """
        perframe_rst = self.evaluate_frame(results=results, metric=metric, metric_options=metric_options, logger=logger, **kwargs)
        rst = dict()
        rst['per_frame'] = perframe_rst
        if self.env_cfg.get("rollout", False):
            perrollout_rst = self.evaluate_rollout(results=results, metric=metric, metric_options=metric_options, logger=logger, **kwargs)
            rst['per_rollout'] = perrollout_rst
        return rst

    def inject_external_forces(self, prev_state, query_pos, radius, decay_radius, delta_x):
        dist = torch.linalg.norm(prev_state - query_pos, dim=-1, keepdim=True)
        selected_mask = dist < radius
        # Apply the forces with decay according to the l2 distance
        assert decay_radius < radius and decay_radius > 0
        scalar_dist = torch.clamp_min(dist/decay_radius, 1.0)
        applied_x = delta_x / scalar_dist
        prev_state_updated = prev_state - selected_mask * applied_x
        return prev_state_updated
    
    def inject_external_forces_tocur(self, cur_state, query_pos, radius, decay_radius, delta_x):
        dist = torch.linalg.norm(cur_state - query_pos, dim=-1, keepdim=True)
        selected_mask = dist < radius
        # Apply the forces with decay according to the l2 distance
        assert decay_radius < radius and decay_radius > 0
        scalar_dist = torch.clamp_min(dist/decay_radius, 1.0)
        applied_x = delta_x / scalar_dist
        cur_state_updated = cur_state + selected_mask * applied_x
        return cur_state_updated

def camera_dataset_collate_fn(batch):
    ret_dict = {
        "cam": [],
        "img_name": [],
    }

    for key in batch[0].keys():
        if key == "cam":
            ret_dict[key].extend([item[key] for item in batch])
        elif key == "img_name":
            ret_dict[key].extend([item[key] for item in batch])
        elif key == "timestamp":
            ret_dict[key] = torch.tensor([item[key] for item in batch])
        else:
            ret_dict[key] = torch.stack([item[key] for item in batch], dim=0)

    return ret_dict


def create_camera(dataset_dir, save_path, *args):
    fname1, fname2, fname3, num_frames = args
    num_frames_each = int(num_frames / 3)

    dataset = MultiviewVideoDataset(dataset_dir, load_imgs=False)

    cam_AB = dataset.interpolate_camera(fname1, fname2, num_frames_each)
    cam_BC = dataset.interpolate_camera(fname2, fname3, num_frames_each)
    cam_CA = dataset.interpolate_camera(fname3, fname1, num_frames_each)

    cam_list = cam_AB + cam_BC + cam_CA
    dataset.save_camera_list(cam_list, save_path)


def test_speed():
    dataset_dir = "../../../../../dataset/3D_capture/purple_branches_colmap"
    dataset_dir = "../../../../../dataset/physics_dreamer/llff_flower_undistorted"

    dataset = MultiviewVideoDataset(dataset_dir)

    data = dataset[0]

    for key, val in data.items():
        if isinstance(val, torch.Tensor):
            print(key, val.shape)
        else:
            print(key, type(val))
