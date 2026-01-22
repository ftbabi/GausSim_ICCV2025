import torch
from time import time
import numpy as np
from sklearn.cluster import KMeans


def downsample_with_kmeans(points_array: np.ndarray, num_points: int):
    """
    Args:
        points_array: [N, 3]
        num_points: int
    Outs:
        downsampled_points: [num_points, 3]
    """

    print(
        "=> staring downsample with kmeans from ",
        points_array.shape[0],
        " points to ",
        num_points,
        " points",
    )
    s_time = time()
    kmeans = KMeans(n_clusters=num_points, random_state=0).fit(points_array)
    cluster_centers = kmeans.cluster_centers_
    e_time = time()

    print("=> downsample with kmeans takes ", e_time - s_time, " seconds")
    return cluster_centers


@torch.no_grad()
def downsample_with_kmeans_gpu(points_array: torch.Tensor, num_points: int):

    from kmeans_gpu import KMeans

    kmeans = KMeans(
        n_clusters=num_points,
        max_iter=100,
        tolerance=1e-4,
        distance="euclidean",
        sub_sampling=None,
        max_neighbors=15,
    )

    features = torch.ones(1, 1, points_array.shape[0], device=points_array.device)
    points_array = points_array.unsqueeze(0)
    # Forward

    print(
        "=> staring downsample with kmeans from ",
        points_array.shape[1],
        " points to ",
        num_points,
        " points",
    )
    s_time = time()
    centroids, features = kmeans(points_array, features)

    ret_points = centroids.squeeze(0)
    e_time = time()
    print("=> downsample with kmeans takes ", e_time - s_time, " seconds")

    # [np_subsample, 3]
    return ret_points


@torch.no_grad()
def downsample_with_kmeans_gpu_with_chunk(points_array: torch.Tensor, num_points: int):
    # split the points_array into chunks, and then do kmeans on each chunk
    #   to save memory.

    from kmeans_gpu import KMeans

    points_array_sum = points_array.sum(dim=1)
    arg_idx = torch.argsort(points_array_sum, descending=True)
    points_array = points_array[arg_idx, :]

    features = torch.ones(1, 1, points_array.shape[0], device=points_array.device)
    points_array = points_array.unsqueeze(0)
    # Forward

    print(
        "=> staring downsample with kmeans from ",
        points_array.shape[1],
        " points to ",
        num_points,
        " points",
        points_array.shape,
    )
    s_time = time()

    num_raw_points = points_array.shape[1]
    chunk_size = 150000

    num_chunks = num_raw_points // chunk_size + 1

    ret_list = []
    for i in range(num_chunks):

        start = i * chunk_size
        end = min((i + 1) * chunk_size, num_raw_points)
        points_chunk = points_array[:, start:end, :]
        features_chunk = features[:, :, start:end]

        num_target_points = min(chunk_size, num_points // num_chunks)

        kmeans = KMeans(
            n_clusters=num_target_points,
            max_iter=100,
            tolerance=1e-4,
            distance="euclidean",
            sub_sampling=None,
            max_neighbors=15,
        )
        centroids, _ = kmeans(points_chunk, features_chunk)
        ret_list.append(centroids.squeeze(0))

    ret_points = torch.cat(ret_list, dim=0)
    e_time = time()
    print("=> downsample with kmeans takes ", e_time - s_time, " seconds")

    # [np_subsample, 3]
    return ret_points

def find_far_points(xyzs, selected_points, thres=0.05):
    """
    Args:
        xyzs: [N, 3]
        selected_points: [M, 3]
    Outs:
        freeze_mask: [N], 1 for points that are far away, 0 for points that are close
                    dtype=torch.int
    """
    chunk_size = 10000

    freeze_mask_list = []
    for i in range(0, xyzs.shape[0], chunk_size):

        end_index = min(i + chunk_size, xyzs.shape[0])
        xyzs_chunk = xyzs[i:end_index]
        # [M, N]
        cdist = torch.cdist(xyzs_chunk, selected_points)

        min_dist, _ = torch.min(cdist, dim=-1)
        freeze_mask = min_dist > thres
        freeze_mask = freeze_mask.type(torch.int)
        freeze_mask_list.append(freeze_mask)

    freeze_mask = torch.cat(freeze_mask_list, dim=0)

    # 1 for points that are far away, 0 for points that are close
    return freeze_mask


import sys
sys.path.append('gaussian-splatting')
from scene.gaussian_model import GaussianModel

def apply_mask_gaussian(gaussian, mask):
    new_xyz = gaussian._xyz[mask]
    if gaussian.xyz_gradient_accum.shape == gaussian._xyz.shape:
        new_xyz_gradient_accum = gaussian.xyz_gradient_accum[mask]
        new_denom = gaussian.denom[mask]
    else:
        new_xyz_gradient_accum = gaussian.xyz_gradient_accum
        new_denom = gaussian.denom
    new_model_args = (
        gaussian.active_sh_degree,
        new_xyz,
        gaussian._features_dc[mask].detach().clone(),
        gaussian._features_rest[mask].detach().clone(),
        gaussian._scaling[mask].detach().clone(),
        gaussian._rotation[mask].detach().clone(),
        gaussian._opacity[mask].detach().clone(),
        gaussian.max_radii2D.detach().clone(),
        new_xyz_gradient_accum.detach().clone(),
        new_denom.detach().clone(),
        None,
        gaussian.spatial_lr_scale,
    )

    ret_gaussian = GaussianModel(gaussian.max_sh_degree)
    ret_gaussian = restore_gaussian(ret_gaussian, new_model_args, None)
    return ret_gaussian

def restore_gaussian(gaussian, model_args, training_args):
    (
        gaussian.active_sh_degree,
        gaussian._xyz,
        gaussian._features_dc,
        gaussian._features_rest,
        gaussian._scaling,
        gaussian._rotation,
        gaussian._opacity,
        gaussian.max_radii2D,
        xyz_gradient_accum,
        denom,
        opt_dict,
        gaussian.spatial_lr_scale,
    ) = model_args

    if training_args is not None:
        gaussian.training_setup(training_args)
    gaussian.xyz_gradient_accum = xyz_gradient_accum
    gaussian.denom = denom
    if opt_dict is not None:
        gaussian.optimizer.load_state_dict(opt_dict)
    return gaussian