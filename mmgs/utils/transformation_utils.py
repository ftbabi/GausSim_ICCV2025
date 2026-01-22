import torch
from mmgs.models.utils.deformation_gradient import DeformationGradient


def get_mat_from_upper(upper_mat):
    upper_mat = upper_mat.reshape(-1, 6)
    mat = torch.zeros((upper_mat.shape[0], 9)).to(upper_mat.device)
    mat[:, :3] = upper_mat[:, :3]
    mat[:, 3] = upper_mat[:, 1]
    mat[:, 4] = upper_mat[:, 3]
    mat[:, 5] = upper_mat[:, 4]
    mat[:, 6] = upper_mat[:, 2]
    mat[:, 7] = upper_mat[:, 4]
    mat[:, 8] = upper_mat[:, 5]

    return mat.view(-1, 3, 3)

def apply_cov_rotation(cov_tensor, rotation_matrix):
    rotated = torch.matmul(cov_tensor, rotation_matrix.transpose(-1, -2))
    rotated = torch.matmul(rotation_matrix, rotated)
    return rotated

def get_uppder_from_mat(mat):
    mat = mat.view(-1, 9)
    upper_mat = torch.zeros((mat.shape[0], 6), device=mat.device)
    upper_mat[:, :3] = mat[:, :3]
    upper_mat[:, 3] = mat[:, 4]
    upper_mat[:, 4] = mat[:, 5]
    upper_mat[:, 5] = mat[:, 8]

    return upper_mat

def apply_cov_rotations_batched(upper_cov_tensor, rotation_matrices, inverse=False):
    cov_tensor = get_mat_from_upper(upper_cov_tensor)
    for i in range(len(rotation_matrices)):
        R = rotation_matrices[len(rotation_matrices) - 1 - i]
        cov_tensor = apply_cov_rotation(cov_tensor, R if not inverse else R.transpose(-1, -2)) # Here is a bit different from original code; original one use .T, we don't need
    return get_uppder_from_mat(cov_tensor)

def get_shs_rotations_batched(deformation_components, inverse=False):
    assert len(deformation_components) > 0
    R = DeformationGradient.get_deformation_gradient_rotation_matrix(deformation_components[0])
    for i in range(1, len(deformation_components)):
        _R = DeformationGradient.get_deformation_gradient_rotation_matrix(deformation_components[i])
        R = torch.bmm(_R, R)
    if inverse:
        R = R.transpose(-1, -2)
    return R