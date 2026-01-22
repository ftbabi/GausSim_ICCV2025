from itertools import chain
import torch
import torch.nn.functional as F


def disable_grad(nn_module):
    for param in nn_module.parameters():
        param.requires_grad = False

def to4x4(matrix):
    '''
        pad 3*3 matrix on bottom and right by 0
    '''
    pad = [[0, 1], [0, 1]]
    pad += [[0, 0] for n in range(len(matrix.shape) - 2)]
    pad = tuple(chain(*pad))
    matrix = F.pad(matrix, pad)
    matrix[:, 3, 3] += 1
    return matrix

def root_loc_to_4x4(root_loc, num_joints):
    '''
        root_loc: n, 3
    '''
    # Change to n, 1, 3, 1
    root_loc = root_loc[:, None, :, None]
    # pad = [[0, 0], [0, num_joints - 1], [0, 1], [3, 0]]
    pad = [[3, 0], [0, 1], [0, num_joints - 1], [0, 0]]
    pad = tuple(chain(*pad))
    # n, 23+1, 1+3, 3+1
    return F.pad(root_loc, pad)