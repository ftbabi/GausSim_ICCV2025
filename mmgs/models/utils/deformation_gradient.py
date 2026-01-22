import torch
import torch.nn as nn


def build_scaling(s):
    '''
        s: n_points, 3
    '''
    L = torch.diag_embed(s, offset=0)
    return L

def build_rotation(r):
    '''
        r: n_points, 4
        # Already normed
    '''
    # norm = torch.linalg.norm(r, dim=-1)
    # norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    # q = r / norm[:, None]
    q = r

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

class DeformationGradient(nn.Module):
    def __init__(self, batched_dg) -> None:
        '''
            batched_gt: bs, 4+3+4
                            U scale V
        '''
        super(DeformationGradient, self).__init__()
        self.batched_dg = batched_dg
        assert batched_dg.shape[-1] == 4+3+4
        self.U = None
        self.Scalar = None
        self.V = None
        self.R = None # Polar decomposition
        self.S = None # Polar decomposition
        self.F = None

    def get_U(self):
        if self.U is None:
            self.U = build_rotation(self.batched_dg[:, 0:4])
        return self.U
    
    def get_Scalar(self):
        if self.Scalar is None:
            self.Scalar = build_scaling(self.batched_dg[:, 4:7])
        return self.Scalar
    
    def get_V(self):
        if self.V is None:
            self.V = build_rotation(self.batched_dg[:, 7:11])
        return self.V
    
    def get_R(self):
        '''
            Left rotation for polar decomposition
        '''
        if self.R is None:
            U = self.get_U()
            V = self.get_V()
            self.R = torch.bmm(U, V.transpose(-1, -2))
        return self.R

    def get_S(self):
        '''
            Scaling for polar decomposition
        '''
        if self.S is None:
            V = self.get_V()
            Scalar = self.get_Scalar()
            self.S = torch.bmm(torch.bmm(V, Scalar), V.transpose(-1, -2))
        return self.S

    def get_F(self):
        '''
            Deformation Gradient Matrix
        '''
        if self.F is None:
            U = self.get_U()
            Scalar = self.get_Scalar()
            V = self.get_V()
            self.F = torch.bmm(torch.bmm(U, Scalar), V.transpose(-1, -2))
        return self.F
    
    @staticmethod
    def get_deformation_gradient_matrix(batched_dg):
        U = build_rotation(batched_dg[:, 0:4])
        Scalar = build_scaling(batched_dg[:, 4:7])
        V = build_rotation(batched_dg[:, 7:11])
        F = torch.bmm(torch.bmm(U, Scalar), V.transpose(-1, -2))
        return F
    
    @staticmethod
    def get_deformation_gradient_rotation_matrix(batched_dg):
        U = build_rotation(batched_dg[:, 0:4])
        V = build_rotation(batched_dg[:, 7:11])
        R = torch.bmm(U, V.transpose(-1, -2))
        return R

