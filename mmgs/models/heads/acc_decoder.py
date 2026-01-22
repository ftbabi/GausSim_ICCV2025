import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.ops import QueryAndGroup, grouping_operation

from ..builder import HEADS
from .sim_head import SimHead
from mmgs.models.utils import FFN, Normalizer
from mmgs.datasets.utils import denormalize
from mmgs.models.losses import L2Loss
from mmgs.core import multi_apply

import numpy as np
from mmgs.datasets.utils import to_numpy_detach
from mmgs.models.utils.dgl_graph import density_activation
from mmgs.models.utils.deformation_gradient import DeformationGradient
from mmgs.models.utils.render import render_gaussian, PipelineParams
import cv2
import plotly.express as px


# torch.autograd.set_detect_anomaly(True)

@HEADS.register_module()
class AccDecoder(SimHead):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        loss (dict): Config of classification loss.
    """  # noqa: W605

    def __init__(self,
                 out_channels=4+3+4, # rot, scale, rot
                 in_channels=128,
                 dt=1/30,
                 add_residual=True, # since not the same size
                 init_cfg=None,
                 eps=1e-7,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 pre_norm=False,
                 render_pipe_cfg=dict(white_bg=False, convert_SHs_python=True, compute_cov3D_python=False, debug=False),
                 scalar_max=5,
                 scalar_min=-5,
                 init_quant=1.0,
                 norm_volumn=True,
                 *args,
                 **kwargs):
        super(AccDecoder, self).__init__(init_cfg=init_cfg, *args, **kwargs)
        # nf_effect
        self.in_channels = in_channels
        # position_dim
        self.out_channels = out_channels
        self.dt = dt
        self.eps = eps
        self.loss_dim = (0, 3)
        self.pre_norm = pre_norm
        self.node_norm = build_norm_layer(norm_cfg, in_channels)[1] if pre_norm else nn.Identity()
        self.scalar_max = scalar_max
        self.scalar_min = scalar_min
        self.init_quant = init_quant
        self.norm_volumn = norm_volumn

        if self.out_channels <= 0:
            raise ValueError(
                f'num_classes={out_channels} must be a positive integer')

        self.dynamic_proj = FFN([in_channels, in_channels//2, in_channels//4, out_channels], final_act=False, act_cfg=act_cfg, add_residual=False)
        self.scalar_activation = torch.exp
        self.scalar_constrain = self.scalar_incompressible_constrain
        self.density_activation = density_activation()

        white_bg = render_pipe_cfg.pop('white_bg', True)
        self.render_background = torch.tensor([1, 1, 1], dtype=torch.float32) if white_bg else torch.tensor([0, 0, 0], dtype=torch.float32)
        self.render_pipe = PipelineParams(**render_pipe_cfg)

    def scalar_incompressible_constrain(self, scalar):
        '''
            The scalar must be activated before this func
            Assume the density is constant: incompressible 
        '''
        assert scalar.shape[-1] == 3
        mult = torch.pow(torch.prod(scalar, dim=-1, keepdim=True), 1/3)
        normed_scalar = scalar / mult
        return normed_scalar
    
    def scalar_ordinary_constrain(self, scalar):
        return scalar

    def init_weights(self):
        super(AccDecoder, self).init_weights()
        pass
    
    def evaluate(self, pred, gt_label, **kwargs):
        acc_dict = self.accuracy(pred, gt_label, **kwargs)
        return acc_dict

    def simple_test(self,
            img_list, gt_label, test_cfg=None, **kwargs):

        rst = dict()
        acc_dict = self.evaluate(
            [pi.permute(1,2,0) for pi in img_list], [gt.permute(1,2,0) for gt in gt_label],
            term_filter=['render'],
            **kwargs
        )
        rst.update(dict(acc=acc_dict))
        return rst

    def pre_predict(self, base_graph, **kwargs):
        pred_pos, pred_dg, pred_dg_mat, base_graph = self.predict(base_graph)
        return pred_pos, pred_dg, pred_dg_mat, base_graph
    
    def pre_render(self, cam, gt_label, white_bg, scene_gaussian, cov3D_precomp, pos, opacity, rotation=None, return_tuple=False, **kwargs):
        active_sh_degree = scene_gaussian.active_sh_degree
        max_sh_degree = scene_gaussian.max_sh_degree
        features = scene_gaussian.get_features
        render_bg = self.render_background.to(pos.device)
        if white_bg is not None:
            if white_bg:
                render_bg = torch.ones_like(render_bg, dtype=torch.float32).to(pos.device)
            else:
                render_bg = torch.zeros_like(render_bg, dtype=torch.float32).to(pos.device)
        # Output: 3(channel) H, W
        img = render_gaussian(
            cam,
            pos,
            opacity, active_sh_degree, max_sh_degree,
            self.render_pipe,
            render_bg,
            cov3D_precomp=cov3D_precomp,
            override_color=None,
            features=features,
            rotation=rotation)["render"]

        if return_tuple:
            return img,
        else:
            return img

    def forward_train_static(self, pred_pos, gt_pos, **kwargs):
        losses = dict()
        loss_static = self.loss(
            [pred_pos], [gt_pos],
            term_filter=['static'], **kwargs)
        losses.update(loss_static)

        return losses

    def forward_train_regularize(self,
            pred_pos, pred_dg, base_graph, connect_graph, **kwargs):
        '''
            Both train and test
        '''
        losses = dict()
        # Compute the momentum
        anchor_pos = base_graph.ndata['anchor_next_state']
        relative_pos = pred_pos - anchor_pos
        density = self.density_activation(base_graph.ndata['attr'][:, 0:1])
        mass = density * base_graph.ndata['diag_volume']
        momentum = (relative_pos * mass).sum(dim=0, keepdim=True).clamp(-100.0, 100.0)
        assert not torch.any(torch.isnan(relative_pos))
        assert not torch.any(torch.isnan(density))
        assert not torch.any(torch.isnan(mass))
        assert not torch.any(torch.isnan(momentum))
        loss_momentum = self.loss(
            momentum, torch.zeros_like(momentum).to(momentum),
            term_filter=['momentum'], **kwargs)
        losses.update(loss_momentum)

        return losses
    
    def forward_train(self,
            pred_img, gt_label, **kwargs):
        losses = dict()
        loss_render = self.loss(
            [pi.permute(1,2,0) for pi in pred_img], [gt.permute(1,2,0) for gt in gt_label],
            term_filter=['render'], **kwargs)
        losses.update(loss_render)

        return losses

    def forward_test(self, **kwargs):
        return self.simple_test(**kwargs)

    def node_pred_pos_dg(self, feature_field, anchor_next_state_field, anchor_template_field, template_field, pin_mask_field, pos_field, deformation_gradient_field, deformation_gradient_matrix_field, apply_pin):
        def func(nodes):
            # Quaternion here is r, xyz
            vert_emb = nodes.data[feature_field]
            feature = self.node_norm(vert_emb)
            # Pred deforamtion gradient
            pred_deformation_gradient = self.dynamic_proj(feature)
            default_quaternion = torch.zeros_like(pred_deformation_gradient[:, 0:4]).to(pred_deformation_gradient)
            default_quaternion[:, 0] = self.init_quant
            pred_U = pred_deformation_gradient[:, 0:4] + default_quaternion
            pred_Scalar = pred_deformation_gradient[:, 4:7].clamp(self.scalar_min, self.scalar_max)
            pred_V = pred_deformation_gradient[:, 7:11] + default_quaternion
            normed_U = pred_U / torch.linalg.norm(pred_U, dim=-1, keepdim=True)
            acted_Scalar = self.scalar_activation(pred_Scalar)
            if self.norm_volumn:
                normed_Scalar = self.scalar_constrain(acted_Scalar)
            else:
                normed_Scalar = acted_Scalar
            normed_V = pred_V / torch.linalg.norm(pred_V, dim=-1, keepdim=True)
            pred_deformation_gradient = torch.cat([normed_U, normed_Scalar, normed_V], dim=-1)

            # Pred new positions
            anchor_next_state = nodes.data[anchor_next_state_field]
            anchor_template = nodes.data[anchor_template_field]
            template = nodes.data[template_field]
            Ft = DeformationGradient.get_deformation_gradient_matrix(pred_deformation_gradient)
            next_pos = anchor_next_state + torch.bmm(Ft, (template-anchor_template).unsqueeze(-1)).squeeze(-1)
            ## Mask out
            pin_mask = nodes.data[pin_mask_field]
            if apply_pin:
                next_pos = next_pos * torch.logical_not(pin_mask) + anchor_next_state * pin_mask
            return {pos_field: next_pos, deformation_gradient_field: pred_deformation_gradient, deformation_gradient_matrix_field: Ft}
        return func
    
    def predict(self, base_graph, apply_pin=False):
        '''
            This is only for training, to adjust to the parent class.
            This ensure vt is gt and pred is in 3D clean space, we ensure the mu is closer to the gt of xt.
            Thus no need to add eps
            No eps for vel to align with the input rules
        '''
        # Patch state
        ## Pred vel
        base_graph.apply_nodes(
            self.node_pred_pos_dg(
                'out_node',
                'anchor_next_state', 'anchor_template_state',
                'template_state', 'pin_mask',
                'pred_pos', 'pred_dg', 'pred_dg_mat', apply_pin=apply_pin)) # HERE*************************************
        
        pred_pos = base_graph.ndata['pred_pos']
        pred_dg = base_graph.ndata['pred_dg']
        pred_dg_mat = base_graph.ndata['pred_dg_mat']

        return pred_pos, pred_dg, pred_dg_mat, base_graph

