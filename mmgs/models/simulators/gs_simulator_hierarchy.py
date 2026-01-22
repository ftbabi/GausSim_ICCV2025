# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from functorch import vmap, jacrev

import numpy as np

from .. import builder
from ..builder import SIMULATORS
from .base import BaseSimulator
from mmgs.core import add_prefix, multi_apply
from mmgs.datasets.utils import to_numpy_detach, readPKL
from collections import defaultdict
from functools import partial
from mmgs.utils import PointCloudViewer

import dgl
import dgl.function as fn
from mmgs.models.utils.dgl_graph import CLUSTER_VERT_ID, P2C_EDGE_ID, POINT_VERT_ID, C2P_EDGE_ID
from mmgs.utils.transformation_utils import apply_cov_rotations_batched, get_shs_rotations_batched
from mmgs.utils.physdreamer_utils import apply_mask_gaussian

import os

import sys
sys.path.append('gaussian-splatting')
from scene.gaussian_model import GaussianModel

import time


@SIMULATORS.register_module()
class GsSimulatorHierarchy(BaseSimulator):
    """Encoder Decoder SIMULATORS.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 gs_scene,
                 processor_cfg,
                 cluster_cfg,
                #  neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 accumulate_gradient=False,
                 forward_last_layer=False,
                 opt_sim=True,
                 opt_vel=False,
                 dt=1/30,
                 static_loss=False,
                 checkpoint_rollout=25,
                 selfsup_loss=False,
                 avg_loss=False,
                 render_mov_only=False, # Render only the moving target object or not
                 data_aug=False,
                 use_rotation=True,
                 pred_vel=False,
                 attr_init=False,
                 force_forward=-1,
                 **kwargs):
        super(GsSimulatorHierarchy, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and simulator set pretrained weight'
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
            # TODO: check pretrain
            # backbone.pretrained = pretrained
        # Update common config
        backbone['dt'] = dt
        decode_head['dt'] = dt
        self.dt = dt
        # 1 for mass;
        self.attr_dim = backbone['attr_dim']
        self.attr_init = attr_init
        self.avg_loss = avg_loss
        self.render_mov_only = render_mov_only
        self.use_rotation = use_rotation
        self.pred_vel = pred_vel
        self.force_forward = force_forward

        self._init_gs_scene(gs_scene)

        # Model init
        self.cluster_cfg = cluster_cfg
        self.forward_last_layer = forward_last_layer

        self.backbone = builder.build_backbone(backbone)
        self.decode_head = self._init_decode_head(decode_head)

        if isinstance(processor_cfg, dict):
            processor_cfg = [processor_cfg]
        generator_cfg = processor_cfg[0]
        self.graph_generator = builder.build_preprocessor(generator_cfg)
        self.preprocessor = []
        for i in range(1, len(processor_cfg)):
            p_cfg = processor_cfg[i]
            self.preprocessor.append(builder.build_preprocessor(p_cfg))

        # This is for preprocess/augment train input
        self.train_cfg = train_cfg
        # This is for preprocess/augment test input
        self.test_cfg = test_cfg

        assert opt_sim or opt_vel
        self.opt_sim = opt_sim
        self.opt_vel = opt_vel
        # Estimating the initial velocity first
        self.is_est_vel = True
        self.num_iter = 0
        self.num_epoch = 0
        self.static_loss = static_loss
        self.checkpoint_rollout = checkpoint_rollout
        self.selfsup_loss = selfsup_loss
        self.data_aug = data_aug
        
        # To control the autoregressive
        self.accumulate_gradient = accumulate_gradient
        assert self.with_decode_head

    def _init_gs_scene(self, gs_scene):
        self.gs_scene_dict = dict() # Only provide initial states and for rendering
        self.gs_scene_dict_render = dict()

        # Trainable
        self.scene_initn1_pos = nn.ParameterDict()
        if self.pred_vel:
            self.scene_init0_pos = nn.ParameterDict()
        ## mass, attr_dim
        self.scene_attr = nn.ParameterDict()

        for scene_idx, scene_cfg in enumerate(gs_scene):
            # Load gaussian scene
            gaussian = GaussianModel(scene_cfg['sh_degree'])
            print(f"Initializing gaussian scene: {scene_cfg['model_path']}")
            gaussian.load_ply(scene_cfg['model_path'])
            self.gs_scene_dict[scene_cfg['name']] = gaussian
            # Load mask
            mask_path = scene_cfg['mov_mask_path']
            mov_mask = readPKL(mask_path)
            init_pos = gaussian.get_xyz[mov_mask, ...]
            attr_vec = torch.zeros((init_pos.shape[0], self.attr_dim)).to(init_pos)
            if self.attr_init:
                attr_vec[:, 1:] += (scene_idx/len(gs_scene))
            self.scene_attr[scene_cfg['name']] = nn.Parameter(attr_vec, requires_grad=True)
            self.scene_initn1_pos[scene_cfg['name']] = nn.ParameterDict()
            for seq_idx in range(scene_cfg['num_seq']):
                self.scene_initn1_pos[scene_cfg['name']][str(seq_idx)] = nn.Parameter(init_pos.detach().clone(), requires_grad=True)
            if self.pred_vel:
                self.scene_init0_pos[scene_cfg['name']] = nn.ParameterDict()
                for seq_idx in range(scene_cfg['num_seq']):
                    self.scene_init0_pos[scene_cfg['name']][str(seq_idx)] = nn.Parameter(init_pos.detach().clone(), requires_grad=True)
            # For render
            if self.render_mov_only:
                cln_mask_path = scene_cfg['cln_mask_path']
                cln_mask = readPKL(cln_mask_path)
                cln_gaussian = apply_mask_gaussian(gaussian, cln_mask)
                self.gs_scene_dict_render[scene_cfg['name']] = cln_gaussian
        return

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        # Build head
        return builder.build_head(decode_head)

    def init_weights(self):
        super(GsSimulatorHierarchy, self).init_weights()
    
    def _pre_maskout_pinverts_rollout(self, input_state, future_state, vert_mask):
        assert input_state.shape == future_state.shape
        assert input_state.shape[0] == vert_mask.shape[0]
        rst_state = input_state * vert_mask + future_state * (1-vert_mask)
        return rst_state,
    
    def _preprocess(self, prev_state, cur_state, template_state, attr, diag_volume, cur_cov, external_forces, p2c_mapping, pin_mask=None, is_training=False):
        '''
            prev_state: n_points, 3
            cur_state: n_points, 3
            material_center: 1, 3
            pin_mask: n_points, 1
        '''

        built_graph, connect_graph = self.graph_generator.batch_preprocess(prev_state, cur_state, template_state, attr, diag_volume, cur_cov, external_forces, p2c_mapping=p2c_mapping, pin_mask=pin_mask, dynamic_base=self.forward_last_layer)

        for i in range(len(self.preprocessor)):
            processor = self.preprocessor[i]
            built_graph = processor.graph_preprocess(built_graph, is_training=is_training)
        
        return built_graph, connect_graph
    
    def _postprocess(self, inputs, pred):
        return pred
    
    def extract_feat(self, backbone_model, input_graph, connect_graph):
        """Extract features from inputs."""
        x = backbone_model(input_graph, connect_graph)
        return x
    
    def encode_decode(self, input_graph, connect_graph, gaussians, cam_list, scene_cov3D, scene_pcs, mov_mask, register_norm=False, gt_label=None, rollout_size=1, cln_mask=None, cln_gaussian=None, white_bg_list=None, opacity_scalar=None):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input.
            connect_graph[i] is for in_graph[i], for moving things(anchor_next_state)  from last(upper) hierachy.
            in_graph: 0 is the leaf, 1 is cluster one, 2 is the root cluster;

        """
        # Forward
        losses = dict()
        outF_lv_list = []
        outDG_lv_list = []
        for i in range(len(input_graph)):
            if self.force_forward > 0 and i >= self.force_forward:
                break
            lv_idx = len(input_graph)-1 - i
            if not self.forward_last_layer and lv_idx <= 0:
                break
            backbone_model = self.backbone
            decode_head = self.decode_head
            # Get the output prediction
            lv_in_g, lv_in_connect_g = input_graph[lv_idx], connect_graph[lv_idx]
            if i == 0:
                # First step initialize
                lv_in_g.ndata['anchor_next_state'] = lv_in_g.ndata['anchor_cur_state']
                lv_in_connect_g.ndata['anchor_next_state'] = lv_in_connect_g.ndata['anchor_cur_state']
            encg_lv_t0 = self.extract_feat(backbone_model, lv_in_g, lv_in_connect_g)
            outx_lv_t1, outF_lv_t1, outFmat_lv_t1, lv_out_g = decode_head.pre_predict(encg_lv_t0, register_norm=register_norm, apply_pin=lv_idx>0)

            if lv_idx > 0:
                next_in_g = input_graph[lv_idx-1]
                next_connect_g = connect_graph[lv_idx-1]
                next_cluster_nids = torch.nonzero(next_connect_g.ndata[CLUSTER_VERT_ID][:, 0], as_tuple=False).squeeze()
                next_point_nids = torch.nonzero(next_connect_g.ndata[POINT_VERT_ID][:, 0], as_tuple=False).squeeze()
                next_c2p_eids = torch.nonzero(next_connect_g.edata[C2P_EDGE_ID][:, 0], as_tuple=False).squeeze()
                next_connect_g.nodes[next_cluster_nids].data['anchor_next_state'] = lv_out_g.ndata['pred_pos']
                next_connect_g.send_and_recv(next_c2p_eids, fn.copy_u('anchor_next_state', 'anchor_next_state'), fn.mean('anchor_next_state', 'anchor_next_state'))
                next_in_g.ndata['anchor_next_state'] = next_connect_g.nodes[next_point_nids].data['anchor_next_state']
                next_in_g.ndata['anchor_template_state'] = next_in_g.ndata['anchor_next_state'][:, :]
            if i > 0:
                lv_loss = decode_head.forward_train_regularize(outx_lv_t1, outF_lv_t1, lv_out_g, lv_in_connect_g)
                losses.update(add_prefix(lv_loss, 'decode'))

            # Broad cast to next level and future level
            for j in range(i+1, len(input_graph)):
                bc_lv_idx = len(input_graph)-1 - j
                bc_cur_connect_g = connect_graph[bc_lv_idx]
                point_nids = torch.nonzero(bc_cur_connect_g.ndata[POINT_VERT_ID][:, 0], as_tuple=False).squeeze()
                cluster_nids = torch.nonzero(bc_cur_connect_g.ndata[CLUSTER_VERT_ID][:, 0], as_tuple=False).squeeze()
                c2p_eids = torch.nonzero(bc_cur_connect_g.edata[C2P_EDGE_ID][:, 0], as_tuple=False).squeeze()
                bc_g = input_graph[bc_lv_idx]
                if j == i+1:
                    bc_cur_connect_g.nodes[cluster_nids].data['hie_anchor_next_state'] = lv_in_g.ndata['anchor_next_state']
                    bc_cur_connect_g.nodes[cluster_nids].data['hie_anchor_template_state'] = lv_in_g.ndata['anchor_template_state']
                    bc_cur_connect_g.nodes[cluster_nids].data['hie_pred_dg_mat'] = lv_out_g.ndata['pred_dg_mat']
                    bc_cur_connect_g.nodes[cluster_nids].data['hie_pred_dg'] = lv_out_g.ndata['pred_dg']
                bc_cur_connect_g.send_and_recv(c2p_eids, fn.copy_u('hie_anchor_next_state', 'hie_anchor_next_state'), fn.mean('hie_anchor_next_state', 'hie_anchor_next_state'))
                bc_cur_connect_g.send_and_recv(c2p_eids, fn.copy_u('hie_anchor_template_state', 'hie_anchor_template_state'), fn.mean('hie_anchor_template_state', 'hie_anchor_template_state'))
                bc_cur_connect_g.send_and_recv(c2p_eids, fn.copy_u('hie_pred_dg_mat', 'hie_pred_dg_mat'), fn.mean('hie_pred_dg_mat', 'hie_pred_dg_mat'))
                bc_cur_connect_g.send_and_recv(c2p_eids, fn.copy_u('hie_pred_dg', 'hie_pred_dg'), fn.mean('hie_pred_dg', 'hie_pred_dg'))
                # Assign to the compute graph
                bc_g.ndata['hie_anchor_next_state'] = bc_cur_connect_g.nodes[point_nids].data['hie_anchor_next_state']
                bc_g.ndata['hie_anchor_template_state'] = bc_cur_connect_g.nodes[point_nids].data['hie_anchor_template_state']
                bc_g.ndata['hie_pred_dg_mat'] = bc_cur_connect_g.nodes[point_nids].data['hie_pred_dg_mat']
                bc_g.ndata['hie_pred_dg'] = bc_cur_connect_g.nodes[point_nids].data['hie_pred_dg']
                # Update current state
                if bc_lv_idx > 0:
                    bc_g.ndata['cur_state'] = bc_g.ndata['hie_anchor_next_state'] + torch.bmm(bc_g.ndata['hie_pred_dg_mat'], (bc_g.ndata['template_state']-bc_g.ndata['hie_anchor_template_state']).unsqueeze(-1)).squeeze(-1)
                else:
                    bc_g.ndata['cur_state'] = torch.logical_not(bc_g.ndata['pin_mask']) * (bc_g.ndata['hie_anchor_next_state'] + torch.bmm(bc_g.ndata['hie_pred_dg_mat'], (bc_g.ndata['template_state']-bc_g.ndata['hie_anchor_template_state']).unsqueeze(-1)).squeeze(-1)) + bc_g.ndata['pin_mask'] * bc_g.ndata['cur_state']

                bc_g.ndata['template_state'] = bc_g.ndata['cur_state'][:, :3]
                if bc_lv_idx <= 0:
                    # Finish already
                    break
                bc_next_connect_g = connect_graph[bc_lv_idx-1]
                cur_cluster_nids = torch.nonzero(bc_next_connect_g.ndata[CLUSTER_VERT_ID][:, 0], as_tuple=False).squeeze()
                bc_next_connect_g.nodes[cur_cluster_nids].data['hie_anchor_next_state'] = bc_g.ndata['hie_anchor_next_state']
                bc_next_connect_g.nodes[cur_cluster_nids].data['hie_anchor_template_state'] = bc_g.ndata['hie_anchor_template_state']
                bc_next_connect_g.nodes[cur_cluster_nids].data['hie_pred_dg_mat'] = bc_g.ndata['hie_pred_dg_mat']
                bc_next_connect_g.nodes[cur_cluster_nids].data['hie_pred_dg'] = bc_g.ndata['hie_pred_dg']

            if self.forward_last_layer and lv_idx <= 0:
                input_graph[0].ndata['hie_pred_dg_mat'] = lv_out_g.ndata['pred_dg_mat']
                input_graph[0].ndata['hie_pred_dg'] = lv_out_g.ndata['pred_dg']
            outF_lv_list = [input_graph[0].ndata['hie_pred_dg_mat']] + outF_lv_list
            outDG_lv_list = [input_graph[0].ndata['hie_pred_dg']] + outDG_lv_list
        
        input_graph[0].ndata['cur_cov'] = apply_cov_rotations_batched(input_graph[0].ndata['cur_cov'], outF_lv_list, inverse=False)
        pred_rot = get_shs_rotations_batched(outDG_lv_list, inverse=True)
        # Render loss
        decode_head = self.decode_head
        
        pred_cov = input_graph[0].ndata['cur_cov']
        pred_pos = input_graph[0].ndata['cur_state'] if not self.forward_last_layer else input_graph[0].ndata['pred_pos']
        scene_rot = None
        if not self.render_mov_only:
            r_gaussians = gaussians
            # Merge cov3D and pos
            scene_cov3D[mov_mask, ...] = pred_cov
            scene_pcs[mov_mask, ...] = pred_pos
            
            if self.use_rotation:
                scene_rot = torch.eye(pred_rot.shape[-1]).to(pred_rot).expand(mov_mask.shape[0], -1, -1).clone()
                scene_rot[mov_mask, ...] = pred_rot
        else:
            # Only apply to training and testing
            assert cln_gaussian is not None
            r_gaussians = cln_gaussian
            scene_cov3D[mov_mask, ...] = pred_cov
            scene_pcs[mov_mask, ...] = pred_pos
            scene_cov3D = scene_cov3D[cln_mask, ...]
            scene_pcs = scene_pcs[cln_mask, ...]

            if self.use_rotation:
                scene_rot = torch.eye(pred_rot.shape[-1]).to(pred_rot).expand(mov_mask.shape[0], -1, -1).clone()
                scene_rot[mov_mask, ...] = pred_rot
                scene_rot = scene_rot[cln_mask, ...]
        
        scene_opacity = r_gaussians.get_opacity
        if opacity_scalar is not None and not self.render_mov_only:
            fg_op = scene_opacity[mov_mask, ...]
            fg_op *= opacity_scalar
            scene_opacity[mov_mask, ...] = fg_op

        if rollout_size < self.checkpoint_rollout:
            img_list = multi_apply(
                decode_head.pre_render,
                cam_list,
                gt_label if gt_label is not None else [None]*len(cam_list), # This one is for debug
                white_bg_list if white_bg_list is not None else [None]*len(cam_list),
                scene_gaussian=r_gaussians, cov3D_precomp=scene_cov3D, pos=scene_pcs, opacity=scene_opacity, rotation=scene_rot, return_tuple=True)[0]
        else:
            img_list = []
            for cam_idx, cam in enumerate(cam_list):
                pred_img = torch.utils.checkpoint.checkpoint(decode_head.pre_render,
                    cam,
                    gt_label[cam_idx] if gt_label is not None else None,
                    white_bg_list[cam_idx] if white_bg_list is not None else None,
                    r_gaussians, scene_cov3D, scene_pcs, scene_opacity, scene_rot, False)
                img_list.append(pred_img)
        
        return pred_pos, pred_cov, outF_lv_list, img_list, losses
    

    def _encode_decode_train(self, img_list, gt_label):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        assert gt_label is not None
        # Render loss
        decode_head = self.decode_head
        render_loss = decode_head.forward_train(img_list, gt_label)
        losses.update(add_prefix(render_loss, 'decode'))
        return losses

    def _encode_decode_train_static(self, pred_pos, gt_label):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        assert gt_label is not None
        # Render loss
        decode_head = self.decode_head
        static_loss = decode_head.forward_train_static(pred_pos, gt_label)
        losses.update(add_prefix(static_loss, 'decode'))
        return losses

    def _encode_decode_test(self, img_list, gt_label):
        """Run forward function and calculate loss for decode head in
        inference."""
        decode_head = self.decode_head
        logits = decode_head.forward_test(img_list=img_list, gt_label=gt_label, test_cfg=self.test_cfg)
        return logits

    def _rollout_steps(self, num_epoch, num_iter, cfg):
        step_increase_interval = cfg.get('step_increase_interval', None)
        step_increase_magnitude = cfg.get('step_increase_magnitude', None)
        max_rollout_steps = cfg.get('max_rollout_step', None)
        by_epoch = cfg.get('by_epoch', None)

        assert step_increase_magnitude is not None
        assert by_epoch is not None
        assert max_rollout_steps is not None and max_rollout_steps > 0, "Should be at least 1"
        assert step_increase_interval is not None

        if by_epoch:
            counter = num_epoch
        else:
            counter = num_iter
        if step_increase_interval == 0:
            rollout_size = max_rollout_steps
        else:
            rollout_size = (counter // step_increase_interval + 1) * step_increase_magnitude
            rollout_size = min(rollout_size, max_rollout_steps)

        return rollout_size, max_rollout_steps

    def forward_train(self, inputs, gt_label, num_epoch=0, num_iter=0, **kwargs):
        """Forward function for training.

        Args:
            inputs:
                img: n_cam, bs==1, 3, H, W
            gt_label: n_cam(for one seq, multiple camera), n_frame, 3, H, W

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        self.num_iter, self.num_epoch = num_iter+1, num_epoch+1
        # This one input the current num_epoch and num_iter
        rollout_size, _ = self._rollout_steps(num_epoch, num_iter, self.train_cfg)
        random_rs = np.random.rand() < 1/(rollout_size+1)
        static_pred = (random_rs or self.static_loss) and self.opt_sim

        losses = defaultdict(list)
        acc_dict = defaultdict(list)
        
        device = inputs['img'][0].device
        for i in range(len(inputs['cam'])):
            inputs['cam'][i].to_device(device)
        
        seq_idx = kwargs['meta']['seq_idx']
        assert seq_idx.shape[0] == 1 and seq_idx.shape[1] == 1
        seq_idx = seq_idx[0, 0].item()

        scene_name = kwargs['meta']['scene_name']
        assert scene_name in self.gs_scene_dict.keys()
        scene_gaussian = self.gs_scene_dict[scene_name]
        scene_render_gaussian = None if not self.render_mov_only else self.gs_scene_dict_render[scene_name]
        # Related scene mask
        mov_mask = inputs['mov_mask'].squeeze(0)
        pin_mask = inputs['pin_mask'].squeeze(0)
        p2c_mapping_list = [p2c.squeeze(0) for p2c in inputs['p2c_mapping']]
        cln_mask = inputs['cln_mask'].squeeze(0)

        original_state = scene_gaussian.get_xyz
        original_mov_state = original_state[mov_mask, ...]
        original_scaling = scene_gaussian.get_scaling
        original_mov_scaling = original_scaling[mov_mask, ...]
        original_cov = scene_gaussian.get_covariance()
        original_mov_cov = original_cov[mov_mask, ...]
        assert scene_name in self.scene_initn1_pos.keys()
        if self.pred_vel:
            assert scene_name in self.scene_init0_pos.keys()
        assert scene_name in self.scene_attr.keys()

        attr = self.scene_attr[scene_name]
        diag_volume = torch.prod(original_mov_scaling*inputs['volume_scalar'].squeeze(0), dim=-1, keepdim=True)
        
        tn1_state = self.scene_initn1_pos[scene_name][str(seq_idx)] * torch.logical_not(pin_mask) + original_mov_state * pin_mask
        if self.pred_vel:
            t0_state = self.scene_init0_pos[scene_name][str(seq_idx)] * torch.logical_not(pin_mask) + original_mov_state * pin_mask
        cur_state = original_mov_state
        cur_cov = original_mov_cov # No need .clone()
        external_forces = inputs['external'].squeeze(0)

        num_frame = min(gt_label[0].shape[1]-1, rollout_size)
        gt_label_idx_offset = 1
        if static_pred:
            gt_label_idx_offset = 0

        for frame_idx in range(num_frame):
            if static_pred:
                if frame_idx == 0:
                    # Input: t0, t0; Output: t'0
                    prev_state = cur_state
                elif frame_idx == 1:
                    # Input: t'0, t-1; Output: t1
                    prev_state = tn1_state
                    cur_state = original_mov_state
                    if self.pred_vel:
                        cur_state = t0_state
            else:
                if frame_idx == 0:
                    # Input: t0, t-1; Output: t1
                    prev_state = tn1_state
                    if self.pred_vel:
                        cur_state = t0_state
            input_graph, connect_graph = self._preprocess(
                prev_state, cur_state, original_mov_state, attr, diag_volume, cur_cov,
                external_forces,
                p2c_mapping=p2c_mapping_list,
                pin_mask=pin_mask)

            cur_label_idx = frame_idx + gt_label_idx_offset
            cur_label = [gt[0, cur_label_idx] for gt in gt_label]
            pred_pos, pred_cov, pred_dg_list, pred_img_list, losses_i = self.encode_decode(
                input_graph, connect_graph,
                scene_gaussian, inputs['cam'],
                original_cov.detach().clone(), original_state.detach().clone(),
                mov_mask, gt_label=cur_label, register_norm=True,
                rollout_size=rollout_size,
                cln_mask=cln_mask, cln_gaussian=scene_render_gaussian,
                white_bg_list=inputs['white_bg'] if 'white_bg' in inputs.keys() else None
                )
            # st4_time = time.time()
            img_loss = self._encode_decode_train(pred_img_list, cur_label)
            losses_i.update(img_loss)
            if static_pred and cur_label_idx == 0:
                # Opt sim, static in and out
                static_loss = self._encode_decode_train_static(pred_pos, original_mov_state)
                losses_i.update(static_loss)

            # Update the state pointer and rollout to next frame
            if not self.accumulate_gradient:
                prev_state = cur_state.detach()
                cur_state = pred_pos.detach()
            else:
                prev_state = cur_state
                cur_state = pred_pos
            # Merge loss
            for key in losses_i.keys():
                if key.startswith('decode.loss'):
                    # Sum loss
                    losses[f'{key}_{scene_name}'].append(losses_i[key])
                else:
                    acc_key = f'{key}_{scene_name}'
                    acc_key = f"{acc_key}_frame{frame_idx}"
                    acc_dict[acc_key].append(losses_i[key])

        # reduce losses
        losses_rst = dict()
        for key, val in losses.items():
            agg_func = torch.sum
            if self.avg_loss:
                agg_func = torch.mean
            losses_rst[key] = agg_func(torch.stack(val))
        for key, val in acc_dict.items():
            losses_rst[key] = torch.mean(torch.stack(val))
        
        # Align keys
        to_pad_static = True
        static_key = f'decode.loss_mse_static_{scene_name}'
        for key in losses_rst.keys():
            if 'static' in key:
                assert static_key == key
                to_pad_static = False
        if to_pad_static:
            assert static_key not in losses_rst.keys()
            losses_rst[static_key] = torch.zeros(1).to(device)[0]

        return losses_rst

    def inference(self, inputs, gt_label=None, is_training=False, **kwargs):
        """Inference with slide/whole style.
        """
        # Can do something using self.test_cfg
        output = self.encode_decode(inputs, gt_label=gt_label, is_training=is_training, **kwargs)
        
        return output

    def evaluate(self, pred, gt_label, meta_info, **kwargs):
        pass

    def _merge_acc(self, pred_rst_dict):
        merged_rst = dict(acc=dict())
        pred_acc = pred_rst_dict.pop('acc')
        pred_step = len(pred_acc)
        laststep_rst_dict = dict()
        for key, acc_val in pred_acc[-1].items():
            laststep_rst_dict[f'{key}_step{pred_step}'] = acc_val
        merged_rst['acc'] = laststep_rst_dict
        for key, val in pred_rst_dict.items():
            merged_rst[key] = val[-1] # Only choose the last one during validation; Test is ok cuz only one step

        if torch.onnx.is_in_onnx_export():
            return merged_rst
        merged_rst = to_numpy_detach(merged_rst)
        return merged_rst

    def simple_test(self,
                    inputs, gt_label=None,
                    prev_state=None, cur_state=None, cur_cov=None, pred_frame_idx=None, zero_init=False, **kwargs):
        """Simple test with single image.
        
            pred_frame_idx: start from 1, given 0

        """
        assert cur_cov == None
        # This one input the current num_epoch and num_iter
        rollout_size, _ = self._rollout_steps(0, 0, self.test_cfg)
        acc_dict = defaultdict(list)
        
        # Init camera given the camera info
        device = inputs['img'][0].device
        for i in range(len(inputs['cam'])):
            inputs['cam'][i].to_device(device)
        
        seq_idx = kwargs['meta']['seq_idx']
        assert seq_idx.shape[0] == 1 and seq_idx.shape[1] == 1
        seq_idx = seq_idx[0, 0].item()

        scene_name = kwargs['meta']['scene_name']
        assert scene_name in self.gs_scene_dict.keys()
        scene_gaussian = self.gs_scene_dict[scene_name]
        scene_render_gaussian = None if not self.render_mov_only else self.gs_scene_dict_render[scene_name]
        # Related scene mask
        mov_mask = inputs['mov_mask'].squeeze(0)
        pin_mask = inputs['pin_mask'].squeeze(0)
        p2c_mapping_list = [p2c.squeeze(0) for p2c in inputs['p2c_mapping']]
        cln_mask = inputs['cln_mask'].squeeze(0)
        
        original_state = scene_gaussian.get_xyz
        original_mov_state = original_state[mov_mask, ...]
        original_scaling = scene_gaussian.get_scaling
        original_mov_scaling = original_scaling[mov_mask, ...]
        original_cov = scene_gaussian.get_covariance()
        original_mov_cov = original_cov[mov_mask, ...]
        assert scene_name in self.scene_initn1_pos.keys()
        if self.pred_vel:
            assert scene_name in self.scene_init0_pos.keys()
        assert scene_name in self.scene_attr.keys()

        attr = self.scene_attr[scene_name]
        diag_volume = torch.prod(original_mov_scaling*inputs['volume_scalar'].squeeze(0), dim=-1, keepdim=True)

        if pred_frame_idx is None or pred_frame_idx == 1:
            assert prev_state is None and cur_state is None and cur_cov is None
            assert str(seq_idx) in self.scene_initn1_pos[scene_name].keys()
            prev_state = self.scene_initn1_pos[scene_name][str(seq_idx)] * torch.logical_not(pin_mask) + original_mov_state * pin_mask
            cur_state = original_mov_state
            if self.pred_vel:
                assert str(seq_idx) in self.scene_init0_pos[scene_name].keys()
                cur_state = self.scene_init0_pos[scene_name][str(seq_idx)] * torch.logical_not(pin_mask) + original_mov_state * pin_mask
        else:
            assert prev_state is not None and cur_state is not None
        if zero_init:
            cur_state = original_mov_state
            prev_state = cur_state
        cur_cov = original_mov_cov
        external_forces = inputs['external'].squeeze(0)

        num_frame = min(gt_label[0].shape[1]-1, rollout_size) if gt_label is not None else rollout_size
        rst_dict = defaultdict(list)
        for frame_idx in range(num_frame):
            start_time = time.time()
            # Build graph for computing
            input_graph, connect_graph = self._preprocess(
                prev_state, cur_state, original_mov_state, attr, diag_volume, cur_cov,
                external_forces,
                p2c_mapping=p2c_mapping_list,
                pin_mask=pin_mask)
            preprocess_time = time.time()
            cur_label_idx = pred_frame_idx if pred_frame_idx is not None else frame_idx+1
            cur_label = [gt[0, cur_label_idx] for gt in gt_label] # Currently all have gt_label
            pred_pos, pred_cov, pred_dg_list, pred_img_list, losses_i = self.encode_decode(
                input_graph, connect_graph,
                scene_gaussian, inputs['cam'],
                original_cov.detach().clone(), original_state.detach().clone(), # For rendering, better not change this two variable
                mov_mask, gt_label=cur_label, register_norm=False,
                cln_mask=cln_mask, cln_gaussian=scene_render_gaussian,
                white_bg_list=inputs['white_bg'] if 'white_bg' in inputs.keys() else None,
                opacity_scalar=inputs.get('opacity_scalar', None))
            img_loss = self._encode_decode_test(pred_img_list, cur_label)
            img_loss['acc'].update(losses_i)
            rst_dict['acc'].append(img_loss['acc'])
            rst_dict['pred_pos'].append(pred_pos.detach().clone())
            rst_dict['pred_cov'].append(pred_cov.detach().clone())
            rst_dict['cur_state'].append(cur_state.detach().clone())
            rst_dict['pred_img_list'].append(pred_img_list)
            if not self.accumulate_gradient:
                prev_state = cur_state.detach()
                cur_state = pred_pos.detach()
            else:
                prev_state = cur_state
                cur_state = pred_pos
            end_time = time.time()
        return_dict = self._merge_acc(rst_dict)
        return_dict['time'] = [start_time, preprocess_time, end_time]
        return return_dict

    def _em_scheme(self, num_epoch, num_iter, cfg):
        step_increase_interval = cfg.get('step_increase_interval', None)
        by_epoch = cfg.get('by_epoch', None)
        em_step_ratio = cfg.get('em_step_ratio', None)

        assert by_epoch is not None
        assert step_increase_interval is not None
        assert em_step_ratio is not None

        if by_epoch:
            counter = num_epoch + 1
        else:
            counter = num_iter + 1
        
        is_est = False
        if counter % step_increase_interval <= step_increase_interval * em_step_ratio:
            is_est = True
        return is_est
    
    def _freeze_stages(self, model):
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return

    def _train_stages(self, model):
        model.train()
        for param in model.parameters():
            param.requires_grad = True
        return
    
    def train(self, mode=True):
        """Set module status before forward computation.
        Args:
            mode (bool): Whether it is train_mode or test_mode
        """
        super(GsSimulatorHierarchy, self).train(mode)
        if not self.opt_sim:
            if isinstance(self.backbone, nn.ModuleList):
                for i in range(len(self.backbone)):
                    self._freeze_stages(self.backbone[i])
            else:
                self._freeze_stages(self.backbone)
            if isinstance(self.decode_head, nn.ModuleList):
                for i in range(len(self.decode_head)):
                    self._freeze_stages(self.decode_head[i])
            else:
                self._freeze_stages(self.decode_head)
        if not self.opt_vel:
            for key in self.scene_attr.keys():
                self.scene_attr[key].requires_grad = False
                for seq_key in self.scene_initn1_pos[key].keys():
                    self.scene_initn1_pos[key][seq_key].requires_grad = False
                if self.pred_vel:
                    for seq_key in self.scene_init0_pos[key].keys():
                        self.scene_init0_pos[key][seq_key].requires_grad = False
        return

    def aug_test(self, **kwargs):
        """
            Refer mmsegmentation
        """
        raise NotImplementedError("aug_test is not implemented")
