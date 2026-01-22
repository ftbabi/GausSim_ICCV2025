import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import (build_activation_layer, build_norm_layer)
from mmcv.runner import BaseModule, ModuleList
from mmcv.ops import QueryAndGroup, grouping_operation

from .. import builder
from ..builder import BACKBONES
from .base_backbone import BaseBackbone
from mmgs.models.utils import (FFN, Normalizer)
from mmgs.datasets.utils import normalize
from mmgs.core import multi_apply
import dgl
import dgl.function as fn


class MeshGraphNetEncoderLayer(BaseModule):
    """Implements one encoder layer in transformer.
    Args:
        embed_dims (int): The feature dimension. Same as `FFN`.
        dropout (float): Probability of an element to be zeroed. Default 0.0.
        order (tuple[str]): The order for encoder layer. Valid examples are
            ('selfattn', 'norm', 'ffn', 'norm') and ('norm', 'selfattn',
            'norm', 'ffn'). Default ('selfattn', 'norm', 'ffn', 'norm').
        act_cfg (dict): The activation config for FFNs. Default ReLU.
        norm_cfg (dict): Config dict for normalization layer. Default
            layer normalization.
    """

    def __init__(self,
                 embed_dims,
                 dropout=0.0,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 pre_norm=False,
                 **kwargs):
        super(MeshGraphNetEncoderLayer, self).__init__()
        self.embed_dims = embed_dims
        self.dropout = dropout
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.pre_norm = pre_norm

        self.norms = ModuleList()
        if self.pre_norm:
            self.norms.append(build_norm_layer(norm_cfg, embed_dims*3)[1])
            self.norms.append(build_norm_layer(norm_cfg, embed_dims*2)[1])
        else:
            self.norms.append(build_norm_layer(norm_cfg, embed_dims)[1])
            self.norms.append(build_norm_layer(norm_cfg, embed_dims)[1])

        # For receiver and sender
        self.edge_weight = FFN([embed_dims*3, embed_dims, embed_dims, embed_dims], bias=True, act_cfg=act_cfg, add_residual=True)
        self.node_weight = FFN([embed_dims*2, embed_dims, embed_dims, embed_dims], bias=True, act_cfg=act_cfg, add_residual=True)

    def interact_feature(self, mlp_func, norm_func, edge_field, src_field, dst_field, out_field):
        '''
            src: sender
            dst: receiver

            edge_field: world/mesh
            src/dst_field: n_feature,
            out_field: n_feature
        '''
        def func(edges):
            # n_edge, n_head, head_dim
            sender = edges.src[src_field]
            receiver = edges.dst[dst_field]
            interactions = edges.data[edge_field]
            in_emb = torch.cat([interactions, receiver, sender], dim=-1)
            if self.pre_norm:
                in_emb = norm_func(in_emb)
            f = mlp_func(in_emb, residual=interactions)
            if not self.pre_norm:
                f = norm_func(f)
            return {out_field: f}
        return func
    
    def node_feature(self, mlp_func, norm_func, node_field, edge_field, out_field):
        def func(nodes):
            node_f = nodes.data[node_field]
            mesh_f = nodes.data[edge_field]
            in_emb = torch.cat([node_f, mesh_f], dim=-1)
            if self.pre_norm:
                in_emb = norm_func(in_emb)
            f = mlp_func(in_emb, residual=node_f)
            if not self.pre_norm:
                f = norm_func(f)
            return {out_field: f}
        return func

    def forward(self, g, out_node_field, out_edge_field):
        """Forward function for `TransformerEncoderLayer`.
        Args:
            x (Tensor): The input query with shape [num_key, bs,
                embed_dims]. Same in `MultiheadAttention.forward`.
            pos (Tensor): The positional encoding for query. Default None.
                Same as `query_pos` in `MultiheadAttention.forward`.
            attn_mask (Tensor): ByteTensor mask with shape [num_key,
                num_key]. Same in `MultiheadAttention.forward`. Default None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_key].
                Same in `MultiheadAttention.forward`. Default None.
            receiver_val_res: n_particles, bs, embed_dims
        """
        # Update nodes
        # Interact
        g.apply_edges(self.interact_feature(self.edge_weight, self.norms[0], out_edge_field, out_node_field, out_node_field, out_edge_field))
        g.send_and_recv(g.edges(), fn.copy_e(out_edge_field, out_edge_field), fn.sum(out_edge_field, out_edge_field))
        if g.num_edges() == 0:
            # Nodes are too far away
            g.ndata[out_edge_field] = torch.zeros_like(g.ndata[out_node_field]).to(g.ndata[out_node_field])
        # Update nodes
        g.apply_nodes(self.node_feature(self.node_weight, self.norms[1], out_node_field, out_edge_field, out_node_field))

        return g


class MeshGraphNetEncoder(BaseModule):
    """Implements the encoder in transformer.
    Args:
        num_layers (iTimeEmbeddingmalization.
    """

    def __init__(self,
                 num_layers,
                 embed_dims,
                 dropout=0.0,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 pre_norm=False,
                 **kwargs):
        super(MeshGraphNetEncoder, self).__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.dropout = dropout
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.layers = ModuleList()
        self.pre_norm = pre_norm

        if pre_norm:
            self.emb_norm = build_norm_layer(norm_cfg, self.embed_dims)[1]
        for _ in range(num_layers):
            self.layers.append(
                MeshGraphNetEncoderLayer(embed_dims, dropout, act_cfg, norm_cfg, pre_norm=pre_norm, **kwargs))

    def forward(self, g, out_node_field, out_edge_field):
        """Forward function for `TransformerEncoder`.
        Args:
            x (Tensor): Input query. Same in `TransformerEncoderLayer.forward`.
            pos (Tensor): Positional encoding for query. Default None.
                Same in `TransformerEncoderLayer.forward`.
            attn_mask (Tensor): ByteTensor attention mask. Default None.
                Same in `TransformerEncoderLayer.forward`.
            key_padding_mask (Tensor): Same in
                `TransformerEncoderLayer.forward`. Default None.
            receiver_val_res: n_particles, bs, embed_dims
        """
        for layer in self.layers:
            g = layer(g, out_node_field, out_edge_field)
        return g


@BACKBONES.register_module()
class MeshGraphNetHie(BaseBackbone):
    """Implements the simulation transformer.
    """

    def __init__(self,
                 attr_dim=5,
                 state_dim=6,
                 position_dim=3,
                 num_frames=1+1,
                 embed_dims=128,
                 num_encoder_layers=4,
                 dropout=0.0,
                 eps=1e-7,
                 num_fcs=2,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 pre_norm=False,
                 dt=1/30,
                 norm_acc_steps=None,
                 **kwargs):
        super(MeshGraphNetHie, self).__init__()
        self.attr_dim = attr_dim
        self.state_dim = state_dim
        self.position_dim = position_dim
        self.embed_dims = embed_dims
        self.eps = eps
        self.num_frames = num_frames
        self.pre_norm = pre_norm
        assert num_frames == 2
        self.dt = dt
        self.norm_acc_steps = norm_acc_steps
        
        # Attribute
        ## attr_dim
        self.attr_encoder = nn.Identity()
        self.attr_normalizer = nn.Identity()
        # States embeddings
        ## vel
        node_dim = 2*position_dim+attr_dim
        self.node_encoder = FFN(
            [node_dim] + [embed_dims for i in range(num_fcs)], 
            final_act=True, bias=True)
        self.node_normalizer = Normalizer(position_dim) if norm_acc_steps is None else Normalizer(position_dim, max_accumulations=norm_acc_steps)
        self.anchor_normalizer = Normalizer(position_dim) if norm_acc_steps is None else Normalizer(position_dim, max_accumulations=norm_acc_steps)
        self.node_norm = build_norm_layer(norm_cfg, embed_dims)[1] if not pre_norm else nn.Identity()
        mesh_in_dim = 3+1
        mesh_in_dim += 3
        mesh_in_enc_dim = mesh_in_dim
        mesh_in_enc_dim += 2 # cos, sin
        self.edge_encoder = FFN(
            [mesh_in_enc_dim] + [embed_dims for i in range(num_fcs)], 
            final_act=True, bias=True)
        self.edge_normalizer = Normalizer(mesh_in_dim) if norm_acc_steps is None else Normalizer(mesh_in_dim, max_accumulations=norm_acc_steps)
        self.edge_norm = build_norm_layer(norm_cfg, embed_dims)[1] if not pre_norm else nn.Identity()

        self.encoder = MeshGraphNetEncoder(num_encoder_layers, embed_dims, dropout, act_cfg, norm_cfg, pre_norm=pre_norm, **kwargs)

    def _edge_theta(self, recv_state, send_state, anchor_recv_state, anchor_send_state):
        recv_vec = recv_state - anchor_recv_state
        send_vec = send_state - anchor_send_state
        normed_recv = recv_vec / torch.linalg.norm(recv_vec, dim=-1, keepdim=True)
        normed_send = send_vec / torch.linalg.norm(send_vec, dim=-1, keepdim=True)
        cos = torch.sum(normed_recv*normed_send, dim=-1, keepdim=True)
        sin = torch.linalg.norm(torch.cross(normed_recv, normed_send, dim=-1), dim=-1, keepdim=True)
        return cos, sin

    def init_edge_features(self, cur_state_field, template_state_field, anchor_next_state_field, anchor_template_state_field, edge_out_field):
        def func(edges):
            # n_edges, 3
            recv_state = edges.dst[cur_state_field]
            send_state = edges.src[cur_state_field]
            recv_tem = edges.dst[template_state_field]
            send_tem = edges.src[template_state_field]

            # n_edges, 3
            delta_state = (recv_state - send_state)
            delta_pos = delta_state
            # n_edges, 1
            norm_delta_pos = torch.linalg.norm(delta_pos, dim=-1, keepdim=True)

            delta_tem = recv_tem - send_tem
            norm_delta_tem = torch.linalg.norm(delta_tem, dim=-1, keepdim=True)

            in_emb = self.edge_normalizer(torch.cat([delta_pos/norm_delta_pos, norm_delta_pos/norm_delta_tem], dim=-1))
            # theta in template state
            anchor_recv_tem = edges.dst[anchor_template_state_field]
            anchor_send_tem = edges.src[anchor_template_state_field]
            tem_cos, tem_sin = self._edge_theta(recv_tem, send_tem, anchor_recv_tem, anchor_send_tem)
            anchor_recv_state = edges.dst[anchor_next_state_field]
            anchor_send_state = edges.src[anchor_next_state_field]
            cur_cos, cur_sin = self._edge_theta(recv_state, send_state, anchor_recv_state, anchor_send_state)
            # cos(a-b) = cosAcosB + sinAsinB
            # sin(A-B) = sinAcosB - cosAsinB
            delta_cos = cur_cos*tem_cos + cur_sin*tem_sin
            delta_sin = cur_sin*tem_cos - cur_cos*tem_sin
            in_emb = torch.cat([in_emb, delta_cos, delta_sin], dim=-1)

            out_emb = self.edge_norm(self.edge_encoder(in_emb))
            
            return {edge_out_field: out_emb}
        return func
    
    def init_edge_features_fixbug(self, cur_state_field, prev_state_field, orig_template_state_field, anchor_next_state_field, anchor_orig_template_state_field, edge_out_field):
        def func(edges):
            # n_edges, 3
            recv_state = edges.dst[cur_state_field]
            send_state = edges.src[cur_state_field]
            recv_prev = edges.dst[prev_state_field]
            send_prev = edges.src[prev_state_field]

            recv_orig_tem = edges.dst[orig_template_state_field]
            send_orig_tem = edges.src[orig_template_state_field]

            # n_edges, 3
            delta_state = recv_state - send_state
            delta_pos = delta_state
            delta_prev = recv_prev - send_prev
            ## under the recv coordinate, the relative vel between recv and send
            delta_rel_vel = (delta_state - delta_prev) / self.dt
            # n_edges, 1
            norm_delta_pos = torch.linalg.norm(delta_pos, dim=-1, keepdim=True)

            delta_orig_tem = recv_orig_tem - send_orig_tem
            norm_delta_orig_tem = torch.linalg.norm(delta_orig_tem, dim=-1, keepdim=True)
            in_emb = self.edge_normalizer(torch.cat([delta_pos/norm_delta_pos, norm_delta_pos/norm_delta_orig_tem, delta_rel_vel], dim=-1))

            anchor_recv_tem = edges.dst[anchor_orig_template_state_field]
            tem_cos, tem_sin = self._edge_theta(recv_orig_tem, send_orig_tem, anchor_recv_tem, anchor_recv_tem)
            anchor_recv_state = edges.dst[anchor_next_state_field]
            cur_cos, cur_sin = self._edge_theta(recv_state, send_state, anchor_recv_state, anchor_recv_state)
            # cos(a-b) = cosAcosB + sinAsinB
            # sin(A-B) = sinAcosB - cosAsinB
            delta_cos = cur_cos*tem_cos + cur_sin*tem_sin
            delta_sin = cur_sin*tem_cos - cur_cos*tem_sin
            in_emb = torch.cat([in_emb, delta_cos, delta_sin], dim=-1)

            out_emb = self.edge_norm(self.edge_encoder(in_emb))
            
            return {edge_out_field: out_emb}
        return func
    
    def init_node_features(self,
                           anchor_prev_state_field, anchor_cur_state_field, anchor_next_state_field,
                           prev_state_field, cur_state_field, 
                           attr_field, external_field, pin_mask_field, node_out_field):
        def func(nodes):
            attr = nodes.data[attr_field]
            pin_mask = nodes.data[pin_mask_field]
            external_forces = nodes.data[external_field]
            anchor_prev_state = nodes.data[anchor_prev_state_field]
            anchor_cur_state = nodes.data[anchor_cur_state_field]
            anchor_next_state = nodes.data[anchor_next_state_field]
            anchor_acc = (anchor_next_state - 2*anchor_cur_state + anchor_prev_state) / (self.dt**2)
            anchor_acc = -anchor_acc + external_forces
            prev_state = nodes.data[prev_state_field]
            cur_state = nodes.data[cur_state_field]
            vel = ((cur_state-anchor_next_state) - (prev_state-anchor_prev_state)) / self.dt * torch.logical_not(pin_mask)

            norm_anchor_acc = self.anchor_normalizer(anchor_acc)
            norm_vel = self.node_normalizer(vel)
            norm_attr = self.attr_normalizer(attr)

            in_dynamic = torch.cat([norm_anchor_acc, norm_vel], dim=-1)
            in_raw = torch.cat([in_dynamic, norm_attr], dim=-1)
            in_emb = self.node_encoder(in_raw)
            in_feature = self.node_norm(in_emb)
            return {node_out_field: in_feature}
        return func

    def init_features(self, graph,
                      anchor_prev_state_field, anchor_cur_state_field, anchor_next_state_field,
                      prev_state_field, cur_state_field,
                      template_state_field, anchor_template_state_field,
                      orig_template_state_field, anchor_orig_template_state_field,
                      attr_field, external_field, pin_mask_field,
                      node_out_field, edge_out_field):
        
        # Apply to all nodes
        graph.apply_nodes(self.init_node_features(
            anchor_prev_state_field, anchor_cur_state_field, anchor_next_state_field,
            prev_state_field, cur_state_field,
            attr_field, external_field, pin_mask_field, node_out_field))
        # Apply to all edges
        graph.apply_edges(self.init_edge_features_fixbug(
            cur_state_field, prev_state_field, orig_template_state_field, anchor_next_state_field, anchor_orig_template_state_field, edge_out_field))
        return graph

    def forward(self, graph, connect_graph, **kwargs):
        # Embed features
        out_node_field = 'out_node'
        out_edge_field = 'out_edge'
        graph = self.init_features(
            graph,
            'anchor_prev_state', 'anchor_cur_state', 'anchor_next_state',
            'prev_state', 'cur_state',
            'template_state', 'anchor_template_state',
            'orig_template_state', 'anchor_orig_template_state',
            'attr', 'external', 'pin_mask',
            out_node_field, out_edge_field)
        # Propagate
        g_enc = self.encoder(
            graph, out_node_field=out_node_field, out_edge_field=out_edge_field)
        return g_enc
