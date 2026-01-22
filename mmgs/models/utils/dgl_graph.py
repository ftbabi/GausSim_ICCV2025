import numpy as np

import torch
import torch.nn as nn

from mmcv.runner import BaseModule, ModuleList
from mmcv.ops import QueryAndGroup, grouping_operation

import dgl
import dgl.function as fn
from mmgs.core import multi_apply

from functools import partial
from collections import OrderedDict

from ..builder import PREPROCESSOR


def density_activation():
    return torch.exp

def density_deactivation():
    return torch.log

RECEIVER_ID = 'receiver'
SENDER_ID = 'sender'

VERT_ID = 'vertices'

FORCE_ID = 'forces'
FORCE_EDGE = 'force_e'
DISTRIBUTE_EDGE = 'tov_e'

# For compute only
CPT_EDGE_NEIGH = 'e_neigh'

REL_DELTA_L2 = [0, 1]
REL_EDGE_DIR = [1, 4]
REL_VEL_L2 = [4, 5]
REL_VEL_DIR = [5, 8]

# For gs sim
P2C_EDGE_ID = 'p2c_mask'
C2P_EDGE_ID = 'c2p_mask'
POINT_VERT_ID = 'point_mask'
CLUSTER_VERT_ID = 'cluster_mask'

    
@PREPROCESSOR.register_module()
class GsHieDynamicDGLProcessor(object):
    def __init__(self, radius, group_cfg, anchor_prefix='anchor_', eps=1e-7, **kwargs) -> None:
        self.radius = radius
        if isinstance(group_cfg, list):
            assert len(group_cfg) == len(radius)
            self.grouper = [
                QueryAndGroup(**g_cfg) for g_cfg in group_cfg]
        else:
            assert isinstance(group_cfg, dict)
            self.grouper = QueryAndGroup(**group_cfg)
        self.eps = eps
        self.graph_builder = BuildGsDGLGraph(**kwargs)
        self.anchor_prefix = anchor_prefix
        self.density_activation = density_activation()
        self.density_deactivation = density_deactivation()

    def _preprocess(self,
                    prev_state, cur_state, template_state, attr, diag_volume, external_forces, pin_mask, dynamic_radius, grouper, dynamic_base):

        if dynamic_base:
            assert dynamic_base is False, "Need debug"
            static_edges = self._dynamic_edges(template_state.unsqueeze(0), template_state.unsqueeze(0), radius=dynamic_radius, grouper=grouper)
            dynamic_edges = self._dynamic_edges(cur_state.unsqueeze(0), cur_state.unsqueeze(0), radius=dynamic_radius, grouper=grouper, non_intersect_relations=static_edges)
            dynamic_edges = torch.cat([static_edges, dynamic_edges], dim=0)
        else:
            dynamic_edges = None

        assert attr is not None
        n_nodes = prev_state.shape[0]
        if external_forces.shape[0] == 1:
            in_external = external_forces.expand(n_nodes, -1)
        else:
            assert external_forces.shape[0] == n_nodes
            in_external = external_forces
        node_wise_dict = {
            'attr': attr,
            'diag_volume': diag_volume,
            'prev_state': prev_state,
            'cur_state': cur_state,
            'template_state': template_state,
            'orig_template_state': template_state.detach().clone(),
            'pin_mask': pin_mask,
            'external': in_external}
        
        # Build Graph
        g = self.graph_builder.base_graph(
            node_wise_dict, dynamic_edges)
        return g

    def _preprocess_hierarchy(self, prev_graph, connect_edges, node_attr_scheme, dynamic_radius, grouper):
        n_points = prev_graph.num_nodes()
        assert n_points == connect_edges.shape[0]
        n_clusters = torch.max(connect_edges)+1

        offseted_connect_edges = connect_edges+n_points
        p2c_edge_pair = torch.stack([offseted_connect_edges, torch.arange(offseted_connect_edges.shape[0]).to(offseted_connect_edges)], dim=-1)
        c2p_edge_pair = torch.stack([torch.arange(offseted_connect_edges.shape[0]).to(offseted_connect_edges), offseted_connect_edges], dim=-1)

        point_vert_mask = torch.zeros((n_points+n_clusters, 1)).float().to(offseted_connect_edges.device)
        point_vert_mask[:n_points] = 1.0
        connect_node_wise_dict = {
            POINT_VERT_ID: point_vert_mask,
            CLUSTER_VERT_ID: 1-point_vert_mask}
        connect_graph = self.graph_builder.connect_graph(n_points, n_clusters, p2c_edge_pair, c2p_edge_pair, connect_node_wise_dict)
        point_nids = torch.nonzero(connect_graph.ndata[POINT_VERT_ID][:, 0], as_tuple=False).squeeze()
        cluster_nids = torch.nonzero(connect_graph.ndata[CLUSTER_VERT_ID][:, 0], as_tuple=False).squeeze()
        p2c_eids = torch.nonzero(connect_graph.edata[P2C_EDGE_ID][:, 0], as_tuple=False).squeeze()
        hie_node_wise_dict = dict()
        for key, mode in node_attr_scheme.items():
            val = prev_graph.ndata[key]
            connect_graph.nodes[point_nids].data[key] = val
            if key == 'attr':
                density_key = 'density'
                volumn_key = 'diag_volume'
                connect_graph.send_and_recv(p2c_eids, fn.copy_u(key, key), fn.mean(key, key))
                connect_graph.apply_nodes(lambda x: {density_key: self.density_activation(x.data['attr'][:, :1])})
                connect_graph.send_and_recv(p2c_eids, lambda x: {density_key: x.src[density_key]*x.src[volumn_key]/x.dst[volumn_key]}, fn.sum(density_key, density_key))
                connect_graph.apply_nodes(lambda x: {key: torch.cat([self.density_deactivation(x.data[density_key]), x.data[key][:, 1:]], dim=-1)}, cluster_nids)
            elif isinstance(mode, dict):
                weighted_key = mode['key']
                assert mode['reduct'] == 'mean', "Currently only support mean"
                assert weighted_key in connect_graph.ndata.keys()
                connect_graph.send_and_recv(p2c_eids, lambda x: {key: x.src[key]*x.src[weighted_key]}, fn.sum(key, key))
                if mode['reduct'] == 'mean':
                    connect_graph.nodes[cluster_nids].data[key] = connect_graph.nodes[cluster_nids].data[key] / connect_graph.nodes[cluster_nids].data[weighted_key]
            else:
                assert isinstance(mode, str)
                if mode == 'mean':
                    connect_graph.send_and_recv(p2c_eids, fn.copy_u(key, key), fn.mean(key, key))
                elif mode == 'or':
                    connect_graph.send_and_recv(p2c_eids, fn.copy_u(key, key), fn.max(key, key))
                elif mode == 'sum':
                    connect_graph.send_and_recv(p2c_eids, fn.copy_u(key, key), fn.sum(key, key))
                else:
                    assert False, "Not support yet"
            hie_node_wise_dict[key] = connect_graph.nodes[cluster_nids].data[key]

        hie_static_edges = self._dynamic_edges(hie_node_wise_dict['template_state'].unsqueeze(0), hie_node_wise_dict['template_state'].unsqueeze(0), radius=dynamic_radius, grouper=grouper)
        ## Dynamic edges
        hie_dynamic_edges = self._dynamic_edges(hie_node_wise_dict['cur_state'].unsqueeze(0), hie_node_wise_dict['cur_state'].unsqueeze(0), radius=dynamic_radius, grouper=grouper, non_intersect_relations=hie_static_edges)
        hie_edges = torch.cat([hie_static_edges, hie_dynamic_edges], dim=0)
        ## Build graph
        hie_g = self.graph_builder.base_graph(
            hie_node_wise_dict, hie_edges)
        return hie_g, connect_graph

    def _preprocess_abs_hierarchy(self, prev_graph, connect_edges, pinned_node_wise_dict):
        n_points = prev_graph.num_nodes()
        assert n_points == connect_edges.shape[0]
        n_clusters = torch.max(connect_edges)+1

        # Move the offset
        offseted_connect_edges = connect_edges+n_points
        p2c_edge_pair = torch.stack([offseted_connect_edges, torch.arange(offseted_connect_edges.shape[0]).to(offseted_connect_edges)], dim=-1)
        c2p_edge_pair = torch.stack([torch.arange(offseted_connect_edges.shape[0]).to(offseted_connect_edges), offseted_connect_edges], dim=-1)

        point_vert_mask = torch.zeros((n_points+n_clusters, 1)).float().to(offseted_connect_edges.device)
        point_vert_mask[:n_points] = 1.0
        connect_node_wise_dict = {
            POINT_VERT_ID: point_vert_mask,
            CLUSTER_VERT_ID: 1-point_vert_mask}
        connect_graph = self.graph_builder.connect_graph(n_points, n_clusters, p2c_edge_pair, c2p_edge_pair, connect_node_wise_dict)

        point_nids = torch.nonzero(connect_graph.ndata[POINT_VERT_ID][:, 0], as_tuple=False).squeeze()
        cluster_nids = torch.nonzero(connect_graph.ndata[CLUSTER_VERT_ID][:, 0], as_tuple=False).squeeze()
        ## Copying, broadcasting, extracting
        for key in pinned_node_wise_dict.keys():
            connect_graph.nodes[point_nids.long()].data[key] = prev_graph.ndata[key]
            connect_graph.nodes[cluster_nids.long()].data[key] = pinned_node_wise_dict[key]
        return connect_graph

    def _dynamic_edges(self, receiver_pos, sender_pos, radius, grouper, non_intersect_relations=None):
        group_xyz_diff, group_idx = grouper(
            sender_pos.contiguous(),
            receiver_pos.contiguous())
        group_xyz_diff = group_xyz_diff.permute(0, 2, 3, 1)
        group_xyz_l2 = torch.sqrt(torch.sum(group_xyz_diff**2, dim=-1, keepdim=True))
        group_xyz_l2_mask = group_xyz_l2 < radius
        group_xyz_l2_mask = group_xyz_l2_mask.squeeze(0).squeeze(-1)
        group_idx = group_idx.squeeze(0)
        valid_idx = torch.where(group_xyz_l2_mask == True)
        valid_neighbor = group_idx[valid_idx].to(torch.int64)
        relation_pair = torch.unique(torch.stack([valid_idx[0], valid_neighbor], dim=-1), dim=0)
        relation_non_loop = torch.where(relation_pair[:, 0] != relation_pair[:, 1])
        relation_pair = relation_pair[relation_non_loop]
        if non_intersect_relations is not None:
            candidates = torch.cat([relation_pair.transpose(-1, -2), non_intersect_relations.transpose(-1, -2), non_intersect_relations.transpose(-1, -2)], dim=-1).transpose(-1, -2)
            r_uniques, r_counts = torch.unique(candidates, dim=0, return_counts=True)
            r_pairs = r_uniques[r_counts == 1]
            relation_pair = r_pairs
        return relation_pair
    
    def _postprocess_hie(self, base_g, connect_g, forward_keys, prefix='anchor_'):
        point_nids = torch.nonzero(connect_g.ndata[POINT_VERT_ID][:, 0], as_tuple=False).squeeze()
        c2p_eids = torch.nonzero(connect_g.edata[C2P_EDGE_ID][:, 0], as_tuple=False).squeeze()
        for key in forward_keys:
            connect_g.send_and_recv(c2p_eids, fn.copy_u(key, key), fn.mean(key, prefix+key))
            base_g.ndata[prefix+key] = connect_g.nodes[point_nids].data[prefix+key]
        return base_g, connect_g

    def batch_preprocess(self,
                         prev_state, cur_state, template_state, attr, diag_volume, cur_cov, external_forces,
                         p2c_mapping=None, pin_mask=None, dynamic_base=True):
        assert len(p2c_mapping) == len(self.radius)
        graph_list = []
        connect_graph_list = []
        base_g = self._preprocess(
            prev_state, cur_state, template_state, attr, diag_volume, external_forces, pin_mask,
            dynamic_radius=self.radius[0],
            grouper=self.grouper[0] if isinstance(self.grouper, list) else self.grouper,
            dynamic_base=dynamic_base)
        graph_list.append(base_g)

        node_attr_scheme = OrderedDict(
            # Static
            diag_volume='sum',
            attr='mean',
            pin_mask='or',
            external='mean',
            # Dynamic
            prev_state=dict(reduct='mean', key='diag_volume'),
            cur_state=dict(reduct='mean', key='diag_volume'),
            template_state=dict(reduct='mean', key='diag_volume'),
            orig_template_state=dict(reduct='mean', key='diag_volume'),
        )

        for cluster_idx in range(len(p2c_mapping)-1):
            p2c_map = p2c_mapping[cluster_idx]
            # Build connection graph
            connect_edges = p2c_map.long()
            prev_graph = graph_list[-1]
            hie_g, connect_g = self._preprocess_hierarchy(prev_graph, connect_edges, node_attr_scheme, dynamic_radius=self.radius[cluster_idx+1], grouper=self.grouper[cluster_idx+1] if isinstance(self.grouper, list) else self.grouper)
            connect_graph_list.append(connect_g)
            graph_list.append(hie_g)
        
        pinned_nodes = base_g.nodes[torch.where(pin_mask.bool()[:, 0])[0]]
        pinned_node_wise_dict = {
            key: val
            for key, val in pinned_nodes.data.items()}
        pin_connect_g = self._preprocess_abs_hierarchy(graph_list[-1], p2c_mapping[-1], pinned_node_wise_dict)
        connect_graph_list.append(pin_connect_g)

        # Post process
        hie_forward_keys = [
            'prev_state',
            'cur_state',
            'template_state',
            'orig_template_state',]
        for i in range(len(graph_list)):
            cur_base_g, cur_connect_g = graph_list[i], connect_graph_list[i]
            update_base_g, update_connect_g = self._postprocess_hie(cur_base_g, cur_connect_g, hie_forward_keys, prefix=self.anchor_prefix)
            graph_list[i], connect_graph_list[i] = update_base_g, update_connect_g

        # Append the cov for only the pc
        graph_list[0].ndata['cur_cov'] = cur_cov
        return graph_list, connect_graph_list
    

class BuildGsDGLGraph:
    def __init__(self, receiver_id=None, sender_id=None) -> None:
        self.receiver_id = VERT_ID
        self.sender_id = VERT_ID
        if receiver_id is not None:
            self.receiver_id = receiver_id
        if sender_id is not None:
            self.sender_id = sender_id

    def build_graph(self,
            node_wise_dict, dynamic_edges):
        # Static graph
        base_g = self.base_graph(node_wise_dict, dynamic_edges)
        return base_g

    def base_graph(self, node_wise_dict, dynamic_edges):
        num_verts = list(node_wise_dict.values())[0].shape[0]
        device = list(node_wise_dict.values())[0].device
        mesh_rel = dynamic_edges
        if mesh_rel is not None:
            g = dgl.graph((mesh_rel[:, 1], mesh_rel[:, 0]), num_nodes=num_verts, idtype=torch.int64)
        else:
            g = dgl.graph(((), ()), num_nodes=num_verts, idtype=torch.int64).to(device)
        for key, val in node_wise_dict.items():
            assert val.shape[0] == num_verts
            g.ndata[key] = val
        return g
    
    def connect_graph(self, num_points, num_clusters, p2c_edges, c2p_edges, connect_node_wise_dict):
        device = None
        g = dgl.graph((p2c_edges[:, 1], p2c_edges[:, 0]), num_nodes=num_points+num_clusters, idtype=torch.int64)
        for key, val in connect_node_wise_dict.items():
            assert val.shape[0] == num_points+num_clusters
            g.ndata[key] = val
            if device == None:
                device = val.device
        g.edata[P2C_EDGE_ID] = torch.ones((p2c_edges.shape[0], 1)).float().to(device)
        g.add_edges(
            c2p_edges[:, 1], c2p_edges[:, 0],
            data={C2P_EDGE_ID: torch.ones((c2p_edges.shape[0], 1)).float().to(device)})
        return g


