model = dict(
    type='GsSimulatorHierarchy',
    cluster_cfg=dict(
        carnations=[
            dict(downsample_rate=0.01), # Init level
            dict(downsample_rate=0.01),],
    ),
    processor_cfg=[
        dict(
            type='GsHieDynamicDGLProcessor',
            anchor_prefix='anchor_',
            radius=[
                0.01, 0.01*10,
                0.01*100
            ], # TODO: DIFFERENT HERE, must match the cluster_cfg and the one in dataset
            group_cfg=dict(
                max_radius=None,
                min_radius=0.0,
                sample_num=8, # TODO: DIFFERENT HERE
                use_xyz=True,
                normalize_xyz=False,
                return_grouped_xyz=False,
                return_grouped_idx=True,
                return_unique_cnt=False,),
        ),],
    accumulate_gradient=False,
    dt=1/30, # Must align with those in dataset
    backbone=dict(
        type='MeshGraphNetHie',
        attr_dim=5,
        state_dim=6, # pos, vel
        position_dim=3,
        num_frames=2,
        embed_dims=128,
        num_encoder_layers=16,
        dropout=0.0,
        eps=1e-7,
        num_fcs=2,
        act_cfg=dict(type='SiLU', inplace=True),
        norm_cfg=dict(type='LN'),
        pre_norm=True,
        norm_acc_steps=None,
        ),
    decode_head=dict(
        type='AccDecoder',
        in_channels=128,
        out_channels=4+3+4,
        add_residual=False,
        init_quant=1.0,
        loss_decode=[
            dict(type='MSELoss', reduction='sum', loss_weight=1.0, loss_name='loss_mse_momentum'),
            dict(type='SSIMLoss', reduction='sum', kernel_size=5, loss_weight=1.0, loss_name='loss_ssim_render'),
            dict(type='L2Loss', reduction='sum', loss_weight=1.0, loss_name='loss_l2_render'),
            dict(type='MSELoss', reduction='sum', loss_weight=1.0, loss_name='loss_mse_static'),
            ],
        accuracy=[
            dict(type='L2Accuracy', reduction='mean', acc_name='acc_l2_render'),
            ],
        ),
    gs_scene=[
        dict(name='carnations', model_path='data_v2/carnations/point_cloud.ply', mov_mask_path='data_v2/carnations/pc_mask.pkl', cln_mask_path='data_v2/carnations/cln_pc_mask.pkl', sh_degree=3, num_seq=36),
    ],
    forward_last_layer=False,
    opt_sim=True,
    opt_vel=False,
    selfsup_loss=False,
)