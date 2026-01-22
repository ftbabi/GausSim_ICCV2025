# 1. loss; 2. build and freeze; 3. Iterbasedrunner
_base_ = [
    '../../_base_/models/gs_simulator_hierarchy.py',
    '../../_base_/datasets/gs_ready_repeat.py',
    '../../_base_/schedules/adam_iter.py',
    '../../_base_/default_runtime.py'
]
n_gpu = 4
find_unused_parameters = True

# New customized option
max_cam_num=4
lr_decay_steps=2
lr_step_start=16
max_epoch=16+30
step_increase_epoch=1
dataset_times = 70

step_increase_interval = step_increase_epoch
step_increase_magnitude = 1
max_rollout_step = 16
by_epoch = True
test_max_frame = 3 + max_rollout_step
test_start_frame = max_rollout_step
assert test_start_frame < test_max_frame
start_frame = 0
max_seq = 30

num_sample = 1
num_worker = 8

num_encoder_layers = 16
selfsup_loss=False
avg_loss=True
data_dir='data_real/'
# For mov only
render_mov_only=True
sub_video_dir='video_mov_images'
sub_img_dir='images_mov'
use_random_background=dict(
    pudding=False,
)
const_white_bg=dict(
    pudding=False,
)

rot_est=dict(
    # scalar-last (x, y, z, w)
    pudding=[0.312, -0.002, 0.000818, 0.950],)
real_dt=dict(
    pudding=1/50, # This is different
)

gs_scene=[
        dict(name='pudding', model_path='data_real/pudding/point_cloud.ply', mov_mask_path='data_real/pudding/pc_mask.pkl', cln_mask_path='data_real/pudding/cln_pc_mask.pkl', sh_degree=3, num_seq=max_seq),
    ]
scene_list=[
    'pudding',
    ]

cluster_type='dis'
cluster_cfg=dict(
        pudding=[
            dict(downsample_rate=0.04),
            dict(downsample_rate=0.4),],
)

use_rotation = False
pred_vel = False # Default
llffhold=0 # For evaluation
volume_scalar=dict(pudding=300)

# Custom model
model = dict(
    pred_vel=pred_vel,
    use_rotation=use_rotation,
    cluster_cfg=cluster_cfg,
    gs_scene=gs_scene,
    processor_cfg=[
        dict(
            type='GsHieDynamicDGLProcessor',
            anchor_prefix='anchor_',
            radius=[
                0.03, 0.04*10,
                0.05*100
            ],
            group_cfg=dict(
                max_radius=None,
                min_radius=0.0,
                sample_num=16,
                use_xyz=True,
                normalize_xyz=False,
                return_grouped_xyz=False,
                return_grouped_idx=True,
                return_unique_cnt=False,),
        ),],
    opt_sim=True,
    opt_vel=True,
    static_loss=False,
    selfsup_loss=selfsup_loss,
    avg_loss=avg_loss,
    render_mov_only=render_mov_only,
    train_cfg=dict(
        by_epoch=by_epoch,
        step_increase_interval=step_increase_interval, # for epoch
        step_increase_magnitude=step_increase_magnitude,
        max_rollout_step=max_rollout_step,),
    test_cfg=dict(
        by_epoch=by_epoch,
        step_increase_interval=step_increase_interval, # for epoch
        step_increase_magnitude=step_increase_magnitude,
        max_rollout_step=max_rollout_step,),
    backbone=dict(
        num_encoder_layers=num_encoder_layers,
        num_fcs=3,
        ),
    decode_head=dict(
        loss_decode=[
            dict(type='MSELoss', reduction='sum', loss_weight=1.0, loss_name='loss_mse_momentum'),
            dict(type='SSIMLoss', reduction='sum', kernel_size=5, loss_weight=0.1, loss_name='loss_ssim_render'),
            dict(type='L2Loss', reduction='sum', loss_weight=0.9, loss_name='loss_l2_render'),
            dict(type='MSELoss', reduction='sum', loss_weight=1.0, loss_name='loss_mse_static'),
            ],
        ),
    )

# Custom dataset
data = dict(
    samples_per_gpu=num_sample,
    workers_per_gpu=num_worker,
    train=dict(
        times=dataset_times,
        dataset=dict(
        phase='all',
        env_cfg=dict(
            volume_scalar=volume_scalar,
            llffhold=llffhold,
            real_dt=real_dt,
            rot_est=rot_est,
            use_random_background=use_random_background,
            const_white_bg=const_white_bg,
            cluster_type=cluster_type,
            cluster_cfg=cluster_cfg,
            scene_list=scene_list,
            data_dir=data_dir,
            sub_video_dir=sub_video_dir,
            sub_img_dir=sub_img_dir,
            max_frame=max_rollout_step+1,
            start_frame=start_frame,
            max_cam_num=max_cam_num,
            max_seq=max_seq,
            max_cam_total=6*max_seq, # Total number of view of cameras
        ),)),
    val=dict(
        phase='all',
        env_cfg=dict(
            volume_scalar=volume_scalar,
            llffhold=llffhold,
            real_dt=real_dt,
            rot_est=rot_est,
            use_random_background=use_random_background,
            const_white_bg=const_white_bg,
            cluster_type=cluster_type,
            cluster_cfg=cluster_cfg,
            scene_list=scene_list,
            data_dir=data_dir,
            sub_video_dir=sub_video_dir,
            sub_img_dir=sub_img_dir,
            max_frame=test_max_frame, # 1 for ground truth
            start_frame=start_frame,
            eval_start_frame=test_start_frame,
            max_cam_num=1,
            max_seq=max_seq,
            max_cam_total=6*max_seq, # Total number of view of cameras
        ),),
    test=dict(
        phase='all',
        env_cfg=dict(
            volume_scalar=volume_scalar,
            llffhold=llffhold,
            real_dt=real_dt,
            rot_est=rot_est,
            use_random_background=use_random_background,
            const_white_bg=const_white_bg,
            cluster_type=cluster_type,
            cluster_cfg=cluster_cfg,
            scene_list=scene_list,
            data_dir=data_dir,
            sub_video_dir=sub_video_dir,
            sub_img_dir=sub_img_dir,
            max_frame=test_max_frame,
            start_frame=start_frame,
            eval_start_frame=test_start_frame,
            max_cam_num=1,
            max_seq=max_seq,
            max_cam_total=6*max_seq, # Total number of view of cameras
        ),),
    )

optimizer = dict(type='Adam', lr=2e-4*num_sample*n_gpu, betas=(0.9, 0.999), weight_decay=0, amsgrad=False)

## FOR ITER BASED
assert by_epoch == True
lr_config = dict(by_epoch=by_epoch, policy='Cus', decay_rate=5e-1, decay_steps=int(step_increase_interval*lr_decay_steps), step_start=int(step_increase_interval*(lr_step_start-lr_decay_steps))) # ITER
runner = dict(type='EpochRunner', max_epochs=int(max_epoch*step_increase_interval), max_iters=None)
checkpoint_config = dict(by_epoch=by_epoch, interval=int(min(step_increase_interval, 32*6)), max_keep_ckpts=10000)
evaluation = dict(by_epoch=by_epoch, interval=int(step_increase_interval))

log_config = dict(
    hooks=[
        dict(type='CusTextLoggerHook'),
        dict(type='TensorboardLoggerHook', by_epoch=by_epoch),
    ]
)