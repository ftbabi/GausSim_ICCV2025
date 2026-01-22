env_cfg = dict(
        data_dir='data_real/',
        sub_video_dir='video_images',
        sub_img_dir='images',
        cam_transform_fn=None,
        scene_list=['mothorchids'], # # Must align with those in models. e.g., gs_scene

        use_random_background=dict(mothorchids_v2=False,),
        const_white_bg=dict(mothorchids_v2=True,),
        resolution=[960, 540],
        scale_x_angle=1.0,
        load_imgs=False,
        volume_scalar=dict(mothorchids_v2=512),
        gravity=[0, 0, -9.8],
        rot_est=dict(
            # scalar-last (x, y, z, w)
            mothorchids=[-0.912, 0.007, 0.015, 0.411],),
        max_cam_num=6,
        pad_cam=False,
        max_cam_total=180,
        max_frame=-1,
        start_frame=0,
        cluster_cfg=dict(
            mothorchids=[
            dict(downsample_rate=0.01),
            dict(downsample_rate=0.01),],
        ),
        dt=1/30,
        real_dt=dict(
            mothorchids=1/29.97, # This is different
            duck=1/50,
        ),
        llffhold=0,
    )

dataset_type = 'MultiviewVideoDataset'

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
	train=dict(
        times=192,
        type='RepeatDataset',
        dataset=dict(
            type=dataset_type,
            phase='train',
		    env_cfg=env_cfg,
        )),
	val=dict(
        type=dataset_type,
        phase='val',
        env_cfg=env_cfg,
        ),
	test=dict(
        type=dataset_type,
        phase='test',
        env_cfg=env_cfg,
        ),
)