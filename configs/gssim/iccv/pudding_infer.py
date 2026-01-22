_base_ = ['./pudding.py']

render_mov_only=False
sub_video_dir='video_images'
sub_img_dir='images'

const_white_bg=dict(
    mothorchids_v3=False,
    pudding=False,
    duck=False,
)

# # Custom model
model = dict(
    render_mov_only=render_mov_only,
    use_rotation=True,
    test_cfg=dict(
        max_rollout_step=1,),
    decode_head=dict(
        accuracy=[
            dict(type='L2Accuracy', reduction='mean', acc_name='acc_l2_render'),],),)
# # Custom dataset
data = dict(
    workers_per_gpu=4,
    test=dict(
        phase='all',
        env_cfg=dict(
            const_white_bg=const_white_bg,
            sub_video_dir=sub_video_dir,
            sub_img_dir=sub_img_dir,
            eval_start_frame=0, # TODO
            max_frame=5,
            max_cam_num=1, # TODO
            rollout=True,),),)