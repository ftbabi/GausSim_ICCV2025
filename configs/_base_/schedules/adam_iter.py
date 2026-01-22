# optimizer
optimizer = dict(type='Adam', lr=5e-5, betas=(0.9, 0.999), weight_decay=0, amsgrad=False)
optimizer_config = dict(grad_clip=dict(max_norm=1.0))