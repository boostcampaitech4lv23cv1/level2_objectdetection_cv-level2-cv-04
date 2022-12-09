# optimizer
optimizer = dict(betas=(0.9, 0.999), lr=0.0001, type='AdamW', weight_decay=0.05)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
