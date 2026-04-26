# Baseline 1: UNet (encoder from scratch) + FCN head, CE+Dice 1:3 loss.
# 5000 iterations, batch_size=8, eval+save best every 500 iter on mDice.

_base_ = [
    '../../_base_/models/fcn_unet_s5-d16.py',
    '../_base_/datasets/student_dataset_256x256.py',
    '../../_base_/default_runtime.py',
]

# Force-import StudentDataset so the @DATASETS.register_module() decorator
# fires before Runner builds the dataloader. Without this, the canonical
# `default_scope='mmseg'` only loads top-level mmseg, not mmseg.datasets.
custom_imports = dict(
    imports=['mmseg.datasets.student_dataset'], allow_failed_imports=False
)


# Для логирования в ClearML был написан специальный бекенд ClearMLVisBackend,
# который мы тут и используем, а также сохраняем всё локально, используя LocalVisBackend
visualizer = dict(
    type='Visualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),      # сохраняем логи локально
        dict(
            type='ClearMLVisBackend',      # дублируем всё в ClearML
            init_kwargs=dict(
                project_name='cv_seg',
                task_name='unet_fcn_bs8_5k',
                reuse_last_task_id=False,
                continue_last_task=False,
                output_uri=None,
                auto_connect_arg_parser=True,
                auto_connect_frameworks=True,
                auto_resource_monitoring=True,
                auto_connect_streams=True,
            )
        )
    ]
)

crop_size = (256, 256)
num_classes = 3

# Single-GPU run → use plain BN (SyncBN degrades to BN here anyway, but
# the explicit BN avoids confusion in logs).
norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    data_preprocessor=dict(size=crop_size),
    backbone=dict(norm_cfg=norm_cfg),
    decode_head=dict(
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0),
        ],
    ),
    auxiliary_head=dict(
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=0.4),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.2),
        ],
    ),
    # Inputs are 256x256 — full image fits, no need for slide-mode TTA.
    test_cfg=dict(mode='whole'),
)

# Training schedule: 5000 iter ≈ 200 epochs at bs=8 over 200 train samples.
train_cfg = dict(type='IterBasedTrainLoop', max_iters=5000, val_interval=500)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=5e-4)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

param_scheduler = [
    dict(type='PolyLR', eta_min=1e-4, power=0.9, begin=0, end=5000, by_epoch=False),
]

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=500,
        save_best='mDice',
        max_keep_ckpts=2,
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'),
)

work_dir = './work_dirs/unet_fcn_bs8_5k'
