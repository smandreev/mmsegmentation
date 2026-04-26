# Baseline 2: DeepLabv3+ on ResNet-18 (ImageNet pretrained), CE+Dice 1:3.
# Same training schedule as Baseline 1 to make the comparison fair.

_base_ = [
    '../../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/student_dataset_256x256.py',
    '../../_base_/default_runtime.py',
]

# Force-import StudentDataset so the @DATASETS.register_module() decorator
# fires before Runner builds the dataloader.
custom_imports = dict(
    imports=['mmseg.datasets.student_dataset'], allow_failed_imports=False
)


# Для логирования в ClearML был написан специальный бекенд ClearMLVisBackend,
# который мы тут и используем, а также сохраняем всё локально, используя LocalVisBackend

crop_size = (256, 256)
num_classes = 3

# Single-GPU run → plain BN.
norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    data_preprocessor=dict(size=crop_size),
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(depth=18, norm_cfg=norm_cfg),
    decode_head=dict(
        # Channel sizes for ResNet-18 → ASPP+decoder, same as the canonical
        # configs/deeplabv3plus/deeplabv3plus_r18-d8_*.py adaptation.
        in_channels=512,
        channels=128,
        c1_in_channels=64,
        c1_channels=12,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0),
        ],
    ),
    auxiliary_head=dict(
        in_channels=256,
        channels=64,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=0.4),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.2),
        ],
    ),
)

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

work_dir = './work_dirs/deeplabv3plus_r18_bs8_5k'
