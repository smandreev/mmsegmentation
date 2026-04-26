# Dataset config for the practicum cat/dog segmentation dataset.
# Images are 256x256 RGB; masks are 256x256 indexed PNG with values
# {0: background, 1: cat, 2: dog}. Splits: train=200 / val=120 / test=120.

dataset_type = 'StudentDataset'
data_root = 'data/train_dataset_for_students'
crop_size = (256, 256)

# Train pipeline: keep it light. Inputs are already 256x256, so we don't
# resize or crop; we only flip horizontally and apply standard photometric
# distortion. PackSegInputs is mandatory at the end.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]

# Test/val pipeline: deterministic. We still call Resize to crop_size
# defensively in case any image diverges from 256x256 (EDA confirmed all
# are 256x256 today).
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='img/train', seg_map_path='labels/train'),
        img_suffix='.jpg',
        seg_map_suffix='.png',
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='img/val', seg_map_path='labels/val'),
        img_suffix='.jpg',
        seg_map_suffix='.png',
        pipeline=test_pipeline,
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='img/test', seg_map_path='labels/test'),
        img_suffix='.jpg',
        seg_map_suffix='.png',
        pipeline=test_pipeline,
    ),
)

val_evaluator = dict(type='IoUMetric', iou_metrics=['mDice', 'mIoU'])
test_evaluator = val_evaluator
