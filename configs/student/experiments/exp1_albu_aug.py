# Experiment 1: Baseline 2 (DeepLabv3+ R18) + сильные аугментации Albu.
#
# Гипотеза: ~5% худших семплов (Dice ≈ 0.33) дают такие низкие числа из-за
# узкого распределения поз/ракурсов в train (200 семплов). Добавим
# геометрические и фотометрические аугментации Albu — особенно полезно
# для класса dog, где разнообразие пород/поз больше.
#
# Меняется ровно одно: train_pipeline. Всё остальное идентично baseline 2.

_base_ = '../baselines/deeplabv3plus_r18_bs8_5k.py'

# ClearML run name для этого эксперимента
visualizer = dict(
    type='Visualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            type='ClearMLVisBackend',
            init_kwargs=dict(
                project_name='cv_seg',
                task_name='exp1_albu_aug',
                reuse_last_task_id=False,
                continue_last_task=False,
            ),
        ),
    ],
)

# --- Аугментации -------------------------------------------------------------

albu_train_transforms = [
    # Albu 2.x: ShiftScaleRotate is deprecated → Affine.
    dict(
        type='Affine',
        translate_percent=(-0.0625, 0.0625),
        scale=(0.85, 1.15),
        rotate=(-20, 20),
        p=0.5,
    ),
    dict(type='RandomBrightnessContrast',
         brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    dict(type='HueSaturationValue',
         hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur',       blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0),
        ],
        p=0.2,
    ),
    # Albu 2.x renamed CoarseDropout's API:
    #   max_holes/min_holes        → num_holes_range
    #   max_height/min_height      → hole_height_range
    #   max_width/min_width        → hole_width_range
    #   fill_value                 → fill
    dict(
        type='CoarseDropout',
        num_holes_range=(1, 4),
        hole_height_range=(8, 24),
        hole_width_range=(8, 24),
        fill=0,
        p=0.3,
    ),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # Albu сам выполняет flip — оставляем здесь только его геометрические/
    # фотометрические преобразования; horizontal flip перенесён в Albu
    # (HorizontalFlip ниже добавляется отдельно).
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Albu',
        keymap={'img': 'image', 'gt_seg_map': 'mask'},
        transforms=albu_train_transforms,
    ),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

work_dir = './work_dirs/exp1_albu_aug'
