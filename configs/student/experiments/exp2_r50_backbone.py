# Experiment 2: тот же DeepLabv3+, но backbone ResNet-50 вместо R18.
#
# Гипотеза: R18 — самый лёгкий ResNet (~12M параметров). Возможно,
# модели не хватает capacity, чтобы аккуратно отделить cat/dog от фона
# в сложных кадрах. R50 (~25M параметров) — следующий стандартный шаг.
#
# Меняется ровно один компонент: backbone (depth, channels, pretrained).
# Всё остальное (пайплайн, лосс, шедулер) идентично baseline 2.

_base_ = '../baselines/deeplabv3plus_r18_bs8_5k.py'

visualizer = dict(
    type='Visualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            type='ClearMLVisBackend',
            init_kwargs=dict(
                project_name='cv_seg',
                task_name='exp2_r50_backbone',
                reuse_last_task_id=False,
                continue_last_task=False,
            ),
        ),
    ],
)

# --- Backbone и каналы голов: возврат к дефолтам R50 --------------------------

model = dict(
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(depth=50),
    decode_head=dict(
        # дефолты deeplabv3plus_r50-d8: in_channels=2048, c1_in=256, c1_ch=48
        in_channels=2048,
        channels=512,
        c1_in_channels=256,
        c1_channels=48,
    ),
    auxiliary_head=dict(in_channels=1024, channels=256),
)

work_dir = './work_dirs/exp2_r50_backbone'
