# Experiment 3: тот же DeepLabv3+ R18, но Dice → Tversky.
#
# Изначальная гипотеза: класс dog (76 Dice) проседает относительно cat
# (83 Dice), и есть длинный хвост 5% худших семплов с Dice ~0.33.
# Хочется лосс, который сильнее «давит» на False Negatives (пропуски
# объекта).
#
# Tversky — обобщение Dice:
#     Tversky(α, β) = 1 - TP / (TP + α·FP + β·FN)
# С α=0.3, β=0.7 штраф за FN втрое больше, чем за FP — модель будет
# охотнее закрашивать «спорные» пиксели как `cat`/`dog`, что должно
# поднять recall этих классов и подтянуть худший хвост Dice.
#
# Меняется ровно один компонент: вторая компонента loss_decode
# (Dice → Tversky). CE-часть и веса 1:3 сохраняются для
# controlled comparison с baseline-2.
#
# Замечание: первоначально планировался Focal+Dice, но mmseg-овский
# `FocalLoss(use_sigmoid=True)` падает с RuntimeError
# "size of tensor a (3) must match size of tensor b (N) at dim 1"
# для multi-class — это баг в `py_sigmoid_focal_loss` (line 295 в
# mmseg/models/losses/focal_loss.py: `target[:, num_classes]` вместо
# `target[:, :num_classes]`). TverskyLoss решает ту же исследовательскую
# задачу без обхода/патча.

_base_ = '../baselines/deeplabv3plus_r18_bs8_5k.py'

visualizer = dict(
    type='Visualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            type='ClearMLVisBackend',
            init_kwargs=dict(
                project_name='cv_seg',
                task_name='exp3_ce_tversky',
                reuse_last_task_id=False,
                continue_last_task=False,
            ),
        ),
    ],
)

# --- Loss --------------------------------------------------------------------

model = dict(
    decode_head=dict(
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
            dict(type='TverskyLoss',     loss_name='loss_tversky',
                 alpha=0.3, beta=0.7,    loss_weight=3.0),
        ],
    ),
    auxiliary_head=dict(
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=0.4),
            dict(type='TverskyLoss',     loss_name='loss_tversky',
                 alpha=0.3, beta=0.7,    loss_weight=1.2),
        ],
    ),
)

work_dir = './work_dirs/exp3_ce_tversky'
