# `configs/student/` — конфиги проектной работы

Здесь лежат mmseg-конфиги для практикума по семантической сегментации
кошек/собак (`StudentDataset`, 3 класса: `background`, `cat`, `dog`).

## Структура

```
configs/student/
├── _base_/
│   └── datasets/student_dataset_256x256.py    # пайплайн данных, dataloaders, evaluator
└── baselines/
    ├── unet_fcn_bs8_5k.py                     # Гипотеза 1: UNet+FCN с нуля
    └── deeplabv3plus_r18_bs8_5k.py            # Гипотеза 2: DeepLabv3+ R18 (ImageNet pretrained)
```

## Запуск обучения

Все запуски — из корня форка mmseg:

```bash
cd cv_seg_final_task/mmsegmentation
PYTHONPATH=. python tools/train.py configs/student/baselines/unet_fcn_bs8_5k.py
PYTHONPATH=. python tools/train.py configs/student/baselines/deeplabv3plus_r18_bs8_5k.py
```

`PYTHONPATH=.` нужен, чтобы `custom_imports` в конфиге смог найти
`mmseg.datasets.student_dataset` (по умолчанию `tools/train.py` не
добавляет корень форка в `sys.path`).

Чекпоинты, логи и `best_mDice_iter_*.pth` появляются в
`work_dirs/<exp_name>/`.

## Анализ качества после обучения

```bash
PYTHONPATH=. python ../practicum_work/src/analysis/compute_metrics.py \
    --config configs/student/baselines/unet_fcn_bs8_5k.py \
    --ckpt   work_dirs/unet_fcn_bs8_5k/best_mDice_iter_*.pth \
    --split  val

PYTHONPATH=. python ../practicum_work/src/analysis/dump_predictions.py \
    --config configs/student/baselines/unet_fcn_bs8_5k.py \
    --ckpt   work_dirs/unet_fcn_bs8_5k/best_mDice_iter_*.pth \
    --split  val \
    --topk   5
```

Метрики попадут в
`practicum_work/supplementary/viz/baselines/<exp>_<split>.csv`,
overlay-картинки — в
`practicum_work/supplementary/viz/baselines/<exp>/<split>/{best,worst}/`.

## Решения, общие для baseline'ов

| | Значение |
|---|---|
| Crop size | 256×256 (= размер входа, не режем) |
| Batch size | 8 |
| Iterations | 5000 (~200 эпох на 200 train семплах) |
| LR | SGD 0.01, momentum 0.9, wd 5e-4, PolyLR power=0.9 |
| Loss | CE (1.0) + Dice (3.0) на decode_head; CE (0.4) + Dice (1.2) на aux head |
| Eval interval | 500 iter |
| Чекпоинты | по best mDice, держим 2 последних |
| Test mode | `whole` (входы 256×256, slide-режим не нужен) |
| Norm | `BN` (single-GPU; SyncBN на 1 GPU деградирует до того же) |
| ImageNet нормализация | mean/std из ImageNet (для совместимости с pretrained ResNet-18) |

## Чем отличаются гипотезы

| | `unet_fcn_bs8_5k` | `deeplabv3plus_r18_bs8_5k` |
|---|---|---|
| Backbone | UNet encoder с нуля | ResNet-18, ImageNet pretrained |
| Decoder | UNet decoder + FCN head | ASPP + lightweight decoder |
| Параметров | ~30 M | ~12 M |
| Skip-connections | да, на каждом уровне | один (low-level из early ResNet stage) |
| Глобальный контекст | через глубину UNet | через ASPP с GAP |
