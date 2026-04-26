# `configs/student/experiments/` — эксперименты Этапа 3

Каждый эксперимент — небольшая инкрементальная правка над baseline-2
(`deeplabv3plus_r18_bs8_5k`), наш чемпион Этапа 2 (test mDice 0.8621).
В одном эксперименте меняется ровно одна сущность.

| # | Конфиг | Что меняем | Гипотеза |
|---|---|---|---|
| 1 | [`exp1_albu_aug.py`](exp1_albu_aug.py) | + Albu pipeline (Affine, BrightnessContrast, HSV, Blur, CoarseDropout) | Худшие 5% семплов и проседание `dog` — от узкого распределения поз/ракурсов в маленьком train (200 семплов) |
| 2 | [`exp2_r50_backbone.py`](exp2_r50_backbone.py) | Backbone R18 → **R50** (тот же decoder, ImageNet pretrained) | Capacity R18 (~12M) может не хватать для сложных кадров; R50 (~25M) — следующий стандартный шаг |
| 3 | [`exp3_ce_tversky.py`](exp3_ce_tversky.py) | Loss CE+**Dice** → CE+**Tversky** (α=0.3, β=0.7) | Tversky штрафует FN втрое сильнее, чем FP — целит в худший хвост Dice и в проседающий `dog` (76 vs cat 83). Заменил FocalLoss из-за бага в mmseg для multi-class CPU-варианта |

## Запуск

```bash
cd cv_seg_final_task/mmsegmentation
PYTHONPATH=. python tools/train.py configs/student/experiments/exp1_albu_aug.py
PYTHONPATH=. python tools/train.py configs/student/experiments/exp2_r50_backbone.py
PYTHONPATH=. python tools/train.py configs/student/experiments/exp3_ce_tversky.py
```

Каждый эксперимент:
- наследуется через `_base_` от `baselines/deeplabv3plus_r18_bs8_5k.py`,
  поэтому шедулер (5000 iter, eval каждые 500, save best by mDice),
  оптимизатор и dataset config не меняются — это controlled comparison;
- свой `task_name` в ClearML (см. `visualizer` блок в каждом конфиге);
- свой `work_dir`, чтобы не конфликтовать с baseline-чекпоинтами.

## Анализ после обучения

Тот же процесс, что и для baseline:

```bash
# на val
PYTHONPATH=. python ../practicum_work/src/analysis/compute_metrics.py \
    --config configs/student/experiments/<exp>.py \
    --ckpt   work_dirs/<exp>/best_mDice_iter_*.pth \
    --split  val

# и на test (для финального решения)
PYTHONPATH=. python ../practicum_work/src/analysis/compute_metrics.py \
    --config configs/student/experiments/<exp>.py \
    --ckpt   work_dirs/<exp>/best_mDice_iter_*.pth \
    --split  test

# и сетки best/worst
PYTHONPATH=. python ../practicum_work/src/analysis/dump_predictions.py \
    --config configs/student/experiments/<exp>.py \
    --ckpt   work_dirs/<exp>/best_mDice_iter_*.pth \
    --split  val \
    --topk   5
```

CSV/PNG падают в `practicum_work/supplementary/viz/baselines/<exp>/`.
Сравнение всех экспериментов — в ноутбуке
`practicum_work/notebooks/02_baseline_analysis.ipynb` (надо добавить
имена в `EXPERIMENTS`-словарь, см. ниже).

## Замечания по реализации

- **Albu 2.x** переименовал API ряда трансформов:
  `ShiftScaleRotate` → `Affine`, `CoarseDropout` принимает диапазоны
  через `*_range`. Старый API молча игнорируется (даёт UserWarning).
  Конфиг `exp1` использует уже актуальный API.
- **mmseg `FocalLoss` для multi-class — багованный.** В
  `py_sigmoid_focal_loss` ошибка на line 295: `target[:, num_classes]`
  должно быть `target[:, :num_classes]` (взять первые `num_classes`
  каналов one-hot, а не один канал с индексом `num_classes`). Из-за
  этого forward падает с
  `RuntimeError: tensor a (3) must match tensor b (N) at dim 1`.
  Поэтому exp3 переключён на **TverskyLoss** — он решает ту же
  исследовательскую задачу (давит FN, целит в редкий класс),
  работает корректно из коробки.
- **R50** + bs=8 на 256×256 — должно поместиться в ~6-8 GB VRAM.
  Если памяти не хватает, `--cfg-options train_dataloader.batch_size=4`.
