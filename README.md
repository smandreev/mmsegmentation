# Проект модуля: Выбор и обучение модели из mmsegmentation для задачи мультиклассовой семантической сегментации.

Учебный проект. Полный цикл: чистка данных → бейзлайн → эксперименты по улучшению → выбор лучшей модели → отчёт.

**Целевая метрика:** mDice > 0.75 на тестовом сплите.

**Датасет:** 3 класса (`background`, `cat`, `dog`), 256×256, splits: train=200 / val=120 / test=120. Лежит в `data/train_dataset_for_students/`

**Стек:** mmsegmentation (форк) + ClearML + кастомные скрипты в `practicum_work/`.

---

## Этап 1. Исследовательский анализ (EDA)

### Анализ качества данных

EDA проведён в [`practicum_work/notebooks/01_eda.ipynb`](practicum_work/notebooks/01_eda.ipynb).
Все семплы парные (img↔mask), 256×256, маски — индексные `L` со значениями из `{0, 1, 2}`. Меток вне набора `{0,1,2}` не найдено; размеры image/mask всегда совпадают.

Подозрительные семплы (категория `tiny-fg`, передний план ≤0.5%) в train — **5 шт.** Все они оказались багом разметки: на изображении явно есть кошка/собака, но в маске закрашено лишь несколько десятков пикселей.

Эти 5 семплов **доразмечены вручную в CVAT**, экспорт в COCO лежит в `practicum_work/data/manual_seg/<stem>/instances_default.json`. Скрипт
[`practicum_work/src/data/apply_relabel.py`](practicum_work/src/data/apply_relabel.py)
растеризует CVAT-полигоны в индексные PNG-маски и применяет их поверх `data/train_dataset_for_students/labels/train/`. 
Сводка изменений до/после — `practicum_work/data/manual_seg/_relabel_log.csv`. 
Бэкапы оригинальных масок — в каждой папке `manual_seg/<stem>/original_mask.png`.

| stem | класс | px ДО | px ПОСЛЕ |
|---|---|---|---|
| `000000028253_7169` | dog | 17 | 4 919 |
| `000000121530_5761` | cat | 86 | 6 771 |
| `000000247301_4455` | cat | 201 | 8 216 |
| `000000275919_4499` | cat | 116 | 12 815 |
| `000000574769_0` | cat | 52 | 5 654 |

`val` и `test` по содержанию **не модифицировались**.

### EDA

См. [`practicum_work/notebooks/01_eda.ipynb`](practicum_work/notebooks/01_eda.ipynb).
Артефакты — в [`practicum_work/supplementary/viz/eda/`](practicum_work/supplementary/viz/eda/):

- `pixel_share_per_split.png` — доля пикселей по классам в каждом сплите;
- `composition_per_split.png` — наличие классов на семплах (cat-only / dog-only / cat+dog / bg-only);
- `foreground_distribution.png` — распределение площади переднего плана;
- `suspicious_grid.png`, `healthy_grid.png` — глазная проверка;
- `sample_stats.csv`, `suspicious.csv` — табличные сводки.

Главное наблюдение: фон сильно доминирует по пикселям (~90%), поэтому для baseline сразу берём **CE+Dice** (а не голый CE).

---

## Этап 2. Формирование первичных гипотез

Выбраны **две контрастные гипотезы**, чтобы за один цикл сравнить encoder-decoder без предобучения и предобученный ResNet с ASPP.

### Стартовая гипотеза 1 — UNet + FCN head, обучение с нуля

**Описание гипотезы.** Самый прямой baseline по структуре урока: backbone UNet `s5-d16`, голова `FCNHead`, без ImageNet-инициализации.
Сильные стороны: skip-connections дают модели хорошую способность к локализации мелких объектов даже на 200 семплах. 

Слабые: глобального контекста через ASPP/PPM нет, и без pretrain ResNet'а сходимость начинает идти дольше.

**Лосс:** `CrossEntropyLoss(1.0) + DiceLoss(3.0)` на decode_head, тот же набор в пропорции 0.4/1.2 на auxiliary_head. Соотношение CE:Dice 1:3
взято из канонических конфигов mmseg (`unet-s5-d16_*_ce-1.0-dice-3.0_*`).

**Аугментации:** `RandomFlip(p=0.5)` + `PhotoMetricDistortion`. Никаких кропов — изображения уже 256×256 (= crop_size).

**Гиперпараметры:** SGD lr=0.01 momentum=0.9 wd=5e-4, PolyLR power=0.9, batch_size=8, **5000 итераций** (~200 эпох), eval каждые 500 iter,
сохраняем лучший по mDice.

**Результаты обучения**

- Конфиг: [`configs/student/baselines/unet_fcn_bs8_5k.py`](configs/student/baselines/unet_fcn_bs8_5k.py)
- ClearML: https://app.clear.ml/projects/837f7091c09f4754a5bb7678dfc8fdd6/experiments/7f820226cf1e46ebaab061a63655d197/output/execution
- best mDice достигнут на iter **3500** (далее — переобучение, на iter 4000–5000 mDice падает до ~60).

| Split | aAcc | mDice | mIoU | bg Dice | cat Dice | dog Dice |
|---|---|---|---|---|---|---|
| val  | 91.85 | **62.33** | 50.53 | 96.36 | 44.25 | 46.39 |
| test | 92.21 | **63.10** | 51.24 | 96.55 | 44.76 | 48.00 |

**Анализ качества**

- Per-image Dice: медиана ≈ **0.50** на val/test → модель «угадывает» грубо везде, но точная маска редкость.
- Сильный разрыв `bg` (96%) vs `cat/dog` (44–48%) — UNet за 5k итераций без предобучения толком не научился отделять зверей от фона.
- Best-картинки (Dice ≈ 0.95+) — крупные контрастные фигуры на однородном фоне.
- Worst-картинки (Dice ≈ 0.31) — модель предсказывает почти весь кадр как `background` либо тотально путает классы.

Артефакты в `practicum_work/supplementary/viz/baselines/unet_fcn_bs8_5k/` — best/worst overlay'и + per-image CSV для val и test.

### Стартовая гипотеза 2 — DeepLabv3+ R18, ImageNet pretrained

**Описание гипотезы.** Современный лёгкий backbone (ResNet-18 ~12M
параметров) с предобучением на ImageNet + ASPP с GAP-веткой для
глобального контекста + декодер с low-level skip из early-stage ResNet.
Pretrained-инициализация должна помочь на маленьком датасете —
backbone не «учится с нуля», а тонко настраивается. ASPP закрывает
дыру, которая есть у baseline-1 (учёт глобального контекста).

**Лосс, аугментации, шедулер** — те же, что в гипотезе 1 (controlled
comparison: меняется только архитектура).

**Результаты обучения**

- Конфиг: [`configs/student/baselines/deeplabv3plus_r18_bs8_5k.py`](configs/student/baselines/deeplabv3plus_r18_bs8_5k.py)
- ClearML: https://app.clear.ml/projects/837f7091c09f4754a5bb7678dfc8fdd6/experiments/c9d9ec3a04a74db8898b5b5f2287f317/output/execution
- best mDice достигнут на iter **4000** (на iter 4500–5000 mDice держится на уровне 85, переобучения не наблюдается).

| Split | aAcc | mDice | mIoU | bg Dice | cat Dice | dog Dice |
|---|---|---|---|---|---|---|
| val  | 96.40 | **85.82** | 76.35 | 98.22 | 82.91 | 76.33 |
| test | 96.74 | **86.21** | 76.92 | 98.53 | 82.81 | 77.30 |

> **Целевая mDice > 0.75 на test уже превышена baseline-моделью** — `86.21 > 75`. Эксперименты Этапа 3 ниже направлены на дальнейшее повышение качества и анализ оставшихся ошибок.

**Анализ качества**

- Per-image Dice: медиана ≈ **0.94** на val/test — модель уверенно справляется на бóльшей части датасета.
- `dog Dice` (76–77) ощутимо ниже `cat Dice` (82–83) — собаки сложнее (вероятно, бóльшее разнообразие пород/поз).
- Длинный левый хвост: ~5% семплов дают Dice ≈ 0.33 (в основном — модель почти весь кадр предсказывает как `background`). Они и есть основная цель Этапа 3.
- Семпл `000000445187_3686` — худший на test для **обеих** моделей; кандидат на ручной разбор/визуальный осмотр.

Артефакты в `practicum_work/supplementary/viz/baselines/deeplabv3plus_r18_bs8_5k/` — best/worst overlay'и + per-image CSV для val и test.

### Сводка по Этапу 2

| Модель | mDice (val) | mDice (**test**) | Δ vs target (0.75) |
|---|---|---|---|
| UNet+FCN, scratch | 0.6233 | 0.6310 | **−0.119** |
| DeepLabv3+ R18, ImageNet pretrained | 0.8582 | **0.8621** | **+0.112** |

**Главные выводы для Этапа 3:**

1. **ImageNet-предобучение — критичный фактор** на маленьком датасете. UNet с нуля проигрывает 23 п.п. mDice на test.
2. **Точная цель уже достигнута**, поэтому Этап 3 — не «выжать ещё», а целиться в **остаточные систематические ошибки**:
   - класс **`dog`** (≈6 п.п. ниже cat) — попробовать аугментации, помогающие именно ему;
   - **5% худших семплов** с Dice ≈ 0.33 — изучить, что общего, и какие аугментации/изменения архитектуры могут их вытащить;
   - переобучение UNet после iter 3500 — намёк, что нужна регуляризация / больше данных.

### Запуск экспериментов

```bash
cd cv_seg_final_task/mmsegmentation
PYTHONPATH=. python tools/train.py configs/student/baselines/unet_fcn_bs8_5k.py
PYTHONPATH=. python tools/train.py configs/student/baselines/deeplabv3plus_r18_bs8_5k.py
```

Подробнее (запуск, метрики, дамп предсказаний) — в
[`configs/student/README.md`](configs/student/README.md).

---

## Этап 3. Эксперименты по улучшению качества

Все три эксперимента надстраиваются над baseline-2 (`deeplabv3plus_r18_bs8_5k`) через `_base_` → меняется ровно одна сущность за раз. Schedule (5000 iter, eval каждые 500, save best by mDice), оптимизатор и dataset-config — неизменны (controlled comparison).

Подробности и инструкции запуска — в [`configs/student/experiments/README.md`](configs/student/experiments/README.md).

### Эксперимент 1 — DeepLabv3+ R18 + Albu аугментации

**Описание эксперимента**

Меняется только `train_pipeline`: добавляется блок `Albu` с геометрическими (`Affine` ±20° / ±15% scale / ±6% shift) и фотометрическими (`RandomBrightnessContrast`, `HueSaturationValue`, `Blur/MedianBlur`) преобразованиями + `CoarseDropout` (1–4 дырки размером 8–24 px). Всё остальное идентично baseline-2.

Цель: 5% худших семплов (Dice ≈ 0.33) и проседание `dog` (76 vs cat 83).
Гипотеза — узкое распределение поз/ракурсов в маленьком train (200 семплов).

**Результаты обучения**

- Конфиг: [`configs/student/experiments/exp1_albu_aug.py`](configs/student/experiments/exp1_albu_aug.py)
- ClearML: https://app.clear.ml/projects/837f7091c09f4754a5bb7678dfc8fdd6/experiments/d41a594b5c9a4bffb04163eaa2beb269/output/execution
- best mDice достигнут на iter **4000**.

| Split | aAcc | mDice | mIoU | bg Dice | cat Dice | dog Dice |
|---|---|---|---|---|---|---|
| val  | 96.57 | **86.41** | 77.17 | 98.33 | 83.44 | 77.47 |
| test | 96.78 | **86.12** | 76.77 | 98.56 | 82.02 | 77.76 |

**Анализ качества**

- На **val** — небольшой прирост (+0.6 п.п.) над baseline-2; на **test** — фактически на том же уровне (-0.1 п.п.). Аугментации помогают только на маргинальную часть случаев.
- Per-image Dice: медиана 0.95 (test) — чуть выше baseline-2 (0.94); p5 без изменений, число «провалов» Dice<0.5 выросло с 10 до 13.
- Вывод: распределение train/val/test однородное, и узкое разнообразие поз — не главная проблема. Albu добавляет шум, который немного «выглаживает» средние семплы, но не вытаскивает худший хвост.

Артефакты в `practicum_work/supplementary/viz/baselines/exp1_albu_aug/`.

### Эксперимент 2 — DeepLabv3+ R50 (вместо R18)

**Описание эксперимента**

Меняется только `model.backbone`: `depth: 18 → 50`, `pretrained: resnet18_v1c → resnet50_v1c`. Каналы decode_head/aux_head возвращены к дефолтам R50 (`in_channels=2048`, `c1_in_channels=256`, `c1_channels=48`, `auxiliary.in_channels=1024`).

Цель: проверить гипотезу «не хватает capacity». R18 — самый лёгкий ResNet (~12M параметров), R50 (~25M) — следующий стандартный шаг.

**Результаты обучения**

- Конфиг: [`configs/student/experiments/exp2_r50_backbone.py`](configs/student/experiments/exp2_r50_backbone.py)
- ClearML: https://app.clear.ml/projects/837f7091c09f4754a5bb7678dfc8fdd6/experiments/e9b2b4bb80bd4c64aa4d6380adb51941/output/execution
- best mDice достигнут уже на iter **2000** — тяжёлый pretrained backbone сходится в 2× быстрее R18.

| Split | aAcc | mDice | mIoU | bg Dice | cat Dice | dog Dice |
|---|---|---|---|---|---|---|
| val  | 97.37 | **89.91** | 82.32 | 98.68 | 87.81 | 83.24 |
| test | **97.62** | **90.40** | **83.07** | 98.85 | 87.66 | 84.68 |

**Анализ качества**

- **Лучший эксперимент Этапа 3.** +4.2 п.п. mDice на test над baseline-2 (86.21 → **90.40**).
- **Cильнее всего вырос `dog`**: 77.3 → 84.7 (+7.4 п.п.) — то есть R50 действительно решал проблему «нехватки capacity» на сложных позах/породах, как и предполагала гипотеза.
- Per-image Dice: число «провальных» семплов (Dice<0.5) уменьшилось 10 → **7**; p5 поднялся 0.39 → **0.49**.
- Сошлись быстрее (best на iter 2000 vs iter 4000 у baseline) — лучшая инициализация ImageNet-весов R50 даёт более эффективное обучение.

Артефакты в `practicum_work/supplementary/viz/baselines/exp2_r50_backbone/`.

### Эксперимент 3 — CE+Tversky вместо CE+Dice

**Описание эксперимента**

Меняется только вторая компонента `loss_decode`: `DiceLoss → TverskyLoss(α=0.3, β=0.7)`. CE-часть и пропорция весов 1:3 сохраняются.

Tversky — обобщение Dice: $1 - \mathrm{TP} / (\mathrm{TP} + \alpha \cdot \mathrm{FP} + \beta \cdot \mathrm{FN})$.
При α=0.3, β=0.7 штраф за **False Negatives** (пропуск объекта) втрое больше, чем за False Positives. Цель — поднять recall на проседающем `dog` и вытащить часть худшего хвоста (там модель именно «не закрашивает» зверя).

**Результаты обучения**

- Конфиг: [`configs/student/experiments/exp3_ce_tversky.py`](configs/student/experiments/exp3_ce_tversky.py)
- ClearML: https://app.clear.ml/projects/837f7091c09f4754a5bb7678dfc8fdd6/experiments/6ea7047e5d5b409e9176650aa35ee79b/output/execution
- best mDice достигнут на iter **5000** (последний — модель «недоучилась» относительно baseline на 5k бюджете).

| Split | aAcc | mDice | mIoU | bg Dice | cat Dice | dog Dice |
|---|---|---|---|---|---|---|
| val  | 94.20 | **76.10** | 64.00 | 97.12 | 66.03 | 65.14 |
| test | 94.92 | **77.43** | 65.63 | 97.67 | 67.28 | 67.33 |

**Анализ качества**

- **Хуже baseline-2 на ~9 п.п. mDice.** Замена Dice на Tversky(α=0.3, β=0.7) **не сработала**: гипотеза о пользе доп. recall-bias оказалась ошибочной.
- Что произошло: модель действительно стала «охотнее закрашивать» спорные пиксели (вырос recall), но за счёт точности — оба класса foreground просели одинаково (~66–67 Dice вместо 82/77).
- CE+Dice 1:3 (наш baseline-лосс) уже даёт хороший баланс precision/recall, и принудительный сдвиг к recall лишь ухудшает общий Dice.
- Число «провальных» семплов (Dice<0.5) выросло 10 → **20**; mean_cat: 0.80 → **0.70**, mean_dog: 0.73 → **0.72**.
- **Вывод:** в задаче со средним фоном ~95% и аккуратным CE+Dice-балансом дополнительная recall-нагрузка лишь портит precision — это типичный артефакт Tversky с β > 0.5.

Артефакты в `practicum_work/supplementary/viz/baselines/exp3_ce_tversky/`.

### Сводная таблица Этапа 3

| Эксперимент | mDice val | mDice **test** | Δ vs baseline-2 (test=0.8621) | bg Dice | cat Dice | dog Dice |
|---|---|---|---|---|---|---|
| baseline-2 (DeepLabv3+ R18 CE+Dice) | 0.8582 | 0.8621 | — | 98.5 | 82.8 | 77.3 |
| **exp1: + Albu** | 0.8641 | 0.8612 | **−0.001** | 98.6 | 82.0 | 77.8 |
| **exp2: R50 backbone** | **0.8991** | **0.9040** | **+0.042** ✅ | 98.9 | 87.7 | 84.7 |
| **exp3: CE+Tversky** | 0.7610 | 0.7743 | −0.088 | 97.7 | 67.3 | 67.3 |

**Победитель — exp2 (R50 backbone), +4.2 п.п. mDice на test.** Главный выигрыш — на классе `dog` (+7.4 п.п.), что подтверждает гипотезу о нехватке capacity у R18 для сложных собачьих поз/пород.

---

## Этап 4. Заключение и выбор лучшего эксперимента

### Лучший эксперимент

**`exp2_r50_backbone`** — DeepLabv3+ на ResNet-50 (ImageNet-pretrained) с тем же decoder/loss/scheduler, что и baseline-2. 
Из всех пяти прогнанных моделей (UNet+FCN, DLv3+ R18, +Albu, R50, +Tversky) этот эксперимент дал **наибольший прирост mDice на test (+4.2 п.п. над baseline-2 R18, +27.3 п.п. над U-Net)** при сравнимом по времени обучении. 
Гипотеза «не хватает capacity» подтвердилась: основной прирост пришёлся именно на проседающий `dog` (+7.4 п.п.), а число «провальных» семплов (per-image Dice<0.5) сократилось 10 → 7.

- Конфиг: [`configs/student/experiments/exp2_r50_backbone.py`](configs/student/experiments/exp2_r50_backbone.py)
- ClearML: https://app.clear.ml/projects/837f7091c09f4754a5bb7678dfc8fdd6/experiments/e9b2b4bb80bd4c64aa4d6380adb51941/output/execution
- Best ckpt: `work_dirs/exp2_r50_backbone/best_mDice_iter_2000.pth`

**mDice (test subset) = 0.9040** (целевая метрика проекта — > 0.75 — превышена на 15.4 п.п.)

| Метрика | Значение |
|---|---|
| mDice (test) | **0.9040** |
| mIoU (test) | 0.8307 |
| aAcc (test) | 0.9762 |
| Dice background | 0.9885 |
| Dice cat | 0.8766 |
| Dice dog | 0.8468 |

### Примеры корректных предсказаний (тестовый датасет)

5 best-overlay'ев в
`practicum_work/supplementary/viz/baselines/exp2_r50_backbone/test/best/`.
Все они — Dice ≈ 0.99 (cat ≈ 0.97, dog = 1.0):

- `000000437537_2563.png` — Dice 0.989
- `000000543836_507.png` — Dice 0.988
- `000000446604_4215.png` — Dice 0.988
- `000000322321_6994.png` — Dice 0.988
- `000000415604_7522.png` — Dice 0.986

Что у них общего: средне-крупные животные на однородном фоне, без
сильных перекрытий и без камуфляжной окраски.

### Примеры ошибок (тестовый датасет)

5 worst-overlay'ев в
`practicum_work/supplementary/viz/baselines/exp2_r50_backbone/test/worst/`:

| stem | Dice | dice_cat | dice_dog | px_cat | px_dog |
|---|---|---|---|---|---|
| `000000576589_5521` | 0.327 | 0.012 | 0.000 | 4133 | 0 |
| `000000445187_3686` | 0.328 | 0.000 | 0.000 | 4590 | 0 |
| `000000364167_7048` | 0.332 | 0.000 | 0.000 | 4588 | 0 |
| `000000284884_6459` | 0.332 | 0.000 | 0.000 | 1541 | 0 |
| `000000436539_4321` | 0.334 | 0.000 | 0.034 | 0 | 4426 |

Характер ошибок:
- В **4 из 5** случаев модель **полностью игнорирует объект** (Dice foreground = 0) и предсказывает почти весь кадр как `background`.
- Объекты не миниатюрные — занимают **2-7%** пикселей, но в нашем baseline и здесь модель «не видит» их;
- 4 из 5 — `cat`. Видимо, это специфические позы/окружения (потенциально: кошка в одеяле, в коробке, сильно затенённая, с нестандартной шерстью/окраской).
- Семпл `000000445187_3686` — худший в **обеих** моделях (R18 и R50); не помог ни pretrained-апгрейд, ни Albu. Кандидат на ручной осмотр — возможно, баг в разметке test (тогда метрика занижена).

### Возможности для улучшения

1. **Test-time augmentation (TTA).** mmseg поддерживает `tta_pipeline` (multi-scale + flip). Free win 0.5–1 п.п. mDice без переобучения.
2. **Более тяжёлый backbone (R101 / Swin-Tiny).** R50 → R101 даст ещё ~5M параметров; на нашем train=200 это уже рискованно (overfit), нужны более сильные аугментации параллельно.
3. **Балансировка классов через class_weight в CE.** В EDA фон занимает ~95% пикселей; явные веса `[0.5, 1.5, 1.5]` могут «вытащить» часть провальных семплов с пропуском объекта.
4. **Ручной разбор worst-5.** Стоит открыть глазами `worst/`-картинки и по каждой решить: ошибка модели, ошибка разметки, или объект действительно маскирован. Если ошибка разметки в test — это систематически занижает все наши метрики.
5. **Self-training / pseudo-labels** на расширенном неразмеченном датасете (например COCO val без cat/dog аннотаций) — рабочий приём для удвоения effective train size без ручной разметки.
6. **DeepLabv3+ с auxiliary backbone (HRNet) или SegFormer.** Если цель — выжать ещё 2–3 п.п. mDice, имеет смысл попробовать transformer-encoder; они обычно лучше на сложных позах foreground-объектов.

---

## Этап 5. Документация кода

Корень репозитория — форк `mmsegmentation`. Всё, что добавлено для
проекта (наш `StudentDataset`, конфиги в `configs/student/`, кастомный
код в `practicum_work/`), помещено внутрь форка, чтобы один архив
содержал и инфраструктуру обучения, и сабмодуль ученика.

```
mmsegmentation/                          ← форк, корень репозитория
├── README.md                            — этот отчёт
│
├── mmseg/datasets/                      — изменения в самой mmseg-библиотеке
│   ├── student_dataset.py               — кастомный класс `StudentDataset`
│   │                                      (3 класса: background, cat, dog).
│   └── __init__.py                      — добавлен import + 'StudentDataset'
│                                          в __all__, чтобы декоратор
│                                          @DATASETS.register_module()
│                                          сработал при импорте mmseg.datasets.
│
├── configs/student/                     — все наши mmseg-конфиги
│   ├── README.md                        — общая шпаргалка по configs/student/.
│   ├── _base_/datasets/
│   │   └── student_dataset_256x256.py   — общий dataset config:
│   │                                      train/val/test dataloaders,
│   │                                      train pipeline (Flip + PMD),
│   │                                      val/test pipeline, IoUMetric
│   │                                      с mDice + mIoU.
│   ├── baselines/                       — Этап 2:
│   │   ├── unet_fcn_bs8_5k.py           — UNet+FCN с нуля, CE+Dice 1:3.
│   │   └── deeplabv3plus_r18_bs8_5k.py  — DeepLabv3+ R18 ImageNet
│   │                                      pretrained, CE+Dice 1:3.
│   └── experiments/                     — Этап 3 (надстройки над baseline-2):
│       ├── README.md                    — описание экспериментов и
│       │                                  инструкции запуска.
│       ├── exp1_albu_aug.py             — + Albu: Affine, BC, HSV, Blur,
│       │                                  CoarseDropout.
│       ├── exp2_r50_backbone.py         — backbone R18 → R50 (winner).
│       └── exp3_ce_tversky.py           — Dice → Tversky(α=0.3, β=0.7).
│
├── data/                                — датасет и доразметка (gitignore)
│   ├── train_dataset_for_students/      — img/{train,val,test} + labels/
│   └── manual_seg/<stem>/               — папки CVAT-экспортов 5
│                                          доразмеченных семплов:
│                                          instances_default.json,
│                                          original_mask.png (бэкап),
│                                          relabeled_mask.png (превью).
│
├── work_dirs/                           — чекпоинты mmseg (gitignore)
│   └── <exp>/best_mDice_iter_*.pth
│
└── practicum_work/                      — сабмодуль студента
    ├── notebooks/
    │   ├── 01_eda.ipynb                 — EDA датасета: размеры, баланс
    │   │                                  классов, поиск подозрительных
    │   │                                  семплов (Этап 1).
    │   └── 02_baseline_analysis.ipynb   — сводный анализ всех 5 моделей:
    │                                      читает CSV/PNG-артефакты,
    │                                      рисует сравнительные таблицы и
    │                                      сетки best/worst (Этапы 2-4).
    ├── src/
    │   ├── data/
    │   │   └── apply_relabel.py         — конвертирует CVAT-экспорты
    │   │                                  (`data/manual_seg/<stem>/instances_default.json`)
    │   │                                  в индексные PNG-маски и
    │   │                                  записывает их поверх
    │   │                                  train-меток. Делает бэкапы
    │   │                                  оригиналов и пишет
    │   │                                  `_relabel_log.csv`.
    │   └── analysis/
    │       ├── compute_metrics.py       — пересчитывает mDice / mIoU /
    │       │                              per-class Dice для пары
    │       │                              (config, ckpt) на val/test.
    │       │                              Пишет CSV в `viz/baselines/`.
    │       └── dump_predictions.py      — прогоняет модель по сплиту,
    │                                      считает per-image Dice,
    │                                      сохраняет CSV со скорами и
    │                                      сетки top-K best / worst
    │                                      overlay'ев (input | gt | pred).
    └── supplementary/viz/
        ├── eda/                         — графики и сетки EDA
        │                                  (pixel_share_per_split.png,
        │                                  composition_per_split.png,
        │                                  foreground_distribution.png,
        │                                  suspicious_grid.png,
        │                                  healthy_grid.png + CSV).
        └── baselines/<exp>/             — артефакты модели <exp>:
                                           `<exp>_<split>.csv` (агрегатные
                                           метрики), `<split>_per_image_dice.csv`
                                           (per-image), `<split>/best/`,
                                           `<split>/worst/` (overlay PNG).
```

### Команды для воспроизведения (из корня форка, `mmsegmentation/`)

```bash
# Этап 1 — чистка данных
python practicum_work/src/data/apply_relabel.py     # применяет CVAT-доразметку 5 семплов

# Этап 2 — baseline-обучение (по ~25 мин на ВМ-GPU)
PYTHONPATH=. python tools/train.py configs/student/baselines/unet_fcn_bs8_5k.py
PYTHONPATH=. python tools/train.py configs/student/baselines/deeplabv3plus_r18_bs8_5k.py

# Этап 3 — улучшения
PYTHONPATH=. python tools/train.py configs/student/experiments/exp1_albu_aug.py
PYTHONPATH=. python tools/train.py configs/student/experiments/exp2_r50_backbone.py
PYTHONPATH=. python tools/train.py configs/student/experiments/exp3_ce_tversky.py

# Этап 4 — оценка любого ckpt на test
PYTHONPATH=. python practicum_work/src/analysis/compute_metrics.py \
    --config configs/student/experiments/exp2_r50_backbone.py \
    --ckpt   work_dirs/exp2_r50_backbone/best_mDice_iter_2000.pth \
    --split  test

# Этап 4 — best/worst overlay'и
PYTHONPATH=. python practicum_work/src/analysis/dump_predictions.py \
    --config configs/student/experiments/exp2_r50_backbone.py \
    --ckpt   work_dirs/exp2_r50_backbone/best_mDice_iter_2000.pth \
    --split  test --topk 5
```
