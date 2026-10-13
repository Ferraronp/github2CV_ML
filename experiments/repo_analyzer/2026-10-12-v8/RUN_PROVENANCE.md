# Исходный прогон: provenance и условия проверки

**Run ID:** `20261012T160658Z` · **Дата:** 12 октября 2026 · **Набор:** `v8-2026-10-12`.

## Источник первичных данных

Исходный архив `benchmark_20261012T160658Z.zip` был получен непосредственно при запуске Colab v8. Он существует как отдельный артефакт исходного эксперимента, **но на момент публикации не включён в этот PR и не размещён по публичному неизменяемому URL**. Поэтому любой читатель GitHub может независимо пересчитать семантическую сводку из включённых аннотаций, но **не может независимо перепроверить raw predictions, структурные показатели и VRAM по одному лишь содержимому PR**. Для полного аудита нужен оригинальный ZIP, совпадающий с хешем ниже.

- Оригинальный ZIP (88 114 байт), SHA-256: `4ad3f16f080a1500d74f717a53504b15142c6881af48fdb9073168fc7c46a6a3`
- `snapshots_20261012T160658Z.json` (147 593 байта), SHA-256: `1d6de61a342a16f9dd53ab2df04370ccfb271c3e62566a2b7028a3ecaeb52aff`
- `predictions_20261012T160658Z.json` (352 000 байт), SHA-256: `5e1366bc82e8aab9dd56f93305c703134e16dde4c8d8d35bd2a92e267ebd087c`
- `metadata_20261012T160658Z.json` (10 363 байта), SHA-256: `ab66f9345ec058fdb2598b728bfdb1817223109b622d3848469352c0b022894e`
- `results_20261012T160658Z.csv` (61 591 байт), SHA-256: `359893c701beb39a9bfbe69238d3b4f92116d1b6ecac3a357eb4fdc1af0a7c66`
- `vram_summary_20261012T160658Z.csv` (1 513 байт), SHA-256: `292b7f68154dda914bfe446c6b19da9ae632cb55f473da9011f44953e8e878fe`

**Содержимое:** 14 frozen public cases, 168 raw predictions (9 локальных моделей × 14 + 3 Groq × 14), 196 попыток (10 T4-конфигураций + 3 Groq + 1 OpenRouter, каждая по 14). Для Phi-4-mini сохранены 14 ошибок загрузки, raw responses отсутствуют.

Воспроизводимая **проверка опубликованных структурных и VRAM-агрегатов** после получения оригинального ZIP:

```bash
cd experiments/repo_analyzer/2026-10-12-v8
python scripts/verify_raw_archive.py /path/to/benchmark_20261012T160658Z.zip
```

Скрипт проверит SHA-256 ZIP и пяти вложенных файлов, повторно вызовет `score()` из опубликованного Colab notebook по сохранённым raw predictions и numbered snapshots, затем сверит `data/structural_summary.csv` и `data/vram_summary.csv` с результатами. **Повторного инференса не требуется.** Проверка была выполнена на оригинальном архиве при подготовке исправлений ревью.

## Фактические параметры полного запуска

Это значения из первичного `metadata_20261012T160658Z.json`, **не дефолтные smoke-настройки** опубликованного ноутбука.

| Параметр | Фактическое значение |
|---|---|
| Количество кейсов | 14 (полный набор) |
| Локальных конфигураций | 10 запрошено, 9 запущено |
| API конфигураций | 3 Groq, 1 OpenRouter |
| Повторов | 1 |
| GPU | Tesla T4 (все локальные модели) |
| Context | 8192 |
| Temperature | 0.1 |
| Max completion tokens | 3000 |
| Budget для 1660 Super | 6144 MiB, резерв 512 MiB |
| System prompt SHA-256 | `1492fe98da2889291810f21107ee3e12db6fe01510509d7d0e7eef8589d0b511` |
| JSON Schema SHA-256 | `8d36cd06fb33019858d83ede1e9057782b7f1477ea20bc61a2a94ca0dd1947af` |

## Frozen commit SHA источников

| Публичный GitHub-репозиторий | Commit SHA |
|---|---|
| [Ferraronp/vlm-nutrition-analyzer](https://github.com/Ferraronp/vlm-nutrition-analyzer/tree/743107a6c5a6b52d3882d6030440fe2c5178e0ea) | `743107a6c5a6b52d3882d6030440fe2c5178e0ea` |
| [Ferraronp/rag_mtuci](https://github.com/Ferraronp/rag_mtuci/tree/b900f06cec56c7fb3161257c7f99b21d5d30dd64) | `b900f06cec56c7fb3161257c7f99b21d5d30dd64` |
| [Ferraronp/github2CV_ML](https://github.com/Ferraronp/github2CV_ML/tree/a37eb108f6fc57d62d9f78fa54f5ddcc9fd407fb) | `a37eb108f6fc57d62d9f78fa54f5ddcc9fd407fb` |
| [Ferraronp/military-charters-rag](https://github.com/Ferraronp/military-charters-rag/tree/0fe4923fb9aaa002838b8114db1445d671d749b0) | `0fe4923fb9aaa002838b8114db1445d671d749b0` |
| [Ferraronp/gpt-pretrain](https://github.com/Ferraronp/gpt-pretrain/tree/d9a05012eaac14a2d2adc6d8a92709def3d9c1ec) | `d9a05012eaac14a2d2adc6d8a92709def3d9c1ec` |
| [Ferraronp/reasoning-memory](https://github.com/Ferraronp/reasoning-memory/tree/04b03516dd51ae477d828da24d81b5bb1ca087d5) | `04b03516dd51ae477d828da24d81b5bb1ca087d5` |
| [Ferraronp/llm.c_tensorboard](https://github.com/Ferraronp/llm.c_tensorboard/tree/57823100f80033b4d4b17323bb6509a136632290) | `57823100f80033b4d4b17323bb6509a136632290` |
| [Ferraronp/CosyVoice-gguf-converter](https://github.com/Ferraronp/CosyVoice-gguf-converter/tree/41e0145d1f91e64d8f9095282c59f6f91a5d9a07) | `41e0145d1f91e64d8f9095282c59f6f91a5d9a07` |

Эти commit SHA фиксируют репозитории-источники, но **сами по себе не восстанавливают выборку файлов, пронумерованные окна и точные raw ответы**. Для этого необходим проверенный исходный ZIP.
