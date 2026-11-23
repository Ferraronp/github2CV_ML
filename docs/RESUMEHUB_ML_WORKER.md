# ResumeHub ↔ ML Core: рабочий протокол и запуск consumer

Дата: 2026-11-23. **Реализация находится в stacked PR поверх #32 и #31.**
Основа: актуальная ветка `dev` репозитория
[`gonnem2/PP-AUT-2026-API`](https://github.com/gonnem2/PP-AUT-2026-API/tree/dev):
`src/common/broker.py`, `src/common/worker.py`,
`src/projects/schemas.py`, `src/projects/service.py`.

## Обязанности и границы

**Backend отвечает:** FastAPI, пользователь/авторизация/ownership, выбор 1–20
проектов, закрепление точных GitHub commit SHA, создание idempotent job и
`request_id`, PostgreSQL outbox, RabbitMQ publish, ожидание результата,
предотвращение stale/superseded отчётов, хранение пользовательского контекста
и PDF, показ статуса. Эту инфраструктуру ML не дублирует.

**ML Core отвечает:** принимать задание из существующей RabbitMQ,
запускать `run_pipeline` для указанных SHA (не брать новый HEAD), сохранять
каждый `RepoAnalysis` немедленно по окончании репозитория, повторно использовать
структурированный кэш, собирать `CandidateProfile`, `ResumeDocument`,
отправлять типизированный результат и метаданные по всем репозиториям.
User-declared `declared_role` — декларация кандидата, не доказанный
источником факт, поэтому не выдаётся за evidence.

**Frontend:** status polling существующего `/api/v1/github/report-jobs/{job_id}`,
открытие сформированного черновика, ручное редактирование и экспорт PDF
через backend. Browser напрямую к RabbitMQ/LLM не подключается.

## Что НЕ меняется в RabbitMQ

Остаётся транспорт из бэка, без новой очереди/Redis broker:

| Назначение | Значение |
| --- | --- |
| exchange | `resumehub.ml` (direct/durable) |
| очередь заданий | `resumehub.ml.requests` (durable) |
| очередь ответов | `resumehub.ml.results` (durable) |
| некорректные сообщения | `resumehub.ml.invalid-results` |
| routing request | `github.report.requested` |
| routing success | `github.report.completed` |
| routing failure | `github.report.failed` |
| свойства | persistent message, publisher confirm, `message_id`, `correlation_id=job_id` |
| parallelism | QoS prefetch=1, один GPU-bound job на worker |

ML Worker подтверждает входящую доставку **только после publisher confirm**
события ответа. Некорректные входные сообщения попадают в существующую
`*.invalid-results` через отдельный обмен `resumehub.ml.dlx`.
Содержимое сообщения не копируется в DLQ — сохраняются лишь причина,
размер и SHA256 (в теле могут быть пользовательские данные).

## Обновление контракта: schema_version = 2

**ВАЖНО: существующий backend `CoreRequest/CoreResult v1` НЕ совместим
с v2.** Бэкендер согласился изменить свои DTO. Оставлять старую валидацию
`StrictDTO(extra="forbid")` и подключать настоящий worker **нельзя**:
она отправит результат в DLQ. Названия event types и маршруты сохраняются.

Единственный источник истины по схеме:
[`worker_contracts.py`](../src/github2cv_ml/worker_contracts.py),
`MLJobRequest`, `MLJobResult`, `MLRepositoryResult`.

Пример request (строковые значения SHA/контекста — пример, не актуальный коммит):

```json
{
  "schema_version": 2,
  "type": "github.report.requested",
  "message_id": "22222222-2222-4222-8222-222222222222",
  "job_id": "11111111-1111-4111-8111-111111111111",
  "input_revision": 1,
  "candidate": {
    "github_login": "developer",
    "display_name": null
  },
  "repositories": [
    {
      "repo_id": "123456789",
      "repository_url": "https://github.com/example/repository",
      "commit_sha": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "context_hash": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      "declared_role": null,
      "started_at": null,
      "ended_at": null
    }
  ],
  "options": {
    "max_steps": 28,
    "max_reads": 12,
    "min_planned_reads": 8,
    "max_skills": 20,
    "max_input_chars": 20000,
    "force_refresh": false
  }
}
```

**Ответ:**

- `schema_version: 2`, `type: github.report.completed | github.report.failed`;
- `job_id`, `request_message_id`, `message_id` (детерминированный UUID
  ответа для дедупликации), `input_revision`, `status: completed|partial|failed`;
- **ровно один `MLRepositoryResult` на каждый входной проект:** прежние
  `repo_id/commit_sha/context_hash`, `status`, `cache_hit`,
  `analysis` (структурированный `RepoAnalysis` либо null), `problems`;
- если CV сформировано: `profile: CandidateProfile` и
  `resume: ResumeDocument` (оба с `evidence_ids/claim_ids`);
- `warnings`, `problems`, `error`. Для терминального failure
  `resume=null`, но результаты репозиториев могут сохраниться.

**Изменение для Backend:** заменить текстовые `report_text/summary_ru`
как единственное содержимое ответа на структурированные `profile/resume`
и результаты каждого репозитория; отказаться от предположения, что
`completed` означает успех **каждого** репозитория. При `status=partial`
показывать предупреждение пользователю, но разрешать редактировать CV,
если `resume` действительно есть. Проверки ID, SHA, context_hash и
`input_revision` сохраняются.

**Важно:** backend должен преобразовывать `ResumeDocument` в свой
`src/resume/schemas.py:Content` с сохранением `claim_ids` в отдельном
аудируемом объекте. Не перезаписывать контакты, образование или
описанный пользователем опыт. Кандидат подтверждает/редактирует черновик.

## Кэш и повторные доставки

Модуль `worker_store.py` создаёт SQLite БД (для MVP один ML consumer и
постоянный Docker volume) с двумя независимыми таблицами:

1. `analyses`: `SHA256(repo + commit_sha + analyzer_revision +
   model_id + max_steps/max_reads/min_planned_reads)` → полный `RepoAnalysis`.
   Сохраняется **сразу после обработки каждого репозитория**.
   `declared_role`/имя пользователя **не входят** в технический ключ.
   Не восстанавливать evidence из старого текстового `cached_report`;
   если структурированного анализа нет, анализировать закреплённый SHA.
2. `messages`: входной `message_id` + хэш всего payload + lease + статус
   + готовый `MLJobResult` и число попыток. Повтор того же сообщения
   публикует тот же результат, не вызывает Qwen повторно. Повтор
   `message_id` с иным payload отвергается.

Одна задача запускает **свежие репозитории** через `run_pipeline`, а
кэшированные `RepoAnalysis` подставляет напрямую. Profile Builder и CV
Writer пересоздаются для выбранного пользователя и набора проектов.
`force_refresh` позволяет игнорировать технический кэш. Смена commit SHA,
версии анализатора/модели или бюджета инвалидирует запись.

**Ограничение:** SQLite находится на ML worker volume. При запуске нескольких
ML-воркеров на разных машинах локальный кэш и inbox не будут общими.
Перед горизонтальным масштабированием перенести их в общий PostgreSQL
и object storage (или отдельный API хранения), не раздавать один SQLite
файл по NFS. Backend может сохранять полученные `RepoAnalysis` отдельно,
но в v2 пока не предусмотрен отдельный URL для их повторной загрузки.

## Обработка сбоев

- Сбой отдельного GitHub репозитория → `MLRepositoryResult.status=failed`,
  остальные продолжают анализироваться. Если из оставшихся данных собралось
  CV, итог `completed` с `status=partial`.
- Нет evidence или Profile/Writer не смогли сформировать CV → `failed`;
  в `repositories[].analysis` могут находиться сохранённые результаты.
- Внутренняя ошибка worker → до **3 persisted attempts** с паузой между
  requeue. Затем детерминированный `failed` event, а не бесконечный poison-loop.
- Брокер временно не подтверждает публикацию → `nack(requeue=true)`,
  при повторе извлекается уже готовый результат из inbox; новые
  GitHub/LLM-запросы не выполняются.
- Если процесс умер без ACK, после истечения lease (180 сек) задача
  может быть захвачена заново. Во время вычислений lease продлевается
  heartbeat каждые 30 сек.
- Размер ответа ограничен `ML_MESSAGE_BYTES` (по умолчанию **4 MB**).
  При превышении вместо чрезмерного события возвращается небольшой
  `failed` event; анализы останутся в локальном кэше.
- Прежний backend expiry `report_result_timeout=1800` секунд может быть
  мал для 20 крупных репозиториев на одной GPU. Нужно согласовать лимит
  числа проектов на запрос и таймауты; это отдельный эксплуатационный
  предел, а не гарантия скорости обработки.

Нет обещания «exactly once»: используется **at-least-once** +
идемпотентность. Публикация ответа происходит после его записи в
inbox. Backend ACK ответа только после DB commit (уже реализовано).

## Развёртывание

```bash
pip install -e ".[worker]"
python -m github2cv_ml.rabbitmq_worker \
  --store ./results/ml-worker.sqlite3 \
  --base-url http://127.0.0.1:8080/v1 \
  --model qwen-3.5
```

Ожидаемые переменные окружения:
`RABBITMQ_URL`, `ML_EXCHANGE`, `ML_REQUEST_QUEUE`, `ML_RESULT_QUEUE`,
`ML_DEAD_LETTER_QUEUE`, `ML_MODEL_BASE_URL`, `ML_MODEL_ID`,
`ML_STORE_PATH`, опциональный `GITHUB_TOKEN`.

Docker worker находится в `deploy/Dockerfile.worker`, пример надстройки
к *backend* `docker-compose.yml` — в `deploy/docker-compose.ml.yml`.
Запуск с локальным llama.cpp вне Docker требует
`host.docker.internal` и явного `--allow-remote-model`;
не открывайте этот endpoint в публичный интернет.

Бэкендер **не должен одновременно запускать** `demo-core` и настоящий ML
worker, иначе они будут конкурировать за одну request queue. Бэкенд
должен мигрировать DTO на v2 и убрать demo consumer из реального контура.

## Observability — база под следующий этап

Логи JSON: `job_id`, `message_id`, `status`, `elapsed_seconds`,
`cache_hits/cache_misses`, без кода или токенов. Рекомендуемые
метрики для Prometheus: `ml_jobs_total{status}`,
`ml_job_duration_seconds`, `ml_repo_cache_hits_total`,
`ml_llm_errors_total{stage}`, RabbitMQ backlog и DLQ.
Не использовать `job_id`, `repo_id`, `commit_sha` как labels
(высокая кардинальность). OpenTelemetry и Grafana можно добавить
после реальной интеграции; текущий PR **не выдаёт** logs за готовую
систему Prometheus/Grafana.

## Проверки / границы MVP

- Python unit/E2E tests без GitHub HTTP и без GPU покрывают версионирование,
  pinned SHA, cache hit/miss, дедупликацию, lease/heartbeat, повторную
  сборку CV, частичные ошибки и некорректные payload.
- Первым live integration smoke должен стать реальный RabbitMQ
  `request → ML consumer → completed event → backend DB commit`.
- Затем прогон с настоящим Qwen/Gemma, проверка потери ACK, рестарта
  ML worker и загрузки многих репозиториев.
- ML **не доказывает личный вклад по репозиторию**. `declared_role`
  остаётся декларацией кандидата; evidence/claim ID дают traceability,
  но не гарантируют семантической истинности. Проверка — задача #29.
