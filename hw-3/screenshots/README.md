# Отчёт по скриншотам (HW-3)

Ниже — краткий отчёт по приложенным скриншотам.

## vLLM

### 1) `vllm-1.jpg` — запуск vLLM
Запуск сервера модели:

```
vllm serve Qwen/Qwen2.5-3B-Instruct \
  --gpu-memory-utilization 0.8 \
  --max-model-len 1024 \
  --dtype float16 \
  --host 0.0.0.0 \
  --port 8000
```

[Открыть файл](./vllm-1.jpg)

![](./vllm-1.jpg)

### 2) `vllm-2.jpg` — пример запроса/ответа через OpenAI-совместимый API (curl)
Пример запроса:

```
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "messages": [{"role": "user", "content": "Привет! Расскажи про ИИ"}],
    "max_tokens": 500
  }'
```

На скриншоте виден корректный ответ от модели.

[Открыть файл](./vllm-2.jpg)

![](./vllm-2.jpg)

### 3) `vllm-3.jpg` — нагрузка GPU
Пример вывода `nvidia-smi` с загрузкой GPU (RTX 2080) во время инференса.

[Открыть файл](./vllm-3.jpg)

![](./vllm-3.jpg)

## MLflow (базовый эксперимент)

### `base-metrics-1.jpg` — общая информация по эксперименту
Общий обзор эксперимента и запусков (runs): список run'ов, базовые поля (дата/статус) и контекст эксперимента.

[Открыть файл](./base-metrics-1.jpg)

![](./base-metrics-1.jpg)

### `base-metrics-2.jpg` — метрики (графики)
Визуализация метрик базового эксперимента в виде графиков (например, средняя/минимальная/максимальная оценка судьи и др.).

[Открыть файл](./base-metrics-2.jpg)

![](./base-metrics-2.jpg)

### `base-metrics-3.jpg` — трейсы (вопросы → ответы → оценки)
Traces/Observability: для каждого запроса видно вход (вопрос), выход (ответ модели) и результат оценки (LLM-as-a-Judge).

[Открыть файл](./base-metrics-3.jpg)

![](./base-metrics-3.jpg)

