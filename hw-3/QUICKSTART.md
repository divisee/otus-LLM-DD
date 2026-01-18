# Быстрый старт

## Описание
Проект для работы с vLLM и MLflow. Модель: **Qwen/Qwen2.5-3B-Instruct** с поддержкой chat-шаблонов.

## Установка

```bash
cd hw-3
make install
```

## Использование

### 1. Запустите vLLM сервер

```bash
vllm serve Qwen/Qwen2.5-3B-Instruct \
  --gpu-memory-utilization 0.8 \
  --max-model-len 1024 \
  --dtype float16 \
  --host 0.0.0.0 \
  --port 8000
```

### 2. Проверьте подключение

```bash
make check
```

### 3. Запустите тесты API

```bash
make test
```

### 4. Запустите эксперимент

```bash
make experiment
```

### 4.1 Эксперимент с make_genai_metric

```bash
make genai
```

### 5. Просмотрите результаты в MLflow

**ВАЖНО!** Запускайте MLflow UI из папки hw-3:

```bash
cd hw-3
make mlflow
```

Или используйте скрипт:
```bash
./start_mlflow.sh
```

Откройте: http://localhost:5000

**MLflow использует SQLite базу данных** (`mlflow.db`)
**Если MLflow UI пустой:**
1. Остановите MLflow UI (Ctrl+C)
2. Убедитесь, что вы в папке hw-3
3. Запустите заново: `make mlflow`
4. Проверьте, что файл `mlflow.db` создан в папке hw-3

## Команды

```bash
make install    # Установить зависимости
make check      # Проверить vLLM сервер
make test       # Тестировать API (requests/httpx/openai)
make experiment # Запустить эксперимент
make genai      # Эксперимент с make_genai_metric
make mlflow     # Запустить MLflow UI
make clean      # Очистить временные файлы
```

## Структура проекта

```
hw-3/
├── config_loader.py
├── vllm_client.py
├── llm_judge.py
├── check_vllm.py
├── test_vllm_api.py
├── run_experiment.py
├── config.yaml
├── Makefile
└── requirements.txt
```

## Что реализовано

### Взаимодействие с vLLM (3 метода)
- **requests** - HTTP запросы
- **httpx** - HTTP запросы
- **openai** - OpenAI SDK

### LLM-as-a-Judge
Оценка качества ответов от 1 до 5

### MLflow
- Логирование параметров
- Сохранение результатов
- Визуализация метрик
