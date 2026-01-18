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
make experiment # Запустить основной эксперимент
make advanced   # Запустить расширенный эксперимент
make mlflow     # Запустить MLflow UI
make clean      # Очистить временные файлы
```

## Структура проекта

```
hw-3/
├── config_loader.py              # Загрузчик конфигурации
├── vllm_client.py               # Клиент vLLM (3 метода)
├── llm_judge.py                 # Метрика LLM-as-a-Judge
├── check_vllm.py                # Проверка подключения
├── test_vllm_api.py             # Тесты API
├── run_experiment.py            # Основной эксперимент
├── run_advanced_experiment.py   # Расширенный эксперимент
├── config.yaml                  # Конфигурация
├── Makefile                     # Команды управления
└── requirements.txt             # Зависимости
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
