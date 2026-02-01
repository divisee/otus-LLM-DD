# Langfuse Demo

Демо с тремя экспериментами по трассировке в Langfuse: полный набор типов наблюдений, запуск через декоратор и ручное логирование без декоратора.

## Настройка

1. Установите зависимости:

   ```bash
   cd ..
   pip install -r requirements.txt
   ```

2. Создайте файл `hw-5/.env` и заполните переменные окружения:

   ```dotenv
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_BASE_URL=https://cloud.langfuse.com
   ```

3. Запустите нужный эксперимент:

   ```bash
   cd ..
   python demo.py
   python demo_observe.py
   ```
Общий список трейсов после запуска экспериментов:

![Traces list](screenshots/traces.png)

Начальная страница Langfuse (дашборд с графиками и диаграммами - трейсов, цен и scores):

![Langfuse dashboard](screenshots/dashboard.png)

## Эксперимент 1: полный набор типов наблюдений

Файл: `hw-5/demo.py`

Сценарий создает один root span и вложенные наблюдения всех доступных типов, включая generation, embedding, agent, tool, retriever, guardrail и ошибки. Скриншот показывает развернутый трейс с таймлайном и статусами.

Примеры нескольких вызовов:

```python
langfuse = get_client()

with langfuse.start_as_current_observation(
    as_type="span",
    name="example_trace",
    input={"started": True},
) as root:
    with root.start_as_current_observation(
        as_type="agent",
        name="planner_agent",
        input={"goal": "Find capital of France"},
    ) as agent_span:
        agent_span.update(output={"status": "completed"})

    with root.start_as_current_observation(
        as_type="tool",
        name="wiki_search",
        input={"query": "capital of France"},
    ) as tool_span:
        tool_span.update(output={"hits": 3})

    with root.start_as_current_observation(
        as_type="generation",
        name="llm_call",
        model="gpt-4o-mini",
        model_parameters={"temperature": "0.7"},
        input={"prompt": "What is the capital of France?"},
    ) as gen:
        gen.update(
            output="The capital of France is Paris.",
            usage_details={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        )

    with root.start_as_current_observation(
        as_type="span",
        name="error_occurred",
        level="ERROR",
        status_message="Connection timeout",
        metadata={"error": "Connection timeout", "code": 500},
    ) as error_span:
        error_span.update(output={"retry": False})
```

![Example trace](screenshots/example_trace.png)

## Эксперимент 2: декоратор @observe

Файл: `hw-5/demo_observe.py`

Сценарий демонстрирует трассировку функций через декоратор `@observe`. Трейс формируется автоматически, а вложенные шаги попадают в дерево.

Пример кода:

```python
from langfuse import observe

@observe(name="process_data_decorated")
def process_data_decorated(data):
    result1 = step_one_decorated(data)
    result2 = step_two_decorated(result1)
    return {"final_result": result2}

@observe(name="step_one_decorated")
def step_one_decorated(data):
    return {**data, "step_one_processed": True}

@observe(name="step_two_decorated")
def step_two_decorated(data):
    return {**data, "step_two_processed": True}
```

![Decorated trace](screenshots/process_data_decorated.png)

## Эксперимент 3: ручное логирование без декоратора

Файл: `hw-5/demo_observe.py`

Сценарий строит трейс вручную через `start_as_current_observation` и `update`. На скриншоте видно два шага, вложенные в root span.

Пример кода:

```python
def process_data_manual(data, trace_name, langfuse):
    with langfuse.start_as_current_observation(
        as_type="span",
        name=trace_name,
        input=data,
    ) as root_span:
        result1 = step_one_manual(data, langfuse)
        result2 = step_two_manual(result1, langfuse)
        root_span.update(output={"final_result": result2})
        return {"final_result": result2}

def step_one_manual(data, langfuse):
    with langfuse.start_as_current_observation(
        as_type="span",
        name="step_one_manual",
        input=data,
    ) as span:
        output = {**data, "step_one_processed": True}
        span.update(output=output)
        return output

def step_two_manual(data, langfuse):
    with langfuse.start_as_current_observation(
        as_type="span",
        name="step_two_manual",
        input=data,
    ) as span:
        output = {**data, "step_two_processed": True}
        span.update(output=output)
        return output
```

![Manual trace](screenshots/process_data_manual.png)

## Вывод

Наиболее информативным выглядит `example_trace`, потому что в одном трейсе показаны все ключевые типы наблюдений (generation, embedding, agent, tool, retriever, guardrail), есть успешные и ошибочные ветки, а также таймлайн по шагам. Это дает наиболее полную картину поведения пайплайна и структуры трассировки по сравнению с минимальными сценариями. 
НО(!) такой трейс сложнее использовать для точечного анализа затрат по вложенным шагам: например, если нужно получить стоимость одной конкретной генерации, а не общую стоимость всего запуска.

## Примечания

- `get_client()` использует переменные окружения `LANGFUSE_*`.
- Для короткоживущих скриптов важно вызывать `flush()` и `shutdown()`.

См. результаты в дашборде Langfuse: https://cloud.langfuse.com
