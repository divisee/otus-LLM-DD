from langfuse import observe, get_client
import time
import uuid
import os
from dotenv import load_dotenv

# --- Конфигурация и инициализация ---
load_dotenv()

required_vars = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"]
missing = [key for key in required_vars if not os.getenv(key)]
if missing:
    print(
        "Ошибка: не найдены переменные окружения: "
        + ", ".join(missing)
        + ". Заполните langfuse-demo/.env или экспортируйте их в окружение."
    )
    exit()

# Инициализация клиента Langfuse
# get_client() использует переменные окружения LANGFUSE_*
langfuse = get_client()

print("Клиент Langfuse инициализирован.")

# --- Пример 1: Использование декоратора @observe ---

@observe(name="process_data_decorated")
def process_data_decorated(data):
    """
    Эта функция и ее вложенные вызовы автоматически отслеживаются с помощью @observe.
    В Langfuse вы увидите трейс 'process_data_decorated' с двумя вложенными спанами.
    """
    print("\n--- Запуск функции с декоратором @observe ---")
    time.sleep(0.1)  # Имитация работы
    result1 = step_one_decorated(data)
    time.sleep(0.1)  # Имитация работы
    result2 = step_two_decorated(result1)
    return {"final_result": result2}

@observe(name="step_one_decorated")
def step_one_decorated(data):
    """Вложенная функция, также с декоратором."""
    print("Шаг 1 (декорированный): обработка данных...")
    return {**data, "step_one_processed": True}

@observe(name="step_two_decorated")
def step_two_decorated(data):
    """Еще одна вложенная функция с декоратором."""
    print("Шаг 2 (декорированный): завершение обработки...")
    return {**data, "step_two_processed": True}


# --- Пример 2: Ручное создание наблюдений (без @observe) ---

def process_data_manual(data, trace_name):
    """
    Эта функция выполняет то же самое, что и декорированная, но все наблюдения
    (spans) создаются вручную с помощью контекстного менеджера.
    В Langfuse вы увидите трейс с такой же структурой.
    """
    print("\n--- Запуск функции с ручным созданием наблюдений ---")
    # Создаем корневой спан, который также создает трейс
    with langfuse.start_as_current_observation(as_type="span", name=trace_name, input=data) as root_span:
        time.sleep(0.1)  # Имитация работы
        result1 = step_one_manual(data)
        time.sleep(0.1)  # Имитация работы
        result2 = step_two_manual(result1)

        root_span.update(output={"final_result": result2})
        return {"final_result": result2}

def step_one_manual(data):
    """Вложенная функция, где спан создается вручную."""
    with langfuse.start_as_current_observation(as_type="span", name="step_one_manual", input=data) as span:
        print("Шаг 1 (ручной): обработка данных...")
        output = {**data, "step_one_processed": True}
        span.update(output=output)
        return output

def step_two_manual(data):
    """Еще одна вложенная функция с ручным спаном."""
    with langfuse.start_as_current_observation(as_type="span", name="step_two_manual", input=data) as span:
        print("Шаг 2 (ручной): завершение обработки...")
        output = {**data, "step_two_processed": True}
        span.update(output=output)
        return output

# --- Запуск демонстрации ---
if __name__ == "__main__":
    initial_data = {"user_id": "user-456", "request_id": str(uuid.uuid4())}

    # Вызов функции с декораторами
    decorated_result = process_data_decorated(initial_data)
    print("Результат (декорированный):", decorated_result)

    # Вызов функции с ручным созданием спанов
    manual_result = process_data_manual(initial_data, trace_name="process_data_manual")
    print("Результат (ручной):", manual_result)

    # Важно: в короткоживущих скриптах нужно вызывать flush/shutdown,
    # чтобы гарантировать отправку всех данных до завершения программы.
    print("\nОтправка данных в Langfuse...")
    langfuse.flush()
    langfuse.shutdown()
    print("Готово. Проверьте дашборд Langfuse.")
    print("Вы должны увидеть два новых трейса: 'process_data_decorated' и 'process_data_manual'.")
