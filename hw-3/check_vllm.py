#!/usr/bin/env python3

import requests
import json


def check_vllm_server():
    print("=" * 80)
    print("Проверка подключения к vLLM серверу")
    print("=" * 80)

    base_url = "http://localhost:8000"

    print(f"\nПроверка доступности сервера: {base_url}")

    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Сервер доступен и работает")
        else:
            print(f"⚠️  Сервер вернул код: {response.status_code}")
    except Exception as e:
        print(f"❌ Ошибка подключения к серверу: {e}")
        print("\nУбедитесь, что vLLM сервер запущен:")
        print("vllm serve Qwen/Qwen2.5-3B-Instruct --gpu-memory-utilization 0.8 --max-model-len 1024 --dtype float16 --host 0.0.0.0 --port 8000")
        return False

    print("\n" + "-" * 80)
    print("Тест простого запроса через requests")
    print("-" * 80)

    url = f"{base_url}/v1/chat/completions"

    payload = {
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "messages": [
            {"role": "user", "content": "What is the capital of Germany?"}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()

        result = response.json()
        answer = result['choices'][0]['message']['content']

        print(f"\nВопрос: What is the capital of Germany?")
        print(f"Ответ: {answer}")
        print("\n✅ Тест пройден успешно!")

        print("\n" + "-" * 80)
        print("Информация о запросе:")
        print("-" * 80)
        print(f"Model: {result.get('model', 'N/A')}")
        print(f"Usage:")
        if 'usage' in result:
            print(f"  - Prompt tokens: {result['usage'].get('prompt_tokens', 'N/A')}")
            print(f"  - Completion tokens: {result['usage'].get('completion_tokens', 'N/A')}")
            print(f"  - Total tokens: {result['usage'].get('total_tokens', 'N/A')}")

        return True

    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP ошибка: {e}")
        print(f"Response: {response.text}")
        return False
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False


if __name__ == "__main__":
    success = check_vllm_server()

    if success:
        print("\n" + "=" * 80)
        print("vLLM сервер работает корректно!")
        print("Можно переходить к запуску основных экспериментов.")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("Обнаружены проблемы с vLLM сервером.")
        print("Устраните ошибки перед запуском экспериментов.")
        print("=" * 80)
