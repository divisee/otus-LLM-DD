#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from config_loader import Config
from vllm_client import VLLMClient


def test_vllm_api():
    print("=" * 80)
    print("Тестирование взаимодействия с vLLM API")
    print("=" * 80)

    config = Config()
    client = VLLMClient(config)

    test_question = "What is the capital of Germany?"

    print(f"\nВопрос: {test_question}\n")

    print("-" * 80)
    print("1. Тест с использованием библиотеки requests")
    print("-" * 80)
    try:
        response_requests = client.chat_with_requests([
            {"role": "user", "content": test_question}
        ])
        print(f"Ответ: {response_requests}\n")
    except Exception as e:
        print(f"Ошибка: {e}\n")

    print("-" * 80)
    print("2. Тест с использованием библиотеки httpx")
    print("-" * 80)
    try:
        response_httpx = client.chat_with_httpx([
            {"role": "user", "content": test_question}
        ])
        print(f"Ответ: {response_httpx}\n")
    except Exception as e:
        print(f"Ошибка: {e}\n")

    print("-" * 80)
    print("3. Тест с использованием библиотеки openai")
    print("-" * 80)
    try:
        response_openai = client.chat_with_openai([
            {"role": "user", "content": test_question}
        ])
        print(f"Ответ: {response_openai}\n")
    except Exception as e:
        print(f"Ошибка: {e}\n")

    print("-" * 80)
    print("4. Тест с chat-шаблоном (system + user)")
    print("-" * 80)
    try:
        response_chat = client.chat_with_openai([
            {"role": "system", "content": "You are a helpful geography expert."},
            {"role": "user", "content": test_question}
        ])
        print(f"Ответ: {response_chat}\n")
    except Exception as e:
        print(f"Ошибка: {e}\n")

    print("=" * 80)
    print("Тестирование завершено")
    print("=" * 80)


if __name__ == "__main__":
    test_vllm_api()
