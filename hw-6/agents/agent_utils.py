from __future__ import annotations

from typing import Any, Dict, Optional, Type, TypeVar

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def parse_llm_output(content: str, model_class: Type[T]) -> T:
    """
    Парсит ответ LLM в Pydantic-модель с использованием PydanticOutputParser.

    Args:
        content: Строка ответа от LLM (JSON)
        model_class: Класс Pydantic-модели для валидации

    Returns:
        Экземпляр Pydantic-модели

    Raises:
        ValueError: Если не удалось распарсить ответ
    """
    parser = PydanticOutputParser(pydantic_object=model_class)
    return parser.parse(content)


def get_format_instructions(model_class: Type[T]) -> str:
    """
    Возвращает инструкции по формату для LLM на основе Pydantic-модели.
    """
    parser = PydanticOutputParser(pydantic_object=model_class)
    return parser.get_format_instructions()


def extract_usage(response) -> Optional[Dict[str, Any]]:
    """
    Извлекает данные об использовании токенов из ответа LLM.
    Работает с LangChain OpenAI/Ollama, которые прокидывают usage в response_metadata.
    """
    if not hasattr(response, "response_metadata"):
        return None

    metadata = response.response_metadata or {}
    token_usage = metadata.get("token_usage") or metadata.get("usage") or {}

    if not token_usage:
        return None

    return {
        "input": token_usage.get("prompt_tokens"),
        "output": token_usage.get("completion_tokens"),
        "total": token_usage.get("total_tokens"),
    }


def extract_user_from_chat_history(raw: str) -> Optional[str]:
    if "<chat_history>" in raw:
        segment = raw.split("<chat_history>", 1)[1].split("</chat_history>", 1)[0]
    elif "Chat History:" in raw:
        segment = raw.split("Chat History:", 1)[1]
    else:
        return None

    lines = segment.splitlines()
    extracted: list[str] = []
    current: Optional[str] = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("USER:"):
            if current:
                extracted.append(current.strip())
            current = stripped[len("USER:"):].strip()
        elif stripped.startswith("ASSISTANT:"):
            if current:
                extracted.append(current.strip())
            current = None
        else:
            if current is not None and stripped:
                current = f"{current}\n{stripped}"
    if current:
        extracted.append(current.strip())

    if extracted:
        return extracted[-1]
    return None
