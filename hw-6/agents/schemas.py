"""Pydantic-модели для структурированных ответов LLM."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class AnalyzerOutput(BaseModel):
    """Результат анализа запроса пользователя."""
    cleaned_query: str = Field(description="Очищенный запрос пользователя без приветствий и лишних слов")
    need_rag: bool = Field(description="Нужно ли искать в локальной базе фильмов (RAG)")
    need_search: bool = Field(description="Нужно ли делать веб-поиск")


class WebSearchResult(BaseModel):
    """Очищенный результат веб-поиска."""
    url: str = Field(description="URL источника")
    title: str = Field(description="Заголовок")
    content: str = Field(description="Краткое описание без лишних тегов")


class AnswerOutput(BaseModel):
    """Ответ кино-агента пользователю."""
    answer: str = Field(description="Текст ответа для пользователя в Markdown")
    sources: List[str] = Field(default_factory=list, description="Список URL источников")


class ReviewOutput(BaseModel):
    """Результат проверки качества ответа."""
    refine_needed: bool = Field(description="Нужно ли дополнительно искать информацию")
    refine_query: Optional[str] = Field(default="", description="Перефразированный запрос для дополнительного поиска")

