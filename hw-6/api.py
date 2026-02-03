"""
FastAPI сервис для RAG-пайплайна поиска фильмов.
Запуск: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config_utils import load_config
from build_graph import build_graph, make_llm, langfuse
from state import User_State
from tools_rag import RagRetriever, make_rag_tool
from tools_tavily import make_web_search_tool
from agents.agent_utils import extract_user_from_chat_history

import os

# Загрузка конфига
CONFIG_PATH = Path("config.yaml")
config = load_config(CONFIG_PATH)

# Инициализация компонентов
llm = make_llm(config)
rag_retriever = RagRetriever(CONFIG_PATH, langfuse=langfuse)
rag_tool = make_rag_tool(rag_retriever)
tavily_key = config.get("tavily", {}).get("api_key") or os.getenv("TAVILY_API_KEY")
web_tool = make_web_search_tool(tavily_key)

# Собираем граф один раз при старте
graph = build_graph(llm, rag_tool, web_tool)


# FastAPI приложение
app = FastAPI(
    title="Movie Agent API",
    description="Мультиагентная система на базе LangGraph для поиска и рекомендации фильмов. Включает 4 агента: Analyzer, Gather, Answer, Review.",
    version="1.0.0",
)

# CORS для Open WebUI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    thread_id: Optional[str] = "default"


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    debug_notes: list[str]


class HealthResponse(BaseModel):
    status: str
    version: str


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка работоспособности сервиса."""
    return HealthResponse(status="ok", version="1.0.0")


@app.post("/query", response_model=QueryResponse)
async def query_movies(request: QueryRequest):
    """
    Основной эндпоинт для запросов к пайплайну.

    - Если запрос про конкретный фильм → поиск в RAG (Qdrant)
    - Если запрос на рекомендации → веб-поиск (Tavily)
    """
    try:
        initial_state: User_State = {
            "messages": [{"role": "user", "content": request.query}],
            "debug_notes": [],
            "refine_iterations": 0,
            "web_search_done": False,
            "status": "analyzing",
        }

        run_config = {"configurable": {"thread_id": request.thread_id}}

        with langfuse.start_as_current_observation(
            as_type="span",
            name="movie_agent_pipeline",
            input={"query": request.query},
        ) as root:
            root.update_trace(name="movie_agent_pipeline", input={"query": request.query})
            result = graph.invoke(initial_state, config=run_config)  # type: ignore[arg-type]
            final_answer = result.get("itinerary", {}).get("answer", "")
            root.update(
                output={"status": result.get("status", "done"), "final_answer": final_answer}
            )

        return QueryResponse(
            answer=final_answer or "Не удалось получить ответ",
            sources=result.get("citations", []),
            debug_notes=result.get("debug_notes", []),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# OpenAI-совместимый API для Open WebUI
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "movie-agent"
    messages: list[ChatMessage]
    stream: bool = False


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    model: str
    choices: list[ChatChoice]


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest):
    """
    OpenAI-совместимый эндпоинт для интеграции с Open WebUI.
    """
    # Берём последнее сообщение пользователя
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found")

    query = user_messages[-1].content
    extracted = extract_user_from_chat_history(query)
    if extracted:
        query = extracted

    # Вызываем основной пайплайн
    result = await query_movies(QueryRequest(query=query))

    # Формируем ответ в формате OpenAI
    response_text = "**Ответ**\n\n" + result.answer
    if result.sources:
        response_text += "\n\n**Источники**\n" + "\n".join(f"- {s}" for s in result.sources)

    return ChatResponse(
        id="chatcmpl-movie-agent",
        model=request.model,
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response_text),
                finish_reason="stop",
            )
        ],
    )


@app.get("/v1/models")
async def list_models():
    """Список доступных моделей для Open WebUI."""
    return {
        "object": "list",
        "data": [
            {
                "id": "movie-agent",
                "object": "model",
                "owned_by": "local",
                "permission": [],
            }
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
