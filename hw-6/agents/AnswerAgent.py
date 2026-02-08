from __future__ import annotations

import time
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from .agent_utils import extract_usage, parse_llm_output
from .prompts import ANSWER_PROMPT
from .schemas import AnswerOutput


class AnswerAgent:
    def __init__(self, llm, langfuse):
        self.llm = llm
        self.langfuse = langfuse

    def __call__(self, state) -> Dict[str, Any]:
        debug = state.get("debug_notes", [])
        user_request = state.get("user_request", "")
        rag_docs = state.get("rag_docs", [])
        web_results = state.get("web_results", [])
        citations = state.get("citations", [])

        rag_text = "\n\n".join([doc.get("text_ru", "") for doc in rag_docs]) or "Локальные данные не найдены."

        web_text_chunks = []
        for res in web_results:
            title = res.get("title", "")
            content = res.get("content", "")
            url = res.get("url", "")
            web_text_chunks.append(f"TITLE: {title}\nURL: {url}\nCONTENT: {content[:1200]}")
        web_text = "\n\n---\n\n".join(web_text_chunks) or "Веб-результаты отсутствуют."

        data = AnswerOutput(answer="Не удалось получить ответ.", sources=citations)

        try:
            with self.langfuse.start_as_current_observation(
                as_type="agent",
                name="answer_agent",
                input={"user_request": user_request},
            ) as agent_span:
                start_time = time.time()
                resp = self.llm.invoke([
                    SystemMessage(content=ANSWER_PROMPT),
                    HumanMessage(content=f"Запрос пользователя:\n{user_request}"),
                    HumanMessage(content=f"Локальные данные (RAG):\n{rag_text}"),
                    HumanMessage(content=f"Результаты веб-поиска (Tavily):\n{web_text}"),
                    HumanMessage(content=f"Доступные URL-источники:\n{citations}"),
                ])
                latency = time.time() - start_time

                with agent_span.start_as_current_observation(
                    as_type="generation",
                    name="answerer_llm",
                    model=self.llm.model_name,
                    input=f"System: {ANSWER_PROMPT}\nUser: Запрос пользователя:\n{user_request}\nЛокальные данные (RAG):\n{rag_text}\nРезультаты веб-поиска (Tavily):\n{web_text}\nДоступные URL-источники:\n{citations}",
                ) as gen:
                    usage = extract_usage(resp)
                    gen.update(output=resp.content, usage=usage, metadata={"latency_s": round(latency, 2)})

                try:
                    data = parse_llm_output(resp.content, AnswerOutput)
                    debug.append("Answerer: parsed JSON successfully.")
                except (ValidationError, ValueError) as e:
                    with agent_span.start_as_current_observation(
                        as_type="span",
                        name="answerer_json_error",
                        input={"error": str(e)},
                    ) as error_span:
                        error_span.update(output={"raw_content": resp.content[:500]})
                    data = AnswerOutput(
                        answer="Не удалось корректно сформировать JSON-ответ.",
                        sources=citations,
                    )
                    debug.append(f"Answerer: validation error ({e}), fallback used.")

                agent_span.update(output={"answer_len": len(data.answer)})
        except Exception:
            pass

        state["itinerary"] = {"answer": data.answer}
        state["citations"] = data.sources
        state["status"] = "answering"
        state["debug_notes"] = debug

        state["answers"] = state.get("answers", []) + [data.answer]

        return state
