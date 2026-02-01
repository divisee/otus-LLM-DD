from __future__ import annotations

import json
import time
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from .prompts import ANSWER_PROMPT

class AnswerAgent:
    def __init__(self, llm, langfuse=None):
        self.llm = llm
        self.langfuse = langfuse

    def __call__(self, state) -> Dict[str, Any]:
        debug = state.get("debug_notes", [])
        user_request = state.get("user_request", "")
        rag_docs = state.get("rag_docs", [])
        web_results = state.get("web_results", [])
        citations = state.get("citations", [])

        rag_text = "\n\n".join([doc.get("text", "") for doc in rag_docs]) or "Локальные данные не найдены."

        web_text_chunks = []
        for res in web_results:
            title = res.get("title", "")
            content = res.get("content", "")
            url = res.get("url", "")
            web_text_chunks.append(f"TITLE: {title}\nURL: {url}\nCONTENT: {content[:1200]}")
        web_text = "\n\n---\n\n".join(web_text_chunks) or "Веб-результаты отсутствуют."

        try:
            start_time = time.time()
            resp = self.llm.invoke([
                SystemMessage(content=ANSWER_PROMPT),
                HumanMessage(content=f"Запрос пользователя:\n{user_request}"),
                HumanMessage(content=f"Локальные данные (RAG):\n{rag_text}"),
                HumanMessage(content=f"Результаты веб-поиска (Tavily):\n{web_text}"),
                HumanMessage(content=f"Доступные URL-источники:\n{citations}"),
            ])
            latency = time.time() - start_time

            if self.langfuse:
                generation_context = self.langfuse.start_observation(
                    as_type="generation",
                    name="answerer_llm",
                    model=self.llm.model_name,
                    input=f"System: {ANSWER_PROMPT}\nUser: Запрос пользователя:\n{user_request}\nЛокальные данные (RAG):\n{rag_text}\nРезультаты веб-поиска (Tavily):\n{web_text}\nДоступные URL-источники:\n{citations}",
                )
                generation_context.update(output=resp.content, metadata={"latency_s": round(latency, 2)})
                generation_context.end()

            try:
                data = json.loads(resp.content)
                debug.append("Answerer: parsed JSON successfully.")
            except Exception as e:
                if self.langfuse:
                    self.langfuse.start_observation(name="answerer_json_error", as_type="span").update(input={"error": str(e), "raw_content": resp.content[:500]}).end()
                data = {
                    "answer": "Не удалось корректно сформировать JSON-ответ.",
                    "sources": citations,
                    "assumptions": ["Не удалось распарсить JSON от LLM, использован fallback-ответ."],
                }
                debug.append("Answerer: invalid JSON, fallback used.")
        except Exception:
            pass

        state["itinerary"] = {"answer": data.get("answer", "")}
        state["assumptions"] = data.get("assumptions", [])
        state["citations"] = data.get("sources", citations)
        state["status"] = "answering"
        state["debug_notes"] = debug

        state["answers"] = state.get("answers", []) + [data.get("answer", "")]

        return state
