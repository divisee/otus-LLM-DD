from __future__ import annotations

import json
import time
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from .prompts import ANALYZER_PROMPT

class AnalyzerAgent:
    def __init__(self, llm, langfuse=None):
        self.llm = llm
        self.langfuse = langfuse

    def __call__(self, state) -> Dict[str, Any]:
        debug = state.get("debug_notes", [])
        messages = state.get("messages", [])

        state["user_request"] = " ".join(m["content"] for m in messages if m["role"] == "user").strip()
        user_request = state.get("user_request", "")

        need_rag = True
        need_search = True

        start_time = time.time()

        try:
            if user_request:
                resp = self.llm.invoke(
                    [
                        SystemMessage(content=ANALYZER_PROMPT),
                        HumanMessage(content=user_request),
                    ]
                )

                decisions = json.loads(resp.content)
                need_rag = bool(decisions.get("need_rag", True))
                need_search = bool(decisions.get("need_search", True))
                user_request = decisions.get("cleaned_query", user_request)
                state["user_request"] = user_request
                debug.append(f"Analyzer: cleaned query: '{user_request}'")
                debug.append(f"Analyzer: need_rag={need_rag}, need_search={need_search}")

        except Exception as e:
            request_lower = user_request.lower()
            if any(word in request_lower for word in ["посовет", "подборк", "рекоменд", "лучшие"]):
                need_rag = False
                need_search = True
            elif any(word in request_lower for word in ["это фильм", "найди", "что за фильм", "описание"]):
                need_rag = True
                need_search = False
            debug.append("Analyzer: fallback heuristic used.")

        latency = time.time() - start_time

        if self.langfuse:
            generation_context = self.langfuse.start_observation(
                as_type="generation",
                name="analyzer",
                model=self.llm.model_name,
                input=f"System: {ANALYZER_PROMPT}\nUser: {user_request}",
            )
            generation_context.update(
                output=json.dumps({"need_rag": need_rag, "need_search": need_search, "cleaned_query": user_request}),
                metadata={"latency_s": round(latency, 2)}
            )
            generation_context.end()

        state["need_rag"] = need_rag
        state["need_search"] = need_search
        state["status"] = "gathering"
        state["debug_notes"] = debug
        return state
