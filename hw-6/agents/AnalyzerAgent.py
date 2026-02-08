from __future__ import annotations

import time
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from .agent_utils import extract_usage, parse_llm_output
from .prompts import ANALYZER_PROMPT
from .schemas import AnalyzerOutput


class AnalyzerAgent:
    def __init__(self, llm, langfuse):
        self.llm = llm
        self.langfuse = langfuse

    def __call__(self, state) -> Dict[str, Any]:
        debug = state.get("debug_notes", [])
        messages = state.get("messages", [])

        state["user_request"] = " ".join(m["content"] for m in messages if m["role"] == "user").strip()
        user_request = state.get("user_request", "")

        need_rag = True
        need_search = True

        with self.langfuse.start_as_current_observation(
            as_type="agent",
            name="analyzer_agent",
            input={"user_request": user_request},
        ) as agent_span:
            start_time = time.time()
            try:
                if user_request:
                    resp = self.llm.invoke(
                        [
                            SystemMessage(content=ANALYZER_PROMPT),
                            HumanMessage(content=user_request),
                        ]
                    )

                    decisions = parse_llm_output(resp.content, AnalyzerOutput)
                    need_rag = decisions.need_rag
                    need_search = decisions.need_search
                    user_request = decisions.cleaned_query
                    state["user_request"] = user_request
                    debug.append(f"Analyzer: cleaned query: '{user_request}'")
                    debug.append(f"Analyzer: need_rag={need_rag}, need_search={need_search}")

                    with agent_span.start_as_current_observation(
                        as_type="generation",
                        name="analyzer_llm",
                        model=self.llm.model_name,
                        input=f"System: {ANALYZER_PROMPT}\nUser: {user_request}",
                    ) as gen:
                        latency = time.time() - start_time
                        usage = extract_usage(resp)
                        gen.update(
                            output=decisions.model_dump_json(),
                            usage=usage,
                            metadata={"latency_s": round(latency, 2)},
                        )

            except (ValidationError, ValueError) as e:
                debug.append(f"Analyzer: parse error ({e}), using fallback heuristic.")
                request_lower = user_request.lower()
                if any(word in request_lower for word in ["посовет", "подборк", "рекоменд", "лучшие"]):
                    need_rag = False
                    need_search = True
                elif any(word in request_lower for word in ["это фильм", "найди", "что за фильм", "описание"]):
                    need_rag = True
                    need_search = False
                debug.append("Analyzer: fallback heuristic used.")

            agent_span.update(
                output={"need_rag": need_rag, "need_search": need_search, "cleaned_query": user_request}
            )

        state["need_rag"] = need_rag
        state["need_search"] = need_search
        state["status"] = "gathering"
        state["debug_notes"] = debug
        return state
