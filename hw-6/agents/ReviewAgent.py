from __future__ import annotations

import json
import time
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from .agent_utils import extract_usage, parse_llm_output
from .prompts import REVIEW_PROMPT
from .schemas import ReviewOutput


class ReviewAgent:
    def __init__(self, llm, config, langfuse):
        self.llm = llm
        self.config = config
        self.langfuse = langfuse

    def __call__(self, state) -> Dict[str, Any]:
        debug = state.get("debug_notes", [])
        user_request = state.get("user_request", "")
        itinerary = state.get("itinerary", {})
        refine_iterations = state.get("refine_iterations", 0)

        max_iterations = self.config.get("agent", {}).get("max_refine_iterations", 3)

        refine_needed = False
        refine_query = None

        try:
            with self.langfuse.start_as_current_observation(
                as_type="agent",
                name="review_agent",
                input={"user_request": user_request, "iteration": refine_iterations},
            ) as agent_span:
                start_time = time.time()
                react_resp = self.llm.invoke([
                    SystemMessage(content=REVIEW_PROMPT),
                    HumanMessage(content=f"Запрос пользователя:\n{user_request}"),
                    HumanMessage(content=f"Текущий JSON-ответ:\n{json.dumps({'itinerary': itinerary}, ensure_ascii=False)}"),
                ])
                latency = time.time() - start_time

                with agent_span.start_as_current_observation(
                    as_type="generation",
                    name="reviewer_llm",
                    model=self.llm.model_name,
                    input=f"System: {REVIEW_PROMPT}\nUser: Запрос пользователя:\n{user_request}\nТекущий JSON-ответ:\n{json.dumps({'itinerary': itinerary}, ensure_ascii=False)}",
                ) as gen:
                    usage = extract_usage(react_resp)
                    gen.update(output=react_resp.content, usage=usage, metadata={"latency_s": round(latency, 2)})

                try:
                    react_data = parse_llm_output(react_resp.content, ReviewOutput)
                    refine_needed = react_data.refine_needed
                    refine_query = (react_data.refine_query or "").strip() or None
                except (ValidationError, ValueError) as e:
                    with agent_span.start_as_current_observation(
                        as_type="span",
                        name="reviewer_json_error",
                        input={"error": str(e)},
                    ) as error_span:
                        error_span.update(output={"raw_content": react_resp.content[:500]})
                    debug.append(f"Reviewer: validation error ({e}), assuming refine_needed=false.")
                    refine_needed = False
                    refine_query = None

                if refine_needed and refine_iterations < max_iterations and refine_query:
                    state["refine_needed"] = True
                    state["refine_query"] = refine_query
                    state["refine_iterations"] = refine_iterations + 1
                    state["status"] = "gathering"
                    debug.append(f"Reviewer: requesting another gather iteration (iteration #{state['refine_iterations']}, query='{refine_query}').")
                else:
                    if refine_needed and refine_iterations >= max_iterations:
                        debug.append("Reviewer: refine limit reached, stopping.")
                    state["refine_needed"] = False
                    state["refine_query"] = None
                    state["status"] = "done"
                    debug.append("Reviewer: done.")

                agent_span.update(output={"refine_needed": refine_needed})
        except Exception:
            pass

        state["debug_notes"] = debug
        return state
