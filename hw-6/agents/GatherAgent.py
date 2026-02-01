from __future__ import annotations

import time
from typing import Any, Dict
import json

from langchain_core.messages import HumanMessage, SystemMessage

from .prompts import GATHER_SEARCH_PROMPT

class GatherAgent:
    def __init__(self, llm, rag_tool, web_tool, langfuse=None):
        self.llm = llm
        self.rag_tool = rag_tool
        self.web_tool = web_tool
        self.langfuse = langfuse

    def __call__(self, state) -> Dict[str, Any]:
        debug = state.get("debug_notes", [])
        user_request = state.get("user_request", "")

        need_rag = state.get("need_rag", True)
        need_search = state.get("need_search", True)
        web_search_done = state.get("web_search_done", False)

        if state.get("refine_query"):
            need_rag = True
            need_search = not web_search_done

        if web_search_done:
            need_search = False
            debug.append("Gather: web search already done, skipping.")

        refine_query = state.get("refine_query")
        effective_query = refine_query or user_request

        rag_docs = []
        web_results = state.get("web_results", [])
        citations = state.get("citations", [])

        try:
            if self.langfuse:
                with self.langfuse.start_as_current_observation(
                    as_type="agent",
                    name="gather_agent",
                    input={"query": effective_query, "need_rag": need_rag, "need_search": need_search},
                ) as agent_span:
                    agent_span.update_trace(name="movie_agent_pipeline")
                    if need_rag:
                        rag_res = self.rag_tool.invoke({"query": effective_query})
                        rag_docs = rag_res.get("results", [])
                        debug.append(f"Gather: RAG retrieved {len(rag_docs)} docs.")

                    if need_search and not web_search_done:
                        with agent_span.start_as_current_observation(
                            as_type="tool",
                            name="tavily_search",
                            input={"query": effective_query},
                        ) as tool_span:
                            tool_span.update_trace(name="movie_agent_pipeline")
                            search_res = self.web_tool.invoke({"query": effective_query})
                            raw_web_results = search_res.get("results", [])
                            tool_span.update(output={"hits": len(raw_web_results)})

                        if raw_web_results:
                            raw_results_text = "\n\n".join([
                                f"Title: {r.get('title', '')}\nURL: {r.get('url', '')}\nContent: {r.get('content', '')[:500]}"
                                for r in raw_web_results
                            ])

                            start_time = time.time()
                            resp = self.llm.invoke([
                                SystemMessage(content=GATHER_SEARCH_PROMPT),
                                HumanMessage(content=f"Запрос: {effective_query}\nРезультаты:\n{raw_results_text}"),
                            ])
                            latency = time.time() - start_time

                            with agent_span.start_as_current_observation(
                                as_type="generation",
                                name="gather_clean_web_results_llm",
                                model=self.llm.model_name,
                                input=f"System: {GATHER_SEARCH_PROMPT}\nUser: Запрос: {effective_query}\nРезультаты:\n{raw_results_text}",
                            ) as gen:
                                gen.update(output=resp.content, metadata={"latency_s": round(latency, 2)})

                            try:
                                web_results = json.loads(resp.content)
                                debug.append(f"Gather: cleaned web results, got {len(web_results)} items.")
                            except Exception as e:
                                debug.append(f"Gather: failed to parse cleaned web results, using raw. Error: {e}")
                                web_results = raw_web_results
                        else:
                            web_results = []
                            debug.append("Gather: no web results.")

                        for res in web_results:
                            url = res.get("url")
                            if url and url not in citations:
                                citations.append(url)

                        debug.append(f"Gather: final web results {len(web_results)} items, citations {len(citations)}.")
                        state["web_search_done"] = True
                    else:
                        debug.append("Gather: search disabled or already done.")

                    agent_span.update(
                        output={
                            "rag_docs": len(rag_docs),
                            "web_results": len(web_results),
                            "citations": len(citations),
                        }
                    )
            else:
                if need_rag:
                    rag_res = self.rag_tool.invoke({"query": effective_query})
                    rag_docs = rag_res.get("results", [])
                    debug.append(f"Gather: RAG retrieved {len(rag_docs)} docs.")

                if need_search and not web_search_done:
                    search_res = self.web_tool.invoke({"query": effective_query})
                    raw_web_results = search_res.get("results", [])

                    if raw_web_results:
                        raw_results_text = "\n\n".join([
                            f"Title: {r.get('title', '')}\nURL: {r.get('url', '')}\nContent: {r.get('content', '')[:500]}"
                            for r in raw_web_results
                        ])

                        resp = self.llm.invoke([
                            SystemMessage(content=GATHER_SEARCH_PROMPT),
                            HumanMessage(content=f"Запрос: {effective_query}\nРезультаты:\n{raw_results_text}"),
                        ])

                        try:
                            web_results = json.loads(resp.content)
                            debug.append(f"Gather: cleaned web results, got {len(web_results)} items.")
                        except Exception as e:
                            debug.append(f"Gather: failed to parse cleaned web results, using raw. Error: {e}")
                            web_results = raw_web_results
                    else:
                        web_results = []
                        debug.append("Gather: no web results.")

                    for res in web_results:
                        url = res.get("url")
                        if url and url not in citations:
                            citations.append(url)

                    debug.append(f"Gather: final web results {len(web_results)} items, citations {len(citations)}.")
                    state["web_search_done"] = True
                else:
                    debug.append("Gather: search disabled or already done.")
        except Exception:
            pass

        state["rag_docs"] = rag_docs
        state["web_results"] = web_results
        state["citations"] = citations
        state["status"] = "answering"
        state["debug_notes"] = debug
        state["refine_query"] = None
        return state
