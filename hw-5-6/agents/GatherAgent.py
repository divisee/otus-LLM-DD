from __future__ import annotations

import time
from typing import Any, Dict

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
            if need_rag:
                if self.langfuse:
                    rag_span_context = self.langfuse.start_observation(
                        as_type="span",
                        name="rag_search",
                        input={"query": effective_query},
                    )
                else:
                    rag_span_context = None

                rag_res = self.rag_tool.invoke({"query": effective_query})
                rag_docs = rag_res.get("results", [])

                if rag_span_context:
                    rag_span_context.update(output={"docs_count": len(rag_docs), "docs": rag_docs})
                    rag_span_context.end()

                debug.append(f"Gather: RAG retrieved {len(rag_docs)} docs.")

            if need_search and not web_search_done:
                start_time = time.time()
                resp = self.llm.invoke(
                    [
                        SystemMessage(content=GATHER_SEARCH_PROMPT),
                        HumanMessage(content=effective_query),
                    ]
                )
                latency = time.time() - start_time

                if self.langfuse:
                    generation_context = self.langfuse.start_observation(
                        as_type="generation",
                        name="gather_search_query_llm",
                        model=self.llm.model_name,
                        input=f"System: {GATHER_SEARCH_PROMPT}\nUser: {effective_query}",
                    )
                    generation_context.update(output=resp.content, metadata={"latency_s": round(latency, 2)})
                    generation_context.end()

                search_query = resp.content.strip()
                debug.append(f"Gather: built search query: {search_query}")

                if self.langfuse:
                    tavily_span_context = self.langfuse.start_observation(
                        as_type="span",
                        name="tavily_search",
                        input={"query": search_query},
                    )
                else:
                    tavily_span_context = None

                search_res = self.web_tool.invoke({"query": search_query})
                web_results = search_res.get("results", [])

                if tavily_span_context:
                    tavily_span_context.update(output={
                        "results_count": len(web_results),
                        "results": [
                            {"title": r.get("title"), "url": r.get("url"), "content": r.get("content", "")[:500]}
                            for r in web_results
                        ],
                    })
                    tavily_span_context.end()

                for res in web_results:
                    url = res.get("url")
                    if url and url not in citations:
                        citations.append(url)

                debug.append(f"Gather: Tavily returned {len(web_results)} results and {len(citations)} urls.")
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
