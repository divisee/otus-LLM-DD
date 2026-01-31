from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from config_utils import load_config
from prompts import ANALYZER_PROMPT, ANSWER_PROMPT, GATHER_SEARCH_PROMPT, REVIEW_PROMPT
from state import User_State
from tools_rag import RagRetriever, make_rag_tool
from tools_tavily import make_web_search_tool

_config = load_config(Path("config.yaml"))
_langfuse_cfg = _config.get("langfuse", {})

from langfuse import get_client, Langfuse

langfuse = Langfuse(
    public_key=_langfuse_cfg["public_key"],
    secret_key=_langfuse_cfg["secret_key"],
    host=_langfuse_cfg["base_url"]
)

class AnalyzerAgent:
    def __init__(self, llm) -> None:
        self.llm = llm

    def __call__(self, state: User_State) -> User_State:
        debug = state.get("debug_notes", [])
        messages = state.get("messages", [])

        state["user_request"] = " ".join(m["content"] for m in messages if m["role"] == "user").strip()
        user_request = state.get("user_request", "")

        need_rag = True
        need_search = True

        with langfuse.start_as_current_observation(
            as_type="span",
            name="analyzer",
            input={"user_request": user_request},
        ) as span:
            if user_request:
                try:
                    with langfuse.start_as_current_observation(
                        as_type="generation",
                        name="analyzer_llm",
                        model=self.llm.model_name,
                        input=[{"role": "system", "content": ANALYZER_PROMPT}, {"role": "user", "content": user_request}],
                    ) as generation:
                        start_time = time.time()
                        resp = self.llm.invoke(
                            [
                                SystemMessage(content=ANALYZER_PROMPT),
                                HumanMessage(content=user_request),
                            ]
                        )
                        latency = time.time() - start_time
                        generation.update(output=resp.content, metadata={"latency_s": round(latency, 2)})

                    decisions = json.loads(resp.content)
                    need_rag = bool(decisions.get("need_rag", True))
                    need_search = bool(decisions.get("need_search", True))
                    debug.append(f"Analyzer: need_rag={need_rag}, need_search={need_search}")

                except Exception as e:
                    langfuse.start_observation(name="analyzer_error", as_type="event").update(input={"error": str(e)}).end()
                    request_lower = user_request.lower()
                    if any(word in request_lower for word in ["посовет", "подборк", "рекоменд", "лучшие"]):
                        need_rag = False
                        need_search = True
                    elif any(word in request_lower for word in ["это фильм", "найди", "что за фильм", "описание"]):
                        need_rag = True
                        need_search = False
                    debug.append("Analyzer: fallback heuristic used.")

            span.update(output={"need_rag": need_rag, "need_search": need_search})

        state["need_rag"] = need_rag
        state["need_search"] = need_search
        state["status"] = "gathering"  # type: ignore[typeddict-item]
        state["debug_notes"] = debug
        return state


class GatherAgent:
    def __init__(self, llm, rag_tool, web_tool) -> None:
        self.llm = llm
        self.rag_tool = rag_tool
        self.web_tool = web_tool

    def __call__(self, state: User_State) -> User_State:
        debug = state.get("debug_notes", [])
        user_request = state.get("user_request", "")

        need_rag = state.get("need_rag", True)
        need_search = state.get("need_search", True)
        web_search_done = state.get("web_search_done", False)

        # Веб-поиск максимум 1 раз за весь запрос
        if web_search_done:
            need_search = False
            debug.append("Gather: web search already done, skipping.")

        refine_query = state.get("refine_query")
        effective_query = refine_query or user_request

        rag_docs: List[Dict[str, Any]] = []
        web_results = state.get("web_results", [])  # сохраняем предыдущие результаты
        citations: List[str] = state.get("citations", [])

        with langfuse.start_as_current_observation(
            as_type="span",
            name="gatherer",
            input={"query": effective_query, "need_rag": need_rag, "need_search": need_search},
        ) as span:
            if need_rag:
                with langfuse.start_as_current_observation(
                    as_type="span",
                    name="rag_search",
                    input={"query": effective_query},
                ) as rag_span:
                    rag_res = self.rag_tool.invoke({"query": effective_query})
                    rag_docs = rag_res.get("results", [])
                    rag_span.update(output={"docs_count": len(rag_docs)})
                debug.append(f"Gather: RAG retrieved {len(rag_docs)} docs.")

            if need_search and not web_search_done:
                with langfuse.start_as_current_observation(
                    as_type="generation",
                    name="gather_search_query_llm",
                    model=self.llm.model_name,
                    input=[{"role": "system", "content": GATHER_SEARCH_PROMPT}, {"role": "user", "content": effective_query}],
                ) as generation:
                    start_time = time.time()
                    resp = self.llm.invoke(
                        [
                            SystemMessage(content=GATHER_SEARCH_PROMPT),
                            HumanMessage(content=effective_query),
                        ]
                    )
                    latency = time.time() - start_time
                    generation.update(output=resp.content, metadata={"latency_s": round(latency, 2)})

                search_query = resp.content.strip()
                debug.append(f"Gather: built search query: {search_query}")

                with langfuse.start_as_current_observation(
                    as_type="span",
                    name="tavily_search",
                    input={"query": search_query},
                ) as tavily_span:
                    search_res = self.web_tool.invoke({"query": search_query})
                    web_results = search_res.get("results", [])
                    tavily_span.update(output={
                        "results_count": len(web_results),
                        "results": [
                            {"title": r.get("title"), "url": r.get("url"), "content": r.get("content", "")[:500]}
                            for r in web_results
                        ],
                    })

                for res in web_results:
                    url = res.get("url")
                    if url and url not in citations:
                        citations.append(url)

                debug.append(f"Gather: Tavily returned {len(web_results)} results and {len(citations)} urls.")
                state["web_search_done"] = True  # Веб-поиск выполнен, больше не повторяем
            else:
                debug.append("Gather: search disabled or already done.")

            span.update(output={"rag_docs_count": len(rag_docs), "web_results_count": len(web_results)})

        state["rag_docs"] = rag_docs
        state["web_results"] = web_results
        state["citations"] = citations
        state["status"] = "answering"  # type: ignore[typeddict-item]
        state["debug_notes"] = debug
        state["refine_query"] = None
        return state


class AnswerAgent:
    def __init__(self, llm) -> None:
        self.llm = llm

    def __call__(self, state: User_State) -> User_State:
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

        messages_input = [
            {"role": "system", "content": ANSWER_PROMPT},
            {"role": "user", "content": f"Запрос пользователя:\n{user_request}"},
            {"role": "user", "content": f"Локальные данные (RAG):\n{rag_text}"},
            {"role": "user", "content": f"Результаты веб-поиска (Tavily):\n{web_text}"},
            {"role": "user", "content": f"Доступные URL-источники:\n{citations}"},
        ]

        with langfuse.start_as_current_observation(
            as_type="span",
            name="answerer",
            input={"user_request": user_request, "rag_docs_count": len(rag_docs), "web_results_count": len(web_results)},
        ) as span:
            with langfuse.start_as_current_observation(
                as_type="generation",
                name="answerer_llm",
                model=self.llm.model_name,
                input=messages_input,
            ) as generation:
                messages = [
                    SystemMessage(content=ANSWER_PROMPT),
                    HumanMessage(content=f"Запрос пользователя:\n{user_request}"),
                    HumanMessage(content=f"Локальные данные (RAG):\n{rag_text}"),
                    HumanMessage(content=f"Результаты веб-поиска (Tavily):\n{web_text}"),
                    HumanMessage(content=f"Доступные URL-источники:\n{citations}"),
                ]

                start_time = time.time()
                resp = self.llm.invoke(messages)
                latency = time.time() - start_time
                generation.update(output=resp.content, metadata={"latency_s": round(latency, 2)})

            try:
                data = json.loads(resp.content)
                debug.append("Answerer: parsed JSON successfully.")
            except Exception as e:
                langfuse.start_observation(name="answerer_json_error", as_type="event").update(input={"error": str(e), "raw_content": resp.content[:500]}).end()
                data = {
                    "answer": "Не удалось корректно сформировать JSON-ответ.",
                    "sources": citations,
                    "assumptions": ["Не удалось распарсить JSON от LLM, использован fallback-ответ."],
                }
                debug.append("Answerer: invalid JSON, fallback used.")

            span.update(output={"answer_length": len(data.get("answer", ""))})

        state["itinerary"] = {"answer": data.get("answer", "")}
        state["assumptions"] = data.get("assumptions", [])
        state["citations"] = data.get("sources", citations)
        state["status"] = "answering"  # type: ignore[typeddict-item]
        state["debug_notes"] = debug
        return state


class ReviewAgent:
    def __init__(self, llm) -> None:
        self.llm = llm

    def __call__(self, state: User_State) -> User_State:
        debug = state.get("debug_notes", [])
        user_request = state.get("user_request", "")
        itinerary = state.get("itinerary", {})
        assumptions = state.get("assumptions", [])
        refine_iterations = state.get("refine_iterations", 0)

        react_input = [
            {"role": "system", "content": REVIEW_PROMPT},
            {"role": "user", "content": f"Запрос пользователя:\n{user_request}"},
            {"role": "user", "content": f"Текущий JSON-ответ:\n{json.dumps({'itinerary': itinerary, 'assumptions': assumptions}, ensure_ascii=False)}"},
        ]

        refine_needed = False
        refine_query = None

        with langfuse.start_as_current_observation(
            as_type="span",
            name="reviewer",
            input={"user_request": user_request, "refine_iterations": refine_iterations},
        ) as span:
            with langfuse.start_as_current_observation(
                as_type="generation",
                name="reviewer_llm",
                model=self.llm.model_name,
                input=react_input,
            ) as generation:
                react_messages = [
                    SystemMessage(content=REVIEW_PROMPT),
                    HumanMessage(content=f"Запрос пользователя:\n{user_request}"),
                    HumanMessage(content=f"Текущий JSON-ответ:\n{json.dumps({'itinerary': itinerary, 'assumptions': assumptions}, ensure_ascii=False)}"),
                ]

                start_time = time.time()
                react_resp = self.llm.invoke(react_messages)
                latency = time.time() - start_time
                generation.update(output=react_resp.content, metadata={"latency_s": round(latency, 2)})

            try:
                react_data = json.loads(react_resp.content)
                refine_needed = bool(react_data.get("refine_needed", False))
                refine_query_raw = react_data.get("refine_query") or ""
                refine_query = refine_query_raw.strip() or None
            except Exception as e:
                langfuse.start_observation(name="reviewer_json_error", as_type="event").update(input={"error": str(e)}).end()
                debug.append("Reviewer: failed to parse refine JSON, assuming refine_needed=false.")
                refine_needed = False
                refine_query = None

            if refine_needed and refine_iterations < 3 and refine_query:
                state["refine_needed"] = True
                state["refine_query"] = refine_query
                state["refine_iterations"] = refine_iterations + 1
                state["status"] = "gathering"  # type: ignore[typeddict-item]
                debug.append(f"Reviewer: requesting another gather iteration (iteration #{state['refine_iterations']}, query='{refine_query}').")
            else:
                if refine_needed and refine_iterations >= 3:
                    debug.append("Reviewer: refine limit reached, stopping.")
                state["refine_needed"] = False
                state["refine_query"] = None
                state["status"] = "done"  # type: ignore[typeddict-item]
                debug.append("Reviewer: done.")

            span.update(output={"refine_needed": refine_needed, "refine_query": refine_query})

        state["debug_notes"] = debug
        return state


def build_graph(llm, rag_tool, web_tool):
    builder = StateGraph(User_State)

    builder.add_node("analyze", AnalyzerAgent(llm))
    builder.add_node("gather", GatherAgent(llm, rag_tool, web_tool))
    builder.add_node("answer", AnswerAgent(llm))
    builder.add_node("review", ReviewAgent(llm))

    builder.set_entry_point("analyze")

    builder.add_edge("analyze", "gather")
    builder.add_edge("gather", "answer")
    builder.add_edge("answer", "review")

    def route_from_review(state: User_State):
        if state.get("refine_needed", False):
            return "gather"
        return END

    builder.add_conditional_edges(
        "review",
        route_from_review,
        {
            "gather": "gather",
            END: END,
        },
    )

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


def make_llm(config: dict):
    openai_cfg = config["openai"]
    api_key = openai_cfg.get("api_key") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key is missing (openai.api_key or OPENAI_API_KEY).")
    return ChatOpenAI(
        api_key=api_key,
        model=openai_cfg.get("model", "gpt-4o-mini"),
        base_url=openai_cfg.get("base_url", "https://api.openai.com/v1"),
        temperature=0.2,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LangGraph RAG pipeline")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--query", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    llm = make_llm(config)
    rag_retriever = RagRetriever(args.config)
    rag_tool = make_rag_tool(rag_retriever)
    tavily_key = config.get("tavily", {}).get("api_key") or os.getenv("TAVILY_API_KEY")
    web_tool = make_web_search_tool(tavily_key)

    graph = build_graph(llm, rag_tool, web_tool)

    initial_state: User_State = {
        "messages": [{"role": "user", "content": args.query}],
        "debug_notes": [],
        "refine_iterations": 0,
        "web_search_done": False,
        "status": "analyzing",
    }

    run_config = {"configurable": {"thread_id": "default"}}

    with langfuse.start_as_current_observation(
        as_type="span",
        name="movie_agent_pipeline",
        input={"query": args.query},
        metadata={"config_path": str(args.config)},
    ) as root_span:
        start_time = time.time()
        result = graph.invoke(initial_state, config=run_config)
        total_latency = time.time() - start_time

        root_span.update(
            output={
                "answer": result.get("itinerary", {}).get("answer", ""),
                "citations": result.get("citations", []),
                "assumptions": result.get("assumptions", []),
                "status": result.get("status"),
            },
            metadata={
                "total_debug_notes": len(result.get("debug_notes", [])),
                "final_status": result.get("status"),
                "total_latency_s": round(total_latency, 2),
            },
        )

    print("=== DEBUG NOTES ===")
    for note in result.get("debug_notes", []):
        print(note)
    print("=== RESULT ===")
    print(json.dumps(result.get("itinerary", {}), ensure_ascii=False, indent=2))

    langfuse.flush()


if __name__ == "__main__":
    main()
