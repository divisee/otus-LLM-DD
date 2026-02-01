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
from state import User_State
from tools_rag import RagRetriever, make_rag_tool
from tools_tavily import make_web_search_tool
from agents.AnalyzerAgent import AnalyzerAgent
from agents.GatherAgent import GatherAgent
from agents.AnswerAgent import AnswerAgent
from agents.ReviewAgent import ReviewAgent

_config = load_config(Path("config.yaml"))
_langfuse_cfg = _config.get("langfuse", {})

from langfuse import get_client, Langfuse

try:
    langfuse = Langfuse(
        public_key=_langfuse_cfg["public_key"],
        secret_key=_langfuse_cfg["secret_key"],
        host=_langfuse_cfg["base_url"]
    )
except Exception as e:
    print(f"Langfuse initialization failed: {e}. Continuing without logging.")
    langfuse = None


def build_graph(llm, rag_tool, web_tool):
    builder = StateGraph(User_State)

    builder.add_node("analyze", AnalyzerAgent(llm, langfuse))
    builder.add_node("gather", GatherAgent(llm, rag_tool, web_tool, langfuse))
    builder.add_node("answer", AnswerAgent(llm, langfuse))
    builder.add_node("review", ReviewAgent(llm, _config, langfuse))

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

    start_time = time.time()
    result = graph.invoke(initial_state, config=run_config)
    total_latency = time.time() - start_time

    print("=== DEBUG NOTES ===")
    for note in result.get("debug_notes", []):
        print(note)
    print("=== RESULT ===")
    answers = result.get("answers", [])
    final_answer = answers[-1] if answers else result.get("itinerary", {}).get("answer", "")
    print(json.dumps({"answer": final_answer}, ensure_ascii=False, indent=2))

    if langfuse:
        try:
            langfuse.flush()
        except Exception as e:
            print(f"Langfuse flush failed: {e}")


if __name__ == "__main__":
    main()
