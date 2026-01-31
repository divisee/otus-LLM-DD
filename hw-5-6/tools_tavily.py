from typing import Any, Dict

from langchain_core.tools import tool
from tavily import TavilyClient



def make_web_search_tool(api_key: str | None):
    tavily = TavilyClient(api_key=api_key or "")

    @tool("web_search")
    def web_search(query: str) -> Dict[str, Any]:
        """
        Поиск через Tavily.
        Обязательно указывать:
        - query: конкретный вопрос (например, "посоветуй лучшие комедии")
        Возвращает:
        - "results": список {"title", "url", "content"}
        """
        res = tavily.search(query=query, max_results=10)
        return res

    return web_search
