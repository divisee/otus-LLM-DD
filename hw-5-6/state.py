from typing import Any, Dict, List, Optional, TypedDict

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class User_State(TypedDict, total=False):
    messages: List[Dict[str, Any]]
    user_request: str

    plan: List[str]
    facts: Dict[str, Any]

    rag_docs: List[Dict[str, Any]]
    citations: List[str]
    web_results: List[Dict[str, Any]]

    debug_notes: List[str]

    status: Literal["analyzing", "gathering", "answering", "done"]

    need_rag: bool
    need_search: bool
    web_search_done: bool

    refine_iterations: int
    refine_needed: bool
    refine_query: Optional[str]

    itinerary: Dict[str, Any]
    assumptions: List[str]
