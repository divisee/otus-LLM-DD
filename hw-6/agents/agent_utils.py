from __future__ import annotations

from typing import Optional


def extract_user_from_chat_history(raw: str) -> Optional[str]:
    if "<chat_history>" in raw:
        segment = raw.split("<chat_history>", 1)[1].split("</chat_history>", 1)[0]
    elif "Chat History:" in raw:
        segment = raw.split("Chat History:", 1)[1]
    else:
        return None

    lines = segment.splitlines()
    extracted: list[str] = []
    current: Optional[str] = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("USER:"):
            if current:
                extracted.append(current.strip())
            current = stripped[len("USER:"):].strip()
        elif stripped.startswith("ASSISTANT:"):
            if current:
                extracted.append(current.strip())
            current = None
        else:
            if current is not None and stripped:
                current = f"{current}\n{stripped}"
    if current:
        extracted.append(current.strip())

    if extracted:
        return extracted[-1]
    return None
