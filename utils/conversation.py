import json
import os
from datetime import datetime
from typing import Optional

HISTORY_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "conversations")

def _ensure_dir():
    os.makedirs(HISTORY_DIR, exist_ok=True)

def get_history_path(user_id: str) -> str:
    _ensure_dir()
    return os.path.join(HISTORY_DIR, f"{user_id}.json")

def load_history(user_id: str) -> list[dict]:
    path = get_history_path(user_id)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_turn(user_id: str, query: str, answer: str, max_turns: int = 10):
    history = load_history(user_id)
    history.append({
        "query": query,
        "answer": answer[:500],
        "timestamp": datetime.now().isoformat(),
    })
    history = history[-max_turns:]
    with open(get_history_path(user_id), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def get_context_block(user_id: str, max_turns: int = 5) -> str:
    history = load_history(user_id)[-max_turns:]
    if not history:
        return ""
    lines = ["*** 用户历史提问（用于理解上下文） ***"]
    for i, h in enumerate(history, 1):
        lines.append(f"{i}. 问：{h['query']} 答：{h['answer'][:200]}")
    return "\n".join(lines)
