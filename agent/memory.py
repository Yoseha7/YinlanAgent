"""Mineradio AI Agent — Conversation persistence via FileChatMessageHistory.

Stores conversation history per conversation_id in the conversations/ directory.
"""

import json
import os
from pathlib import Path
from typing import Optional

from langchain_community.chat_message_histories import FileChatMessageHistory

CONVERSATIONS_DIR = Path(__file__).parent / "conversations"


def ensure_conversations_dir():
    CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)


def get_message_history(conversation_id: str) -> FileChatMessageHistory:
    """Get or create a FileChatMessageHistory for the given conversation ID."""
    ensure_conversations_dir()
    file_path = CONVERSATIONS_DIR / f"{conversation_id}.json"
    return FileChatMessageHistory(str(file_path))


def list_conversations() -> list[dict]:
    """List all saved conversations with basic metadata."""
    ensure_conversations_dir()
    results = []
    for f in sorted(CONVERSATIONS_DIR.glob("*.json"), key=os.path.getmtime, reverse=True):
        try:
            data = json.loads(f.read_text("utf-8"))
            messages = data if isinstance(data, list) else []
            results.append({
                "id": f.stem,
                "message_count": len(messages),
                "updated_at": os.path.getmtime(f),
            })
        except Exception:
            continue
    return results


def delete_conversation(conversation_id: str) -> bool:
    """Delete a saved conversation file."""
    file_path = CONVERSATIONS_DIR / f"{conversation_id}.json"
    if file_path.exists():
        file_path.unlink()
        return True
    return False
