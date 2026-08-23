"""
The three things that live as long as the process, in one place.

    registry   the model bundles, loaded once and kept
    db         the conversation store
    CHATS      slot state per chat

Slot state is keyed by chat, not by connection: with the socket gone there is no connection to
key it on, and a reload that reuses its chat id continues the conversation. A chat this process
has never seen is rebuilt from the stored turns on first use (`backend.store.last_state`), so a
restart does not lose where "there" is.
"""

from __future__ import annotations

from backend import store
from backend.nlu import Registry
from src.schema import ConversationState

registry = Registry()
db = store.connect()
CHATS: dict[str, ConversationState] = {}


def conversation_state(chat_id: str) -> ConversationState:
    """The slots for this chat: in memory, else rebuilt from the log, else fresh."""
    if (state := CHATS.get(chat_id)) is not None:
        return state
    stored = store.last_state(db, chat_id)
    return ConversationState(**stored) if stored else ConversationState()
