"""
session.py
----------

In-memory session store.  Maintains up to HISTORY_LIMIT turns per user.
Each turn is a pair: {"role": "user"|"assistant", "content": "..."}

"""

from collections import deque
from typing import Optional

HISTORY_LIMIT = 6   # 3 user + 3 assistant turns = 6 messages


class SessionManager:

    def __init__(self):
        self._sessions: dict[str, deque] = {}

    def _key(self, user_id) -> str:
        return str(user_id)

    def get_history(self, user_id) -> list[dict]:
        return list(self._sessions.get(self._key(user_id), []))

    def add_turn(self, user_id, role: str, content: str) -> None:
        k = self._key(user_id)
        if k not in self._sessions:
            self._sessions[k] = deque(maxlen=HISTORY_LIMIT)
        self._sessions[k].append({"role": role, "content": content})

    def clear(self, user_id) -> None:
        k = self._key(user_id)
        self._sessions.pop(k, None)



sessions = SessionManager()



#----------------------------------END---------------------------------------