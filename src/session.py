"""
session.py
----------

In-memory session store.  Maintains up to HISTORY_LIMIT turns per user.
Each turn is a pair: {"role": "user"|"assistant", "content": "..."}

"""

from collections import deque
from typing import Optional
from src.rag_engine import get_connection
import json

HISTORY_LIMIT = 6   # 3 user + 3 assistant turns = 6 messages


class SessionManager:

    """def __init__(self):
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
        self._sessions.pop(k, None)"""


    def get_user_row(self, user_id: str) -> Optional[dict]:
        """Fetch the row for this user from DB."""
        conn = get_connection()
        row = conn.execute(
            "SELECT * FROM sessions WHERE user_id = ?", (user_id,)
        ).fetchone()
        conn.close()
        return row

    def upsert(self, user_id: str, history: list):
        """Insert or update a user's session in DB."""
        conn = get_connection()
        conn.execute("""
            INSERT INTO sessions (user_id, history)
            VALUES (?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                history = excluded.history
        """, (user_id, json.dumps(history)))

        conn.commit()
        conn.close()

    

    def get_history(self, user_id) -> list:
        """Return conversation history for the user."""
        row = self.get_user_row(str(user_id))
        if not row:
            return []
        return json.loads(row["history"])

    def add_turn(self, user_id, role: str, content: str) -> None:
        """Add one message to this user's history."""
        uid = str(user_id)
        history = self.get_history(uid)

        #keep only last MAX_HISTORY turns
        history.append({"role": role, "content": content})
        if len(history) > HISTORY_LIMIT:
            history = history[-HISTORY_LIMIT:]

 
        self.upsert(uid, history)

    def clear(self, user_id) -> None:
        """Wipe everything for this user."""
        conn = get_connection()
        conn.execute(
            "DELETE FROM sessions WHERE user_id = ?", (str(user_id),)
        )
        conn.commit()
        conn.close()


sessions = SessionManager()



#----------------------------------END---------------------------------------