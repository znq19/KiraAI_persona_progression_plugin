import json
from pathlib import Path
from typing import Dict, Any


class StateManager:
    def __init__(self, data_dir: Path, scope: str = "session"):
        self.data_dir = data_dir
        self.scope = scope
        self.state_file = data_dir / "states.json"
        self._states: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    self._states = json.load(f)
            except Exception:
                self._states = {}
        else:
            self._states = {}

    def _save(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self._states, f, indent=2, ensure_ascii=False)

    def _get_key(self, session_id: str) -> str:
        if self.scope == "global":
            return "global"
        return session_id

    def get_state(self, session_id: str) -> Dict[str, Any]:
        key = self._get_key(session_id)
        return self._states.get(key, {})

    def update_state(self, session_id: str, updates: Dict[str, Any]):
        key = self._get_key(session_id)
        if key not in self._states:
            self._states[key] = {}
        self._states[key].update(updates)
        self._save()

    def get_stage(self, session_id: str, default_stage: int = 0) -> int:
        return self.get_state(session_id).get("stage", default_stage)

    def set_stage(self, session_id: str, stage: int):
        self.update_state(session_id, {"stage": stage})

    def get_turn_counter(self, session_id: str) -> int:
        return self.get_state(session_id).get("turn_counter", 0)

    def set_turn_counter(self, session_id: str, value: int):
        self.update_state(session_id, {"turn_counter": value})

    def increment_turn(self, session_id: str):
        cur = self.get_turn_counter(session_id)
        self.set_turn_counter(session_id, cur + 1)

    def reset_turn_counter(self, session_id: str):
        self.set_turn_counter(session_id, 0)

    def get_consecutive_stay(self, session_id: str) -> int:
        return self.get_state(session_id).get("consecutive_stay", 0)

    def set_consecutive_stay(self, session_id: str, value: int):
        self.update_state(session_id, {"consecutive_stay": value})