import hashlib
from typing import Optional, Dict, Any

class ExactCache:
    def __init__(self):
        self._cache: Dict[str, Any] = {}

    def _hash(self, prompt: str) -> str:
        return hashlib.sha256(prompt.strip().encode("utf-8")).hexdigest()

    def get(self, prompt: str) -> Optional[Any]:
        key = self._hash(prompt)
        return self._cache.get(key)

    def set(self, prompt: str, response: Any) -> None:
        key = self._hash(prompt)
        self._cache[key] = response
