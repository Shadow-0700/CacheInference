import os
import json
import time
from datetime import datetime
from typing import Optional

LOG_FILE = os.getenv("LOG_FILE", "cache_requests.jsonl")

def log_request(
    outcome: str,
    similarity: float,
    latency_ms: float,
    prompt: Optional[str] = None
) -> None:
    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "outcome": outcome,
        "similarity": round(similarity, 4),
        "latency_ms": round(latency_ms, 2),
        "prompt": prompt
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
