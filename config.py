import os
from dotenv import load_dotenv  # type: ignore

load_dotenv()

UPSTREAM_API_KEY = os.getenv("UPSTREAM_API_KEY", "")
UPSTREAM_URL = os.getenv("UPSTREAM_URL", "https://api.openai.com/v1")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.97"))
VERIFY_THRESHOLD = float(os.getenv("VERIFY_THRESHOLD", "0.90"))
VERIFY_MODEL = os.getenv("VERIFY_MODEL", "gpt-4o-mini")
