import json
import httpx  # type: ignore
from typing import AsyncGenerator, Tuple, Dict, Any
from config import UPSTREAM_API_KEY, UPSTREAM_URL, VERIFY_MODEL

async def forward_upstream_stream(
    payload: Dict[str, Any]
) -> AsyncGenerator[str, None]:
    url = f"{UPSTREAM_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {UPSTREAM_API_KEY}",
        "Content-Type": "application/json"
    }

    # Ensure stream is True for streaming upstream call
    payload_copy = dict(payload)
    payload_copy["stream"] = True

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST", url, headers=headers, json=payload_copy) as response:
            if response.status_code != 200:
                err_text = await response.aread()
                yield f"data: {json.dumps({'error': f'Upstream error {response.status_code}: {err_text.decode()}'})}\n\n"
                yield "data: [DONE]\n\n"
                return

            async for line in response.aiter_lines():
                if line:
                    yield f"{line}\n\n"

async def forward_upstream_json(
    payload: Dict[str, Any]
) -> Dict[str, Any]:
    url = f"{UPSTREAM_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {UPSTREAM_API_KEY}",
        "Content-Type": "application/json"
    }
    payload_copy = dict(payload)
    payload_copy["stream"] = False

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, headers=headers, json=payload_copy)
        response.raise_for_status()
        return response.json()

async def verify_similarity(cached_prompt: str, new_prompt: str) -> bool:
    if not UPSTREAM_API_KEY:
        # Fallback if no API key configured in mock/test environment
        return True

    url = f"{UPSTREAM_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {UPSTREAM_API_KEY}",
        "Content-Type": "application/json"
    }
    
    prompt = (
        f"Do these two questions expect the same answer?\n"
        f"Q1: {cached_prompt}\n"
        f"Q2: {new_prompt}\n"
        f"Answer with only 'yes' or 'no'."
    )
    
    body = {
        "model": VERIFY_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 5
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, headers=headers, json=body)
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"].strip().lower()
                return "yes" in content
            return False
    except Exception:
        return False
