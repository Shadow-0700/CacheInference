import time
import json
import uuid
import asyncio
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Request, BackgroundTasks  # type: ignore
from fastapi.responses import StreamingResponse, JSONResponse  # type: ignore

from config import SIMILARITY_THRESHOLD, VERIFY_THRESHOLD
from cache.exact_cache import ExactCache
from cache.semantic_cache import SemanticCache
from upstream import forward_upstream_stream, forward_upstream_json, verify_similarity
from logger import log_request

app = FastAPI(title="CacheInference Proxy")

exact_cache = ExactCache()
semantic_cache = SemanticCache()

def extract_prompt(messages: List[Dict[str, Any]]) -> str:
    user_msgs = [m.get("content", "") for m in messages if m.get("role") == "user"]
    if user_msgs:
        return user_msgs[-1]
    return json.dumps(messages)

def create_cached_json_response(content: str, model: str = "cached-model") -> Dict[str, Any]:
    return {
        "id": f"chatcmpl-cache-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }

async def stream_cached_response(content: str, model: str = "cached-model"):
    chat_id = f"chatcmpl-cache-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    # Stream in word chunks for realistic SSE behavior
    words = content.split(" ")
    for idx, word in enumerate(words):
        chunk_text = word + (" " if idx < len(words) - 1 else "")
        data = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": chunk_text},
                    "finish_reason": None
                }
            ]
        }
        yield f"data: {json.dumps(data)}\n\n"
        await asyncio.sleep(0.01)

    finish_data = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }
        ]
    }
    yield f"data: {json.dumps(finish_data)}\n\n"
    yield "data: [DONE]\n\n"

def update_caches_task(prompt: str, embedding: Any, response_text: str):
    exact_cache.set(prompt, response_text)
    if embedding is not None:
        semantic_cache.add(prompt, embedding, response_text)

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, background_tasks: BackgroundTasks):
    start_time = time.perf_counter()
    body = await request.json()
    
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    model = body.get("model", "gpt-4o-mini")
    prompt = extract_prompt(messages)

    # 1. Check Exact Cache
    cached_exact = exact_cache.get(prompt)
    if cached_exact is not None:
        latency_ms = (time.perf_counter() - start_time) * 1000
        log_request(outcome="exact_hit", similarity=1.0, latency_ms=latency_ms, prompt=prompt)
        
        if stream:
            return StreamingResponse(stream_cached_response(cached_exact, model=model), media_type="text/event-stream")
        return JSONResponse(content=create_cached_json_response(cached_exact, model=model))

    # 2. Check Semantic Cache
    cached_prompt, cached_response, similarity, embedding = semantic_cache.search(prompt)
    
    # High similarity hit
    if similarity > SIMILARITY_THRESHOLD and cached_response is not None:
        latency_ms = (time.perf_counter() - start_time) * 1000
        log_request(outcome="semantic_hit", similarity=similarity, latency_ms=latency_ms, prompt=prompt)
        exact_cache.set(prompt, cached_response)
        
        if stream:
            return StreamingResponse(stream_cached_response(cached_response, model=model), media_type="text/event-stream")
        return JSONResponse(content=create_cached_json_response(cached_response, model=model))

    # Verification zone: 0.90 <= similarity <= 0.97
    if VERIFY_THRESHOLD <= similarity <= SIMILARITY_THRESHOLD and cached_prompt is not None and cached_response is not None:
        is_verified = await verify_similarity(cached_prompt, prompt)
        if is_verified:
            latency_ms = (time.perf_counter() - start_time) * 1000
            log_request(outcome="verified_hit", similarity=similarity, latency_ms=latency_ms, prompt=prompt)
            exact_cache.set(prompt, cached_response)
            
            if stream:
                return StreamingResponse(stream_cached_response(cached_response, model=model), media_type="text/event-stream")
            return JSONResponse(content=create_cached_json_response(cached_response, model=model))
        else:
            verify_rejected = True
    else:
        verify_rejected = False

    # 3. Cache Miss / Rejected -> Forward to Upstream
    outcome = "verify_rejected" if verify_rejected else "miss"

    if stream:
        async def streaming_generator():
            full_response_text = []
            async for chunk in forward_upstream_stream(body):
                yield chunk
                # Extract text chunks to cache full response
                if chunk.startswith("data: ") and not chunk.startswith("data: [DONE]"):
                    try:
                        chunk_json = json.loads(chunk[6:].strip())
                        delta = chunk_json.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if delta:
                            full_response_text.append(delta)
                    except Exception:
                        pass

            latency_ms = (time.perf_counter() - start_time) * 1000
            log_request(outcome=outcome, similarity=similarity, latency_ms=latency_ms, prompt=prompt)
            
            complete_text = "".join(full_response_text)
            if complete_text:
                update_caches_task(prompt, embedding, complete_text)

        return StreamingResponse(streaming_generator(), media_type="text/event-stream")
    else:
        upstream_resp = await forward_upstream_json(body)
        latency_ms = (time.perf_counter() - start_time) * 1000
        log_request(outcome=outcome, similarity=similarity, latency_ms=latency_ms, prompt=prompt)

        try:
            content = upstream_resp["choices"][0]["message"]["content"]
            background_tasks.add_task(update_caches_task, prompt, embedding, content)
        except Exception:
            pass

        return JSONResponse(content=upstream_resp)
