"""Mineradio AI Agent — FastAPI sidecar service (port 3001).

Endpoints:
  GET  /health  → Health check
  POST /chat    → SSE streaming chat
"""

import logging
import sys
from pathlib import Path
from contextlib import asynccontextmanager

# 确保项目根目录在 Python 路径中，使 agent 包可导入
_project_dir = str(Path(__file__).parent.parent)
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from agent.config import settings
from agent.schemas import ChatRequest, HealthResponse, SSEMessage
from agent.graph import get_or_build_agent_graph, stream_agent_events
from agent.tools import _close_client

logging.basicConfig(
    level=logging.INFO,
    format="[Agent] %(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("mineradio-agent")


# ── Shared agent graph (singleton, reused across requests) ──
#
# The graph is built ONCE with:
#   - context_schema=PlaybackContext  →  enables runtime context injection
#   - middleware=[_playback_context_prompt]  →  @dynamic_prompt middleware
#
# At model-call time, the middleware reads request.runtime.context (the raw
# playback state dict passed via graph.astream(..., context=...)) and
# dynamically generates a formatted system prompt — so the prompt always
# reflects the current song, queue, weather, etc., without rebuilding the
# graph.
#
# The shared MemorySaver preserves conversation history when the same
# thread_id (conversation_id) is used across requests.


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Mineradio AI Agent starting on port %d", settings.agent_port)
    logger.info("LLM provider: %s, model: %s", settings.llm_provider, settings.effective_model)
    # Pre-warm the shared graph on startup (singleton, built once)
    get_or_build_agent_graph()
    logger.info("Agent graph initialized (shared MemorySaver, ^_playback_context_prompt middleware)")
    yield
    logger.info("Mineradio AI Agent shutting down")
    await _close_client()


app = FastAPI(title="Mineradio AI Agent", version="1.0.0", lifespan=lifespan)


# ── Health ─────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        provider=settings.llm_provider,
        model=settings.effective_model,
    )


# ── Chat (SSE) ────────────────────────────────────────────────


@app.post("/chat")
async def chat(body: ChatRequest, request: Request):
    # 1. Get the shared singleton graph (built once, reused across requests).
    graph = get_or_build_agent_graph()

    # 2. Pass raw frontend context directly to the streaming function.
    #    The _playback_context_prompt middleware will read it at model-call
    #    time via request.runtime.context and format the system prompt
    #    dynamically — no need to format/rebuild here.
    context = body.context or {}

    # 3. Build conversation ID for memory threading
    conversation_id = body.conversation_id or None

    async def event_generator():
        try:
            async for etype, econtent in stream_agent_events(
                graph,
                body.message,
                conversation_id=conversation_id,
                context=context,
            ):
                if await request.is_disconnected():
                    break
                payload = SSEMessage(type=etype, content=econtent)
                # Manually format SSE: "data: {json}\n\n"
                yield f"data: {payload.model_dump_json()}\n\n"
        except Exception as e:
            logger.exception("Chat streaming error")
            error_payload = SSEMessage(type="error", content=f"抱歉，AI 回复出了点问题：{e!s}")
            yield f"data: {error_payload.model_dump_json()}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Main ───────────────────────────────────────────────────────


def main():
    uvicorn.run(
        "agent.main:app",
        host="127.0.0.1",
        port=settings.agent_port,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()
