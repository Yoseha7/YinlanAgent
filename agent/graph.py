"""Mineradio AI Agent — Build LLM and agent graph (LangChain v1.x API)."""

import logging
from typing import Any, AsyncGenerator

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt
from langchain.agents.middleware.types import ModelRequest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph

from .config import settings
from .tools import ALL_TOOLS
from .prompts import SYSTEM_PROMPT
from .schemas import PlaybackContext

logger = logging.getLogger("mineradio-agent")

# ── Module-level shared instances (reused across requests) ──
_llm: BaseChatModel | None = None
_checkpointer: MemorySaver | None = None
_graph: CompiledStateGraph | None = None


# ── Dynamic system prompt (reads runtime context at model-call time) ──


@dynamic_prompt
def _playback_context_prompt(request: ModelRequest) -> str:
    """在每次模型调用时，从 runtime.context 动态生成 system prompt。

    LangChain v1 的 @dynamic_prompt 装饰器会将此函数包装为 AgentMiddleware，
    在 model_node 执行前拦截 ModelRequest，用 runtime.context 中的实时
    播放状态替换 system_message，从而避免 system prompt 被编译时固化。
    """
    ctx: dict[str, Any] = request.runtime.context or {}  # type: ignore[assignment]
    current_song = ctx.get("currentSong", None) or {}
    current_song_str = (
        f"{current_song.get('name', '未知')} - {current_song.get('artist', '未知')}"
        if current_song
        else "无"
    )
    queue_count = ctx.get("queue", 0)
    playlist_count = ctx.get("playlistCount", "?")
    weather = ctx.get("weather", "未知")
    login_status = ctx.get("loginStatus", "未知")

    return SYSTEM_PROMPT.format(
        current_song=current_song_str,
        queue_count=queue_count,
        playlist_count=playlist_count,
        weather=weather,
        login_status=login_status,
    )


def _get_or_build_llm() -> BaseChatModel:
    """Return the shared LLM instance, creating it lazily on first call."""
    global _llm
    if _llm is None:
        _llm = build_llm()
    return _llm


def build_llm() -> BaseChatModel:
    """Build a chat model based on the active provider (adapter pattern)."""
    provider = settings.llm_provider

    if provider == "deepseek":
        logger.info("Using DeepSeek: model=%s base=%s", settings.deepseek_model, settings.deepseek_api_base)
        return ChatOpenAI(
            model=settings.deepseek_model,
            openai_api_key=settings.deepseek_api_key,
            openai_api_base=settings.deepseek_api_base,
            temperature=0.7,
            streaming=True,
            request_timeout=45.0,
            max_retries=1,
        )

    elif provider == "openai":
        logger.info("Using OpenAI: model=%s base=%s", settings.openai_model, settings.openai_api_base)
        return ChatOpenAI(
            model=settings.openai_model,
            openai_api_key=settings.openai_api_key,
            openai_api_base=settings.openai_api_base,
            temperature=0.7,
            streaming=True,
            request_timeout=45.0,
            max_retries=1,
        )

    elif provider == "ollama":
        from langchain_ollama import ChatOllama  # lazy import

        logger.info("Using Ollama: model=%s base=%s", settings.ollama_model, settings.ollama_api_base)
        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_api_base,
            temperature=0.7,
            streaming=True,
        )

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def _get_or_build_checkpointer() -> MemorySaver:
    """Return the shared MemorySaver, creating it lazily on first call."""
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = MemorySaver()
    return _checkpointer


def build_checkpointer() -> MemorySaver:
    """Build an in-memory checkpointer for conversation history."""
    return _get_or_build_checkpointer()


def build_agent_graph(
    llm: BaseChatModel | None = None,
    tools: list[BaseTool] | None = None,
    system_prompt: str | None = None,
    checkpointer: MemorySaver | None = None,
) -> CompiledStateGraph:
    """Build and return a configured agent graph (LangChain v1.x CompiledStateGraph).

    Parameters:
        llm: Language model instance. Created via build_llm() if omitted.
        tools: List of tools. Uses ALL_TOOLS if omitted.
        system_prompt: System prompt **template** (with ``{placeholder}`` variables).
            Uses SYSTEM_PROMPT from prompts.py if omitted.  The actual values are
            filled at **model-call time** by ``_playback_context_prompt`` middleware,
            which reads ``runtime.context`` to get the live playback state.
        checkpointer: MemorySaver for conversation persistence. Created if omitted.

    Notes:
        The graph is built with ``context_schema=PlaybackContext`` and the
        ``_playback_context_prompt`` middleware.  This means the system prompt
        is **not** baked in at compile time — it's dynamically regenerated
        on every model call using the runtime context passed to
        ``graph.astream(..., context=...)``.
    """
    if llm is None:
        llm = _get_or_build_llm()
    if tools is None:
        tools = ALL_TOOLS
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT
    if checkpointer is None:
        checkpointer = _get_or_build_checkpointer()

    graph = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        middleware=[_playback_context_prompt],
        context_schema=PlaybackContext,
        checkpointer=checkpointer,
    )
    return graph


def get_or_build_agent_graph() -> CompiledStateGraph:
    """Return the shared agent graph instance (lazy singleton).

    The graph is immutable after creation in LangChain v1.x, so a single
    instance can be safely reused across requests.  The checkpointer
    (MemorySaver) is shared, so conversation history persists across
    requests when the same thread_id is used.

    Because the system prompt is now generated dynamically via the
    ``_playback_context_prompt`` middleware (which reads runtime.context),
    the same graph instance can safely serve all requests — even when
    playback state differs between requests.
    """
    global _graph
    if _graph is None:
        _graph = build_agent_graph()
    return _graph


async def stream_agent_events(
    graph: CompiledStateGraph,
    user_input: str,
    *,
    conversation_id: str | None = None,
    context: dict[str, Any] | None = None,
) -> AsyncGenerator[tuple[str, str], None]:
    """Stream agent execution events as (type, content) tuples.

    Uses the LangGraph astream() API with stream_mode="messages" for
    true token-level streaming from the LLM.

    Parameters:
        graph: The compiled agent graph.
        user_input: User message text.
        conversation_id: Conversation ID for memory threading.
        context: Runtime context dict (playback state, weather, etc.).
            Passed to ``graph.astream(..., context=...)`` and becomes
            available as ``request.runtime.context`` in middleware.

    Yields:
        ("token", token_text)           — chat model streaming tokens
        ("tool_start", tool_name)       — a tool has been invoked
        ("tool_end", "name|content")    — a tool has finished, content is the tool output
        ("chain_end", "")               — agent has finished
        ("error", error_msg)            — an error occurred
    """
    config = {"configurable": {"thread_id": conversation_id or "default"}}

    try:
        tool_invocation_names: list[str] = []

        async for event in graph.astream(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
            context=context,
            stream_mode="messages",
        ):
            if not isinstance(event, tuple) or len(event) != 2:
                continue
            msg_chunk, metadata = event

            if isinstance(msg_chunk, AIMessageChunk):
                if msg_chunk.content:
                    yield "token", msg_chunk.content
                if hasattr(msg_chunk, 'tool_call_chunks') and msg_chunk.tool_call_chunks:
                    for tc_chunk in msg_chunk.tool_call_chunks:
                        name = (tc_chunk.get("name") or "").strip()
                        if name and name not in tool_invocation_names:
                            tool_invocation_names.append(name)
                            yield "tool_start", name

            elif isinstance(msg_chunk, AIMessage):
                if msg_chunk.tool_calls:
                    for tc in msg_chunk.tool_calls:
                        name = (tc.get("name") or "").strip()
                        if name and name not in tool_invocation_names:
                            tool_invocation_names.append(name)
                            yield "tool_start", name

            elif isinstance(msg_chunk, ToolMessage):
                tool_name = getattr(msg_chunk, "name", "") or ""
                tool_content = msg_chunk.content or ""
                yield "tool_end", f"{tool_name}|{tool_content}"

        yield "chain_end", ""

    except Exception as e:
        logger.exception("Agent streaming error")
        yield "error", f"{e!s}"
