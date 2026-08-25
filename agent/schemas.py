"""Mineradio AI Agent — Pydantic request/response models & runtime context schema."""

from typing import Any, TypedDict

from pydantic import BaseModel, Field


# ── Runtime context schema (for LangChain create_agent context_schema) ──


class PlaybackContext(TypedDict, total=False):
    """运行时上下文：前端传入的播放状态，通过 LangGraph Runtime.context 注入 middleware。

    所有字段都是可选的（total=False），确保即使部分字段缺失也不会崩溃。
    """
    currentSong: dict[str, Any] | None
    """当前正在播放的歌曲对象 {name, artist, album, cover, id, source, ...}"""
    queue: int
    """播放队列中待播的歌曲数量"""
    playing: bool
    """是否正在播放"""
    currentIdx: int
    """当前播放索引"""
    weather: str | None
    """天气信息（如 '晴', '🌧 小雨'）"""
    playlistCount: int
    """用户歌单总数"""
    loginStatus: str
    """登录状态（'logged_in' | 'not_logged_in'）"""


# ── API 请求/响应模型 ──


class ChatRequest(BaseModel):
    """Incoming chat request from the frontend."""
    message: str = Field(..., description="User message text")
    context: dict = Field(default_factory=dict, description="Extra context from frontend")
    conversation_id: str = Field(default="", description="Optional conversation ID for memory")


class SSEMessage(BaseModel):
    """A single SSE event payload sent to the client."""
    type: str = Field(..., description="Event type: token | tool_start | tool_end | chain_end | error")
    content: str = Field(default="", description="Event content")


class HealthResponse(BaseModel):
    """Response for GET /health."""
    status: str = "ok"
    provider: str = ""
    model: str = ""
