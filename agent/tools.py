"""Mineradio AI Agent — LangChain tools that call Node.js API endpoints."""

import logging
from typing import Any

from langchain_core.tools import tool
import httpx
from langgraph.runtime import get_runtime, Runtime

from .config import settings

logger = logging.getLogger("mineradio-agent")

# ─── Tavily search client (lazy init) ────────────────────────────
_tavily_client: Any = None


def _get_tavily_client() -> Any:
    """Get or create the shared Tavily client."""
    global _tavily_client
    if _tavily_client is None:
        from tavily import TavilyClient

        _tavily_client = TavilyClient(api_key=settings.tavily_api_key)
    return _tavily_client

_client: httpx.AsyncClient | None = None


async def _get_client() -> httpx.AsyncClient:
    """Get or create the shared httpx async client."""
    global _client
    if _client is None:
        _client = httpx.AsyncClient(base_url=settings.node_server_base, timeout=15.0)
    return _client


async def _close_client():
    """Close the shared httpx client."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


# ─── Mood keywords ────────────────────────────────────────────
_MOOD_KEYWORDS = {
    "开心": ["快乐", "欢快", "阳光", "喜悦", "happy", "joy", "sunny", " upbeat", "愉快", "轻快"],
    "伤感": ["悲伤", "难过", "伤感", "忧郁", "sad", "blue", "melancholy", "心痛", "流泪", "寂寞"],
    "安静": ["安静", "宁静", "平和", "放松", "calm", "peaceful", "relax", "舒缓", "轻柔", "催眠"],
    "激昂": ["燃", "热血", "激情", "振奋", "epic", "powerful", "激昂", "摇滚", "战斗", "澎湃"],
    "浪漫": ["浪漫", "甜蜜", "爱情", "温柔", "romantic", "love", "sweet", "心动", "告白", "婚礼"],
}


def _classify_mood_by_text(text: str) -> list[str]:
    """Classify text into mood categories based on keyword matching."""
    text_lower = text.lower()
    moods = []
    for mood, keywords in _MOOD_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in text_lower:
                moods.append(mood)
                break
    return moods if moods else ["未知"]


# ─── Web search via Tavily ──────────────────────────────────────


async def _search_web(query: str, max_results: int = 5) -> str:
    """使用 Tavily AI Search API 搜索互联网，返回格式化文本。

    Tavily 是专为 LLM Agent 优化的搜索引擎，返回结构化、高相关性的结果。
    无需缓存，Tavily 自带去重和相关性排序。

    Args:
        query: 搜索关键词
        max_results: 返回结果数量（1-10）
    """
    try:
        client = _get_tavily_client()
        # Tavily 同步 API，在 executor 中运行避免阻塞事件循环
        import asyncio

        def _search() -> dict[str, Any]:
            return client.search(
                query=query,
                max_results=max_results,
                search_depth="advanced",  # 深度搜索，获取更完整的结果
                include_answer=True,  # 包含 AI 摘要回答
                include_domains=None,
                exclude_domains=None,
            )

        result = await asyncio.get_event_loop().run_in_executor(None, _search)
    except Exception as exc:
        logger.warning("Tavily search failed: %s", exc)
        return "❌ 网络搜索失败，请稍后重试。"

    results = result.get("results", [])
    ai_answer = result.get("answer", "")

    if not results:
        return "❌ 未找到相关搜索结果，请尝试更换关键词。"

    lines: list[str] = [f"🔍 网络搜索结果（{query}）："]

    # 如果有 AI 摘要回答，优先展示
    if ai_answer:
        lines.append(f"\n📝 摘要：{ai_answer}\n")

    # 格式化每条结果
    for i, r in enumerate(results, 1):
        title = r.get("title", "").strip() or "(无标题)"
        content = r.get("content", "").strip()
        url = r.get("url", "").strip()
        score = r.get("score", None)

        lines.append(f"\n  {i}. **{title}**")
        if content:
            content_short = content[:300] + "…" if len(content) > 300 else content
            lines.append(f"     > {content_short}")
        if url:
            lines.append(f"     🔗 {url}")
        if score is not None:
            lines.append(f"     (相关度: {round(score * 100)}%)")

    return "\n".join(lines)


# ─── Tools ────────────────────────────────────────────────────


@tool
async def get_user_playlists() -> str:
    """获取当前登录用户的网易云歌单列表。当你需要了解用户有哪些歌单时使用此工具。"""
    client = await _get_client()
    try:
        resp = await client.get("/api/user/playlists?limit=60")
        data = resp.json()
        if not data.get("loggedIn"):
            return "用户未登录，无法获取歌单。"
        playlists = data.get("playlists", [])
        if not playlists:
            return "暂无歌单。"
        lines = [f"📋 共 {len(playlists)} 个歌单："]
        for pl in playlists:
            cover_info = f"[封面]{pl.get('cover', '')[:40]} " if pl.get("cover") else ""
            lines.append(
                f"- **{pl['name']}** (ID: {pl['id']}) "
                f"{cover_info}"
                f"· {pl.get('trackCount', 0)} 首 · 播放 {pl.get('playCount', 0)} 次"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"获取歌单失败：{e!s}"


@tool
async def get_playlist_tracks(playlist_id: str) -> str:
    """获取指定歌单的曲目列表。参数 playlist_id 是歌单的数字 ID。"""
    client = await _get_client()
    try:
        resp = await client.get(f"/api/playlist/tracks?id={playlist_id}")
        data = resp.json()
        playlist = data.get("playlist", {})
        tracks = data.get("tracks", [])
        if not tracks:
            return f"歌单「{playlist.get('name', '未知')}」暂无曲目。"
        lines = [
            f"📀 **{playlist.get('name', '未知歌单')}** — {len(tracks)} 首歌曲："
        ]
        for i, track in enumerate(tracks[:30], 1):
            artists = track.get("artist", track.get("artists", [{"name": "未知"}]))
            if isinstance(artists, list):
                artist_str = " / ".join(a.get("name", "未知") for a in artists)
            else:
                artist_str = str(artists)
            lines.append(f"  {i}. 《{track.get('name', '未知')}》- {artist_str}")
        if len(tracks) > 30:
            lines.append(f"  ... 还有 {len(tracks) - 30} 首")
        return "\n".join(lines)
    except Exception as e:
        return f"获取歌单曲目失败：{e!s}"


@tool
async def search_music(keyword: str) -> str:
    """搜索歌曲。参数 keyword 是搜索关键词（歌曲名或歌手名）。"""
    client = await _get_client()
    try:
        resp = await client.get(f"/api/search?keywords={keyword}&limit=10")
        data = resp.json()
        songs = data.get("songs", [])
        if not songs:
            return f"没有找到与「{keyword}」相关的歌曲。"
        lines = [f"🔍 搜索「{keyword}」结果："]
        for i, song in enumerate(songs[:10], 1):
            artist = song.get("artist", "未知")
            album = song.get("album", "")
            sname = song.get("name", "未知")
            album_info = f" [{album}]" if album else ""
            lines.append(f"  {i}. **《{sname}》** - {artist}{album_info}")
        return "\n".join(lines)
    except Exception as e:
        return f"搜索失败：{e!s}"


@tool
async def get_artist_info(artist_id: str) -> str:
    """获取歌手详情和热门歌曲。参数 artist_id 是歌手的数字 ID。"""
    client = await _get_client()
    try:
        resp = await client.get(f"/api/artist/detail?id={artist_id}")
        data = resp.json()
        artist = data.get("artist", {})
        songs = data.get("songs", [])
        if not artist.get("name"):
            return f"未找到 ID 为 {artist_id} 的歌手。"
        lines = [
            f"🎤 **{artist['name']}**"
        ]
        brief = artist.get("brief", "")
        if brief:
            lines.append(f"> {brief[:150]}")
        lines.append(f"歌曲数：{artist.get('musicSize', '?')} · 专辑数：{artist.get('albumSize', '?')}")
        if songs:
            lines.append(f"\n热门歌曲 TOP {min(len(songs), 10)}：")
            for i, song in enumerate(songs[:10], 1):
                lines.append(f"  {i}. 《{song.get('name', '未知')}》")
        return "\n".join(lines)
    except Exception as e:
        return f"获取歌手信息失败：{e!s}"


@tool
async def get_current_playback_context() -> str:
    """获取当前播放上下文（正在播放的歌曲、队列状态等）。了解用户正在听什么时使用此工具。

    优先从 LangGraph Runtime context（前端实时传入的播放状态）读取，
    如果不可用则回退到 HTTP 调用 /api/agent/context。
    """
    # 尝试从 runtime context 读取（由 main.py 通过 graph.astream(context=...) 注入）
    try:
        runtime: Runtime[Any] = get_runtime()
        ctx: dict[str, Any] = runtime.context or {}  # type: ignore[assignment]
        song = ctx.get("currentSong")
        if song:
            name = song.get("name", "未知")
            artist = song.get("artist", "未知")
            album = song.get("album", "")
            queue_count = ctx.get("queue", 0)
            playing = ctx.get("playing", False)
            status = "▶️ 播放中" if playing else "⏸️ 暂停"
            lines = [
                f"{status} **《{name}》** - {artist}",
                f"专辑：{album}" if album else "",
                f"队列中还有 {queue_count} 首待播",
            ]
            return "\n".join(filter(None, lines))
    except Exception:
        pass  # fallback to HTTP call below

    # Fallback: 通过 HTTP 调用 Node.js 服务端
    client = await _get_client()
    try:
        resp = await client.get("/api/agent/context")
        data = resp.json()
        song = data.get("currentSong")
        if not song:
            return "当前没有在播放任何歌曲。"
        name = song.get("name", "未知")
        artist = song.get("artist", "未知")
        album = song.get("album", "")
        queue_count = data.get("queue", 0)
        playing = data.get("playing", False)
        status = "▶️ 播放中" if playing else "⏸️ 暂停"
        lines = [
            f"{status} **《{name}》** - {artist}",
            f"专辑：{album}" if album else "",
            f"队列中还有 {queue_count} 首待播",
        ]
        return "\n".join(filter(None, lines))
    except Exception as e:
        return f"获取播放上下文失败：{e!s}"


@tool
async def analyze_playlist_mood(playlist_id: str) -> str:
    """分析指定歌单的情绪氛围（开心/伤感/安静/激昂/浪漫）。
    基于歌单中歌曲的名称和歌手名做关键词情绪分类。参数 playlist_id 是歌单的数字 ID。"""
    # 先获取歌单曲目
    client = await _get_client()
    try:
        resp = await client.get(f"/api/playlist/tracks?id={playlist_id}")
        data = resp.json()
        playlist = data.get("playlist", {})
        tracks = data.get("tracks", [])
        if not tracks:
            return f"歌单「{playlist.get('name', '未知')}」暂无曲目，无法分析情绪。"
        # 分析每首歌曲的情绪
        mood_counts: dict[str, int] = {}
        for track in tracks:
            text = f"{track.get('name', '')} {track.get('artist', '')}"
            moods = _classify_mood_by_text(text)
            for m in moods:
                mood_counts[m] = mood_counts.get(m, 0) + 1
        total = len(tracks)
        sorted_moods = sorted(mood_counts.items(), key=lambda x: -x[1])
        lines = [
            f"🎭 **{playlist.get('name', '未知歌单')}** — 情绪分析",
            f"共分析 {total} 首歌曲："
        ]
        for mood, count in sorted_moods:
            pct = round(count / total * 100)
            bar = "█" * (pct // 10) + "░" * (10 - pct // 10) if total > 0 else ""
            lines.append(f"  {mood}：{bar} {pct}% ({count} 首)")
        # 主要情绪
        if sorted_moods:
            main_mood = sorted_moods[0][0]
            mood_emojis = {"开心": "😊", "伤感": "😢", "安静": "🧘", "激昂": "🔥", "浪漫": "💕", "未知": "🎵"}
            emoji = mood_emojis.get(main_mood, "🎵")
            lines.append(f"\n**主要情绪**：{emoji} {main_mood}")
        return "\n".join(lines)
    except Exception as e:
        return f"分析歌单情绪失败：{e!s}"


@tool
async def get_existing_playlist_features(playlist_id: str) -> str:
    """深度分析指定歌单的"听觉基因"：情绪分布、语种分布、歌手多样性、高频标签、节奏特征等。

    基于歌单曲目数据提取多维特征向量，用于个性化推荐中的"内部审美向量"计算。
    参数 playlist_id 是歌单的数字 ID。"""
    client = await _get_client()
    try:
        resp = await client.get(f"/api/playlist/tracks?id={playlist_id}")
        data = resp.json()
        playlist = data.get("playlist", {})
        tracks = data.get("tracks", [])
        if not tracks:
            return f"歌单「{playlist.get('name', '未知')}」暂无曲目。"
        pl_name = playlist.get('name', '未知歌单')

        # --- 基础统计 ---
        total = len(tracks)
        unique_artists: set[str] = set()
        for t in tracks:
            artist = t.get("artist", "") or ""
            if artist:
                unique_artists.add(artist)
        artist_count = len(unique_artists)
        diversity = round(artist_count / total, 2) if total else 0

        # --- 情绪分布 (复用 _classify_mood_by_text) ---
        mood_counts: dict[str, int] = {}
        for t in tracks:
            text = f"{t.get('name', '')} {t.get('artist', '')}"
            moods = _classify_mood_by_text(text)
            for m in moods:
                mood_counts[m] = mood_counts.get(m, 0) + 1
        sorted_moods = sorted(mood_counts.items(), key=lambda x: -x[1])

        # --- 语种分布 ---
        import re
        lang_counts: dict[str, int] = {"中文": 0, "英文": 0, "日文": 0, "韩文": 0, "其他": 0}
        for t in tracks:
            name = t.get('name', '')
            cjk = len(re.findall(r'[\u4e00-\u9fff]', name))
            latin = len(re.findall(r'[a-zA-Z]', name))
            jp = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', name))
            kr = len(re.findall(r'[\uac00-\ud7af]', name))
            if jp > latin and jp > cjk:
                lang_counts["日文"] += 1
            elif kr > latin and kr > cjk:
                lang_counts["韩文"] += 1
            elif cjk > latin:
                lang_counts["中文"] += 1
            elif latin > 0:
                lang_counts["英文"] += 1
            else:
                lang_counts["其他"] += 1

        # --- 高频标签 (从歌名提取关键词) ---
        stop_words = {"的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一", "个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "那", "些", "什么", "怎么", "如何"}
        word_counts: dict[str, int] = {}
        for t in tracks:
            name = str(t.get('name', ''))
            for w in re.findall(r'[\u4e00-\u9fff\w]+', name):
                if w.lower() not in stop_words and len(w) > 1:
                    word_counts[w] = word_counts.get(w, 0) + 1
        top_tags = sorted(word_counts.items(), key=lambda x: -x[1])[:6]

        # --- 节奏风格提示 ---
        tempo_keywords = {
            "快": ["快", "急速", "加速", "奔跑", "冲刺"],
            "慢": ["慢", "缓", "轻柔", "舒缓", "摇篮"],
            "燃": ["燃", "热血", "战斗", "激昂", "摇滚", "Metal", "Rock"],
            "电子": ["电子", "电音", "Dubstep", "EDM", "Techno"],
            " acoustic": ["吉他", "钢琴", "acoustic", "不插电", "纯音乐", "钢琴曲"],
        }
        tempo_hints: list[str] = []
        for t in tracks:
            text = f"{t.get('name', '')} {t.get('artist', '')}"
            for category, kws in tempo_keywords.items():
                for kw in kws:
                    if kw.lower() in text.lower():
                        if category not in tempo_hints:
                            tempo_hints.append(category)
                        break

        # --- 平均时长 ---
        total_duration_ms = 0
        duration_count = 0
        for t in tracks:
            dur = t.get('duration', 0) or 0
            if dur > 0:
                total_duration_ms += dur
                duration_count += 1
        if duration_count > 0:
            avg_sec = (total_duration_ms // duration_count) // 1000
            avg_min = avg_sec // 60
            avg_sec_remain = avg_sec % 60
            avg_dur = f"{avg_min:02d}:{avg_sec_remain:02d}"
        else:
            avg_dur = "未知"

        # --- 格式化输出 ---
        lines = [
            f"🎵 歌单「{pl_name}」听觉基因分析",
            "━━━━━━━━━━━━━━━━━━",
            f"📊 **基础数据**：{total} 首 · {artist_count} 位歌手 · 多样性 {diversity}",
        ]
        # 情绪
        mood_line = " · ".join(f"{m} {round(c/total*100)}%" for m, c in sorted_moods[:4])
        lines.append(f"🎭 **情绪分布**：{mood_line}")
        # 语种
        active_langs = [f"{k} {v}" for k, v in lang_counts.items() if v > 0]
        lines.append(f"🌐 **语种分布**：{' · '.join(active_langs)}")
        # 高频标签
        if top_tags:
            tags_str = " · ".join(f"#{w}" for w, _ in top_tags)
            lines.append(f"🏷️ **高频标签**：{tags_str}")
        # 节奏提示
        if tempo_hints:
            lines.append(f"🥁 **节奏特征**：{' / '.join(tempo_hints)}")
        # 时长
        lines.append(f"⏱ **平均曲长**：{avg_dur}")

        # 主要情绪结论
        if sorted_moods:
            main_mood = sorted_moods[0][0]
            mood_emojis = {"开心": "😊", "伤感": "😢", "安静": "🧘", "激昂": "🔥", "浪漫": "💕", "未知": "🎵"}
            lines.append(f"\n**核心审美**：{mood_emojis.get(main_mood, '🎵')} 以「{main_mood}」为主基调，"
                        f"听众偏好{'/'.join(tempo_hints[:2]) if tempo_hints else '多元'}风格")

        return "\n".join(lines)
    except Exception as e:
        return f"分析歌单特征失败：{e!s}"


@tool
async def play_next_song() -> str:
    """切换到下一首歌曲。用户说「下一首」「切歌」「换一首」时使用此工具。"""
    client = await _get_client()
    try:
        resp = await client.post("/api/agent/playback/next")
        data = resp.json()
        if data.get("ok"):
            return "✅ 已切换到下一首。"
        return f"切歌失败：{data.get('error', '未知错误')}"
    except Exception as e:
        return f"切歌失败：{e!s}"


@tool
async def play_previous_song() -> str:
    """切换到上一首歌曲。用户说「上一首」「回到上一首」时使用此工具。"""
    client = await _get_client()
    try:
        resp = await client.post("/api/agent/playback/prev")
        data = resp.json()
        if data.get("ok"):
            return "✅ 已切换到上一首。"
        return f"切换失败：{data.get('error', '未知错误')}"
    except Exception as e:
        return f"切换失败：{e!s}"


@tool
async def toggle_playback() -> str:
    """切换播放/暂停状态。用户说「暂停」「继续播放」「停一下」时使用此工具。"""
    client = await _get_client()
    try:
        resp = await client.post("/api/agent/playback/toggle-play")
        data = resp.json()
        if data.get("ok"):
            return "✅ 已切换播放状态。"
        return f"操作失败：{data.get('error', '未知错误')}"
    except Exception as e:
        return f"操作失败：{e!s}"


@tool
async def play_song(song_name: str, artist: str = "") -> str:
    """搜索并播放指定歌曲。用户说「播放《xxx》」「我想听xxx」「放一首xxx」时使用此工具。

    注意：对需要精确匹配的场景（如歌名+歌手），请同时提供 song_name 和 artist。

    Args:
        song_name: 歌曲名称，如「晴天」「不能说的秘密」
        artist: 歌手名称（可选），如「周杰伦」
    """
    keyword = f"{song_name} {artist}".strip()
    client = await _get_client()
    try:
        # 1. 搜索歌曲
        resp = await client.get(f"/api/search?keywords={keyword}&limit=5")
        data = resp.json()
        songs = data.get("songs", [])
        if not songs:
            return f"🔍 没有找到「{keyword}」相关的歌曲，试试换个关键词？"

        # 2. 取第一个搜索结果
        song = songs[0]
        song_id = song.get("id")
        song_name_found = song.get("name", "未知")
        song_artist = song.get("artist", "")

        # 3. 播放权限预检：先调用 /api/song/url 检查歌曲是否可播
        try:
            url_check = await client.get(f"/api/song/url?id={song_id}&quality=standard")
            url_data = url_check.json()
        except Exception:
            url_data = {}

        playable = url_data.get("playable", False)
        trial = url_data.get("trial", False)
        restriction = url_data.get("restriction", {})
        reason = url_data.get("reason", "") or restriction.get("category", "")
        message = url_data.get("message", "") or restriction.get("message", "")
        fee = url_data.get("fee") or restriction.get("fee")
        logged_in = url_data.get("loggedIn", False)

        # 判断是否真的可播（有完整 URL 且不是试听片段）
        # 优先级顺序对齐 server.js classifyNeteasePlaybackRestriction:
        #   login_required → trial_only → vip_required → paid_required → copyright → url_unavailable
        if not playable or trial:
            # ① 未登录
            if not logged_in or reason == "login_required":
                return (
                    f"⚠️ 无法播放《{song_name_found}》"
                    + (f" - {song_artist}" if song_artist else "")
                    + "：当前网易云账号未登录，需要登录后才能播放完整歌曲。"
                    "请在客户端中登录网易云账号后再试。"
                )
            # ② 试听片段（仅返回片段，无完整播放权限）
            if trial or reason == "trial_only":
                return (
                    f"⚠️ 无法完整播放《{song_name_found}》"
                    + (f" - {song_artist}" if song_artist else "")
                    + "：网易云仅返回了试听片段，完整播放需要会员或购买。"
                )
            # ③ VIP 会员歌曲
            if reason == "vip_required" or fee == 1:
                return (
                    f"⚠️ 无法播放《{song_name_found}》"
                    + (f" - {song_artist}" if song_artist else "")
                    + "：这是一首 VIP 付费歌曲，当前账号没有 VIP 权限，"
                    "无法获取完整播放地址。"
                )
            # ④ 付费购买歌曲（单曲/专辑）
            if reason in ("paid_required",) or fee in (4, 8):
                return (
                    f"⚠️ 无法播放《{song_name_found}》"
                    + (f" - {song_artist}" if song_artist else "")
                    + "：这是一首需要单独购买（单曲/专辑）的歌曲，"
                    "当前权限不足。"
                )
            # ⑤ 版权不可用 / 无播放地址
            if reason in ("copyright_unavailable", "url_unavailable") or not url_data.get("url"):
                return (
                    f"⚠️ 无法播放《{song_name_found}》"
                    + (f" - {song_artist}" if song_artist else "")
                    + "：该歌曲版权暂不可用，网易云没有返回可播放的音频地址。"
                )
            # ⑥ 兜底：有自定义消息
            if message:
                return (
                    f"⚠️ 无法播放《{song_name_found}》"
                    + (f" - {song_artist}" if song_artist else "")
                    + f"：{message}"
                )
            # ⑦ 兜底：未知原因
            return (
                f"⚠️ 无法播放《{song_name_found}》"
                + (f" - {song_artist}" if song_artist else "")
                + "：此歌曲暂时没有可播放的音频源。"
            )

        # 4. 权限预检通过 → 通过 Node.js 服务端发送到渲染进程播放
        play_resp = await client.post("/api/agent/playback/play", json={"song": song})
        play_data = play_resp.json()
        if play_data.get("ok"):
            artist_display = f" - {song_artist}" if song_artist else ""
            play_msg = f"🎵 正在播放：《{song_name_found}》{artist_display}"

            # 5. 自动联网搜索歌曲相关信息（创作背景、乐评、歌手介绍等）
            try:
                search_query = f"{song_name_found} {song_artist} 歌曲 介绍 乐评 创作背景"
                web_info = await _search_web(search_query, max_results=3)
                if web_info and "❌" not in web_info:
                    lines = [play_msg, "", "🌐 **相关信息**"]
                    for line in web_info.split("\n"):
                        stripped = line.strip()
                        if stripped and not stripped.startswith("🔍"):
                            lines.append(f"  {stripped}")
                    return "\n".join(lines)
            except Exception:
                pass  # 信息搜索失败不影响播放，仅返回播放成功消息

            return play_msg
        return f"播放失败：{play_data.get('error', '未知错误')}"
    except Exception as e:
        return f"播放失败：{e!s}"


@tool
async def get_song_detail(song_id: str) -> str:
    """获取歌曲的详细信息，包括专辑、发行时间、歌词摘要、评论数等。

    当用户问「这首歌的详细信息」「这是哪张专辑」「什么时候发行的」
    「这首歌的歌词是什么」「这首歌有多少评论」等问题时使用此工具。

    Args:
        song_id: 歌曲的数字 ID
    """
    client = await _get_client()
    try:
        # 1. 获取歌曲详情
        resp = await client.get(f"/api/song/detail?id={song_id}")
        data = resp.json()
        song = data.get("name") and data or {}
        if not song.get("name"):
            return f"未找到 ID 为 {song_id} 的歌曲。"

        name = song.get("name", "未知")
        artists_list = song.get("artists", [])
        artist_str = " / ".join(a.get("name", "") for a in artists_list) if artists_list else "未知"
        album = song.get("album", {}) or {}
        album_name = album.get("name", "未知专辑")
        album_pic = album.get("picUrl", "")
        duration_ms = song.get("duration", 0) or 0
        minutes = duration_ms // 60000
        seconds = (duration_ms % 60000) // 1000
        duration_str = f"{minutes:02d}:{seconds:02d}"
        fee = song.get("fee", 0)
        fee_map = {0: "免费", 1: "VIP", 4: "付费专辑", 8: "付费单曲"}
        fee_str = fee_map.get(fee, "未知")

        lines = [
            f"📀 **《{name}》** - {artist_str}",
            f"",
            f"📝 **基本信息**",
            f"  • 专辑：《{album_name}》",
            f"  • 时长：{duration_str}",
            f"  • 权限：{fee_str}",
        ]
        if album_pic:
            lines[0] = f"📀 **《{name}》** - {artist_str}  [封面]({album_pic})"

        # 2. 获取歌词
        try:
            lyric_resp = await client.get(f"/api/lyric?id={song_id}")
            lyric_data = lyric_resp.json()
            lyric_text = lyric_data.get("lyric", "")
            if lyric_text:
                lyric_lines = lyric_text.strip().split("\n")[:6]
                # 过滤时间戳行，只保留纯歌词
                import re
                clean_lines = []
                for l in lyric_lines:
                    clean = re.sub(r'\[\d{2}:\d{2}(?:\.\d{2,3})?\]', '', l).strip()
                    if clean:
                        clean_lines.append(clean)
                if clean_lines:
                    lines.append(f"")
                    lines.append(f"📄 **歌词摘要**")
                    for cl in clean_lines[:4]:
                        lines.append(f"  {cl}")
        except Exception:
            pass

        # 3. 获取评论数
        try:
            comment_resp = await client.get(f"/api/song/comments?id={song_id}&limit=1")
            comment_data = comment_resp.json()
            total = comment_data.get("total", 0)
            if total:
                if total > 10000:
                    total_str = f"{total / 10000:.1f}万+"
                else:
                    total_str = str(total)
                lines.append(f"")
                lines.append(f"💬 **数据**")
                lines.append(f"  • 评论数：{total_str}")
        except Exception:
            pass

        return "\n".join(lines)

    except Exception as e:
        return f"获取歌曲详情失败：{e!s}"


@tool
async def get_personalized_recommendations(limit: int = 10) -> str:
    """获取网易云根据你的听歌历史生成的个性化推荐歌曲。

    基于你的播放记录和偏好，从网易云获取每日推荐歌曲。
    适合场景：用户说「推荐一些歌」「今天听什么」「有什么好听的」

    Args:
        limit: 推荐数量（1-20），默认10首
    """
    limit = max(1, min(20, limit))
    client = await _get_client()
    try:
        resp = await client.get("/api/recommend/songs")
        data = resp.json()
        if not data.get("loggedIn"):
            return "用户未登录，无法获取个性化推荐。请先在客户端登录网易云账号。"

        songs = data.get("recommend", [])
        if not songs:
            return "暂无个性化推荐，多听一些歌后网易云会为你生成推荐。"

        lines = [
            f"🎯 **今日推荐**（共 {len(songs)} 首）：",
            f"根据你的听歌偏好生成的个性化推荐：",
        ]
        for i, song in enumerate(songs[:limit], 1):
            sname = song.get("name", "未知")
            sartist = song.get("artist", "")
            album_info = f" [{song.get('album', '')}]" if song.get("album") else ""
            lines.append(f"  {i}. **《{sname}》** - {sartist}{album_info}")

        if len(songs) > limit:
            lines.append(f"  ... 还有 {len(songs) - limit} 首")

        lines.append(f"")
        lines.append(f("💡 试试说「播放第X首」来听具体的歌曲"))

        return "\n".join(lines)
    except Exception as e:
        return f"获取推荐失败：{e!s}"


@tool
async def recommend_from_current_mood(mood: str = "") -> str:
    """根据当前情绪/场景推荐音乐（结合联网搜索+网易云歌单）。

    使用 Tavily 搜索该场景的经典推荐，适合情绪化推荐场景。
    如果不提供 mood 参数，会自动从当前播放上下文推断场景。

    Args:
        mood: 情绪/场景关键词，如「开心」「伤感」「安静」「激昂」「浪漫」
              或具体场景如「下雨」「工作」「学习」「运动」「睡前」「旅行」
    """
    # 如果没有提供 mood，尝试从播放上下文推断
    if not mood:
        try:
            runtime: Runtime[Any] = get_runtime()
            ctx: dict[str, Any] = runtime.context or {}
            weather = ctx.get("weather", "")
            current_song = ctx.get("currentSong", None) or {}
            if weather:
                mood = f"{weather}天气"
            elif current_song.get("name"):
                song_name = current_song.get("name", "")
                artist = current_song.get("artist", "")
                mood = f"类似《{song_name}》- {artist} 的风格"
            else:
                mood = "放松"
        except Exception:
            mood = "放松"

    # 先用 Tavily 搜索场景推荐
    search_query = f"{mood} 适合听的歌 音乐推荐 歌单"
    search_result = await _search_web(search_query, max_results=5)

    # 再用网易云搜索相关歌曲
    client = await _get_client()
    try:
        search_resp = await client.get(f"/api/search?keywords={mood} 歌曲&limit=5")
        search_data = search_resp.json()
        songs = search_data.get("songs", [])
    except Exception:
        songs = []

    lines = [f"🎵 **{mood} 音乐推荐**\n"]

    # 展示联网搜索结果
    if search_result and "❌" not in search_result:
        lines.append("🌐 **网络推荐**")
        # Extract just the title/snippet parts, skip the header line
        for line in search_result.split("\n"):
            if line.strip() and not line.startswith("🔍"):
                lines.append(f"  {line}")
        lines.append("")

    # 展示网易云搜索结果
    if songs:
        lines.append(f"📀 **网易云相关歌曲**")
        for i, song in enumerate(songs[:5], 1):
            sname = song.get("name", "未知")
            sartist = song.get("artist", "")
            lines.append(f"  {i}. **《{sname}》** - {sartist}")
        lines.append("")
        lines.append("💡 试试说「播放第X首」来听歌")

    return "\n".join(lines) if len(lines) > 2 else "暂无相关推荐，试试其他关键词。"


@tool
async def web_search(query: str, max_results: int = 5) -> str:
    """搜索互联网获取最新音乐资讯、乐评推荐、歌手动态等信息。

    基于 Tavily AI Search 引擎，专为 AI Agent 优化，会返回 AI 生成的摘要总结。
    相比传统搜索引擎，Tavily 返回的结果更精准、相关性更高。

    当以下场景时优先使用此工具：
    - 用户请求「推荐类似风格的歌曲」时，结合搜索结果的风格标签来推荐
    - 用户询问「最近有什么好听的歌」「有什么新歌推荐」
    - 需要了解歌曲的创作背景、歌手最新动态
    - 需要基于实时信息（天气、节日、热点）推荐场景音乐
    - 用户的问题超出网易云 API 能提供的范围

    使用技巧：
    - 中文关键词优先，可组合「歌曲」「推荐」「风格」等词
    - 例如搜索「周杰伦 晴天 相似风格 推荐」
    - 例如搜索「2025年 热门 华语 新歌 榜单」
    - 例如搜索「下雨天 适合听的歌 推荐」
    - 例如搜索「Taylor Swift similar artists recommend」

    Args:
        query: 搜索关键词，建议包含音乐相关词汇以获得更精准结果
        max_results: 返回结果数量（1-10），默认5条
    """
    max_results = max(1, min(10, max_results))
    return await _search_web(query, max_results=max_results)


@tool
async def batch_play_songs(songs_json: str) -> str:
    """批量搜索并播放歌曲列表。生成歌单后，用户同意播放时使用此工具。

    接受 JSON 格式的歌曲列表，依次搜索、校验权限、入队，然后从第一首开始播放。

    Args:
        songs_json: JSON 字符串，格式为 [{"name": "歌曲名1", "artist": "歌手1"}, {"name": "歌曲名2", "artist": "歌手2"}, ...]
    """
    import json as _json
    try:
        songs_list = _json.loads(songs_json)
    except Exception:
        return "❌ 歌曲列表格式错误，请提供有效的 JSON 数组。"

    if not songs_list or not isinstance(songs_list, list):
        return "❌ 歌曲列表为空。"
    if len(songs_list) > 30:
        songs_list = songs_list[:30]

    client = await _get_client()
    queued: list[dict] = []
    failed: list[str] = []

    for i, s in enumerate(songs_list):
        song_name = str(s.get("name", "")).strip()
        artist = str(s.get("artist", "")).strip()
        if not song_name:
            failed.append(f"第{i+1}首（缺少歌名）")
            continue
        keyword = f"{song_name} {artist}".strip()
        try:
            resp = await client.get(f"/api/search?keywords={keyword}&limit=3")
            data = resp.json()
            songs = data.get("songs", [])
            if not songs:
                failed.append(f"《{song_name}》")
                continue
            best = songs[0]
            # 权限预检
            song_id = best.get("id")
            try:
                url_check = await client.get(f"/api/song/url?id={song_id}&quality=standard")
                url_data = url_check.json()
            except Exception:
                url_data = {}
            playable = url_data.get("playable", False)
            trial = url_data.get("trial", False)
            if not playable or trial:
                reason_str = ""
                if not url_data.get("loggedIn", False) or url_data.get("reason") == "login_required":
                    reason_str = "（未登录）"
                elif trial or url_data.get("reason") == "trial_only":
                    reason_str = "（仅试听）"
                elif url_data.get("reason") in ("vip_required",) or url_data.get("fee") == 1:
                    reason_str = "（VIP）"
                else:
                    reason_str = "（不可播）"
                failed.append(f"《{song_name}》{reason_str}")
                continue
            queued.append({
                "id": song_id,
                "name": best.get("name", song_name),
                "artist": best.get("artist", artist),
                "cover": best.get("cover", ""),
            })
        except Exception as e:
            failed.append(f"《{song_name}》({e!s})")
            continue

    if not queued:
        return "❌ 未能找到任何可播放的歌曲。" + (f" 失败：{', '.join(failed[:5])}" if failed else "")

    # 调用批量入队接口
    try:
        play_resp = await client.post("/api/agent/playback/queue", json={"songs": queued, "startIndex": 0})
        play_data = play_resp.json()
        if not play_data.get("ok"):
            return f"❌ 批量入队失败：{play_data.get('error', '未知错误')}"
    except Exception as e:
        return f"❌ 批量入队请求失败：{e!s}"

    first = queued[0]
    result = f"✅ 已队列 {len(queued)} 首歌曲"
    if failed:
        result += f"，{len(failed)} 首无法播放"
    result += f"，从《{first.get('name', '未知')}」- {first.get('artist', '')}》开始播放。"
    return result


# Export all tools as a list
ALL_TOOLS = [
    get_user_playlists,
    get_playlist_tracks,
    search_music,
    get_artist_info,
    get_current_playback_context,
    analyze_playlist_mood,
    get_existing_playlist_features,
    play_next_song,
    play_previous_song,
    toggle_playback,
    play_song,
    batch_play_songs,
    web_search,
    get_song_detail,
    get_personalized_recommendations,
    recommend_from_current_mood,
]
