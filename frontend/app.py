from __future__ import annotations

import hashlib
from html import escape
import json
from urllib.parse import quote
import uuid

import httpx
import streamlit as st
import streamlit.components.v1 as components

from core.settings import get_settings


settings = get_settings()
BACKEND_URL = settings.backend_url.rstrip("/")
ASSISTANT_AVATAR = ":material/auto_awesome:"

st.set_page_config(page_title="BuddyAi", layout="wide")
st.markdown(
    """
    <style>
    :root {
        --nexus-accent: #7c3aed;
        --nexus-accent-2: #0d9488;
        --nexus-border: rgba(148, 163, 184, 0.24);
        --nexus-border: color-mix(in srgb, var(--text-color) 14%, transparent);
        --nexus-control: rgba(148, 163, 184, 0.10);
        --nexus-control: color-mix(in srgb, var(--secondary-background-color) 88%, var(--primary-color) 12%);
        --nexus-hover: rgba(148, 163, 184, 0.16);
        --nexus-hover: color-mix(in srgb, var(--secondary-background-color) 72%, var(--primary-color) 28%);
        --nexus-active: rgba(124, 58, 237, 0.16);
        --nexus-active: color-mix(in srgb, var(--primary-color) 18%, var(--secondary-background-color) 82%);
        --nexus-muted: rgba(100, 116, 139, 0.95);
        --nexus-muted: color-mix(in srgb, var(--text-color) 62%, transparent);
        --nexus-panel: color-mix(in srgb, var(--secondary-background-color) 90%, var(--background-color) 10%);
        --nexus-shadow: 0 18px 48px rgba(15, 23, 42, 0.12);
    }
    .stApp {
        background:
            linear-gradient(135deg, color-mix(in srgb, var(--primary-color) 8%, transparent), transparent 32%),
            linear-gradient(180deg, var(--background-color), color-mix(in srgb, var(--background-color) 90%, var(--secondary-background-color) 10%));
    }
    .block-container {
        max-width: 1160px;
        padding-top: 2.75rem;
    }
    .app-header {
        align-items: center;
        background:
            linear-gradient(135deg, color-mix(in srgb, var(--nexus-accent) 12%, var(--nexus-panel)), color-mix(in srgb, var(--nexus-accent-2) 9%, var(--nexus-panel)));
        border: 1px solid var(--nexus-border);
        border-radius: 8px;
        box-shadow: var(--nexus-shadow);
        display: flex;
        gap: 0.85rem;
        margin: 0 0 1.35rem;
        min-height: 4.25rem;
        overflow: visible;
        padding: 0.75rem 1rem;
        width: 100%;
    }
    .app-logo {
        align-items: center;
        background: linear-gradient(135deg, var(--nexus-accent), var(--nexus-accent-2));
        border-radius: 8px;
        box-shadow: var(--nexus-shadow);
        color: #ffffff;
        display: flex;
        flex: 0 0 2.35rem;
        font-size: 1.15rem;
        height: 2.35rem;
        justify-content: center;
        line-height: 1;
        overflow: visible;
        width: 2.35rem;
    }
    .app-title {
        color: var(--text-color);
        flex: 1 1 auto;
        font-size: 1.7rem;
        font-weight: 760;
        line-height: 1.45;
        min-width: 0;
        overflow: visible;
        padding: 0.1rem 0;
        white-space: normal;
        word-break: keep-all;
    }
    div[data-testid="stChatMessage"] {
        border-radius: 8px;
        margin: 0.42rem 0;
        padding: 0.7rem 0.85rem;
        transition: background 120ms ease, border-color 120ms ease, transform 120ms ease;
    }
    div[data-testid="stChatMessage"]:hover {
        transform: translateY(-1px);
    }
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: color-mix(in srgb, var(--secondary-background-color) 78%, var(--background-color) 22%);
        border: 1px solid var(--nexus-border);
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.07);
    }
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: color-mix(in srgb, var(--primary-color) 14%, var(--secondary-background-color) 86%);
        border: 1px solid color-mix(in srgb, var(--primary-color) 24%, var(--nexus-border));
    }
    div[data-testid="chatAvatarIcon-assistant"] {
        background: linear-gradient(135deg, var(--nexus-accent), var(--nexus-accent-2)) !important;
        color: #ffffff !important;
    }
    div[data-testid="chatAvatarIcon-user"] {
        box-shadow: 0 10px 24px rgba(239, 68, 68, 0.20);
    }
    div[data-testid="stChatInput"] textarea {
        border-radius: 8px;
        border: 1px solid var(--nexus-border);
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
    }
    div[data-testid="stChatInput"] textarea:focus {
        border-color: color-mix(in srgb, var(--primary-color) 48%, var(--nexus-border));
        box-shadow: 0 0 0 3px color-mix(in srgb, var(--primary-color) 15%, transparent);
    }
    div[data-testid="stExpander"] {
        border: 1px solid var(--nexus-border);
        border-radius: 8px;
        overflow: hidden;
    }
    .weather-card {
        background:
            linear-gradient(135deg, color-mix(in srgb, #f59e0b 12%, transparent), transparent 34%),
            linear-gradient(145deg, color-mix(in srgb, var(--secondary-background-color) 86%, #0ea5e9 14%), color-mix(in srgb, var(--background-color) 72%, var(--secondary-background-color) 28%));
        border: 1px solid color-mix(in srgb, var(--nexus-border) 74%, #ffffff 26%);
        border-radius: 8px;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.16);
        margin: 0 0 0.8rem;
        overflow: hidden;
        padding: 1rem;
    }
    .weather-top {
        align-items: flex-start;
        display: flex;
        gap: 1rem;
        justify-content: space-between;
    }
    .weather-place {
        color: var(--text-color);
        font-size: 1rem;
        font-weight: 760;
        line-height: 1.25;
        margin-bottom: 0.4rem;
        overflow-wrap: anywhere;
    }
    .weather-temp-row {
        align-items: baseline;
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
    }
    .weather-temp {
        color: var(--text-color);
        font-size: 3rem;
        font-weight: 780;
        letter-spacing: 0;
        line-height: 1;
    }
    .weather-unit,
    .weather-muted {
        color: var(--nexus-muted);
        font-size: 0.86rem;
        font-weight: 650;
    }
    .weather-desc {
        color: var(--text-color);
        font-size: 1rem;
        line-height: 1.35;
        margin-top: 0.55rem;
        overflow-wrap: anywhere;
    }
    .weather-hero-icon {
        align-items: center;
        background: color-mix(in srgb, var(--background-color) 48%, transparent);
        border: 1px solid var(--nexus-border);
        border-radius: 8px;
        display: flex;
        flex: 0 0 3.4rem;
        font-size: 2rem;
        height: 3.4rem;
        justify-content: center;
        width: 3.4rem;
    }
    .weather-stats {
        display: grid;
        gap: 0.5rem;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        margin-top: 0.9rem;
    }
    .weather-stat {
        background: color-mix(in srgb, var(--background-color) 44%, transparent);
        border: 1px solid var(--nexus-border);
        border-radius: 8px;
        min-width: 0;
        padding: 0.55rem 0.65rem;
        transition: transform 120ms ease, border-color 120ms ease, background 120ms ease;
    }
    .weather-stat:hover,
    .forecast-day:hover {
        background: color-mix(in srgb, var(--primary-color) 14%, var(--secondary-background-color) 86%);
        border-color: color-mix(in srgb, var(--primary-color) 42%, var(--nexus-border));
        transform: translateY(-1px);
    }
    .weather-stat-label {
        color: var(--nexus-muted);
        font-size: 0.72rem;
        font-weight: 720;
        line-height: 1.1;
        margin-bottom: 0.22rem;
        text-transform: uppercase;
    }
    .weather-stat-value {
        color: var(--text-color);
        font-size: 0.95rem;
        font-weight: 760;
        line-height: 1.2;
        overflow-wrap: anywhere;
    }
    .forecast-strip {
        display: grid;
        gap: 0.45rem;
        grid-template-columns: repeat(6, minmax(4.6rem, 1fr));
        margin-top: 1rem;
        overflow-x: auto;
        padding-bottom: 0.15rem;
    }
    .forecast-day {
        background: color-mix(in srgb, var(--background-color) 36%, transparent);
        border: 1px solid var(--nexus-border);
        border-radius: 8px;
        min-width: 4.6rem;
        padding: 0.62rem 0.45rem;
        text-align: center;
        transition: transform 120ms ease, border-color 120ms ease, background 120ms ease;
    }
    .forecast-day-name {
        color: var(--text-color);
        font-size: 0.82rem;
        font-weight: 760;
        line-height: 1.1;
    }
    .forecast-icon {
        font-size: 1.35rem;
        line-height: 1.35;
        margin: 0.2rem 0;
    }
    .forecast-high {
        color: var(--text-color);
        font-size: 0.95rem;
        font-weight: 780;
        line-height: 1.15;
    }
    .forecast-low {
        color: var(--nexus-muted);
        font-size: 0.82rem;
        font-weight: 640;
        line-height: 1.15;
    }
    .weather-chart {
        background: color-mix(in srgb, var(--background-color) 34%, transparent);
        border: 1px solid var(--nexus-border);
        border-radius: 8px;
        margin-top: 0.8rem;
        overflow: hidden;
        padding: 0.7rem 0.65rem 0.55rem;
    }
    .weather-chart-title {
        color: var(--text-color);
        font-size: 0.85rem;
        font-weight: 760;
        margin-bottom: 0.35rem;
    }
    .weather-chart svg {
        display: block;
        height: 5.1rem;
        overflow: visible;
        width: 100%;
    }
    .weather-hour-labels {
        display: grid;
        gap: 0.2rem;
        grid-template-columns: repeat(8, minmax(2.2rem, 1fr));
        margin-top: 0.2rem;
    }
    .weather-hour-label {
        color: var(--nexus-muted);
        font-size: 0.72rem;
        font-weight: 650;
        overflow: hidden;
        text-align: center;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    section[data-testid="stSidebar"] {
        background: color-mix(in srgb, var(--secondary-background-color) 90%, var(--background-color) 10%);
        border-right: 1px solid var(--nexus-border);
    }
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        color: var(--text-color);
    }
    .generating-row {
        align-items: center;
        color: var(--nexus-muted);
        display: flex;
        font-size: 0.95rem;
        gap: 0.5rem;
        min-height: 2rem;
    }
    .typing-spinner {
        border: 2px solid var(--nexus-border);
        border-right-color: var(--nexus-accent-2);
        border-top-color: var(--nexus-accent);
        border-radius: 999px;
        display: inline-block;
        height: 1rem;
        width: 1rem;
        animation: nexus-spin 0.7s linear infinite;
    }
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
        gap: 0.55rem;
    }
    section[data-testid="stSidebar"] div[data-testid="stButton"] button {
        background: var(--nexus-control);
        border: 1px solid var(--nexus-border);
        border-radius: 8px;
        color: var(--text-color);
        font-weight: 650;
        min-height: 2.35rem;
    }
    section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover {
        background: var(--nexus-hover);
        border-color: color-mix(in srgb, var(--primary-color) 45%, var(--nexus-border));
        color: var(--text-color);
    }
    section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"] {
        background: linear-gradient(135deg, var(--nexus-accent), var(--nexus-accent-2));
        border-color: transparent;
        color: #ffffff;
    }
    .chat-history-title {
        color: var(--text-color);
        font-size: 1.35rem;
        font-weight: 780;
        line-height: 1.2;
        margin: 0.05rem 0 0.65rem;
    }
    .chat-row-wrap {
        align-items: center;
        border: 1px solid transparent;
        border-radius: 8px;
        display: flex;
        gap: 0.2rem;
        height: 2.12rem;
        margin: 0 0 0.15rem;
        padding: 0 0.2rem 0 0.65rem;
        transition: background 120ms ease, border-color 120ms ease;
        width: 100%;
    }
    .chat-row-wrap:hover {
        background: var(--nexus-hover);
        border-color: var(--nexus-border);
    }
    .chat-row-wrap.active {
        background: var(--nexus-active);
        border-color: color-mix(in srgb, var(--primary-color) 42%, var(--nexus-border));
    }
    .chat-row-link {
        color: var(--text-color) !important;
        flex: 1 1 auto;
        font-size: 0.86rem;
        font-weight: 520;
        line-height: 2.12rem;
        min-width: 0;
        overflow: hidden;
        text-decoration: none !important;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .chat-row-wrap.active .chat-row-link {
        font-weight: 720;
    }
    .chat-delete-link {
        align-items: center;
        border-radius: 6px;
        color: var(--nexus-muted) !important;
        display: flex;
        flex: 0 0 1.55rem;
        font-size: 1.05rem;
        height: 1.55rem;
        justify-content: center;
        opacity: 0;
        text-decoration: none !important;
        transition: background 120ms ease, color 120ms ease, opacity 120ms ease;
        width: 1.55rem;
    }
    .chat-row-wrap:hover .chat-delete-link,
    .chat-row-wrap.active .chat-delete-link {
        opacity: 1;
    }
    .chat-delete-link:hover {
        background: color-mix(in srgb, #ef4444 18%, var(--secondary-background-color) 82%);
        color: #ef4444 !important;
        opacity: 1;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: color-mix(in srgb, var(--secondary-background-color) 94%, var(--background-color) 6%);
        border-color: var(--nexus-border);
        border-radius: 8px;
    }
    div[data-testid="stFileUploader"] section {
        border-color: var(--nexus-border);
        border-radius: 8px;
    }
    @keyframes nexus-spin {
        to { transform: rotate(360deg); }
    }
    @media (max-width: 760px) {
        .weather-top {
            flex-direction: column;
        }
        .weather-stats {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .forecast-strip {
            grid-template-columns: repeat(6, minmax(4.8rem, 1fr));
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="app-header">
        <div class="app-logo">✦</div>
        <div class="app-title">BuddyAi</div>
    </div>
    """,
    unsafe_allow_html=True,
)


def query_param(name: str) -> str | None:
    value = st.query_params.get(name)
    if isinstance(value, list):
        return value[0] if value else None
    return value


query_chat_id = query_param("chat")
delete_chat_id = query_param("delete_chat")
if "session_id" not in st.session_state:
    st.session_state.session_id = query_chat_id or uuid.uuid4().hex
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_bind_keys" not in st.session_state:
    st.session_state.uploaded_bind_keys = set()


def api_get(path: str):
    response = httpx.get(f"{BACKEND_URL}{path}", timeout=20)
    response.raise_for_status()
    return response.json()


def upload_file(file) -> dict:
    content = file.getvalue()
    response = httpx.post(
        f"{BACKEND_URL}/ingest/upload",
        data={"session_id": st.session_state.session_id},
        files={"file": (file.name, content, file.type or "application/octet-stream")},
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


def api_delete(path: str):
    response = httpx.delete(f"{BACKEND_URL}{path}", timeout=20)
    response.raise_for_status()
    return response.json()


def load_conversation(session_id: str) -> None:
    rows = api_get(f"/conversations/{session_id}/messages")
    st.session_state.session_id = session_id
    st.query_params["chat"] = session_id
    st.session_state.messages = [
        {
            "role": row["role"],
            "content": row["content"],
            "metadata": row.get("metadata") if row["role"] == "assistant" else None,
        }
        for row in rows
    ]
    st.session_state.loaded_session_id = session_id


def new_chat() -> None:
    session_id = uuid.uuid4().hex
    st.session_state.session_id = session_id
    st.query_params["chat"] = session_id
    st.session_state.messages = []
    st.session_state.loaded_session_id = session_id


def chat(message: str) -> dict:
    response = httpx.post(
        f"{BACKEND_URL}/chat",
        json={"session_id": st.session_state.session_id, "message": message},
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


def stream_chat(message: str):
    timeout = httpx.Timeout(180.0, connect=10.0, read=180.0)
    with httpx.stream(
        "POST",
        f"{BACKEND_URL}/chat/stream",
        json={"session_id": st.session_state.session_id, "message": message},
        timeout=timeout,
    ) as response:
        response.raise_for_status()
        for chunk in response.iter_text():
            if chunk:
                yield chunk


def generating_html(status: str = "Generating response") -> str:
    return (
        "<div class='generating-row'>"
        "<span class='typing-spinner'></span>"
        f"<span>{escape(status)}</span>"
        "</div>"
    )


def as_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def temp_label(value) -> str:
    number = as_float(value)
    return "N/A" if number is None else f"{round(number)}°"


def value_label(value, suffix: str = "") -> str:
    number = as_float(value)
    if number is None:
        return "N/A"
    if abs(number - round(number)) < 0.05:
        return f"{round(number)}{suffix}"
    return f"{number:.1f}{suffix}"


def weather_icon(icon_code: str | None, main: str | None = None, description: str | None = None) -> str:
    icon = (icon_code or "").lower()
    text = f"{main or ''} {description or ''}".lower()
    if icon.startswith("11") or "thunder" in text:
        return "⛈️"
    if icon.startswith("13") or "snow" in text:
        return "❄️"
    if icon.startswith("50") or any(word in text for word in ("mist", "fog", "haze")):
        return "🌫️"
    if icon.startswith(("09", "10")) or "rain" in text or "shower" in text:
        return "🌧️"
    if icon.startswith("01") or "clear" in text:
        return "☀️"
    if icon.startswith("02") or "few clouds" in text or "partly" in text:
        return "🌤️"
    if icon.startswith(("03", "04")) or "cloud" in text:
        return "☁️"
    return "🌡️"


def weather_visibility(value) -> str:
    number = as_float(value)
    if number is None:
        return "N/A"
    return f"{number / 1000:.1f} km"


def weather_chance(value) -> str:
    number = as_float(value)
    if number is None:
        return "N/A"
    return f"{round(number * 100)}%"


def weather_stat(label: str, value: str) -> str:
    return (
        "<div class='weather-stat'>"
        f"<div class='weather-stat-label'>{escape(label)}</div>"
        f"<div class='weather-stat-value'>{escape(value)}</div>"
        "</div>"
    )


def weather_chart_html(hourly: list[dict]) -> str:
    valid = [item for item in hourly[:8] if as_float(item.get("temp")) is not None]
    if len(valid) < 2:
        return ""
    temps = [as_float(item.get("temp")) or 0 for item in valid]
    low = min(temps)
    high = max(temps)
    spread = max(high - low, 1)
    points = []
    markers = []
    for index, item in enumerate(valid):
        temp = as_float(item.get("temp")) or 0
        x = 4 + (index * 92 / max(len(valid) - 1, 1))
        y = 50 - ((temp - low) / spread * 32)
        points.append(f"{x:.2f},{y:.2f}")
        markers.append(
            "<g>"
            f"<circle cx='{x:.2f}' cy='{y:.2f}' r='1.8' fill='var(--text-color)' "
            "stroke='var(--secondary-background-color)' stroke-width='0.9'></circle>"
            f"<text x='{x:.2f}' y='{max(y - 5, 7):.2f}' text-anchor='middle' "
            "font-size='5.2' font-weight='700' fill='var(--text-color)'>"
            f"{escape(temp_label(temp))}</text>"
            "</g>"
        )
    labels = "".join(
        f"<div class='weather-hour-label'>{escape(str(item.get('time') or ''))}</div>"
        for item in valid
    )
    label_grid_style = f"grid-template-columns: repeat({len(valid)}, minmax(2.2rem, 1fr));"
    return (
        "<div class='weather-chart'>"
        "<div class='weather-chart-title'>Next hours</div>"
        "<svg viewBox='0 0 100 62' preserveAspectRatio='none' aria-hidden='true'>"
        "<defs>"
        "<linearGradient id='weatherFill' x1='0' y1='0' x2='0' y2='1'>"
        "<stop offset='0%' stop-color='currentColor' stop-opacity='0.22'></stop>"
        "<stop offset='100%' stop-color='currentColor' stop-opacity='0'></stop>"
        "</linearGradient>"
        "</defs>"
        f"<polyline points='4,58 {' '.join(points)} 96,58' fill='url(#weatherFill)' stroke='none'></polyline>"
        f"<polyline points='{' '.join(points)}' fill='none' stroke='var(--text-color)' stroke-width='1.2' "
        "stroke-linecap='round' stroke-linejoin='round'></polyline>"
        f"{''.join(markers)}"
        "</svg>"
        f"<div class='weather-hour-labels' style='{label_grid_style}'>{labels}</div>"
        "</div>"
    )


def weather_component_html(tool_data: dict) -> str:
    payload_json = json.dumps(tool_data).replace("</", "<\\/")
    html = """
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
    :root {
        color-scheme: light dark;
        --panel: #ffffff;
        --panel-gloss: rgba(17, 24, 39, 0.018);
        --panel-raised: #f5f4f0;
        --border: rgba(17, 24, 39, 0.13);
        --surface: rgba(17, 24, 39, 0.045);
        --surface-hover: rgba(17, 24, 39, 0.075);
        --surface-border: rgba(17, 24, 39, 0.095);
        --unit-active: rgba(17, 24, 39, 0.12);
        --text: #111217;
        --muted: rgba(17, 18, 23, 0.56);
        --faint: rgba(17, 18, 23, 0.26);
        --warm: #f5d27a;
        --rain: #8ec8ff;
        --line: #16181d;
        --chart-fill: #efc37a;
        --point-stroke: #ffffff;
        --edge-highlight: rgba(255, 255, 255, 0.82);
        --frame-bg: #ffffff;
        --shadow: 0 14px 38px rgba(17, 24, 39, 0.12), 0 2px 8px rgba(17, 24, 39, 0.06);
    }
    body.theme-dark {
        --panel: #303030;
        --panel-gloss: rgba(255, 255, 255, 0.055);
        --panel-raised: #42444d;
        --border: rgba(255, 255, 255, 0.16);
        --surface: rgba(255, 255, 255, 0.08);
        --surface-hover: rgba(255, 255, 255, 0.10);
        --surface-border: rgba(255, 255, 255, 0.12);
        --unit-active: rgba(255, 255, 255, 0.20);
        --text: #f7f7f7;
        --muted: rgba(247, 247, 247, 0.62);
        --faint: rgba(247, 247, 247, 0.34);
        --line: #f2f2f2;
        --chart-fill: #f5d27a;
        --point-stroke: #303030;
        --edge-highlight: rgba(255, 255, 255, 0.08);
        --frame-bg: #111111;
        --shadow: 0 16px 48px rgba(0, 0, 0, 0.34), 0 2px 10px rgba(0, 0, 0, 0.22);
    }
    @media (prefers-color-scheme: dark) {
        body:not(.theme-light) {
            --panel: #303030;
            --panel-gloss: rgba(255, 255, 255, 0.055);
            --panel-raised: #42444d;
            --border: rgba(255, 255, 255, 0.16);
            --surface: rgba(255, 255, 255, 0.08);
            --surface-hover: rgba(255, 255, 255, 0.10);
            --surface-border: rgba(255, 255, 255, 0.12);
            --unit-active: rgba(255, 255, 255, 0.20);
            --text: #f7f7f7;
            --muted: rgba(247, 247, 247, 0.62);
            --faint: rgba(247, 247, 247, 0.34);
            --line: #f2f2f2;
            --chart-fill: #f5d27a;
            --point-stroke: #303030;
            --edge-highlight: rgba(255, 255, 255, 0.08);
            --frame-bg: #111111;
            --shadow: 0 16px 48px rgba(0, 0, 0, 0.34), 0 2px 10px rgba(0, 0, 0, 0.22);
        }
    }
    * { box-sizing: border-box; }
    body {
        background: var(--frame-bg);
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        margin: 0;
        overflow: hidden;
        padding: 5px;
    }
    .weather-shell {
        background:
            linear-gradient(180deg, var(--panel-gloss), transparent 48%),
            var(--panel);
        border: 1px solid var(--border);
        border-radius: 8px;
        box-shadow: var(--shadow);
        color: var(--text);
        max-width: 100%;
        min-height: 500px;
        overflow: hidden;
        padding: 18px 20px 32px;
        position: relative;
    }
    .weather-shell::before {
        border: 1px solid var(--edge-highlight);
        border-radius: inherit;
        box-shadow: inset 0 1px 0 var(--edge-highlight);
        content: "";
        inset: 0;
        pointer-events: none;
        position: absolute;
        z-index: 0;
    }
    .weather-shell > * {
        position: relative;
        z-index: 1;
    }
    .weather-header {
        align-items: flex-start;
        display: flex;
        gap: 16px;
        justify-content: space-between;
        min-width: 0;
    }
    .place {
        font-size: 16px;
        font-weight: 720;
        line-height: 1.25;
        overflow-wrap: anywhere;
    }
    .updated {
        color: var(--muted);
        font-size: 12px;
        font-weight: 620;
        margin-top: 4px;
    }
    .unit-toggle {
        align-items: center;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 999px;
        display: flex;
        flex: 0 0 auto;
        padding: 3px;
    }
    .unit-button {
        background: transparent;
        border: 0;
        border-radius: 999px;
        color: var(--muted);
        cursor: pointer;
        font-size: 12px;
        font-weight: 760;
        height: 26px;
        min-width: 32px;
        padding: 0 10px;
        transition: background 140ms ease, color 140ms ease;
    }
    .unit-button.active {
        background: var(--unit-active);
        color: var(--text);
    }
    .hero {
        align-items: flex-start;
        display: flex;
        gap: 18px;
        justify-content: space-between;
        margin-top: 18px;
    }
    .temp-wrap {
        align-items: flex-start;
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        min-width: 0;
    }
    .current-temp {
        font-size: 50px;
        font-weight: 680;
        letter-spacing: 0;
        line-height: 0.92;
    }
    .condition {
        color: var(--text);
        font-size: 16px;
        font-weight: 520;
        line-height: 1.38;
        margin-top: 10px;
        max-width: 760px;
        overflow-wrap: anywhere;
    }
    .big-icon {
        align-items: center;
        background:
            radial-gradient(circle at 38% 30%, rgba(245, 210, 122, 0.24), transparent 42%),
            var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        display: flex;
        flex: 0 0 56px;
        font-size: 32px;
        height: 56px;
        justify-content: center;
        width: 56px;
    }
    .daily-strip {
        display: grid;
        gap: 8px;
        grid-auto-columns: minmax(68px, 1fr);
        grid-auto-flow: column;
        margin-top: 20px;
        overflow-x: auto;
        padding-bottom: 4px;
        scrollbar-color: var(--faint) transparent;
        scrollbar-width: thin;
    }
    .day-card {
        background: transparent;
        border: 1px solid transparent;
        border-radius: 8px;
        color: var(--text);
        cursor: pointer;
        min-width: 68px;
        padding: 8px 7px 10px;
        text-align: center;
        transition: background 140ms ease, border-color 140ms ease, transform 140ms ease;
    }
    .day-card:hover {
        background: var(--surface-hover);
        border-color: var(--surface-border);
        transform: translateY(-1px);
    }
    .day-card.active {
        background: var(--panel-raised);
        border-color: var(--surface-border);
    }
    .day-name {
        font-size: 13px;
        font-weight: 740;
        line-height: 1.1;
    }
    .day-icon {
        font-size: 22px;
        line-height: 1;
        margin: 9px 0 8px;
    }
    .day-high {
        font-size: 16px;
        font-weight: 760;
        line-height: 1.1;
    }
    .day-low {
        color: var(--muted);
        font-size: 13px;
        font-weight: 620;
        line-height: 1.2;
        margin-top: 6px;
    }
    .details {
        display: grid;
        gap: 8px;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        margin-top: 18px;
    }
    .detail-card {
        background: var(--surface);
        border: 1px solid var(--surface-border);
        border-radius: 8px;
        min-width: 0;
        padding: 9px 10px;
    }
    .detail-label {
        color: var(--muted);
        font-size: 11px;
        font-weight: 780;
        line-height: 1.1;
        text-transform: uppercase;
    }
    .detail-value {
        color: var(--text);
        font-size: 15px;
        font-weight: 760;
        line-height: 1.2;
        margin-top: 5px;
        overflow-wrap: anywhere;
    }
    .chart-block {
        margin-top: 20px;
    }
    .chart-head {
        align-items: center;
        display: flex;
        justify-content: space-between;
        margin-bottom: 8px;
    }
    .chart-title {
        font-size: 15px;
        font-weight: 760;
    }
    .selected-summary {
        color: var(--muted);
        font-size: 12px;
        font-weight: 650;
        max-width: 56%;
        overflow: hidden;
        text-align: right;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .chart-frame {
        height: 160px;
        margin-top: 2px;
        overflow: visible;
        position: relative;
        width: 100%;
    }
    .chart-svg {
        display: block;
        height: 100%;
        inset: 0;
        overflow: visible;
        position: absolute;
        width: 100%;
    }
    .chart-area {
        opacity: 0.9;
        transform-origin: 50% 100%;
    }
    .chart-path {
        filter: drop-shadow(0 1px 0 rgba(255, 255, 255, 0.18));
        pathLength: 100;
    }
    .chart-frame.flowing .chart-path {
        animation: line-flow 620ms cubic-bezier(0.22, 1, 0.36, 1) both;
    }
    .chart-frame.flowing .chart-area {
        animation: area-rise 620ms cubic-bezier(0.22, 1, 0.36, 1) both;
    }
    .chart-value-layer {
        inset: 0;
        pointer-events: none;
        position: absolute;
    }
    .chart-value {
        color: var(--text);
        font-size: 14px;
        font-weight: 760;
        line-height: 1;
        position: absolute;
        text-shadow:
            0 1px 0 var(--panel),
            0 -1px 0 var(--panel),
            1px 0 0 var(--panel),
            -1px 0 0 var(--panel);
        transform: translate(-50%, -85%);
        white-space: nowrap;
    }
    .chart-frame.flowing .chart-value {
        animation: value-rise 360ms ease both;
    }
    .axis-labels {
        display: grid;
        gap: 4px;
        margin-top: 2px;
        padding: 0 2px;
    }
    .axis-label {
        color: var(--muted);
        font-size: 11px;
        font-weight: 650;
        overflow: hidden;
        text-align: center;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    @keyframes line-flow {
        from {
            opacity: 0.35;
            stroke-dasharray: 100;
            stroke-dashoffset: 100;
            transform: translateX(-1.5%);
        }
        to {
            opacity: 1;
            stroke-dasharray: 100;
            stroke-dashoffset: 0;
            transform: translateX(0);
        }
    }
    @keyframes area-rise {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 0.9;
            transform: translateY(0);
        }
    }
    @keyframes value-rise {
        from {
            opacity: 0;
            transform: translate(-50%, -55%);
        }
        to {
            opacity: 1;
            transform: translate(-50%, -85%);
        }
    }
    @media (max-width: 700px) {
        .weather-shell {
            min-height: 560px;
            padding: 16px 16px 30px;
        }
        .hero {
            margin-top: 16px;
        }
        .current-temp {
            font-size: 44px;
        }
        .condition {
            font-size: 15px;
        }
        .details {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .selected-summary {
            max-width: 45%;
        }
    }
</style>
</head>
<body>
<div class="weather-shell">
    <div class="weather-header">
        <div>
            <div class="place" id="place"></div>
            <div class="updated" id="updated"></div>
        </div>
        <div class="unit-toggle" aria-label="Temperature unit">
            <button type="button" class="unit-button active" data-unit="c">C</button>
            <button type="button" class="unit-button" data-unit="f">F</button>
        </div>
    </div>
    <div class="hero">
        <div>
            <div class="temp-wrap">
                <div class="current-temp" id="currentTemp"></div>
            </div>
            <div class="condition" id="condition"></div>
        </div>
        <div class="big-icon" id="bigIcon"></div>
    </div>
    <div class="daily-strip" id="dailyStrip"></div>
    <div class="details" id="details"></div>
    <div class="chart-block">
        <div class="chart-head">
            <div class="chart-title" id="chartTitle">Temperature</div>
            <div class="selected-summary" id="selectedSummary"></div>
        </div>
        <div class="chart-frame" id="chartFrame">
            <svg class="chart-svg" id="chart" viewBox="0 0 100 76" preserveAspectRatio="none" aria-hidden="true"></svg>
            <div class="chart-value-layer" id="chartValues"></div>
        </div>
        <div class="axis-labels" id="axisLabels"></div>
    </div>
</div>
<script>
function parseColor(value) {
    const match = String(value || "").match(/rgba?\\((\\d+),\\s*(\\d+),\\s*(\\d+)/i);
    if (!match) return null;
    return [Number(match[1]), Number(match[2]), Number(match[3])];
}
function isDarkColor(value) {
    const color = parseColor(value);
    if (!color) return null;
    const [red, green, blue] = color.map(channel => {
        const normalized = channel / 255;
        return normalized <= 0.03928 ? normalized / 12.92 : Math.pow((normalized + 0.055) / 1.055, 2.4);
    });
    return (0.2126 * red + 0.7152 * green + 0.0722 * blue) < 0.42;
}
function detectTheme() {
    try {
        const parentDoc = window.parent.document;
        const rootStyle = window.parent.getComputedStyle(parentDoc.documentElement);
        const bodyStyle = window.parent.getComputedStyle(parentDoc.body);
        const candidates = [
            rootStyle.getPropertyValue("--background-color"),
            bodyStyle.backgroundColor,
            rootStyle.backgroundColor
        ];
        for (const candidate of candidates) {
            const dark = isDarkColor(candidate);
            if (dark !== null) return dark ? "dark" : "light";
        }
    } catch (error) {
        // Streamlit may sandbox component frames differently across versions.
    }
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches
        ? "dark"
        : "light";
}
function applyTheme() {
    const theme = detectTheme();
    document.body.classList.toggle("theme-dark", theme === "dark");
    document.body.classList.toggle("theme-light", theme === "light");
}
applyTheme();
if (window.matchMedia) {
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", applyTheme);
}
window.setInterval(applyTheme, 1200);

const weatherData = __WEATHER_DATA__;
let unit = "c";
let selectedDay = 0;

const current = weatherData.current || {};
const daily = Array.isArray(weatherData.daily) ? weatherData.daily.slice(0, 6) : [];
const hourly = Array.isArray(weatherData.hourly) ? weatherData.hourly.slice(0, 8) : [];
const timeline = Array.isArray(weatherData.timeline) ? weatherData.timeline : hourly;

function number(value) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
}
function toUnit(value) {
    const parsed = number(value);
    if (parsed === null) return null;
    return unit === "f" ? parsed * 9 / 5 + 32 : parsed;
}
function temp(value) {
    const converted = toUnit(value);
    return converted === null ? "N/A" : `${Math.round(converted)}°`;
}
function label(value, suffix = "") {
    const parsed = number(value);
    if (parsed === null) return "N/A";
    return `${Math.abs(parsed - Math.round(parsed)) < 0.05 ? Math.round(parsed) : parsed.toFixed(1)}${suffix}`;
}
function chance(value) {
    const parsed = number(value);
    if (parsed === null) return "N/A";
    return `${Math.round(parsed * 100)}%`;
}
function visibility(value) {
    const parsed = number(value);
    if (parsed === null) return "N/A";
    return `${(parsed / 1000).toFixed(1)} km`;
}
function weatherIcon(iconCode, main, description, pop = null, tempValue = null, windValue = null) {
    const icon = String(iconCode || "").toLowerCase();
    const text = `${main || ""} ${description || ""}`.toLowerCase();
    const rainChance = number(pop);
    const warmTemp = number(tempValue);
    const wind = number(windValue);
    if (icon.startsWith("11") || text.includes("thunder")) return "⛈️";
    if (icon.startsWith("13") || text.includes("snow")) return "❄️";
    if (rainChance !== null && rainChance >= 0.55) return "🌧️";
    if (rainChance !== null && rainChance >= 0.25) return "🌦️";
    if (icon.startsWith("50") || ["mist", "fog", "haze", "smoke", "dust"].some(word => text.includes(word))) return "🌫️";
    if (icon.startsWith("09") || icon.startsWith("10") || text.includes("rain") || text.includes("shower")) {
        return text.includes("light") || text.includes("drizzle") ? "🌦️" : "🌧️";
    }
    if (wind !== null && wind >= 11) return "🌬️";
    if (warmTemp !== null && warmTemp >= 38 && (icon.startsWith("01") || icon.startsWith("02") || icon.startsWith("03"))) return "☀️";
    if (icon.startsWith("01") || text.includes("clear")) return "☀️";
    if (icon.startsWith("02") || text.includes("few clouds") || text.includes("partly")) return "🌤️";
    if (icon.startsWith("03") || text.includes("scattered")) return warmTemp !== null && warmTemp >= 32 ? "🌤️" : "⛅";
    if (icon.startsWith("04") || text.includes("broken") || text.includes("overcast")) return "☁️";
    if (text.includes("cloud")) return warmTemp !== null && warmTemp >= 32 ? "🌥️" : "☁️";
    return "🌡️";
}
function escapeHtml(value) {
    return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}
function titleCase(text) {
    return String(text || "Current conditions").replace(/\\w\\S*/g, word => word.charAt(0).toUpperCase() + word.slice(1));
}
function renderHeader() {
    document.getElementById("place").textContent = weatherData.place || current.name || "Weather";
    document.getElementById("updated").textContent = current.local_time ? `Updated ${current.local_time}` : "Live weather";
    document.getElementById("currentTemp").textContent = temp(current.temp);
    document.getElementById("condition").textContent = titleCase(current.description);
    document.getElementById("bigIcon").textContent = weatherIcon(
        current.icon,
        current.main,
        current.description,
        null,
        current.temp,
        current.wind_speed
    );
}
function renderDays() {
    const strip = document.getElementById("dailyStrip");
    const days = daily.length ? daily : [{
        day: "Now",
        temp_max: current.temp_max ?? current.temp,
        temp_min: current.temp_min ?? current.feels_like,
        icon: current.icon,
        description: current.description,
        pop: null
    }];
    strip.innerHTML = days.map((day, index) => `
        <button type="button" class="day-card ${index === selectedDay ? "active" : ""}" data-index="${index}" title="${escapeHtml(titleCase(day.description))}">
            <div class="day-name">${escapeHtml(day.day || "Day")}</div>
            <div class="day-icon">${weatherIcon(day.icon, day.main, day.description, day.pop, day.temp_max)}</div>
            <div class="day-high">${temp(day.temp_max)}</div>
            <div class="day-low">${temp(day.temp_min)}</div>
        </button>
    `).join("");
    strip.querySelectorAll(".day-card").forEach(button => {
        button.addEventListener("click", () => {
            selectedDay = Number(button.dataset.index) || 0;
            renderAll();
        });
    });
}
function detail(labelText, valueText) {
    return `
        <div class="detail-card">
            <div class="detail-label">${escapeHtml(labelText)}</div>
            <div class="detail-value">${escapeHtml(valueText)}</div>
        </div>
    `;
}
function renderDetails() {
    const day = daily[selectedDay] || {};
    const daySeries = selectedForecastSeries();
    const dayHumidity = daySeries.map(item => number(item.humidity)).filter(value => value !== null);
    const dayWind = daySeries.map(item => number(item.wind_speed)).filter(value => value !== null);
    const dayFeels = daySeries.map(item => number(item.feels_like)).filter(value => value !== null);
    const selectedDetails = selectedDay === 0 ? [
        detail("Feels Like", temp(current.feels_like)),
        detail("Humidity", label(current.humidity, "%")),
        detail("Wind", label(current.wind_speed, " m/s")),
        detail("Visibility", visibility(current.visibility)),
        detail("Sunrise", current.sunrise || "N/A"),
        detail("Sunset", current.sunset || "N/A"),
        detail("Pressure", label(current.pressure, " hPa")),
        detail("Clouds", label(current.clouds, "%"))
    ] : [
        detail("High", temp(day.temp_max)),
        detail("Low", temp(day.temp_min)),
        detail("Rain Chance", chance(day.pop)),
        detail("Condition", titleCase(day.description)),
        detail("Feels Like", dayFeels.length ? temp(Math.max(...dayFeels)) : temp(current.feels_like)),
        detail("Wind", dayWind.length ? label(Math.max(...dayWind), " m/s") : label(current.wind_speed, " m/s")),
        detail("Humidity", dayHumidity.length ? label(Math.round(dayHumidity.reduce((sum, value) => sum + value, 0) / dayHumidity.length), "%") : label(current.humidity, "%")),
        detail("Timeline", daySeries.length ? `${daySeries[0].time || ""} - ${daySeries[daySeries.length - 1].time || ""}` : "Forecast")
    ];
    document.getElementById("details").innerHTML = selectedDetails.join("");
    document.getElementById("selectedSummary").textContent = day.description
        ? `${titleCase(day.description)} · rain ${chance(day.pop)}`
        : titleCase(current.description);
}
function selectedForecastSeries() {
    const selected = daily[selectedDay] || {};
    if (selectedDay === 0 && hourly.length >= 2) {
        return hourly;
    }
    const byDate = selected.date
        ? timeline.filter(item => item.date === selected.date && number(item.temp) !== null)
        : [];
    if (byDate.length >= 2) {
        return byDate;
    }
    if (hourly.length >= 2) {
        return hourly;
    }
    return daily
        .map(item => ({
            temp: item.temp_max,
            time: item.day,
            time_short: item.day,
            description: item.description,
            pop: item.pop
        }))
        .filter(item => number(item.temp) !== null);
}
function smoothPath(points) {
    if (!points.length) return "";
    if (points.length === 1) return `M ${points[0].x.toFixed(2)} ${points[0].y.toFixed(2)}`;
    let path = `M ${points[0].x.toFixed(2)} ${points[0].y.toFixed(2)}`;
    for (let index = 0; index < points.length - 1; index += 1) {
        const previous = points[index - 1] || points[index];
        const currentPoint = points[index];
        const next = points[index + 1];
        const afterNext = points[index + 2] || next;
        const cp1x = currentPoint.x + (next.x - previous.x) / 6;
        const cp1y = currentPoint.y + (next.y - previous.y) / 6;
        const cp2x = next.x - (afterNext.x - currentPoint.x) / 6;
        const cp2y = next.y - (afterNext.y - currentPoint.y) / 6;
        path += ` C ${cp1x.toFixed(2)} ${cp1y.toFixed(2)}, ${cp2x.toFixed(2)} ${cp2y.toFixed(2)}, ${next.x.toFixed(2)} ${next.y.toFixed(2)}`;
    }
    return path;
}
function renderChart() {
    const svg = document.getElementById("chart");
    const labels = document.getElementById("axisLabels");
    const valuesLayer = document.getElementById("chartValues");
    const frame = document.getElementById("chartFrame");
    const series = selectedForecastSeries()
        .map(item => ({
            value: toUnit(item.temp),
            label: item.time_short || item.time || item.day || "",
            fullLabel: item.time || item.day || "",
        }))
        .filter(item => item.value !== null)
        .slice(0, 8);

    if (series.length < 2) {
        svg.innerHTML = "";
        valuesLayer.innerHTML = "";
        labels.innerHTML = "";
        return;
    }

    const values = series.map(item => item.value);
    const low = Math.min(...values);
    const high = Math.max(...values);
    const spread = Math.max(high - low, 1);
    const points = series.map((item, index) => {
        const x = 5 + index * 90 / Math.max(series.length - 1, 1);
        const y = 57 - ((item.value - low) / spread * 34);
        return { x, y, value: item.value, label: item.label };
    });
    const linePath = smoothPath(points);
    const areaPath = `${linePath} L ${points[points.length - 1].x.toFixed(2)} 68 L ${points[0].x.toFixed(2)} 68 Z`;
    svg.innerHTML = `
        <defs>
            <linearGradient id="fill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="var(--chart-fill)" stop-opacity="0.32"></stop>
                <stop offset="100%" stop-color="var(--chart-fill)" stop-opacity="0"></stop>
            </linearGradient>
        </defs>
        <path class="chart-area" d="${areaPath}" fill="url(#fill)" stroke="none"></path>
        <path class="chart-path" d="${linePath}" fill="none" stroke="var(--line)" stroke-width="1.45" stroke-linecap="round" stroke-linejoin="round" pathLength="100"></path>
        ${points.map(point => `
            <g class="chart-dot">
                <circle cx="${point.x.toFixed(2)}" cy="${point.y.toFixed(2)}" r="2.1" fill="var(--line)" stroke="var(--point-stroke)" stroke-width="1"></circle>
            </g>
        `).join("")}
    `;
    valuesLayer.innerHTML = points.map(point => {
        const left = point.x;
        const top = Math.min(Math.max((point.y / 76) * 100 - 4, 7), 82);
        return `<div class="chart-value" style="left:${left.toFixed(2)}%;top:${top.toFixed(2)}%">${Math.round(point.value)}°</div>`;
    }).join("");
    labels.style.gridTemplateColumns = `repeat(${series.length}, minmax(28px, 1fr))`;
    labels.innerHTML = series.map(item => `<div class="axis-label">${escapeHtml(item.label)}</div>`).join("");
    frame.classList.remove("flowing");
    void frame.offsetWidth;
    frame.classList.add("flowing");
}
function renderAll() {
    renderHeader();
    renderDays();
    renderDetails();
    renderChart();
    document.querySelectorAll(".unit-button").forEach(button => {
        button.classList.toggle("active", button.dataset.unit === unit);
    });
}
document.querySelectorAll(".unit-button").forEach(button => {
    button.addEventListener("click", () => {
        unit = button.dataset.unit || "c";
        renderAll();
    });
});
renderAll();
</script>
</body>
</html>
"""
    return html.replace("__WEATHER_DATA__", payload_json)


def render_weather_card(metadata: dict | None) -> bool:
    tool_data = (metadata or {}).get("tool_data") or {}
    if not isinstance(tool_data, dict) or tool_data.get("type") != "weather":
        return False
    components.html(weather_component_html(tool_data), height=700, scrolling=False)
    return True


STATUS_MARKER = "[[status]]"
ANSWER_START_MARKER = "[[answer_start]]"
METADATA_MARKER = "[[metadata]]"
STREAM_MARKERS = (STATUS_MARKER, ANSWER_START_MARKER, METADATA_MARKER)


def split_safe_stream_text(buffer: str) -> tuple[str, str]:
    keep = 0
    for marker in STREAM_MARKERS:
        max_size = min(len(marker) - 1, len(buffer))
        for size in range(1, max_size + 1):
            if buffer.endswith(marker[:size]):
                keep = max(keep, size)
    if keep:
        return buffer[:-keep], buffer[-keep:]
    return buffer, ""


def compact_text(value: str | None, limit: int = 46) -> str:
    text = (value or "Untitled chat").strip().replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def conversation_link(session_id: str) -> str:
    return f"?chat={quote(session_id)}"


def delete_conversation_link(session_id: str) -> str:
    return f"?chat={quote(st.session_state.session_id)}&delete_chat={quote(session_id)}"


def render_chat_history() -> None:
    st.markdown("<div class='chat-history-title'>Chats</div>", unsafe_allow_html=True)
    if st.button("+ New chat", key="new-chat", type="primary", use_container_width=True):
        new_chat()
        st.rerun()

    try:
        conversations = api_get("/conversations")
        current_in_history = any(
            conversation["session_id"] == st.session_state.session_id
            for conversation in conversations
        )

        with st.container(height=285, border=True):
            if not conversations:
                st.caption("No saved chats yet.")
            elif not current_in_history:
                st.caption("New chat. Send a message or upload a file to save it.")

            for conversation in conversations:
                full_title = (
                    conversation.get("title")
                    or conversation.get("topic")
                    or conversation.get("last_message")
                    or "Untitled chat"
                )
                title = compact_text(
                    full_title,
                    limit=25,
                )
                is_current = conversation["session_id"] == st.session_state.session_id
                active_class = " active" if is_current else ""
                session_id = conversation["session_id"]
                st.markdown(
                    f"""
                    <div class="chat-row-wrap{active_class}">
                        <a
                            class="chat-row-link"
                            href="{conversation_link(session_id)}"
                            target="_self"
                            title="{escape(full_title)}"
                        >{escape(title)}</a>
                        <a
                            class="chat-delete-link"
                            href="{delete_conversation_link(session_id)}"
                            target="_self"
                            title="Delete chat"
                        >&times;</a>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    except Exception:
        st.caption("Chat history is unavailable.")


if delete_chat_id:
    try:
        api_delete(f"/conversations/{delete_chat_id}")
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code != 404:
            st.warning(f"Delete failed: {exc}")
    if delete_chat_id == st.session_state.session_id:
        st.session_state.session_id = uuid.uuid4().hex
        st.session_state.messages = []
        st.session_state.loaded_session_id = st.session_state.session_id
    st.query_params.clear()
    st.query_params["chat"] = st.session_state.session_id
    st.rerun()


if query_chat_id and query_chat_id != st.session_state.get("loaded_session_id"):
    try:
        load_conversation(query_chat_id)
    except Exception:
        st.session_state.loaded_session_id = st.session_state.session_id
elif not query_chat_id:
    st.query_params["chat"] = st.session_state.session_id


with st.sidebar:
    render_chat_history()
    st.divider()

    st.subheader("Knowledge Base")
    uploaded = st.file_uploader(
        "Upload PDF, PowerPoint, Excel, CSV, or image",
        type=["pdf", "ppt", "pptx", "xlsx", "xls", "csv", "png", "jpg", "jpeg", "tif", "tiff"],
        key=f"knowledge-upload-{st.session_state.session_id}",
    )
    if uploaded:
        upload_hash = hashlib.sha256(uploaded.getvalue()).hexdigest()
        upload_key = f"{st.session_state.session_id}:{upload_hash}"
        if upload_key not in st.session_state.uploaded_bind_keys:
            st.session_state.uploaded_bind_keys.add(upload_key)
            with st.spinner("Ingesting uploaded file..."):
                try:
                    result = upload_file(uploaded)
                    st.success(result["summary"])
                    st.json(result)
                except Exception as exc:
                    st.session_state.uploaded_bind_keys.discard(upload_key)
                    st.error(f"Ingestion failed: {exc}")
        else:
            st.caption("Uploaded file is attached to this chat.")

    st.subheader("Google Drive")
    if st.button("Sync Drive folder"):
        try:
            response = httpx.post(f"{BACKEND_URL}/google-drive/sync", timeout=60)
            response.raise_for_status()
            st.json(response.json())
        except Exception as exc:
            st.error(f"Google Drive sync failed: {exc}")

    st.subheader("Indexed Documents")
    if st.button("Reindex documents", use_container_width=True):
        with st.spinner("Reindexing active documents..."):
            try:
                response = httpx.post(f"{BACKEND_URL}/ingest/reindex", timeout=600)
                response.raise_for_status()
                st.json(response.json())
            except Exception as exc:
                st.error(f"Reindex failed: {exc}")
    try:
        docs = api_get("/documents")
        for doc in docs:
            st.caption(f"{doc['file_name']} | active version {doc.get('active_version_id')}")
    except Exception:
        st.caption("Backend not reachable or database not initialized.")


for message in st.session_state.messages:
    avatar = ASSISTANT_AVATAR if message["role"] == "assistant" else None
    with st.chat_message(message["role"], avatar=avatar):
        if message.get("metadata"):
            render_weather_card(message["metadata"])
        st.markdown(message["content"])
        if message.get("metadata"):
            with st.expander("Sources and audit trace"):
                st.json(message["metadata"])


prompt = st.chat_input("Ask about documents, spreadsheets, weather, web, or general topics")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        placeholder = st.empty()
        indicator = st.empty()
        try:
            answer = ""
            metadata_raw = ""
            metadata_started = False
            answer_started = False
            stream_buffer = ""
            indicator.markdown(generating_html(), unsafe_allow_html=True)

            for chunk in stream_chat(prompt):
                stream_buffer += chunk
                while stream_buffer:
                    if metadata_started:
                        metadata_raw += stream_buffer
                        stream_buffer = ""
                        break

                    if stream_buffer.startswith(STATUS_MARKER):
                        status_end = stream_buffer.find("\n\n")
                        if status_end < 0:
                            break
                        status = stream_buffer[len(STATUS_MARKER) : status_end].strip()
                        if not answer_started:
                            indicator.markdown(
                                generating_html(status or "Generating response"),
                                unsafe_allow_html=True,
                            )
                        stream_buffer = stream_buffer[status_end + 2 :]
                        continue

                    if stream_buffer.startswith(ANSWER_START_MARKER):
                        answer_started = True
                        indicator.empty()
                        stream_buffer = stream_buffer[len(ANSWER_START_MARKER) :]
                        continue

                    metadata_index = stream_buffer.find(METADATA_MARKER)
                    if metadata_index >= 0:
                        answer_part = stream_buffer[:metadata_index]
                        if answer_part:
                            answer += answer_part
                            answer_started = True
                            indicator.empty()
                            placeholder.markdown(answer)
                        metadata_raw += stream_buffer[metadata_index + len(METADATA_MARKER) :]
                        metadata_started = True
                        stream_buffer = ""
                        break

                    emit_text, stream_buffer = split_safe_stream_text(stream_buffer)
                    if not emit_text:
                        break
                    answer += emit_text
                    answer_started = True
                    indicator.empty()
                    placeholder.markdown(answer)

            if stream_buffer:
                if metadata_started:
                    metadata_raw += stream_buffer
                else:
                    answer += stream_buffer
                    answer_started = True
                    indicator.empty()
                    placeholder.markdown(answer)

            indicator.empty()
            try:
                result = json.loads(metadata_raw) if metadata_raw.strip() else {"answer": answer}
            except json.JSONDecodeError:
                result = {"answer": answer}
            final_answer = result.get("answer") or answer
            metadata = {
                "confidence": result.get("confidence"),
                "sources": result.get("sources", []),
                "route": result.get("route"),
                "fallback": result.get("fallback"),
                "audit_trace": result.get("audit_trace", []),
                "tool_data": result.get("tool_data", {}),
            }
            if render_weather_card(metadata):
                placeholder.empty()
                st.markdown(final_answer)
            else:
                placeholder.markdown(final_answer)
            with st.expander("Sources and audit trace"):
                st.json(metadata)
            st.session_state.messages.append(
                {"role": "assistant", "content": final_answer, "metadata": metadata}
            )
        except Exception as exc:
            indicator.empty()
            error = f"Request failed: {exc}"
            placeholder.error(error)
            st.session_state.messages.append({"role": "assistant", "content": error})
