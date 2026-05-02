from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


WEATHER_ALIASES = {
    "banglore": "Bangalore,IN",
    "bangalore": "Bangalore,IN",
    "bengaluru": "Bangalore,IN",
    "bnglr": "Bangalore,IN",
    "blr": "Bangalore,IN",
    "delhi": "Delhi,IN",
    "new delhi": "New Delhi,IN",
    "mumbai": "Mumbai,IN",
    "kolkata": "Kolkata,IN",
    "chennai": "Chennai,IN",
    "hyderabad": "Hyderabad,IN",
    "hyd": "Hyderabad,IN",
    "pune": "Pune,IN",
}


@dataclass
class ConversationFrame:
    user_query: str | None = None
    assistant_answer: str | None = None
    tool_name: str | None = None
    intent: str | None = None
    document_name: str | None = None
    document_id: str | None = None
    location: str | None = None

    @property
    def domain(self) -> str:
        if self.tool_name == "weather_api" or self.intent == "weather" or self.location:
            return "weather"
        if self.tool_name == "kb_retriever" or self.intent in {"kb_qa", "cross_source"}:
            return "kb"
        if self.tool_name == "excel_calculation" or self.intent == "excel_analysis":
            return "excel"
        if self.tool_name == "web_search" or self.intent == "web_search":
            return "web"
        if self.tool_name == "version_compare" or self.intent == "version_compare":
            return "version"
        if self.tool_name == "general_llm" or self.intent == "general_qa":
            return "general"
        return "unknown"


@dataclass
class ConversationState:
    frames: list[ConversationFrame]

    @classmethod
    def from_context(cls, context: Any) -> "ConversationState":
        frames: list[ConversationFrame] = []
        last_user: str | None = None

        for message in context.recent_messages or []:
            role = message.get("role")
            content = message.get("content") or ""
            if role == "user":
                last_user = content
                continue
            if role != "assistant":
                continue

            metadata = message.get("metadata") or {}
            route = metadata.get("route") or {}
            sources = metadata.get("sources") or []
            first_source = sources[0] if sources else {}
            is_weather_route = route.get("tool_name") == "weather_api" or route.get("intent") == "weather"
            location = _extract_weather_location(
                last_user or "",
                allow_bare_location=is_weather_route,
            ) or _extract_weather_location(
                content,
                allow_bare_location=is_weather_route,
            )
            frames.append(
                ConversationFrame(
                    user_query=last_user,
                    assistant_answer=content,
                    tool_name=route.get("tool_name"),
                    intent=route.get("intent"),
                    document_name=first_source.get("file_name"),
                    document_id=first_source.get("document_id"),
                    location=location,
                )
            )

        if not frames and (context.previous_query or context.previous_answer or context.last_tool):
            frames.append(
                ConversationFrame(
                    user_query=context.previous_query,
                    assistant_answer=context.previous_answer,
                    tool_name=context.last_tool,
                    document_name=context.last_document_name,
                    document_id=context.last_document,
                    location=_extract_weather_location(
                        context.previous_query or "",
                        allow_bare_location=context.last_tool == "weather_api",
                    )
                    or _extract_weather_location(
                        context.previous_answer or "",
                        allow_bare_location=context.last_tool == "weather_api",
                    ),
                )
            )

        if frames and context.last_tool and not frames[-1].tool_name:
            frames[-1].tool_name = context.last_tool
        if frames and context.last_document_name and not frames[-1].document_name:
            frames[-1].document_name = context.last_document_name
        if frames and context.last_document and not frames[-1].document_id:
            frames[-1].document_id = context.last_document

        return cls(frames=frames)

    @property
    def latest(self) -> ConversationFrame | None:
        return self.frames[-1] if self.frames else None

    def latest_domain(self, domain: str, within: int | None = None) -> ConversationFrame | None:
        candidates = self.frames if within is None else self.frames[-within:]
        for frame in reversed(candidates):
            if frame.domain == domain:
                return frame
        return None

    def has_domain(self, domain: str, within: int | None = None) -> bool:
        return self.latest_domain(domain, within=within) is not None

    @property
    def last_document_name(self) -> str | None:
        frame = self.latest_domain("kb")
        return frame.document_name if frame else None

    @property
    def last_weather_location(self) -> str | None:
        frame = self.latest_domain("weather")
        return frame.location if frame else None

    def compact_context(self, max_frames: int = 4) -> str:
        lines: list[str] = []
        for frame in self.frames[-max_frames:]:
            label = frame.domain
            details = []
            if frame.document_name:
                details.append(f"document={frame.document_name}")
            if frame.location:
                details.append(f"location={frame.location}")
            suffix = f" ({', '.join(details)})" if details else ""
            if frame.user_query:
                lines.append(f"User[{label}{suffix}]: {frame.user_query}")
            if frame.assistant_answer:
                lines.append(f"Assistant[{label}{suffix}]: {frame.assistant_answer}")
        return "\n".join(lines)


def _extract_weather_location(text: str, allow_bare_location: bool = False) -> str | None:
    lowered = text.lower()
    match = re.search(r"\bweather for\s+([^:.\n]+)", lowered)
    if not match and re.search(r"\b(weather|temperature|forecast)\b", lowered):
        match = re.search(
            r"\b(?:weather|temperature|forecast)\b[^.\n:]{0,40}?\b(?:in|for|at)\s+([a-z ,.-]+)",
            lowered,
        )
    if not match and allow_bare_location:
        match = re.search(r"\b(?:in|for|at)\s+([a-z ,.-]+)", lowered)
    if not match:
        candidate = lowered.strip(" ?.!,")
        return WEATHER_ALIASES.get(candidate)

    location = re.sub(r"\s+", " ", match.group(1)).strip(" .,?!")
    if not location or location in {"india", "in"}:
        return None
    return WEATHER_ALIASES.get(location, location.title())
