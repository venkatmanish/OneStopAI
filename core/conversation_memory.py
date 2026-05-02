from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


STOPWORDS = {
    "a",
    "about",
    "again",
    "all",
    "also",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "been",
    "by",
    "can",
    "could",
    "day",
    "did",
    "do",
    "does",
    "for",
    "from",
    "get",
    "give",
    "has",
    "have",
    "here",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "out",
    "please",
    "should",
    "some",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "you",
}

DOMAIN_TERMS = {
    "travel": {
        "accommodation",
        "activity",
        "beach",
        "dinner",
        "goa",
        "hotel",
        "itinerary",
        "restaurant",
        "seafood",
        "sightseeing",
        "trip",
        "visit",
    },
    "health": {
        "cancer",
        "disease",
        "doctor",
        "harmful",
        "health",
        "medical",
        "medicine",
        "symptom",
        "treatment",
    },
    "finance": {"gold", "karat", "price", "rate", "stock", "market"},
    "weather": {"celsius", "clouds", "forecast", "temperature", "weather"},
    "exam": {"beta", "certification", "exam", "result", "score"},
    "profile": {
        "applicant",
        "candidate",
        "cgpa",
        "cv",
        "cybersecurity",
        "education",
        "experience",
        "generative",
        "profile",
        "project",
        "resume",
        "skill",
    },
}


@dataclass
class MemorySnippet:
    label: str
    user_query: str
    assistant_answer: str
    terms: set[str]
    document_name: str | None = None
    tool_name: str | None = None
    intent: str | None = None

    def to_context_lines(self) -> list[str]:
        details = []
        if self.document_name:
            details.append(f"document={self.document_name}")
        suffix = f" ({', '.join(details)})" if details else ""
        return [
            f"User[memory {self.label}{suffix}]: {self.user_query}",
            f"Assistant[memory {self.label}{suffix}]: {self.assistant_answer}",
        ]


def build_memory_snippets(messages: list[Any], max_snippets: int = 18) -> list[dict[str, Any]]:
    snippets: list[MemorySnippet] = []
    last_user: str | None = None
    for message in messages:
        role = _message_value(message, "role")
        content = _message_value(message, "content") or ""
        if role == "user":
            last_user = content
            continue
        if role != "assistant" or not last_user or not content:
            continue

        metadata = _message_value(message, "message_metadata") or _message_value(message, "metadata") or {}
        route = metadata.get("route") or {}
        sources = metadata.get("sources") or []
        first_source = sources[0] if sources else {}
        document_name = first_source.get("file_name")
        text = f"{last_user}\n{content}\n{document_name or ''}"
        terms = content_terms(text)
        label = _label_for_terms(terms, document_name=document_name, route=route)
        if label == "general" and not _has_memory_value(text, terms):
            continue
        snippets.append(
            MemorySnippet(
                label=label,
                user_query=_compact(last_user, 260),
                assistant_answer=_compact(content, 700),
                terms=terms,
                document_name=document_name,
                tool_name=route.get("tool_name"),
                intent=route.get("intent"),
            )
        )

    deduped: list[MemorySnippet] = []
    seen: set[tuple[str, str, str | None]] = set()
    for snippet in reversed(snippets):
        key = (snippet.label, snippet.user_query.lower(), snippet.document_name)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(snippet)
    deduped.reverse()
    return [
        {
            "label": snippet.label,
            "user_query": snippet.user_query,
            "assistant_answer": snippet.assistant_answer,
            "terms": sorted(snippet.terms),
            "document_name": snippet.document_name,
            "tool_name": snippet.tool_name,
            "intent": snippet.intent,
        }
        for snippet in deduped[-max_snippets:]
    ]


def relevant_memory_snippets(query: str, snippets: list[dict[str, Any]] | None, limit: int = 3) -> list[dict[str, Any]]:
    if not snippets:
        return []
    query_terms = content_terms(query)
    query_lower = query.lower()
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for index, snippet in enumerate(snippets):
        snippet_terms = set(snippet.get("terms") or [])
        text = f"{snippet.get('user_query', '')} {snippet.get('assistant_answer', '')}".lower()
        document_name = str(snippet.get("document_name") or "").lower()
        score = float(len(query_terms & snippet_terms))

        if re.search(r"\b(day\s*\d+|itinerary|plan|trip|restaurant|dinner|evening)\b", query_lower):
            if snippet.get("label") == "travel" or re.search(r"\b(day\s*\d+|itinerary|trip)\b", text):
                score += 3
        if re.search(r"\b(cancer|harmful|health|medical|disease|treatment)\b", query_lower):
            if snippet.get("label") == "health":
                score += 3
        if re.search(
            r"\b(he|him|his|guy|person|candidate|applicant|resume|cv|profile|cgpa|"
            r"gen\s*ai|generative|cyber\s*security|cybersecurity|skill|experience|rate)\b",
            query_lower,
        ):
            if snippet.get("label") == "profile" or re.search(r"\b(resume|cv)\b", document_name):
                score += 3
        if re.search(r"\b(that|this|our|previous|earlier|we were|from our|that plan)\b", query_lower):
            score += min(2, len(snippet_terms))
        day_match = re.search(r"\bday\s*(\d+)\b", query_lower)
        if day_match and re.search(rf"\bday\s*{day_match.group(1)}\b", text):
            score += 4
        if score > 0:
            scored.append((score, index, snippet))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [snippet for _, _, snippet in scored[:limit]]


def format_memory_context(query: str, snippets: list[dict[str, Any]] | None, limit: int = 3) -> str:
    selected = relevant_memory_snippets(query, snippets, limit=limit)
    if not selected:
        return ""
    lines = ["Relevant earlier context:"]
    for snippet in selected:
        memory = MemorySnippet(
            label=snippet.get("label", "general"),
            user_query=snippet.get("user_query", ""),
            assistant_answer=snippet.get("assistant_answer", ""),
            terms=set(snippet.get("terms") or []),
            document_name=snippet.get("document_name"),
            tool_name=snippet.get("tool_name"),
            intent=snippet.get("intent"),
        )
        lines.extend(memory.to_context_lines())
    return "\n".join(lines) + "\n"


def content_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        if token in STOPWORDS or len(token) < 3:
            continue
        if token.endswith("s") and len(token) > 4:
            token = token[:-1]
        terms.add(token)
    return terms


def _message_value(message: Any, attr: str) -> Any:
    if isinstance(message, dict):
        return message.get(attr)
    return getattr(message, attr, None)


def _label_for_terms(
    terms: set[str],
    document_name: str | None = None,
    route: dict[str, Any] | None = None,
) -> str:
    doc = (document_name or "").lower()
    if re.search(r"\b(resume|cv)\b", doc):
        return "profile"
    if (route or {}).get("tool_name") == "kb_retriever" and terms & DOMAIN_TERMS["profile"]:
        return "profile"
    best_label = "general"
    best_score = 0
    for label, domain_terms in DOMAIN_TERMS.items():
        score = len(terms & domain_terms)
        if score > best_score:
            best_label = label
            best_score = score
    return best_label


def _has_memory_value(text: str, terms: set[str]) -> bool:
    return bool(
        len(terms) >= 5
        or re.search(
            r"\b(day\s*\d+|itinerary|trip|restaurant|cancer|gold|weather|exam|certificate|resume|cv)\b",
            text,
            re.I,
        )
    )


def _compact(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."
