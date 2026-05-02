from __future__ import annotations

import re
from typing import Any
from dataclasses import dataclass

from core.conversation_memory import format_memory_context, relevant_memory_snippets
from core.conversation_state import ConversationState
from core.llm import LLMClient
from core.schemas import RouteDecision


@dataclass
class SessionContext:
    previous_query: str | None = None
    previous_answer: str | None = None
    recent_messages: list[dict[str, Any]] | None = None
    topic: str | None = None
    last_tool: str | None = None
    last_document: str | None = None
    last_document_name: str | None = None
    available_files: list[str] | None = None
    active_versions: dict[str, str] | None = None
    memory_snippets: list[dict[str, Any]] | None = None


class IntentRouter:
    DOCUMENT_TYPE_PREFIX = "__document_type__:"
    DOCUMENT_TYPE_EXTENSIONS = {
        "pdf": {".pdf"},
        "presentation": {".ppt", ".pptx"},
        "spreadsheet": {".csv", ".xls", ".xlsx"},
        "image": {".jpeg", ".jpg", ".png", ".tif", ".tiff"},
    }
    FILE_MATCH_STOPWORDS = {
        "about",
        "case",
        "doc",
        "docs",
        "file",
        "from",
        "pdf",
        "study",
        "the",
    }
    ORDINALS = {
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": 4,
        "fifth": 5,
        "sixth": 6,
        "seventh": 7,
        "eighth": 8,
        "ninth": 9,
        "tenth": 10,
    }
    DOCUMENT_TYPE_TERMS = {
        "pdf": {"pdf"},
        "certificate": {"certificate", "cert"},
        "invoice": {"invoice", "bill"},
        "receipt": {"receipt"},
        "report": {"report"},
        "policy": {"policy"},
        "contract": {"contract", "agreement"},
        "presentation": {"presentation", "ppt", "pptx", "slide", "slides", "deck"},
        "spreadsheet": {"csv", "excel", "sheet", "spreadsheet", "xls", "xlsx"},
        "image": {"image", "jpeg", "jpg", "photo", "picture", "png"},
        "resume": {"resume", "cv"},
        "transcript": {"transcript"},
        "marksheet": {"marksheet", "mark", "sheet"},
        "statement": {"statement"},
        "letter": {"letter"},
    }
    WEATHER_LOCATION_ALIASES = {
        "banglore",
        "bangalore",
        "bengaluru",
        "bnglr",
        "blr",
        "delhi",
        "new delhi",
        "mumbai",
        "kolkata",
        "chennai",
        "hyderabad",
        "hyd",
        "pune",
    }
    FOLLOW_UP_HINTS = {
        "it",
        "that",
        "this",
        "they",
        "them",
        "those",
        "their",
        "there",
        "he",
        "him",
        "his",
        "guy",
        "person",
        "same",
        "above",
        "earlier",
        "previous",
        "context",
        "meant",
    }
    CONTEXT_STOPWORDS = {
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

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def route(self, query: str, context: SessionContext) -> RouteDecision:
        rule = self._rule_route(query, context)
        if rule.confidence >= 0.85:
            return rule

        semantic = self._semantic_route(query, context)
        combined = self._combine(rule, semantic)
        if 0.6 <= combined.confidence < 0.85:
            return self._validate(combined, query, context)
        if combined.confidence < 0.6:
            return RouteDecision(
                intent="clarify",
                tool_name="general_llm",
                needs_memory=True,
                confidence=combined.confidence,
                is_follow_up=combined.is_follow_up,
                is_topic_shift=combined.is_topic_shift,
                reason="Router confidence below clarification threshold.",
            )
        return combined

    def rewrite_follow_up(self, query: str, context: SessionContext, route: RouteDecision) -> str:
        numbered_item = self.resolve_numbered_reference(query, context)
        if numbered_item:
            return (
                f"Follow-up question: {query}\n"
                "The referenced numbered item from the previous answer is exactly:\n"
                f"{numbered_item}\n"
                "Answer about that exact item. Do not reinterpret the number as a quantity, page number, or a different numbered item."
            )
        term = self._term_definition_follow_up(query, context)
        if term and route.is_follow_up:
            recent_context = self._format_context_for_query(query, context)
            return (
                "Use the conversation context to answer this term follow-up.\n"
                f"Term to explain: {term}\n"
                "Task: explain what this term means here. Do not answer the previous question again.\n"
                f"{recent_context}"
                f"Follow-up question: {query}"
            )
        if not route.is_follow_up or not context.previous_query:
            return query
        recent_context = self._format_context_for_query(query, context)
        return (
            "Use the conversation context to answer this follow-up.\n"
            f"{recent_context}"
            f"Follow-up question: {query}"
        )

    def _rule_route(self, query: str, context: SessionContext) -> RouteDecision:
        q = query.lower().strip()
        conversation_state = ConversationState.from_context(context)
        follow_up = self._looks_like_follow_up(q, context, conversation_state)
        topic_shift = self._topic_shift(q, context)
        sub_intents: list[str] = []

        if re.match(r"^(hi|hello|hey|namaste|good morning|good afternoon|good evening)[!. ]*$", q):
            return RouteDecision(
                intent="general_qa",
                tool_name="general_llm",
                needs_memory=True,
                is_follow_up=False,
                is_topic_shift=topic_shift,
                confidence=0.95,
                reason="Greeting matched.",
            )
        if self._looks_like_rag_system_question(q):
            return RouteDecision(
                intent="general_qa",
                tool_name="general_llm",
                needs_memory=True,
                needs_rag=False,
                is_follow_up=False,
                is_topic_shift=topic_shift,
                confidence=0.9,
                reason="Question asks how RAG grounding works, not for content from uploaded files.",
            )
        if conversation_state.has_domain("weather", within=3) and self._looks_like_weather_follow_up(q):
            return RouteDecision(
                intent="weather",
                tool_name="weather_api",
                needs_web=True,
                is_follow_up=True,
                is_topic_shift=False,
                confidence=0.9,
                reason="Follow-up corrected or continued a weather query.",
            )
        if self._looks_like_weather_location_fragment(q) and self._recent_weather_context(context, conversation_state):
            return RouteDecision(
                intent="weather",
                tool_name="weather_api",
                needs_web=True,
                is_follow_up=True,
                is_topic_shift=False,
                confidence=0.88,
                reason="Short location continued a recent weather thread.",
            )
        if self._looks_like_health_query(q, context):
            return RouteDecision(
                intent="web_search",
                sub_intents=["health"],
                needs_web=True,
                tool_name="web_search",
                is_follow_up=follow_up,
                is_topic_shift=topic_shift,
                confidence=0.9,
                reason="Health or medical query should use non-KB general/web evidence.",
            )
        if self._looks_like_recommendation_query(q, context):
            return RouteDecision(
                intent="web_search",
                sub_intents=["recommendation"],
                needs_web=True,
                tool_name="web_search",
                is_follow_up=follow_up,
                is_topic_shift=topic_shift,
                confidence=0.88,
                reason="Recommendation query should use current web evidence and conversation context.",
            )
        if self._should_escalate_follow_up_to_web(q, context, conversation_state, follow_up):
            sub_intents.append("freshness")
            return RouteDecision(
                intent="web_search",
                sub_intents=sub_intents,
                needs_web=True,
                tool_name="web_search",
                is_follow_up=True,
                is_topic_shift=False,
                confidence=0.87,
                reason="Follow-up asks for a current or exact detail missing from indexed evidence.",
            )
        cross_source_intents = self._cross_source_sub_intents(q, context)
        if cross_source_intents:
            target_source = self._matched_available_file(q, context)
            active_document = context.last_document_name or conversation_state.last_document_name
            if not target_source and active_document and self._looks_like_current_document_query(q):
                target_source = active_document
            return RouteDecision(
                intent="cross_source",
                sub_intents=cross_source_intents,
                needs_rag=any(intent in cross_source_intents for intent in {"kb", "excel"}),
                needs_web="web" in cross_source_intents,
                tool_name="kb_retriever",
                is_follow_up=follow_up,
                is_topic_shift=topic_shift,
                target_source=target_source,
                confidence=0.9,
                reason="Complex request needs a controlled multi-step plan across tools.",
            )
        numbered_item = self.resolve_numbered_reference(q, context)
        if numbered_item:
            target_source = (
                context.last_document_name
                or conversation_state.last_document_name
                or self._matched_available_file(
                    numbered_item,
                    context,
                )
            )
            return RouteDecision(
                intent="kb_qa" if target_source or context.last_tool == "kb_retriever" else "general_qa",
                needs_rag=bool(target_source or context.last_tool == "kb_retriever"),
                tool_name="kb_retriever" if target_source or context.last_tool == "kb_retriever" else "general_llm",
                is_follow_up=True,
                is_topic_shift=False,
                target_source=target_source,
                confidence=0.9,
                reason="Follow-up referenced a numbered item from a previous answer.",
            )
        person_document = self._person_profile_document_target(q, context, conversation_state)
        if person_document:
            return RouteDecision(
                intent="kb_qa",
                needs_rag=True,
                tool_name="kb_retriever",
                is_follow_up=follow_up,
                is_topic_shift=False,
                target_source=person_document,
                confidence=0.89,
                reason="Person/profile reference resolves to an indexed document from the conversation.",
            )
        spreadsheet_file = self._matched_spreadsheet_file(q, context)
        if spreadsheet_file and self._looks_like_spreadsheet_analysis_query(q):
            return RouteDecision(
                intent="excel_analysis",
                sub_intents=["calculation"],
                needs_rag=True,
                tool_name="excel_calculation",
                is_follow_up=follow_up,
                is_topic_shift=topic_shift,
                target_source=spreadsheet_file,
                confidence=0.91,
                reason="Spreadsheet file matched with analysis/calculation intent.",
            )
        matched_file = self._matched_available_file(q, context)
        if matched_file:
            return RouteDecision(
                intent="kb_qa",
                needs_rag=True,
                tool_name="kb_retriever",
                is_follow_up=follow_up,
                is_topic_shift=topic_shift,
                target_source=matched_file,
                confidence=0.89,
                reason="Query matched an indexed file name.",
            )
        if self._has_ambiguous_available_file_match(q, context):
            return RouteDecision(
                intent="kb_qa",
                needs_rag=True,
                tool_name="kb_retriever",
                is_follow_up=follow_up,
                is_topic_shift=topic_shift,
                confidence=0.86,
                reason="Query matched multiple indexed file names; searching across KB for the closest chunk.",
            )
        requested_document_type = self._requested_document_type(q)
        typed_scope = self._document_type_target(q, context)
        active_document = context.last_document_name or conversation_state.last_document_name
        if (
            active_document
            and self._looks_like_current_document_query(q)
            and (
                not requested_document_type
                or self._file_matches_document_type(active_document, requested_document_type)
            )
        ):
            return RouteDecision(
                intent="kb_qa",
                needs_rag=True,
                tool_name="kb_retriever",
                is_follow_up=follow_up,
                is_topic_shift=False,
                target_source=active_document,
                confidence=0.88,
                reason="Query refers to the current chat document.",
            )
        if self._looks_like_active_document_content_query(
            q,
            context,
            conversation_state,
            active_document,
            requested_document_type,
        ):
            return RouteDecision(
                intent="kb_qa",
                needs_rag=True,
                tool_name="kb_retriever",
                is_follow_up=follow_up,
                is_topic_shift=False,
                target_source=active_document,
                confidence=0.87,
                reason="Question appears related to the active chat document.",
            )
        if typed_scope:
            return RouteDecision(
                intent="kb_qa",
                needs_rag=True,
                tool_name="kb_retriever",
                is_follow_up=follow_up,
                is_topic_shift=topic_shift,
                target_source=typed_scope,
                confidence=0.88,
                reason="Query explicitly requested an indexed document type.",
            )
        typed_file = self._matched_file_by_document_type(q, context)
        if typed_file:
            return RouteDecision(
                intent="kb_qa",
                needs_rag=True,
                tool_name="kb_retriever",
                is_follow_up=follow_up,
                is_topic_shift=topic_shift,
                target_source=typed_file,
                confidence=0.88,
                reason="Query matched an indexed document type.",
            )
        if any(word in q for word in ["weather", "temperature", "forecast"]):
            return RouteDecision(
                intent="weather",
                tool_name="weather_api",
                needs_web=True,
                is_follow_up=follow_up,
                is_topic_shift=topic_shift,
                confidence=0.92,
                reason="Weather keyword matched.",
            )
        if self._looks_like_web_search_request(q):
            sub_intents.append("freshness")
            return RouteDecision(
                intent="web_search",
                sub_intents=sub_intents,
                needs_web=True,
                tool_name="web_search",
                is_follow_up=follow_up,
                is_topic_shift=topic_shift,
                confidence=0.88,
                reason="Freshness or web-search keyword matched.",
            )
        if any(word in q for word in ["excel", "sheet", "sum", "average", "total", "count", "column"]):
            return RouteDecision(
                intent="excel_analysis",
                sub_intents=["calculation"],
                needs_rag=True,
                tool_name="excel_calculation",
                is_follow_up=follow_up,
                is_topic_shift=topic_shift,
                confidence=0.9,
                reason="Spreadsheet or calculation keyword matched.",
            )
        if any(word in q for word in ["version", "compare revision", "old document", "previous document"]):
            return RouteDecision(
                intent="version_compare",
                needs_rag=True,
                tool_name="version_compare",
                is_follow_up=follow_up,
                is_topic_shift=topic_shift,
                confidence=0.87,
                reason="Version comparison keyword matched.",
            )
        if any(word in q for word in ["pdf", "document", "file", "source", "page", "policy", "contract", "ppt", "slide"]):
            return RouteDecision(
                intent="kb_qa",
                needs_rag=True,
                tool_name="kb_retriever",
                is_follow_up=follow_up,
                is_topic_shift=topic_shift,
                confidence=0.82,
                reason="Knowledge-base keyword matched.",
            )
        if context.available_files and any(
            word in q
            for word in ["certificate", "college", "university", "provider", "issuer", "institution"]
        ):
            return RouteDecision(
                intent="kb_qa",
                needs_rag=True,
                tool_name="kb_retriever",
                is_follow_up=follow_up,
                is_topic_shift=topic_shift,
                confidence=0.83,
                reason="Question asks about institutional details likely stored in indexed files.",
            )
        if context.available_files and any(
            phrase in q
            for phrase in ["problem statement", "problem statements", "case study", "case studies"]
        ):
            return RouteDecision(
                intent="kb_qa",
                needs_rag=True,
                tool_name="kb_retriever",
                is_follow_up=follow_up,
                is_topic_shift=topic_shift,
                confidence=0.86,
                reason="Question asks about case/problem-statement content in indexed files.",
            )
        if follow_up and context.last_tool == "kb_retriever" and context.available_files and not self._standalone_topic_shift(q):
            return RouteDecision(
                intent="kb_qa",
                needs_rag=True,
                tool_name="kb_retriever",
                is_follow_up=True,
                is_topic_shift=False,
                target_source=active_document,
                confidence=0.86,
                reason="Follow-up continues the previous knowledge-base answer.",
            )
        if (
            follow_up
            and conversation_state.has_domain("kb", within=5)
            and context.available_files
            and not self._standalone_topic_shift(q)
        ):
            return RouteDecision(
                intent="kb_qa",
                needs_rag=True,
                tool_name="kb_retriever",
                is_follow_up=True,
                is_topic_shift=False,
                target_source=conversation_state.last_document_name,
                confidence=0.84,
                reason="Follow-up continues a recent knowledge-base answer.",
            )
        return RouteDecision(
            intent="general_qa",
            tool_name="general_llm",
            is_follow_up=follow_up,
            is_topic_shift=topic_shift,
            confidence=0.72,
            reason="No specialized rule matched.",
        )

    def _semantic_route(self, query: str, context: SessionContext) -> RouteDecision:
        prompt = f"""
Return only JSON matching this schema:
intent: one of general_qa,kb_qa,excel_analysis,web_search,weather,version_compare,cross_source,clarify
sub_intents: array of strings
needs_memory, needs_rag, needs_web: booleans
tool_name: one of kb_retriever,web_search,weather_api,excel_calculation,version_compare,general_llm
is_follow_up, is_topic_shift: booleans
target_source: string or null
target_version: string
confidence: number 0-1
reason: short string

Query: {query}
Previous query: {context.previous_query}
Topic: {context.topic}
Last tool: {context.last_tool}
Last document: {context.last_document}
Available files: {context.available_files or []}
"""
        data = self.llm.json_complete(prompt, system="You are an intent router for an agentic RAG system.")
        try:
            return RouteDecision(**data)
        except Exception:
            return RouteDecision(
                intent="general_qa",
                tool_name="general_llm",
                confidence=0.55,
                reason="LLM router unavailable or invalid.",
            )

    def _combine(self, rule: RouteDecision, semantic: RouteDecision) -> RouteDecision:
        if semantic.confidence > rule.confidence + 0.1:
            chosen = semantic
        else:
            chosen = rule
        chosen.confidence = round(max(rule.confidence, semantic.confidence) * 0.92, 3)
        chosen.sub_intents = sorted(set(rule.sub_intents + semantic.sub_intents))
        return chosen

    def _validate(self, route: RouteDecision, query: str, context: SessionContext) -> RouteDecision:
        if route.needs_rag and not context.available_files:
            route.intent = "clarify"
            route.tool_name = "general_llm"
            route.needs_rag = False
            route.reason = "RAG route requested but no knowledge-base files are indexed."
            route.confidence = min(route.confidence, 0.58)
        if (
            route.needs_rag
            and not self._has_kb_affinity(query.lower(), context)
            and not (route.is_follow_up and context.last_tool == "kb_retriever")
        ):
            route.intent = "general_qa"
            route.tool_name = "general_llm"
            route.needs_rag = False
            route.reason = "RAG route requested without explicit document affinity; using general context."
            route.confidence = min(route.confidence, 0.74)
        return route

    @staticmethod
    def _topic_shift(query: str, context: SessionContext) -> bool:
        if not context.topic:
            return False
        topic_tokens = set(re.findall(r"[a-z0-9]{4,}", context.topic.lower()))
        query_tokens = set(re.findall(r"[a-z0-9]{4,}", query.lower()))
        if not topic_tokens or not query_tokens:
            return False
        return len(topic_tokens & query_tokens) == 0 and len(query_tokens) >= 3

    @classmethod
    def resolve_numbered_reference(cls, query: str, context: SessionContext) -> str | None:
        index = cls._referenced_number(query)
        if index is None:
            return None
        for answer in cls._assistant_answers_newest_first(context):
            items = cls._numbered_items(answer)
            if index in items:
                return items[index]
        return None

    @classmethod
    def _assistant_answers_newest_first(cls, context: SessionContext) -> list[str]:
        answers: list[str] = []
        if context.previous_answer:
            answers.append(context.previous_answer)
        for message in reversed(context.recent_messages or []):
            if message.get("role") == "assistant" and message.get("content"):
                content = message["content"]
                if content not in answers:
                    answers.append(content)
        return answers

    @classmethod
    def _referenced_number(cls, query: str) -> int | None:
        q = query.lower().strip()
        if re.search(r"[\+\-*/=]", q):
            return None
        number_match = re.match(
            r"^(?:tell me about|explain|expand on|more about|details on|what about|elaborate on)?\s*"
            r"(?:point|option|number|item|#)?\s*(\d{1,2})(?:st|nd|rd|th)?(?:\s+one)?[?.! ]*$",
            q,
        )
        if number_match:
            return int(number_match.group(1))
        ordinal_match = re.match(
            r"^(?:tell me about|explain|expand on|more about|details on|what about|elaborate on)?\s*"
            r"(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)(?:\s+one)?[?.! ]*$",
            q,
        )
        if ordinal_match:
            return cls.ORDINALS[ordinal_match.group(1)]
        return None

    @staticmethod
    def _numbered_items(answer: str) -> dict[int, str]:
        pattern = re.compile(
            r"(?ms)^\s*(?:[-*]\s*)?(?:\*\*)?(\d{1,2})[.)]\s*(.*?)(?=^\s*(?:[-*]\s*)?(?:\*\*)?\d{1,2}[.)]\s+|\Z)"
        )
        items: dict[int, str] = {}
        for match in pattern.finditer(answer):
            text = re.sub(r"\s+", " ", match.group(2)).strip()
            text = text.strip("* ")
            if text:
                items[int(match.group(1))] = text[:600]
        return items

    def _looks_like_follow_up(
        self,
        query: str,
        context: SessionContext,
        conversation_state: ConversationState | None = None,
    ) -> bool:
        conversation_state = conversation_state or ConversationState.from_context(context)
        if not (context.previous_query or context.previous_answer):
            return False
        if self._referenced_number(query) is not None and self.resolve_numbered_reference(query, context):
            return True
        query_tokens = self._loose_tokens(query)
        if query_tokens & self.FOLLOW_UP_HINTS:
            return True
        if re.search(r"\b(you meant|you said|meant earlier|in this context|from earlier)\b", query):
            return True
        if self._contextual_completion_follow_up(query, context, conversation_state):
            return True
        if self._short_fragment(query):
            if context.last_tool in {"kb_retriever", "excel_calculation", "version_compare"}:
                return True
            if conversation_state.has_domain("kb", within=5):
                return True
            if conversation_state.has_domain("weather", within=3) and self._looks_like_weather_location_fragment(query):
                return True
            if self._overlaps_recent_context(query, context):
                return True
            if self._parallel_acronym_follow_up(query, context):
                return True
        return bool(
            re.match(
                r"^(what about|and |also|compare that|how about|it|they|those|that|this|"
                r"tell me more|explain more|more details|what is it called again|"
                r"what's it called again)\b",
                query,
            )
        )

    @staticmethod
    def _looks_like_weather_follow_up(query: str) -> bool:
        cleaned = query.lower().strip(" ?.!,")
        return bool(
            re.search(r"\b(weather|temperature|forecast)\b", query)
            or re.match(r"^(i meant|in |for |at |what about|how about)\b", query)
            or IntentRouter._looks_like_weather_location_fragment(query)
        )

    @staticmethod
    def _looks_like_rag_system_question(query: str) -> bool:
        normalized = re.sub(r"\binn\b", "in", query.lower())
        if not re.search(r"\brag\b|retrieval\s+augmented\s+generation", normalized):
            return False
        return bool(
            re.search(
                r"\b(how|why|what|explain|describe)\b.*\b(ground|grounded|grounding|cite|"
                r"citation|source|sources|evidence|retriev|answer|answers)\b",
                normalized,
            )
            or re.search(
                r"\b(ground|grounded|grounding|cite|citation|source|sources|evidence|retriev)\b"
                r".*\b(rag|retrieval\s+augmented\s+generation)\b",
                normalized,
            )
        )

    @staticmethod
    def _looks_like_web_search_request(query: str) -> bool:
        return bool(
            re.search(r"\b(latest|today|news|web|internet|search online|online search)\b", query)
            or re.search(
                r"\b(find|look|check|search|browse|verify)\b.{0,40}\b(online|web|internet)\b",
                query,
            )
            or re.search(
                r"\b(online|web|internet)\b.{0,40}\b(find|look|check|search|browse|verify)\b",
                query,
            )
        )

    @classmethod
    def _matched_spreadsheet_file(cls, query: str, context: SessionContext) -> str | None:
        matched = cls._matched_available_file(query, context)
        if matched and cls._file_matches_document_type(matched, "spreadsheet"):
            return matched
        typed = cls._matched_file_by_document_type(query, context)
        if typed and cls._file_matches_document_type(typed, "spreadsheet"):
            return typed
        return None

    @staticmethod
    def _looks_like_spreadsheet_analysis_query(query: str) -> bool:
        return bool(
            re.search(
                r"\b(calculate|calculation|compute|sum|average|avg|total|count|margin|gross|net|"
                r"revenue|sales|target|highest|lowest|rank|group|grouped|by|join|joined|normalize|"
                r"normalise|messy|sku|orders?|returns?|products?|master|columns?|sheets?|schema|"
                r"preview|available|contain|contains|flag|unmatched|match)\b",
                query,
            )
        )

    @classmethod
    def _looks_like_health_query(cls, query: str, context: SessionContext) -> bool:
        if cls._has_kb_affinity(query, context):
            return False
        if re.search(
            r"\b(cancer|tumou?r|disease|diagnosis|symptom|treatment|medicine|medical|doctor|"
            r"harmful|harmfull|pain|infection|therapy|risk|cough|cold|fever|tablet|tablets|"
            r"medication|drug|dose|dosage)\b",
            query,
        ):
            return True
        if re.search(r"\b(harmful|harmfull|dangerous|serious|risk)\b", query):
            memory_text = cls._recent_context_text(context, max_messages=8).lower()
            memory_text += " " + " ".join(
                f"{snippet.get('user_query', '')} {snippet.get('assistant_answer', '')}"
                for snippet in (context.memory_snippets or [])
                if snippet.get("label") == "health"
            ).lower()
            return bool(re.search(r"\b(cancer|tumou?r|disease|medical|health)\b", memory_text))
        return False

    @classmethod
    def _looks_like_recommendation_query(cls, query: str, context: SessionContext) -> bool:
        if cls._has_kb_affinity(query, context):
            return False
        if not re.search(r"\b(recommend|suggest|restaurant|hotel|accommodation|where should|where to|best)\b", query):
            return False
        if re.search(r"\b(restaurant|hotel|accommodation|stay|dinner|lunch|cafe|bar|beach|trip|itinerary)\b", query):
            return True
        return bool(relevant_memory_snippets(query, context.memory_snippets, limit=1))

    @classmethod
    def _should_escalate_follow_up_to_web(
        cls,
        query: str,
        context: SessionContext,
        conversation_state: ConversationState,
        follow_up: bool,
    ) -> bool:
        if not follow_up or not conversation_state.has_domain("kb", within=5):
            return False
        if cls._standalone_topic_shift(query):
            return False
        recent_text = cls._recent_context_text(context, max_messages=8).lower()
        if not cls._recent_answer_has_evidence_gap(recent_text):
            return False
        return cls._asks_for_current_or_exact_timing(query)

    @staticmethod
    def _recent_answer_has_evidence_gap(recent_text: str) -> bool:
        return bool(
            re.search(
                r"\b(indexed evidence|provided evidence|evidence)\b.{0,80}\b(does not|doesn't|not|no)\b"
                r"|"
                r"\b(no exact|not specified|not available|not found|couldn'?t find|could not find|"
                r"not give an exact|not show the exact|does not show the exact)\b",
                recent_text,
            )
        )

    @staticmethod
    def _asks_for_current_or_exact_timing(query: str) -> bool:
        return bool(
            re.search(
                r"\b(official|exact|specific|expected|estimated|available|release|released|"
                r"date|month|timeline|when|end|ends|out|status)\b",
                query,
            )
        )

    @classmethod
    def _cross_source_sub_intents(cls, query: str, context: SessionContext) -> list[str]:
        q = query.lower()
        intents: list[str] = []
        mentions_kb = bool(
            context.available_files
            and re.search(
                r"\b(pdf|document|documents|doc|docs|file|files|uploaded|source|sources|"
                r"kb|knowledge base|resume|cv|presentation|ppt|pptx|slide|slides|deck)\b",
                q,
            )
        )
        mentions_web = bool(
            re.search(
                r"\b(web|internet|online|latest|current|currently|today|live|news|external|official)\b",
                q,
            )
        )
        mentions_excel = bool(
            context.available_files
            and re.search(r"\b(excel|spreadsheet|sheet|sheets|csv|table|tables|worksheet)\b", q)
        )
        compare_or_combine = bool(
            re.search(
                r"\b(compare|contrast|combine|cross-source|across|against|with|using|use both|"
                r"from both|answer using|explain using|correlate|reconcile)\b",
                q,
            )
        )
        multi_doc_summary = bool(
            context.available_files
            and re.search(
                r"\b(all|multiple|many|several|across)\s+(?:indexed\s+)?(?:docs|documents|files|sources)\b",
                q,
            )
            and re.search(r"\b(search|summari[sz]e|cite|citation|citations|answer|extract|find)\b", q)
        )

        if multi_doc_summary:
            intents.append("kb")
        if compare_or_combine and mentions_kb:
            intents.append("kb")
        if compare_or_combine and mentions_web:
            intents.append("web")
        if mentions_excel and (
            compare_or_combine
            or re.search(r"\b(analy[sz]e|calculate|sum|average|total|count|explain)\b", q)
        ):
            intents.append("excel")
            if mentions_kb or re.search(r"\b(explain|pdf|document|doc|source)\b", q):
                intents.append("kb")

        unique = list(dict.fromkeys(intents))
        if len(unique) >= 2 or multi_doc_summary:
            return unique
        if "web" in unique and "kb" not in unique and mentions_kb:
            return ["kb", "web"]
        return []

    @staticmethod
    def _looks_like_weather_location_fragment(query: str) -> bool:
        cleaned = query.lower().strip(" ?.!,")
        tokens = IntentRouter._loose_tokens(cleaned)
        return bool(
            cleaned in IntentRouter.WEATHER_LOCATION_ALIASES
            or (
                0 < len(tokens) <= 2
                and not (tokens & {"what", "is", "why", "how", "who", "when", "where"})
                and re.fullmatch(r"[a-z][a-z .,-]{1,40}", cleaned)
            )
        )

    @staticmethod
    def _recent_weather_context(
        context: SessionContext,
        conversation_state: ConversationState | None = None,
    ) -> bool:
        conversation_state = conversation_state or ConversationState.from_context(context)
        if context.last_tool == "weather_api" or conversation_state.has_domain("weather", within=5):
            return True
        recent_text = " ".join(
            [
                context.previous_query or "",
                context.previous_answer or "",
                " ".join(
                    message.get("content", "")
                    for message in (context.recent_messages or [])[-6:]
                ),
            ]
        ).lower()
        return bool(
            re.search(r"\bweather\b", recent_text)
            or re.search(r"\bweather for\b", recent_text)
            or re.search(r"\b\d{1,2}(?:\.\d+)?\s*c\b", recent_text)
        )

    @classmethod
    def _person_profile_document_target(
        cls,
        query: str,
        context: SessionContext,
        conversation_state: ConversationState,
    ) -> str | None:
        if not context.available_files:
            return None
        if not cls._looks_like_person_profile_query(query):
            return None
        if cls._looks_like_health_query(query, context):
            return None

        explicit_resume = cls._matched_file_by_document_type(query, context)
        if explicit_resume:
            return explicit_resume

        active_document = context.last_document_name or conversation_state.last_document_name
        if active_document and cls._file_looks_like_profile(active_document):
            return active_document

        latest_kb = conversation_state.latest_domain("kb")
        if latest_kb and latest_kb.document_name and cls._file_looks_like_profile(latest_kb.document_name):
            return latest_kb.document_name

        memory_documents = [
            snippet.get("document_name")
            for snippet in relevant_memory_snippets(query, context.memory_snippets, limit=5)
            if snippet.get("document_name") and snippet.get("label") == "profile"
        ]
        unique_memory_documents = list(dict.fromkeys(memory_documents))
        if len(unique_memory_documents) == 1:
            return unique_memory_documents[0]

        profile_files = [file_name for file_name in context.available_files if cls._file_looks_like_profile(file_name)]
        if len(profile_files) == 1 and cls._has_prior_profile_context(context, conversation_state):
            return profile_files[0]
        return None

    @staticmethod
    def _looks_like_person_profile_query(query: str) -> bool:
        return bool(
            re.search(r"\b(he|him|his|guy|person|candidate|applicant)\b", query)
            and re.search(
                r"\b(good|rate|rating|score|skill|skills|experience|project|projects|cgpa|"
                r"education|college|degree|resume|cv|profile|gen\s*ai|generative|ai|"
                r"cyber\s*security|cybersecurity|github|linkedin|link|url|contact)\b",
                query,
            )
        )

    @staticmethod
    def _file_looks_like_profile(file_name: str) -> bool:
        return bool(re.search(r"\b(resume|cv)\b", file_name.lower()))

    @classmethod
    def _has_prior_profile_context(
        cls,
        context: SessionContext,
        conversation_state: ConversationState,
    ) -> bool:
        if context.last_document_name and cls._file_looks_like_profile(context.last_document_name):
            return True
        for frame in conversation_state.frames:
            if frame.document_name and cls._file_looks_like_profile(frame.document_name):
                return True
        for snippet in context.memory_snippets or []:
            if snippet.get("label") == "profile" or cls._file_looks_like_profile(str(snippet.get("document_name") or "")):
                return True
        return False

    @classmethod
    def _has_kb_affinity(cls, query: str, context: SessionContext) -> bool:
        q = query.lower()
        if cls.resolve_numbered_reference(q, context):
            return True
        if cls._matched_available_file(q, context):
            return True
        if context.last_document_name and cls._looks_like_current_document_query(q):
            return True
        if cls._matched_file_by_document_type(q, context):
            return True
        if re.search(
            r"\b(pdf|document|file|source|page|policy|contract|uploaded|current document|this document|"
            r"ppt|pptx|presentation|slide|deck|link|url|hyperlink|hypertext|github|linkedin)\b",
            q,
        ):
            return True
        if context.available_files and re.search(
            r"\b(certificate|college|university|provider|issuer|institution|problem statement|case study)\b",
            q,
        ):
            return True
        return False

    @staticmethod
    def _looks_like_current_document_query(query: str) -> bool:
        if re.search(r"\b(this|that|it|uploaded|current|above|same)\b", query) and re.search(
            r"\b(file|document|pdf|ppt|pptx|presentation|slide|deck|certificate|report|doc|source|"
            r"college|provider|issuer|summari[sz]e|explain|name|details|about|link|url|"
            r"hyperlink|hypertext|github|linkedin)\b",
            query,
        ):
            return True
        return bool(
            re.match(
                r"^(summari[sz]e|explain)\s+(?:this|that|current|uploaded|the)\s+(?:document|file|pdf|source)\b|"
                r"^(what is this|what does this|who issued|"
                r"who is the provider|what's the provider|what is the provider)\b",
                query,
            )
        )

    @classmethod
    def _looks_like_active_document_content_query(
        cls,
        query: str,
        context: SessionContext,
        conversation_state: ConversationState,
        active_document: str | None,
        requested_document_type: str | None,
    ) -> bool:
        if not active_document or not context.available_files:
            return False
        if requested_document_type and not cls._file_matches_document_type(
            active_document,
            requested_document_type,
        ):
            return False
        if cls._standalone_topic_shift(query) or cls._explicit_non_document_query(query):
            return False
        if not cls._asks_about_document_content(query):
            return False

        query_terms = cls._context_terms(query)
        file_terms = cls._search_tokens(active_document)
        if query_terms & file_terms:
            return True

        if (context.topic or "").lower().startswith("uploaded "):
            return True

        if cls._file_matches_document_type(active_document, "presentation"):
            return True

        return conversation_state.has_domain("kb", within=3) and cls._overlaps_recent_context(
            query,
            context,
        )

    @staticmethod
    def _asks_about_document_content(query: str) -> bool:
        return bool(
            re.search(
                r"\b(what|which|who|where|when|how|list|summari[sz]e|explain|describe|"
                r"tell|show|give|steps?|points?|message|architecture|problem|statement|"
                r"approach|section|slide|title|name|details?|benefits?|risks?)\b",
                query,
            )
        )

    @staticmethod
    def _explicit_non_document_query(query: str) -> bool:
        return bool(
            re.search(
                r"\b(weather|forecast|temperature|gold|stock|price|rate|restaurant|hotel|"
                r"trip|itinerary|goa|cancer|disease|medical|medicine|tablet|tablets|"
                r"cough|cold|fever|exam prep|study tips|career advice)\b",
                query,
            )
        )

    @staticmethod
    def _format_recent_context(context: SessionContext) -> str:
        conversation_state = ConversationState.from_context(context)
        compact = conversation_state.compact_context(max_frames=4)
        if compact:
            return "Recent conversation:\n" + compact + "\n"
        messages = context.recent_messages or []
        if messages:
            relevant = messages[-4:]
            lines = [
                f"{message.get('role', 'unknown').title()}: {message.get('content', '')}"
                for message in relevant
                if message.get("content")
            ]
            return "Recent conversation:\n" + "\n".join(lines) + "\n"
        return (
            f"Previous question: {context.previous_query}\n"
            f"Previous answer: {context.previous_answer or ''}\n"
        )

    @classmethod
    def _format_context_for_query(cls, query: str, context: SessionContext) -> str:
        memory_context = format_memory_context(query, context.memory_snippets)
        recent_context = cls._format_recent_context(context)
        if memory_context:
            return memory_context + recent_context
        return recent_context

    @classmethod
    def _matched_available_file(cls, query: str, context: SessionContext) -> str | None:
        if not context.available_files:
            return None
        query_tokens = cls._search_tokens(query)
        if not query_tokens:
            return None
        active_document = context.last_document_name
        best_overlap, best_files = cls._available_file_title_matches(query_tokens, context)
        if best_overlap < 2:
            return None
        if active_document and active_document in best_files:
            return active_document
        if len(best_files) == 1:
            return best_files[0]
        return None

    @classmethod
    def _has_ambiguous_available_file_match(cls, query: str, context: SessionContext) -> bool:
        if not context.available_files:
            return False
        query_tokens = cls._search_tokens(query)
        if not query_tokens:
            return False
        best_overlap, best_files = cls._available_file_title_matches(query_tokens, context)
        return best_overlap >= 2 and len(best_files) > 1

    @classmethod
    def _available_file_title_matches(
        cls,
        query_tokens: set[str],
        context: SessionContext,
    ) -> tuple[int, list[str]]:
        best_overlap = 0
        best_files: list[str] = []
        for file_name in context.available_files:
            file_tokens = cls._search_tokens(file_name)
            overlap = len(query_tokens & file_tokens)
            if overlap > best_overlap:
                best_files = [file_name]
                best_overlap = overlap
            elif overlap == best_overlap and overlap > 0:
                best_files.append(file_name)
        return best_overlap, best_files

    @classmethod
    def _matched_file_by_document_type(cls, query: str, context: SessionContext) -> str | None:
        if not context.available_files:
            return None
        query_tokens = cls._search_tokens(query)
        candidate_files: list[str] = []
        for canonical, aliases in cls.DOCUMENT_TYPE_TERMS.items():
            if not (query_tokens & aliases or canonical in query_tokens):
                continue
            matching_files = cls._files_matching_document_type(canonical, context, aliases)
            if len(matching_files) == 1:
                candidate_files.extend(matching_files)
        unique_candidates = list(dict.fromkeys(candidate_files))
        return unique_candidates[0] if len(unique_candidates) == 1 else None

    @classmethod
    def _document_type_target(cls, query: str, context: SessionContext) -> str | None:
        requested_document_type = cls._requested_document_type(query)
        if not requested_document_type or not context.available_files:
            return None
        matching_files = cls._files_matching_document_type(requested_document_type, context)
        if len(matching_files) == 1:
            return matching_files[0]
        return f"{cls.DOCUMENT_TYPE_PREFIX}{requested_document_type}"

    @classmethod
    def _requested_document_type(cls, query: str) -> str | None:
        query_tokens = cls._search_tokens(query)
        for canonical, aliases in cls.DOCUMENT_TYPE_TERMS.items():
            if canonical not in cls.DOCUMENT_TYPE_EXTENSIONS:
                continue
            if query_tokens & aliases or canonical in query_tokens:
                return canonical
        return None

    @classmethod
    def _files_matching_document_type(
        cls,
        document_type: str,
        context: SessionContext,
        aliases: set[str] | None = None,
    ) -> list[str]:
        aliases = aliases or cls.DOCUMENT_TYPE_TERMS.get(document_type, set())
        matching_files = []
        for file_name in context.available_files or []:
            file_tokens = cls._search_tokens(file_name)
            if file_tokens & aliases or cls._file_matches_document_type(file_name, document_type):
                matching_files.append(file_name)
        return matching_files

    @classmethod
    def _file_matches_document_type(cls, file_name: str, document_type: str) -> bool:
        suffix = "." + file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
        return suffix in cls.DOCUMENT_TYPE_EXTENSIONS.get(document_type, set())

    @classmethod
    def _search_tokens(cls, text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]{3,}", text.lower())
            if token not in cls.FILE_MATCH_STOPWORDS
        }

    @staticmethod
    def _loose_tokens(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    @classmethod
    def _short_fragment(cls, query: str) -> bool:
        q = query.strip().lower()
        tokens = cls._loose_tokens(q)
        if not tokens or len(tokens) > 4 or len(q) > 48:
            return False
        if re.search(r"[\+\-*/=]", q):
            return False
        if q.startswith(("can you ", "please ", "give me ", "show me ", "tell me ")):
            return False
        return True

    @classmethod
    def _overlaps_recent_context(cls, query: str, context: SessionContext) -> bool:
        query_tokens = cls._loose_tokens(query)
        if not query_tokens:
            return False
        recent_text = cls._recent_context_text(context, max_messages=4)
        context_tokens = cls._loose_tokens(recent_text)
        return bool(query_tokens & context_tokens)

    @classmethod
    def _contextual_completion_follow_up(
        cls,
        query: str,
        context: SessionContext,
        conversation_state: ConversationState,
    ) -> bool:
        q = query.lower().strip()
        if cls._standalone_topic_shift(q):
            return False
        recent_text = cls._recent_context_text(context, max_messages=8)
        memory_text = " ".join(
            f"{snippet.get('user_query', '')} {snippet.get('assistant_answer', '')}"
            for snippet in relevant_memory_snippets(query, context.memory_snippets, limit=3)
        )
        if memory_text:
            recent_text = f"{recent_text} {memory_text}"
        if not recent_text.strip():
            return False

        query_terms = cls._context_terms(q)
        recent_terms = cls._context_terms(recent_text)
        if not query_terms or not recent_terms:
            return False

        overlap = query_terms & recent_terms
        elliptical = cls._elliptical_question(q)
        if overlap and (elliptical or len(query_terms) <= 4):
            return True

        if (
            conversation_state.has_domain("kb", within=4)
            and cls._asks_for_missing_detail(q)
            and len(query_terms) <= 5
        ):
            return True
        return False

    @classmethod
    def _recent_context_text(cls, context: SessionContext, max_messages: int = 6) -> str:
        return " ".join(
            [
                context.previous_query or "",
                context.previous_answer or "",
                " ".join(
                    message.get("content", "")
                    for message in (context.recent_messages or [])[-max_messages:]
                ),
            ]
        )

    @classmethod
    def _context_terms(cls, text: str) -> set[str]:
        terms = set()
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            if token in cls.CONTEXT_STOPWORDS or len(token) < 3:
                continue
            if token.endswith("s") and len(token) > 4:
                token = token[:-1]
            terms.add(token)
        return terms

    @staticmethod
    def _elliptical_question(query: str) -> bool:
        return bool(
            re.match(
                r"^(when|where|which|who|what|why|how|any|is there|are there|do we|can we|"
                r"will it|will they|would it|could it)\b",
                query,
            )
        )

    @staticmethod
    def _asks_for_missing_detail(query: str) -> bool:
        return bool(
            re.search(
                r"\b(any|exact|estimated|expected|specific|rough|approx(?:imate)?|official|when|where|who|which)\b.*"
                r"\b(date|time|timeline|details?|name|person|place|amount|value|reason|status)\b",
                query,
            )
        )

    @classmethod
    def _standalone_topic_shift(cls, query: str) -> bool:
        terms = cls._context_terms(query)
        asks_fresh_fact = bool(
            re.search(r"\b(current|currently|latest|today|now|live|price|rate|weather|news)\b", query)
        )
        has_named_topic = bool(re.search(r"\b[A-Z][A-Za-z0-9-]*\b", query)) or len(terms) >= 3
        return asks_fresh_fact and has_named_topic

    @classmethod
    def _parallel_acronym_follow_up(cls, query: str, context: SessionContext) -> bool:
        tokens = cls._loose_tokens(query)
        if len(tokens) != 1:
            return False
        token = next(iter(tokens))
        if not (2 <= len(token) <= 5):
            return False
        previous = (context.previous_query or "").lower()
        return bool(re.search(r"\b(what is|what's|define|meaning of)\s+[a-z0-9]{1,5}\b", previous))

    @classmethod
    def _term_definition_follow_up(cls, query: str, context: SessionContext) -> str | None:
        if not (context.previous_query or context.previous_answer):
            return None
        q = query.lower().strip()
        q = re.sub(r"[?.!]+$", "", q).strip()
        patterns = [
            r"^(?:what does|what did)\s+([a-z0-9-]{2,30})\s+mean(?:\s+(?:here|in this context|in that context|you meant earlier))?$",
            r"^(?:what\s+i(?:s)?\s+)?([a-z0-9-]{2,30})\s+meaning(?:\s+(?:here|in this context|in that context|you meant earlier))?$",
            r"^(?:what is|what's|what does|define|meaning of)\s+([a-z0-9-]{2,30})(?:\s+(?:here|mean|means|in this context|you meant earlier))?$",
            r"^([a-z0-9-]{2,30})$",
            r"^([a-z0-9-]{2,30})\s+(?:here|in this context)$",
        ]
        for pattern in patterns:
            match = re.match(pattern, q)
            if not match:
                continue
            term = match.group(1).strip()
            if term in {"what", "where", "when", "why", "how", "who", "the", "this", "that"}:
                return None
            context_text = " ".join(
                [
                    context.previous_query or "",
                    context.previous_answer or "",
                    " ".join(
                        message.get("content", "")
                        for message in (context.recent_messages or [])[-6:]
                    ),
                ]
            ).lower()
            if term in context_text or context.last_tool in {"kb_retriever", "general_llm"}:
                return term
        if cls._asks_about_referenced_term(q):
            return cls._latest_salient_term(context)
        return None

    @staticmethod
    def _asks_about_referenced_term(query: str) -> bool:
        return bool(
            re.match(
                r"^(?:what is|what's|what was|explain|define)\s+"
                r"(?:that|this|the)\s+(?:term|word|phrase|acronym)"
                r"(?:\s+(?:here|in this context|in that context|you meant earlier))?$",
                query,
            )
            or re.match(
                r"^(?:what does|what did)\s+"
                r"(?:that|this|it|that term|this term|the term|that word|this word|the word)\s+"
                r"mean(?:\s+(?:here|in this context|in that context|you meant earlier))?$",
                query,
            )
        )

    @classmethod
    def _latest_salient_term(cls, context: SessionContext) -> str | None:
        texts: list[str] = []
        if context.previous_answer:
            texts.append(context.previous_answer)
        for message in reversed(context.recent_messages or []):
            if message.get("role") == "assistant" and message.get("content"):
                content = message["content"]
                if content not in texts:
                    texts.append(content)
        for text in texts:
            term = cls._salient_term_from_text(text)
            if term:
                return term
        return None

    @classmethod
    def _salient_term_from_text(cls, text: str) -> str | None:
        for pattern in [
            r"`([^`\n]{2,40})`",
            r"\bcalled\s+(?:as\s+)?([A-Za-z0-9][A-Za-z0-9-]{1,30})\b",
            r"\bterm\s+([A-Za-z0-9][A-Za-z0-9-]{1,30})\b",
        ]:
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            for candidate in reversed(matches):
                cleaned = cls._clean_term_candidate(candidate)
                if cleaned:
                    return cleaned

        parentheticals = re.findall(r"\(([^()]{2,60})\)", text)
        for candidate in reversed(parentheticals):
            cleaned = cls._clean_term_candidate(candidate)
            if cleaned:
                return cleaned

        acronyms = re.findall(r"\b[A-Z][A-Z0-9-]{1,8}\b", text)
        for candidate in reversed(acronyms):
            cleaned = cls._clean_term_candidate(candidate)
            if cleaned:
                return cleaned
        return None

    @staticmethod
    def _clean_term_candidate(candidate: str) -> str | None:
        candidate = re.sub(r"\s+", " ", candidate).strip(" `\"'.,;:!?")
        lowered = candidate.lower()
        if not candidate or len(candidate) > 40:
            return None
        if lowered in {
            "source",
            "sources",
            "page",
            "version",
            "pdf",
            "http",
            "https",
            "www",
        }:
            return None
        if any(marker in lowered for marker in [".pdf", "page ", "version=", "http", "source:"]):
            return None
        if len(candidate.split()) > 4:
            return None
        return candidate
