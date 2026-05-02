from __future__ import annotations

import asyncio
import re
import threading
from typing import TypedDict

from core.conversation_memory import build_memory_snippets
from core.db import session_scope
from core.llm import LLMClient
from core.repository import Repository
from core.router import IntentRouter, SessionContext
from core.schemas import AuditEvent, ChatResponse, RouteDecision, SourceRef
from core.tools import ToolResult, ToolRouter


class AgentState(TypedDict, total=False):
    session_id: str
    query: str
    rewritten_query: str
    context: SessionContext
    route: RouteDecision
    tool_result: ToolResult
    response: ChatResponse


class AgentService:
    def __init__(self) -> None:
        self.router = IntentRouter()
        self.tools = ToolRouter()
        self.llm = LLMClient()
        self.graph = self._build_graph()

    def run(self, session_id: str, query: str) -> ChatResponse:
        result = self.graph.invoke({"session_id": session_id, "query": query})
        return result["response"]

    async def stream(self, session_id: str, query: str):
        state: AgentState = {"session_id": session_id, "query": query}
        try:
            yield self._status_chunk("Reading chat context")
            state = await asyncio.to_thread(self._load_context, state)

            yield self._status_chunk("Understanding request")
            state = await asyncio.to_thread(self._route, state)

            route = state["route"]
            yield self._status_chunk(self._execution_status(route))
            if self._should_use_multi_step_planner(route):
                state = await asyncio.to_thread(self._plan_execute, state)
            else:
                state = await asyncio.to_thread(self._execute, state)
            state = await asyncio.to_thread(self._validate, state)

            prompt, fallback = self._response_prompt(state)
            yield self._status_chunk("Writing answer")
            forced_answer = self._forced_answer(state, fallback)
            if forced_answer:
                yield "[[answer_start]]"
                yield forced_answer
                response = self._chat_response(state, forced_answer, fallback)
                state["response"] = response
                await asyncio.to_thread(self._memory, state)
                yield "\n\n[[metadata]]" + response.model_dump_json()
                return

            answer_parts: list[str] = []
            started = False
            async for chunk in self._stream_llm_answer(
                prompt,
                system=self._response_system(),
                temperature=0.2,
            ):
                if not chunk:
                    continue
                if not started:
                    started = True
                    yield "[[answer_start]]"
                answer_parts.append(chunk)
                yield chunk

            answer = "".join(answer_parts)
            if not answer.strip():
                answer = await asyncio.to_thread(
                    self.llm.complete,
                    prompt,
                    self._response_system(),
                    0.2,
                )
                if not started:
                    yield "[[answer_start]]"
                yield answer

            final_answer = self._finalize_answer(state, answer, fallback)
            response = self._chat_response(state, final_answer, fallback)
            state["response"] = response
            await asyncio.to_thread(self._memory, state)
            yield "\n\n[[metadata]]" + response.model_dump_json()
        except Exception as exc:
            yield f"Request failed: {exc}"

    @staticmethod
    def _status_chunk(status: str) -> str:
        return f"[[status]]{status}\n\n"

    @staticmethod
    def _execution_status(route: RouteDecision) -> str:
        if route.intent == "cross_source":
            return "Running multi-step analysis"
        if route.tool_name == "kb_retriever" or route.needs_rag:
            return "Searching indexed files"
        if route.tool_name == "excel_calculation":
            return "Analyzing spreadsheet data"
        if route.tool_name == "web_search" or route.needs_web:
            return "Searching the web"
        if route.tool_name == "weather_api":
            return "Checking weather"
        return "Thinking"

    async def _stream_llm_answer(self, prompt: str, system: str, temperature: float):
        queue: asyncio.Queue[object] = asyncio.Queue()
        sentinel = object()
        loop = asyncio.get_running_loop()

        def worker() -> None:
            try:
                for chunk in self.llm.stream_complete(
                    prompt,
                    system=system,
                    temperature=temperature,
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        while True:
            item = await queue.get()
            if item is sentinel:
                break
            if isinstance(item, Exception):
                raise item
            yield str(item)

    def _build_graph(self):
        from langgraph.graph import END, StateGraph

        graph = StateGraph(AgentState)
        graph.add_node("load_context", self._load_context)
        graph.add_node("route", self._route)
        graph.add_node("execute", self._execute)
        graph.add_node("plan_execute", self._plan_execute)
        graph.add_node("validate", self._validate)
        graph.add_node("respond", self._respond)
        graph.add_node("memory", self._memory)

        graph.set_entry_point("load_context")
        graph.add_edge("load_context", "route")
        graph.add_conditional_edges(
            "route",
            self._execution_branch,
            {"planner": "plan_execute", "single": "execute"},
        )
        graph.add_edge("execute", "validate")
        graph.add_edge("plan_execute", "validate")
        graph.add_edge("validate", "respond")
        graph.add_edge("respond", "memory")
        graph.add_edge("memory", END)
        return graph.compile()

    def _load_context(self, state: AgentState) -> AgentState:
        with session_scope() as session:
            repo = Repository(session)
            conversation = repo.get_or_create_conversation(state["session_id"])
            history_messages = repo.recent_messages(state["session_id"], limit=80)
            messages = history_messages[-16:]
            documents = repo.list_documents()
            document_names_by_id = {doc.document_id: doc.file_name for doc in documents}
            previous_user = next((m.content for m in reversed(messages) if m.role == "user"), None)
            previous_answer = next((m.content for m in reversed(messages) if m.role == "assistant"), None)
            state["context"] = SessionContext(
                previous_query=previous_user,
                previous_answer=previous_answer,
                recent_messages=[
                    {
                        "role": m.role,
                        "content": m.content,
                        "metadata": m.message_metadata or {},
                    }
                    for m in messages
                ],
                topic=conversation.topic,
                last_tool=conversation.last_tool,
                last_document=conversation.last_document_id,
                last_document_name=document_names_by_id.get(conversation.last_document_id or ""),
                available_files=[doc.file_name for doc in documents],
                active_versions={doc.document_id: doc.active_version_id or "" for doc in documents},
                memory_snippets=build_memory_snippets(history_messages),
            )
        return state

    def _route(self, state: AgentState) -> AgentState:
        route = self.router.route(state["query"], state["context"])
        state["route"] = route
        state["rewritten_query"] = self.router.rewrite_follow_up(state["query"], state["context"], route)
        return state

    @staticmethod
    def _execution_branch(state: AgentState) -> str:
        route = state["route"]
        return "planner" if AgentService._should_use_multi_step_planner(route) else "single"

    @staticmethod
    def _should_use_multi_step_planner(route: RouteDecision) -> bool:
        return route.intent == "cross_source"

    def _execute(self, state: AgentState) -> AgentState:
        route = state["route"]
        if route.intent == "clarify":
            state["tool_result"] = ToolResult(
                "I need a bit more detail to choose the right source or tool.",
                route.confidence,
                audit=[AuditEvent(stage="clarify", detail=route.reason)],
            )
        else:
            try:
                state["tool_result"] = self.tools.execute(
                    route, state["rewritten_query"], state["session_id"]
                )
            except Exception as exc:
                state["tool_result"] = ToolResult(
                    f"The tool '{route.tool_name}' failed: {exc}",
                    0.35,
                    audit=[AuditEvent(stage="tool_error", detail=str(exc))],
                )
        return state

    def _plan_execute(self, state: AgentState) -> AgentState:
        route = state["route"]
        query = state["rewritten_query"]
        plan = self._build_tool_plan(route)
        if not plan:
            return self._execute(state)

        sections = ["Controlled multi-step tool plan:"]
        sources: list[SourceRef] = []
        audit = [
            AuditEvent(
                stage="planner",
                detail="Executing controlled multi-step plan for cross-source request.",
                metadata={"steps": plan, "sub_intents": route.sub_intents},
            )
        ]
        confidences: list[float] = []

        for index, step in enumerate(plan, start=1):
            tool_name = step["tool_name"]
            reason = step["reason"]
            try:
                result = self._execute_planned_tool(tool_name, query, route)
            except Exception as exc:
                result = ToolResult(
                    f"The planned tool '{tool_name}' failed: {exc}",
                    0.35,
                    audit=[AuditEvent(stage="tool_error", detail=str(exc))],
                )

            sections.append(f"\nStep {index}: {tool_name}\nReason: {reason}\n{result.answer_context}")
            sources.extend(result.sources)
            audit.append(
                AuditEvent(
                    stage="planner_step",
                    detail=f"Completed {tool_name}.",
                    metadata={"step": index, "confidence": result.confidence},
                )
            )
            audit.extend(result.audit)
            confidences.append(result.confidence)

        state["tool_result"] = ToolResult(
            "\n".join(sections),
            self._planner_confidence(confidences),
            sources=self._dedupe_sources(sources),
            audit=audit,
            raw={"plan": plan},
        )
        return state

    @staticmethod
    def _build_tool_plan(route: RouteDecision) -> list[dict[str, str]]:
        intents = set(route.sub_intents)
        plan: list[dict[str, str]] = []
        if "excel" in intents:
            plan.append(
                {
                    "tool_name": "excel_calculation",
                    "reason": "Analyze indexed spreadsheet/table data first.",
                }
            )
        if "kb" in intents or route.needs_rag:
            plan.append(
                {
                    "tool_name": "kb_retriever",
                    "reason": "Retrieve indexed document evidence with citations.",
                }
            )
        if "web" in intents or route.needs_web:
            plan.append(
                {
                    "tool_name": "web_search",
                    "reason": "Fetch current external evidence for comparison.",
                }
            )

        deduped: list[dict[str, str]] = []
        seen: set[str] = set()
        for step in plan:
            if step["tool_name"] in seen:
                continue
            seen.add(step["tool_name"])
            deduped.append(step)
        return deduped[:3]

    def _execute_planned_tool(self, tool_name: str, query: str, route: RouteDecision) -> ToolResult:
        if tool_name == "excel_calculation":
            return self.tools.excel_calculation(query)
        if tool_name == "kb_retriever":
            return self.tools.kb_retriever(query, target_source=route.target_source)
        if tool_name == "web_search":
            return self.tools.web_search(query)
        return self.tools.execute(route, query, "")

    @staticmethod
    def _planner_confidence(confidences: list[float]) -> float:
        if not confidences:
            return 0.35
        successful = [confidence for confidence in confidences if confidence >= 0.55]
        if successful:
            return min(0.9, sum(successful) / len(successful))
        return max(confidences)

    @staticmethod
    def _dedupe_sources(sources: list[SourceRef]) -> list[SourceRef]:
        deduped: list[SourceRef] = []
        seen: set[tuple[str | None, str | None, int | None, str | None]] = set()
        for source in sources:
            key = (source.document_id, source.chunk_id, source.page, source.sheet)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(source)
        return deduped

    def _validate(self, state: AgentState) -> AgentState:
        route = state["route"]
        result = state["tool_result"]
        if route.needs_rag and route.intent != "cross_source" and not result.sources:
            result.confidence = min(result.confidence, 0.45)
            result.audit.append(
                AuditEvent(
                    stage="self_correction",
                    detail="RAG route produced no sources; answer will fallback.",
                )
            )
        state["tool_result"] = result
        return state

    def _respond(self, state: AgentState) -> AgentState:
        prompt, fallback = self._response_prompt(state)
        answer = self._forced_answer(state, fallback)
        if answer is None:
            answer = self.llm.complete(
                prompt,
                system=self._response_system(),
                temperature=0.2,
            )
        answer = self._finalize_answer(state, answer, fallback)
        state["response"] = self._chat_response(state, answer, fallback)
        return state

    @staticmethod
    def _response_system() -> str:
        return "You generate explainable, evidence-grounded chatbot answers."

    def _response_prompt(self, state: AgentState) -> tuple[str, str | None]:
        route = state["route"]
        result = state["tool_result"]
        fallback = None
        if route.needs_rag and result.confidence < 0.55 and not result.sources:
            fallback = "I couldn't find enough evidence to answer confidently."

        prompt = f"""
User question: {state['rewritten_query']}
Original user wording: {state['query']}
Route: {route.model_dump()}
Evidence/tool output:
{result.answer_context}

Write a direct answer. If sources exist, use only the evidence/tool output and prefer the highest-ranked evidence blocks first.
If the route does not require external evidence, answer normally from general knowledge and do not mention indexed evidence.
The original user wording is authoritative for the current intent; use recent conversation only to resolve references.
If the current user asks what a term means, explain that term in the current context instead of repeating the previous answer.
For cross-source planner output, synthesize across the step sections and clearly separate indexed document evidence from web/tool evidence.
Cite file/page/sheet for every factual claim that comes from sources.
If the evidence does not contain the requested fact, say that the indexed evidence does not show it instead of guessing.
If fallback is present, be transparent and ask for the missing detail.
Fallback: {fallback}
"""
        return prompt, fallback

    def _finalize_answer(self, state: AgentState, answer: str, fallback: str | None) -> str:
        result = state["tool_result"]
        forced_answer = self._forced_answer(state, fallback)
        if forced_answer:
            return forced_answer
        if self._llm_failed(answer) and result.sources:
            answer = self._extractive_answer(state["query"], state["rewritten_query"], result)
        if fallback and fallback not in answer:
            answer = f"{fallback}\n\n{answer}"
        return answer

    @staticmethod
    def _forced_answer(state: AgentState, fallback: str | None) -> str | None:
        result = state["tool_result"]
        if fallback and not result.sources and result.answer_context.lower().startswith("no indexed"):
            return f"{fallback}\n\n{result.answer_context}"
        if any(event.stage == "direct_response" for event in result.audit) and not result.sources:
            return result.answer_context
        return None

    def _chat_response(
        self,
        state: AgentState,
        answer: str,
        fallback: str | None,
    ) -> ChatResponse:
        route = state["route"]
        result = state["tool_result"]

        audit = [
            AuditEvent(stage="router", detail=route.reason, metadata=route.model_dump()),
            *result.audit,
        ]
        return ChatResponse(
            answer=answer,
            confidence=round(result.confidence, 3),
            sources=result.sources,
            audit_trace=audit,
            route=route,
            fallback=fallback,
            tool_data=result.raw,
        )

    @staticmethod
    def _llm_failed(answer: str) -> bool:
        return answer.startswith("LLM backend is not configured") or "Backend error:" in answer

    @classmethod
    def _extractive_answer(cls, original_query: str, rewritten_query: str, result: ToolResult) -> str:
        direct = cls._evidence_answer(original_query, rewritten_query, result)
        if direct:
            return direct

        lines = ["From the indexed evidence:"]
        for index, source in enumerate(result.sources[:3], start=1):
            location = cls._source_location(source)
            excerpt = cls._clean_excerpt(source.excerpt or "")
            if len(excerpt) > 360:
                excerpt = excerpt[:357].rstrip() + "..."
            lines.append(f"{index}. {source.file_name}, {location}: {excerpt}")
        return "\n".join(lines)

    @classmethod
    def _evidence_answer(
        cls,
        original_query: str,
        rewritten_query: str,
        result: ToolResult,
    ) -> str | None:
        if not result.sources:
            return None

        term_match = re.search(r"(?im)^Term to explain:\s*(.+?)\s*$", rewritten_query)
        term = term_match.group(1).strip() if term_match else ""
        if term:
            return cls._term_answer(term, result.sources)

        query = cls._current_question(original_query, rewritten_query)
        evidence_query = cls._evidence_query(original_query, rewritten_query)
        ranked_sources = cls._rank_sources_for_question(evidence_query, result.sources)
        if not ranked_sources:
            return None

        query_terms = cls._content_terms(evidence_query)
        if cls._asks_for_specific_timing(query):
            return cls._timing_answer(query_terms, ranked_sources)
        if cls._asks_for_entity(query):
            entity_answer = cls._entity_answer(query_terms, ranked_sources)
            if entity_answer:
                return entity_answer

        return cls._closest_evidence_answer(query_terms, ranked_sources)

    @staticmethod
    def _clean_excerpt(excerpt: str) -> str:
        excerpt = re.sub(r"\s+", " ", excerpt).strip()
        excerpt = excerpt.replace("Follow-up interpreted as:", "")
        return excerpt

    @staticmethod
    def _source_location(source: SourceRef) -> str:
        return f"page {source.page}" if source.page else f"sheet {source.sheet}" if source.sheet else "source"

    @classmethod
    def _citation(cls, source: SourceRef | None) -> str:
        if source is None:
            return ""
        return f"({source.file_name}, {cls._source_location(source)})"

    @staticmethod
    def _source_for_term(term: str, sources: list[SourceRef]) -> SourceRef | None:
        term_lower = term.lower()
        return next(
            (source for source in sources if term_lower in (source.excerpt or "").lower()),
            None,
        )

    @classmethod
    def _term_answer(cls, term: str, sources: list[SourceRef]) -> str | None:
        source = cls._source_for_term(term, sources)
        if not source:
            return None
        sentence = cls._sentence_for_term(term, source.excerpt)
        expansion = cls._term_expansion(term, sentence)
        citation = cls._citation(source)
        if expansion:
            return f"In this context, `{term}` means {expansion}. {sentence} {citation}".strip()
        return f"In this context, `{term}` is used in this retrieved evidence: {sentence} {citation}".strip()

    @classmethod
    def _sentence_for_term(cls, term: str, text: str | None) -> str:
        cleaned = cls._clean_excerpt(text or "")
        if not cleaned:
            return ""
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        term_lower = term.lower()
        for index, sentence in enumerate(sentences):
            if term_lower in sentence.lower():
                return cls._with_paired_answer(sentences, index)[:360].strip()
        return cleaned[:360].strip()

    @staticmethod
    def _term_expansion(term: str, sentence: str) -> str | None:
        escaped = re.escape(term)
        patterns = [
            rf"\b([A-Z][A-Za-z0-9 /&-]{{2,80}}?)\s*\(\s*{escaped}\s*\)",
            rf"\b{escaped}\s*\(\s*([A-Za-z][A-Za-z0-9 /&-]{{2,80}}?)\s*\)",
            rf"\b{escaped}\s+stands for\s+([^.;,\n]{{2,100}})",
            rf"\b{escaped}\s+means\s+([^.;,\n]{{2,100}})",
        ]
        for pattern in patterns:
            match = re.search(pattern, sentence, flags=re.IGNORECASE)
            if match:
                expansion = re.sub(r"\s+", " ", match.group(1)).strip(" .,:;")
                return re.sub(r"^(?:and|or)\s+", "", expansion, flags=re.IGNORECASE)
        return None

    @staticmethod
    def _current_question(original_query: str, rewritten_query: str) -> str:
        if "Follow-up question:" in rewritten_query:
            return rewritten_query.rsplit("Follow-up question:", 1)[1].splitlines()[0].strip()
        return original_query.strip()

    @classmethod
    def _evidence_query(cls, original_query: str, rewritten_query: str) -> str:
        current = cls._current_question(original_query, rewritten_query)
        assistant_matches = re.findall(
            r"Assistant(?:\[[^\]]+\])?:\s*(.*?)(?=\n(?:User|Assistant)(?:\[[^\]]+\])?:|\nFollow-up question:|\Z)",
            rewritten_query,
            flags=re.DOTALL,
        )
        previous_answer = re.sub(r"\s+", " ", assistant_matches[-1]).strip() if assistant_matches else ""
        return f"{current} {previous_answer}".strip()

    @classmethod
    def _rank_sources_for_question(cls, query: str, sources: list[SourceRef]) -> list[SourceRef]:
        terms = cls._content_terms(query)
        if not terms:
            return sources[:3]
        return sorted(
            sources,
            key=lambda source: (
                len(terms & cls._content_terms(source.excerpt or "")),
                source.score or 0,
            ),
            reverse=True,
        )

    @staticmethod
    def _content_terms(text: str) -> set[str]:
        stopwords = {
            "a",
            "about",
            "again",
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
        terms = set()
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            if token in stopwords or len(token) < 3:
                continue
            if token.endswith("s") and len(token) > 4:
                token = token[:-1]
            terms.add(token)
        return terms

    @staticmethod
    def _asks_for_specific_timing(query: str) -> bool:
        return bool(
            re.search(
                r"\b(when|date|timeline|time|expected|estimated|release|released|available|out)\b",
                query.lower(),
            )
        )

    @staticmethod
    def _asks_for_entity(query: str) -> bool:
        return bool(
            re.search(
                r"\b(who|whom|which|what is the name|what's the name|provider|issuer|conducted by)\b",
                query.lower(),
            )
        )

    @classmethod
    def _timing_answer(cls, query_terms: set[str], sources: list[SourceRef]) -> str | None:
        for source in sources:
            excerpt = cls._clean_excerpt(source.excerpt or "")
            sentence = cls._best_sentence(query_terms, excerpt)
            if not sentence:
                continue
            overlap = query_terms & cls._content_terms(sentence)
            if not overlap and not cls._has_specific_date(sentence):
                continue
            citation = cls._citation(source)
            if cls._has_specific_date(sentence):
                return f"The indexed evidence gives this timing: {sentence} {citation}".strip()
            if re.search(
                r"\b(no|not|without|after|before|later|well after|pending|typically|usually|expected|estimated)\b",
                sentence,
                flags=re.IGNORECASE,
            ):
                return (
                    "The indexed evidence does not show a specific date. "
                    f"The closest timing evidence says: {sentence} {citation}"
                ).strip()
            return f"The closest timing evidence says: {sentence} {citation}".strip()
        return None

    @staticmethod
    def _has_specific_date(text: str) -> bool:
        return bool(
            re.search(
                r"\b(?:\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|\d{4}|"
                r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
                r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
                r"nov(?:ember)?|dec(?:ember)?)\b",
                text,
                flags=re.IGNORECASE,
            )
        )

    @classmethod
    def _entity_answer(cls, query_terms: set[str], sources: list[SourceRef]) -> str | None:
        for source in sources:
            excerpt = cls._clean_excerpt(source.excerpt or "")
            sentence = cls._best_sentence(query_terms, excerpt)
            if not sentence:
                continue
            entity = cls._entity_candidate(sentence)
            if entity:
                citation = cls._citation(source)
                return f"The indexed evidence points to {entity}. {sentence} {citation}".strip()
        return None

    @staticmethod
    def _entity_candidate(sentence: str) -> str | None:
        patterns = [
            r"\b([A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*){0,5})\s+(?:Certified|certification|credential|exam|university|college|institute|institution)\b",
            r"\b(?:by|from|provider|issuer|conducted by|offered by)\s+([A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*){0,5})\b",
            r"\b([A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*){1,5})\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, sentence)
            if match:
                candidate = re.sub(r"\s+", " ", match.group(1)).strip(" .,:;")
                if len(candidate) > 1 and not candidate.lower().startswith(("this ", "the ")):
                    return candidate
        return None

    @classmethod
    def _closest_evidence_answer(cls, query_terms: set[str], sources: list[SourceRef]) -> str | None:
        lines = []
        for source in sources[:3]:
            sentence = cls._best_sentence(query_terms, source.excerpt or "")
            if sentence:
                lines.append(f"- {sentence} {cls._citation(source)}")
        if not lines:
            return None
        return "The closest indexed evidence says:\n" + "\n".join(lines)

    @classmethod
    def _best_sentence(cls, query_terms: set[str], text: str | None) -> str:
        cleaned = cls._clean_excerpt(text or "")
        if not cleaned:
            return ""
        sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", cleaned) if sentence.strip()]
        if not sentences:
            return cleaned[:360].strip()
        if not query_terms:
            return sentences[0][:360].strip()
        best_index = max(
            range(len(sentences)),
            key=lambda index: len(query_terms & cls._content_terms(sentences[index])),
        )
        return cls._with_paired_answer(sentences, best_index)[:360].strip()

    @staticmethod
    def _with_paired_answer(sentences: list[str], index: int) -> str:
        sentence = sentences[index].strip()
        if re.match(r"^(?:q|question)\s*:", sentence, flags=re.IGNORECASE) and index + 1 < len(sentences):
            next_sentence = sentences[index + 1].strip()
            if re.match(r"^(?:a|answer)\s*:", next_sentence, flags=re.IGNORECASE):
                return f"{sentence} {next_sentence}"
        return sentence

    def _memory(self, state: AgentState) -> AgentState:
        response = state["response"]
        first_source: SourceRef | None = response.sources[0] if response.sources else None
        with session_scope() as session:
            repo = Repository(session)
            repo.add_message(state["session_id"], "user", state["query"])
            repo.add_message(
                state["session_id"],
                "assistant",
                response.answer,
                {
                    "route": response.route.model_dump(),
                    "confidence": response.confidence,
                    "sources": [source.model_dump() for source in response.sources],
                    "fallback": response.fallback,
                    "audit_trace": [event.model_dump(mode="json") for event in response.audit_trace],
                    "tool_data": response.tool_data,
                },
            )
            repo.update_memory(
                state["session_id"],
                topic=state["query"][:240] if not response.route.is_follow_up else None,
                last_tool=response.route.tool_name,
                last_document_id=first_source.document_id if first_source else None,
            )
            repo.add_tool_run(
                state["session_id"],
                response.route.tool_name,
                state["query"],
                response.answer,
            )
            repo.add_source_traces(state["session_id"], response.sources)
        return state
