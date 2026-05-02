from core.agent import AgentService
from core.router import SessionContext
from core.schemas import AuditEvent, RouteDecision, SourceRef
from core.tools import ToolResult


def test_direct_response_tool_result_bypasses_llm_generation():
    result = ToolResult(
        "RAG means Retrieval-Augmented Generation.",
        0.9,
        audit=[AuditEvent(stage="direct_response", detail="Answered directly.")],
    )

    answer = AgentService._forced_answer({"tool_result": result}, fallback=None)

    assert answer == "RAG means Retrieval-Augmented Generation."


def test_extractive_fallback_answers_beta_term_without_internal_prompt_leak():
    source = SourceRef(
        file_name="DP-800 Developing AI-Enabled Database Solutions (beta) Gunshot Study Plan.pdf",
        page=14,
        excerpt=(
            "Q: Beta exam results timing (general beta policy)? "
            "A: Scores aren't immediate; typically released well after beta ends. "
            "If you're in India and DP-800 is still beta, you may be ineligible to take the exam."
        ),
    )
    result = ToolResult("context", 0.8, sources=[source])
    rewritten = """Use the conversation context to answer this term follow-up.
Term to explain: beta
Task: explain what this term means here. Do not answer the previous question again.
Recent conversation:
User: this exam is being conducted by whom?
Assistant: This exam is being conducted by Microsoft.
Follow-up question: what is beta here"""

    answer = AgentService._extractive_answer("what is beta here", rewritten, result)

    assert "In this context, `beta` is used in this retrieved evidence" in answer
    assert "Scores aren't immediate" in answer
    assert "Follow-up interpreted" not in answer
    assert "Use the conversation context" not in answer
    assert "Microsoft" not in answer


def test_extractive_fallback_current_term_intent_beats_previous_conductor_context():
    source = SourceRef(
        file_name="DP-800 Developing AI-Enabled Database Solutions (beta) Gunshot Study Plan.pdf",
        page=14,
        excerpt=(
            "Microsoft's beta exam policy says candidates located in India are not eligible "
            "to participate in beta certification exams. Beta exam results timing: scores "
            "aren't immediate and are typically released after beta ends."
        ),
    )
    result = ToolResult("context", 0.8, sources=[source])
    rewritten = """Use the conversation context to answer this term follow-up.
Term to explain: beta
Task: explain what this term means here. Do not answer the previous question again.
Recent conversation:
User: this exam is being conducted by whom?
Assistant: The exam is conducted by Microsoft.
Follow-up question: what's beta here"""

    answer = AgentService._extractive_answer("what's beta here", rewritten, result)

    assert "In this context, `beta` is used in this retrieved evidence" in answer
    assert not answer.startswith("The exam is conducted by Microsoft")


def test_extractive_fallback_answers_exam_conductor_directly():
    source = SourceRef(
        file_name="DP-800 Developing AI-Enabled Database Solutions (beta) Gunshot Study Plan.pdf",
        page=1,
        excerpt=(
            "This exam is for DP-800: Developing AI-Enabled Database Solutions (beta), "
            "which aligns to the Microsoft Certified: SQL AI Developer Associate (beta) credential."
        ),
    )
    result = ToolResult("context", 0.8, sources=[source])

    answer = AgentService._extractive_answer(
        "this exam is being conducted by whom?",
        "Use the conversation context to answer this follow-up.\nFollow-up question: this exam is being conducted by whom?",
        result,
    )

    assert "Microsoft" in answer
    assert "Use the conversation context" not in answer


def test_extractive_fallback_explains_generic_term_from_evidence():
    source = SourceRef(
        file_name="DP-800 Developing AI-Enabled Database Solutions (beta) Gunshot Study Plan.pdf",
        page=19,
        excerpt=(
            "Security approaches include Dynamic Data Masking (DDM), Always Encrypted, "
            "and Row-Level Security (RLS) for controlling database access."
        ),
    )
    result = ToolResult("context", 0.8, sources=[source])
    rewritten = """Use the conversation context to answer this term follow-up.
Term to explain: RLS
Task: explain what this term means here. Do not answer the previous question again.
Follow-up question: what does that term mean here"""

    answer = AgentService._extractive_answer("what does that term mean here", rewritten, result)

    assert "`RLS` means Row-Level Security" in answer
    assert "controlling database access" in answer
    assert "Use the conversation context" not in answer


def test_extractive_fallback_answers_beta_result_release_timing_without_exact_date():
    source = SourceRef(
        file_name="DP-800 Developing AI-Enabled Database Solutions (beta) Gunshot Study Plan.pdf",
        page=14,
        excerpt=(
            "Q: Beta exam results timing (general beta policy)? "
            "A: Scores aren't immediate; typically released well after beta ends."
        ),
    )
    result = ToolResult("context", 0.8, sources=[source])
    rewritten = """Use the conversation context to answer this follow-up.
Recent conversation:
User[kb]: beta?
Assistant[kb]: Beta exam results timing says scores are not immediate.
Follow-up question: any estimated or expected date"""

    answer = AgentService._extractive_answer("any estimated or expected date", rewritten, result)

    assert "does not show a specific date" in answer
    assert "Scores aren't immediate" in answer


def test_timing_fallback_uses_contextual_query_and_skips_weak_timeline_page():
    weak = SourceRef(
        file_name="DP-800 Developing AI-Enabled Database Solutions (beta) Gunshot Study Plan.pdf",
        page=1,
        excerpt=(
            "DP-800 Developing AI-Enabled Database Solutions beta study pack with quick links, "
            "timeline, flashcards, and practice questions."
        ),
    )
    relevant = SourceRef(
        file_name="DP-800 Developing AI-Enabled Database Solutions (beta) Gunshot Study Plan.pdf",
        page=99,
        excerpt=(
            "Beta exam results timing: scores are not immediate and are typically released "
            "well after the beta exam ends."
        ),
    )
    result = ToolResult("context", 0.8, sources=[weak, relevant])
    rewritten = """Use the conversation context to answer this follow-up.
Recent conversation:
User[kb]: when will the results get released for this
Assistant[kb]: The results for the DP-800 beta exam are typically released well after the beta exam ends.
Follow-up question: any expected date out?"""

    answer = AgentService._extractive_answer("any expected date out?", rewritten, result)

    assert "page 99" in answer
    assert "does not show a specific date" in answer
    assert "timeline, flashcards" not in answer


class FakePlannerTools:
    def __init__(self):
        self.calls = []

    def excel_calculation(self, query):
        self.calls.append(("excel_calculation", query))
        return ToolResult("Excel calculation SQL:\nSELECT 1\n\nResult:\n1", 0.82)

    def kb_retriever(self, query, target_source=None):
        self.calls.append(("kb_retriever", query, target_source))
        return ToolResult(
            "[1] strategy.pdf page 2\nThe PDF explains the revenue driver.",
            0.8,
            sources=[
                SourceRef(
                    document_id="strategy",
                    file_name="strategy.pdf",
                    page=2,
                    chunk_id="chunk-1",
                    excerpt="The PDF explains the revenue driver.",
                )
            ],
        )

    def web_search(self, query):
        self.calls.append(("web_search", query))
        return ToolResult("- Current market update: demand increased. (https://example.com)", 0.78)


def test_cross_source_route_uses_controlled_planner_steps():
    service = AgentService()
    service.tools = FakePlannerTools()
    route = RouteDecision(
        intent="cross_source",
        sub_intents=["excel", "kb", "web"],
        needs_rag=True,
        needs_web=True,
        tool_name="kb_retriever",
        target_source="strategy.pdf",
        confidence=0.9,
    )
    state = {
        "session_id": "test-session",
        "query": "compare excel with pdf and latest web",
        "rewritten_query": "compare excel with pdf and latest web",
        "context": SessionContext(),
        "route": route,
    }

    service._plan_execute(state)

    assert service.tools.calls == [
        ("excel_calculation", "compare excel with pdf and latest web"),
        ("kb_retriever", "compare excel with pdf and latest web", "strategy.pdf"),
        ("web_search", "compare excel with pdf and latest web"),
    ]
    assert "Controlled multi-step tool plan" in state["tool_result"].answer_context
    assert "Step 1: excel_calculation" in state["tool_result"].answer_context
    assert "Step 2: kb_retriever" in state["tool_result"].answer_context
    assert "Step 3: web_search" in state["tool_result"].answer_context
    assert state["tool_result"].sources[0].file_name == "strategy.pdf"
    assert any(event.stage == "planner" for event in state["tool_result"].audit)


def test_normal_kb_route_does_not_use_multi_step_planner():
    route = RouteDecision(intent="kb_qa", needs_rag=True, tool_name="kb_retriever")

    assert AgentService._should_use_multi_step_planner(route) is False


def test_finalize_answer_preserves_typed_no_evidence_message():
    service = AgentService()
    state = {
        "query": "in ppt i meant",
        "rewritten_query": "in ppt i meant",
        "tool_result": ToolResult(
            "No indexed presentation evidence was found. Upload or sync a .pptx file, then ask again.",
            0.35,
        ),
    }

    answer = service._finalize_answer(
        state,
        "Here is how to create a PowerPoint presentation...",
        "I couldn't find enough evidence to answer confidently.",
    )

    assert "No indexed presentation evidence was found" in answer
    assert "Here is how to create" not in answer
