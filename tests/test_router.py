from core.conversation_memory import build_memory_snippets
from core.router import IntentRouter, SessionContext


def test_weather_routes_to_weather_tool():
    route = IntentRouter().route("what is the weather in Mumbai", SessionContext())

    assert route.intent == "weather"
    assert route.tool_name == "weather_api"
    assert route.confidence > 0.85


def test_excel_routes_to_calculation_tool():
    route = IntentRouter().route("sum the revenue column in the Excel sheet", SessionContext())

    assert route.intent == "excel_analysis"
    assert route.tool_name == "excel_calculation"
    assert route.needs_rag is True


def test_spreadsheet_file_name_with_calculation_routes_to_excel_not_kb():
    route = IntentRouter().route(
        (
            "Using complex_multi_sheet_sales_messy.xlsx, calculate net gross margin by "
            "normalized region and product category for Closed orders only. Use Orders + "
            "Product Master + Returns and flag unmatched SKU values."
        ),
        SessionContext(available_files=["complex_multi_sheet_sales_messy.xlsx"]),
    )

    assert route.intent == "excel_analysis"
    assert route.tool_name == "excel_calculation"
    assert route.needs_rag is True
    assert route.target_source == "complex_multi_sheet_sales_messy.xlsx"


def test_rag_grounding_question_stays_general_even_with_uploaded_excel():
    route = IntentRouter().route(
        "how are answers grounded inn rag",
        SessionContext(available_files=["complex_multi_sheet_sales_messy.xlsx"]),
    )

    assert route.intent == "general_qa"
    assert route.tool_name == "general_llm"
    assert route.needs_rag is False
    assert route.confidence >= 0.85


def test_missing_files_validates_rag_to_clarify():
    route = IntentRouter().route("what does the document say about payment terms", SessionContext())

    assert route.intent == "clarify"
    assert route.confidence < 0.6


def test_follow_up_detection():
    route = IntentRouter().route(
        "what about the next sheet",
        SessionContext(previous_query="show revenue", topic="revenue report", available_files=["a.xlsx"]),
    )

    assert route.is_follow_up is True


def test_available_file_title_routes_to_kb():
    route = IntentRouter().route(
        "what are the problem statements in Hero Campus Challenge",
        SessionContext(available_files=["Hero Campus Challenge Engineering Case.pdf"]),
    )

    assert route.intent == "kb_qa"
    assert route.tool_name == "kb_retriever"
    assert route.needs_rag is True
    assert route.target_source == "Hero Campus Challenge Engineering Case.pdf"


def test_certificate_query_targets_unique_certificate_file():
    route = IntentRouter().route(
        "what's the college name for this certificate provider",
        SessionContext(
            available_files=[
                "degree_certificate_2025.pdf",
                "AAVA_eSLM_Hybrid_SLM_Java_Documentation.pdf",
            ]
        ),
    )

    assert route.intent == "kb_qa"
    assert route.tool_name == "kb_retriever"
    assert route.target_source == "degree_certificate_2025.pdf"


def test_invoice_query_targets_unique_invoice_file():
    route = IntentRouter().route(
        "what is the invoice total",
        SessionContext(
            available_files=[
                "vendor_invoice_april.pdf",
                "engineering_notes.pdf",
            ]
        ),
    )

    assert route.intent == "kb_qa"
    assert route.tool_name == "kb_retriever"
    assert route.target_source == "vendor_invoice_april.pdf"


def test_current_uploaded_document_query_targets_last_document():
    route = IntentRouter().route(
        "summarize this document",
        SessionContext(
            last_document_name="annual_report_2026.pdf",
            available_files=[
                "annual_report_2026.pdf",
                "AAVA_eSLM_Hybrid_SLM_Java_Documentation.pdf",
            ],
        ),
    )

    assert route.intent == "kb_qa"
    assert route.tool_name == "kb_retriever"
    assert route.target_source == "annual_report_2026.pdf"


def test_explicit_ppt_scope_does_not_fall_back_to_pdf_when_no_ppt_is_indexed():
    route = IntentRouter().route(
        "in ppt i meant",
        SessionContext(
            previous_query="what are 4 steps in gxs nexus one empathic message",
            previous_answer="The closest evidence came from GXS Nexus Deep Research Review and Improvement Plan.pdf.",
            last_tool="kb_retriever",
            last_document_name="GXS Nexus Deep Research Review and Improvement Plan.pdf",
            available_files=["GXS Nexus Deep Research Review and Improvement Plan.pdf"],
        ),
    )

    assert route.intent == "kb_qa"
    assert route.tool_name == "kb_retriever"
    assert route.target_source == "__document_type__:presentation"


def test_explicit_presentation_query_targets_unique_pptx_by_extension():
    route = IntentRouter().route(
        "what are the 4 steps in the presentation",
        SessionContext(
            available_files=[
                "GXS Nexus Deep Research Review and Improvement Plan.pdf",
                "GXS Nexus Pitch Deck.pptx",
            ],
        ),
    )

    assert route.intent == "kb_qa"
    assert route.tool_name == "kb_retriever"
    assert route.target_source == "GXS Nexus Pitch Deck.pptx"


def test_active_ppt_document_query_targets_ppt_without_saying_ppt():
    route = IntentRouter().route(
        "what are 4 steps in gxs nexus one empathic message",
        SessionContext(
            topic="Uploaded GXS Nexus Pitch Deck.pptx",
            last_document_name="GXS Nexus Pitch Deck.pptx",
            available_files=[
                "GXS Nexus Deep Research Review and Improvement Plan.pdf",
                "GXS Nexus Pitch Deck.pptx",
            ],
        ),
    )

    assert route.intent == "kb_qa"
    assert route.tool_name == "kb_retriever"
    assert route.target_source == "GXS Nexus Pitch Deck.pptx"


def test_active_document_overlap_targets_current_file_over_global_kb():
    route = IntentRouter().route(
        "explain gxs nexus empathic message",
        SessionContext(
            last_document_name="GXS Nexus Pitch Deck.pptx",
            available_files=[
                "GXS Nexus Deep Research Review and Improvement Plan.pdf",
                "GXS Nexus Pitch Deck.pptx",
            ],
        ),
    )

    assert route.intent == "kb_qa"
    assert route.tool_name == "kb_retriever"
    assert route.target_source == "GXS Nexus Pitch Deck.pptx"


def test_ambiguous_gxs_title_match_searches_kb_without_locking_to_first_file():
    route = IntentRouter().route(
        "what are 4 steps in gxs nexus one empathic message",
        SessionContext(
            available_files=[
                "GXS Nexus Deep Research Review and Improvement Plan.pdf",
                "GXS Nexus Pitch Deck.pptx",
            ],
        ),
    )

    assert route.intent == "kb_qa"
    assert route.tool_name == "kb_retriever"
    assert route.target_source is None
    assert "multiple indexed file names" in route.reason


def test_active_document_non_document_query_does_not_force_kb():
    route = IntentRouter().route(
        "can you help me with some exam prep tips",
        SessionContext(
            topic="Uploaded GXS Nexus Pitch Deck.pptx",
            last_document_name="GXS Nexus Pitch Deck.pptx",
            available_files=["GXS Nexus Pitch Deck.pptx"],
        ),
    )

    assert route.tool_name == "general_llm"


def test_this_pdf_query_keeps_current_pdf_when_document_type_matches():
    route = IntentRouter().route(
        "summarize this pdf",
        SessionContext(
            last_document_name="GXS Nexus Deep Research Review and Improvement Plan.pdf",
            available_files=[
                "GXS Nexus Deep Research Review and Improvement Plan.pdf",
                "Other Notes.pdf",
            ],
        ),
    )

    assert route.intent == "kb_qa"
    assert route.tool_name == "kb_retriever"
    assert route.target_source == "GXS Nexus Deep Research Review and Improvement Plan.pdf"


def test_specific_certificate_query_targets_matching_certificate_file():
    route = IntentRouter().route(
        "what's the college name for this degree certificate provider",
        SessionContext(
            available_files=[
                "degree_certificate_2025.pdf",
                "AAVA_eSLM_Hybrid_SLM_Java_Documentation.pdf",
            ]
        ),
    )

    assert route.intent == "kb_qa"
    assert route.tool_name == "kb_retriever"
    assert route.target_source == "degree_certificate_2025.pdf"


def test_numbered_follow_up_routes_to_referenced_kb_item():
    previous_answer = """
1. "An eco-friendly alternate fuel/powertrain/safety solution for ICE-based 2-wheelers" (Hero Campus Challenge Engineering Case.pdf, page 2)
2. "AI-Driven Image-Based Vehicle Damage Detection & Intelligent Assessment" (Hero Campus Challenge Engineering Case.pdf, page 3)
"""
    context = SessionContext(
        previous_query="what are the problem statements",
        previous_answer=previous_answer,
        recent_messages=[{"role": "assistant", "content": previous_answer}],
        last_tool="kb_retriever",
        last_document_name="Hero Campus Challenge Engineering Case.pdf",
        available_files=["Hero Campus Challenge Engineering Case.pdf"],
    )

    router = IntentRouter()
    route = router.route("tell me about 2", context)
    rewritten = router.rewrite_follow_up("tell me about 2", context, route)

    assert route.intent == "kb_qa"
    assert route.tool_name == "kb_retriever"
    assert route.is_follow_up is True
    assert route.target_source == "Hero Campus Challenge Engineering Case.pdf"
    assert "AI-Driven Image-Based Vehicle Damage Detection" in rewritten


def test_numbered_follow_up_keeps_last_document_over_generic_filename_match():
    previous_answer = """
1. AI-Driven Image-Based Vehicle Damage Detection & Intelligent Assessment.
2. An eco-friendly alternate fuel/powertrain/safety solution for ICE-based 2-wheelers.
"""
    route = IntentRouter().route(
        "tell me about 2",
        SessionContext(
            previous_answer=previous_answer,
            recent_messages=[{"role": "assistant", "content": previous_answer}],
            last_tool="kb_retriever",
            last_document_name="Hero Campus Challenge Engineering Case.pdf",
            available_files=[
                "Hero Campus Challenge Engineering Case.pdf",
                "DP-800 Developing AI-Enabled Database Solutions Study Plan.pdf",
            ],
        ),
    )

    assert route.target_source == "Hero Campus Challenge Engineering Case.pdf"


def test_math_query_is_not_numbered_follow_up():
    route = IntentRouter().route(
        "what is 2+2",
        SessionContext(
            previous_answer="1. First item\n2. Second item",
            available_files=["Hero Campus Challenge Engineering Case.pdf"],
        ),
    )

    assert route.is_follow_up is False


def test_pronoun_follow_up_after_kb_answer_routes_to_kb():
    route = IntentRouter().route(
        "what problem does it solve",
        SessionContext(
            previous_query="tell me about 2",
            previous_answer="The second problem statement is an eco-friendly alternate fuel solution.",
            last_tool="kb_retriever",
            last_document_name="Hero Campus Challenge Engineering Case.pdf",
            available_files=["Hero Campus Challenge Engineering Case.pdf"],
        ),
    )

    assert route.intent == "kb_qa"
    assert route.tool_name == "kb_retriever"
    assert route.is_follow_up is True
    assert route.is_topic_shift is False
    assert route.target_source == "Hero Campus Challenge Engineering Case.pdf"


def test_unrelated_exam_prep_question_does_not_follow_hero_context():
    route = IntentRouter().route(
        "can you help me with some exam prep tips",
        SessionContext(
            previous_query="what data would be needed for that",
            previous_answer="For that vehicle problem, you would need image data.",
            last_tool="kb_retriever",
            last_document_name="Hero Campus Challenge Engineering Case.pdf",
            available_files=[
                "Hero Campus Challenge Engineering Case.pdf",
                "DP-800 Developing AI-Enabled Database Solutions Study Plan.pdf",
            ],
        ),
    )

    assert route.is_follow_up is False
    assert route.tool_name == "general_llm"


def test_weather_location_correction_routes_to_weather():
    route = IntentRouter().route(
        "i meant in india",
        SessionContext(
            previous_query="what's the weather today",
            previous_answer="Weather for London: 12 C, clear sky.",
            last_tool="weather_api",
        ),
    )

    assert route.intent == "weather"
    assert route.tool_name == "weather_api"
    assert route.is_follow_up is True


def test_weather_city_shorthand_follow_up_routes_to_weather():
    route = IntentRouter().route(
        "hyd",
        SessionContext(
            previous_query="weather in Bangalore",
            previous_answer="Weather for Bangalore, IN: 33 C, scattered clouds.",
            last_tool="weather_api",
        ),
    )

    assert route.intent == "weather"
    assert route.tool_name == "weather_api"
    assert route.is_follow_up is True


def test_weather_city_shorthand_uses_recent_weather_even_after_general_follow_up():
    route = IntentRouter().route(
        "bnglr",
        SessionContext(
            previous_query="what is c",
            previous_answer="In weather context, C means Celsius.",
            recent_messages=[
                {"role": "user", "content": "in bangalore"},
                {"role": "assistant", "content": "Weather for Bangalore, IN: 33 C."},
                {"role": "user", "content": "what is c"},
                {"role": "assistant", "content": "In weather context, C means Celsius."},
            ],
            last_tool="general_llm",
        ),
    )

    assert route.intent == "weather"
    assert route.tool_name == "weather_api"
    assert route.is_follow_up is True


def test_goa_itinerary_restaurant_follow_up_uses_memory_after_topic_shift():
    context = SessionContext(
        previous_query="what is that c",
        previous_answer='In the context of temperature, "C" stands for Celsius.',
        recent_messages=[
            {"role": "user", "content": "what's the weather today?"},
            {
                "role": "assistant",
                "content": "Please provide the city.",
                "metadata": {"route": {"tool_name": "general_llm", "intent": "general_qa"}},
            },
            {"role": "user", "content": "bnglr"},
            {
                "role": "assistant",
                "content": "The current weather in Bengaluru is 30 C.",
                "metadata": {"route": {"tool_name": "weather_api", "intent": "weather"}},
            },
            {"role": "user", "content": "what is that c"},
            {
                "role": "assistant",
                "content": 'In the context of temperature, "C" stands for Celsius.',
                "metadata": {"route": {"tool_name": "general_llm", "intent": "general_qa"}},
            },
        ],
        last_tool="general_llm",
        memory_snippets=[
            {
                "label": "travel",
                "user_query": "why don't you plan for 3 days my entire itinerary",
                "assistant_answer": (
                    "Day 1: North Goa. Day 2: South Goa with Palolem, Old Goa, "
                    "and a seafood dinner. Day 3: Dudhsagar and wellness."
                ),
                "terms": [
                    "day",
                    "dinner",
                    "goa",
                    "itinerary",
                    "palolem",
                    "seafood",
                    "south",
                ],
            }
        ],
    )

    router = IntentRouter()
    route = router.route("which restaurant you recommend for day 2 evening time", context)
    rewritten = router.rewrite_follow_up(
        "which restaurant you recommend for day 2 evening time",
        context,
        route,
    )

    assert route.intent == "web_search"
    assert route.tool_name == "web_search"
    assert route.is_follow_up is True
    assert "Relevant earlier context" in rewritten
    assert "Day 2: South Goa" in rewritten


def test_weather_unit_question_after_weather_is_general_follow_up():
    route = IntentRouter().route(
        "what is c",
        SessionContext(
            previous_query="in bangalore",
            previous_answer="Weather for Bangalore, IN: 33 C.",
            last_tool="weather_api",
        ),
    )

    assert route.intent == "general_qa"
    assert route.tool_name == "general_llm"
    assert route.is_follow_up is True


def test_short_general_follow_up_uses_recent_context():
    context = SessionContext(
        previous_query="this exam is being conducted by whom?",
        previous_answer="The exam is DP-800 Developing AI-Enabled Database Solutions (beta).",
        recent_messages=[
            {"role": "user", "content": "this exam is being conducted by whom?"},
            {
                "role": "assistant",
                "content": "The exam is DP-800 Developing AI-Enabled Database Solutions (beta).",
            },
        ],
        last_tool="kb_retriever",
        available_files=["DP-800 Developing AI-Enabled Database Solutions Study Plan.pdf"],
    )

    router = IntentRouter()
    route = router.route("beta?", context)
    rewritten = router.rewrite_follow_up("beta?", context, route)

    assert route.is_follow_up is True
    assert "DP-800" in rewritten
    assert "Term to explain: beta" in rewritten
    assert "Do not answer the previous question again" in rewritten
    assert "Follow-up question: beta?" in rewritten


def test_term_definition_follow_up_with_context_phrase_is_explicit():
    context = SessionContext(
        previous_query="this exam is being conducted by whom?",
        previous_answer="The exam is DP-800 Developing AI-Enabled Database Solutions (beta).",
        recent_messages=[
            {"role": "user", "content": "this exam is being conducted by whom?"},
            {
                "role": "assistant",
                "content": "The exam is DP-800 Developing AI-Enabled Database Solutions (beta).",
            },
        ],
        last_tool="kb_retriever",
        available_files=["DP-800 Developing AI-Enabled Database Solutions Study Plan.pdf"],
    )

    router = IntentRouter()
    route = router.route("what is beta here", context)
    rewritten = router.rewrite_follow_up("what is beta here", context, route)

    assert route.is_follow_up is True
    assert "Term to explain: beta" in rewritten
    assert "Do not answer the previous question again" in rewritten
    assert "Follow-up question: what is beta here" in rewritten


def test_term_definition_follow_up_supports_rough_meaning_phrase():
    context = SessionContext(
        previous_query="this exam is being conducted by whom?",
        previous_answer="The exam is DP-800 Developing AI-Enabled Database Solutions (beta).",
        recent_messages=[
            {
                "role": "assistant",
                "content": "The exam is DP-800 Developing AI-Enabled Database Solutions (beta).",
            },
        ],
        last_tool="kb_retriever",
        available_files=["DP-800 Developing AI-Enabled Database Solutions Study Plan.pdf"],
    )

    router = IntentRouter()
    route = router.route("what i beta meaning", context)
    rewritten = router.rewrite_follow_up("what i beta meaning", context, route)

    assert route.is_follow_up is True
    assert route.tool_name == "kb_retriever"
    assert "Term to explain: beta" in rewritten


def test_term_definition_follow_up_supports_what_does_term_mean_phrase():
    context = SessionContext(
        previous_query="give some AI-enabled database solution approaches",
        previous_answer="Vector search uses embeddings to find semantically similar records.",
        last_tool="kb_retriever",
        available_files=["DP-800 Developing AI-Enabled Database Solutions Study Plan.pdf"],
    )

    router = IntentRouter()
    route = router.route("what does vector mean in this context", context)
    rewritten = router.rewrite_follow_up("what does vector mean in this context", context, route)

    assert route.is_follow_up is True
    assert route.tool_name == "kb_retriever"
    assert "Term to explain: vector" in rewritten


def test_referenced_term_follow_up_infers_salient_recent_term():
    context = SessionContext(
        previous_query="this exam is being conducted by whom?",
        previous_answer="The exam is DP-800 Developing AI-Enabled Database Solutions (beta).",
        recent_messages=[
            {
                "role": "assistant",
                "content": "The exam is DP-800 Developing AI-Enabled Database Solutions (beta).",
            },
        ],
        last_tool="kb_retriever",
        available_files=["DP-800 Developing AI-Enabled Database Solutions Study Plan.pdf"],
    )

    router = IntentRouter()
    route = router.route("what does that term mean here", context)
    rewritten = router.rewrite_follow_up("what does that term mean here", context, route)

    assert route.is_follow_up is True
    assert "Term to explain: beta" in rewritten
    assert "Follow-up question: what does that term mean here" in rewritten


def test_results_release_question_continues_beta_exam_context():
    context = SessionContext(
        previous_query="beta?",
        previous_answer=(
            "The term beta means the DP-800 exam is still in beta. "
            "Beta exam results timing says scores are not immediate."
        ),
        recent_messages=[
            {"role": "user", "content": "beta?"},
            {
                "role": "assistant",
                "content": (
                    "The term beta means the DP-800 exam is still in beta. "
                    "Beta exam results timing says scores are not immediate."
                ),
                "metadata": {
                    "route": {"tool_name": "kb_retriever", "intent": "kb_qa"},
                    "sources": [{"file_name": "DP-800 Developing AI-Enabled Database Solutions Study Plan.pdf"}],
                },
            },
        ],
        last_tool="kb_retriever",
        last_document_name="DP-800 Developing AI-Enabled Database Solutions Study Plan.pdf",
        available_files=["DP-800 Developing AI-Enabled Database Solutions Study Plan.pdf"],
    )

    router = IntentRouter()
    route = router.route("when will be the results be out", context)
    rewritten = router.rewrite_follow_up("when will be the results be out", context, route)

    assert route.intent == "kb_qa"
    assert route.tool_name == "kb_retriever"
    assert route.is_follow_up is True
    assert route.target_source == "DP-800 Developing AI-Enabled Database Solutions Study Plan.pdf"
    assert "Follow-up question: when will be the results be out" in rewritten


def test_expected_date_question_continues_results_release_context():
    context = SessionContext(
        previous_query="when will the results get released for this",
        previous_answer=(
            "The results for the DP-800 beta exam are typically released well after "
            "the beta exam ends."
        ),
        recent_messages=[
            {"role": "user", "content": "beta?"},
            {
                "role": "assistant",
                "content": "Beta exam results timing says scores are not immediate.",
                "metadata": {
                    "route": {"tool_name": "kb_retriever", "intent": "kb_qa"},
                    "sources": [{"file_name": "DP-800 Developing AI-Enabled Database Solutions Study Plan.pdf"}],
                },
            },
            {"role": "user", "content": "when will the results get released for this"},
            {
                "role": "assistant",
                "content": (
                    "The results for the DP-800 beta exam are typically released well after "
                    "the beta exam ends."
                ),
                "metadata": {
                    "route": {"tool_name": "kb_retriever", "intent": "kb_qa"},
                    "sources": [{"file_name": "DP-800 Developing AI-Enabled Database Solutions Study Plan.pdf"}],
                },
            },
        ],
        last_tool="kb_retriever",
        last_document_name="DP-800 Developing AI-Enabled Database Solutions Study Plan.pdf",
        available_files=["DP-800 Developing AI-Enabled Database Solutions Study Plan.pdf"],
    )

    router = IntentRouter()
    route = router.route("any estimated or expected date", context)
    rewritten = router.rewrite_follow_up("any estimated or expected date", context, route)

    assert route.intent == "kb_qa"
    assert route.tool_name == "kb_retriever"
    assert route.is_follow_up is True
    assert "Follow-up question: any estimated or expected date" in rewritten


def test_find_it_online_routes_to_web_search_from_kb_context():
    context = SessionContext(
        previous_query="i meant the exam official results release date anything available like the month",
        previous_answer=(
            "The results for the DP-800 beta exam are typically released well after "
            "the beta exam ends."
        ),
        recent_messages=[
            {"role": "user", "content": "i meant the exam official results release date anything available like the month"},
            {
                "role": "assistant",
                "content": (
                    "The results for the DP-800 beta exam are typically released well after "
                    "the beta exam ends."
                ),
                "metadata": {
                    "route": {"tool_name": "kb_retriever", "intent": "kb_qa"},
                    "sources": [{"file_name": "DP-800 Developing AI-Enabled Database Solutions Study Plan.pdf"}],
                },
            },
        ],
        last_tool="kb_retriever",
        last_document_name="DP-800 Developing AI-Enabled Database Solutions Study Plan.pdf",
        available_files=["DP-800 Developing AI-Enabled Database Solutions Study Plan.pdf"],
    )

    router = IntentRouter()
    route = router.route("find it online", context)
    rewritten = router.rewrite_follow_up("find it online", context, route)

    assert route.intent == "web_search"
    assert route.tool_name == "web_search"
    assert route.needs_web is True
    assert route.is_follow_up is True
    assert "Follow-up question: find it online" in rewritten

    official_route = router.route("find official release date online", context)
    assert official_route.intent == "web_search"
    assert official_route.tool_name == "web_search"
    assert official_route.is_follow_up is True


def test_missing_exact_date_follow_up_escalates_to_web_without_web_keyword():
    context = SessionContext(
        previous_query="when will the results be out",
        previous_answer=(
            "The indexed evidence does not show the exact month. It only says the "
            "official results are typically released well after the beta exam ends."
        ),
        recent_messages=[
            {"role": "user", "content": "when will the results be out"},
            {
                "role": "assistant",
                "content": (
                    "The indexed evidence does not show the exact month. It only says the "
                    "official results are typically released well after the beta exam ends."
                ),
                "metadata": {
                    "route": {"tool_name": "kb_retriever", "intent": "kb_qa"},
                    "sources": [{"file_name": "DP-800 Developing AI-Enabled Database Solutions Study Plan.pdf"}],
                },
            },
        ],
        last_tool="kb_retriever",
        last_document_name="DP-800 Developing AI-Enabled Database Solutions Study Plan.pdf",
        available_files=["DP-800 Developing AI-Enabled Database Solutions Study Plan.pdf"],
    )

    router = IntentRouter()
    for query in [
        "any expected date?",
        "i meant the exam official results release date anything available like the month",
        "when is that when will it end",
    ]:
        route = router.route(query, context)
        assert route.intent == "web_search"
        assert route.tool_name == "web_search"
        assert route.needs_web is True
        assert route.is_follow_up is True


def test_unrelated_current_topic_after_beta_context_is_not_kb_follow_up():
    route = IntentRouter().route(
        "what is gold rate currently",
        SessionContext(
            previous_query="beta?",
            previous_answer="Beta exam results timing says scores are not immediate.",
            last_tool="kb_retriever",
            last_document_name="DP-800 Developing AI-Enabled Database Solutions Study Plan.pdf",
            available_files=["DP-800 Developing AI-Enabled Database Solutions Study Plan.pdf"],
        ),
    )

    assert route.is_follow_up is False
    assert route.tool_name != "kb_retriever"


def test_general_health_topic_shift_does_not_route_to_indexed_pdf():
    route = IntentRouter().route(
        "tell me about cancer",
        SessionContext(
            previous_query="what does this resume say",
            previous_answer="The resume mentions Java and AWS.",
            last_tool="kb_retriever",
            last_document_name="Manish_Resume_Feb26.pdf",
            available_files=["Manish_Resume_Feb26.pdf"],
        ),
    )

    assert route.intent == "web_search"
    assert route.tool_name == "web_search"
    assert route.needs_rag is False


def test_health_pronoun_follow_up_uses_recent_health_context_not_pdf():
    route = IntentRouter().route(
        "how harmful is that",
        SessionContext(
            previous_query="tell me about cancer",
            previous_answer="Cancer is a disease involving uncontrolled abnormal cell growth.",
            recent_messages=[
                {"role": "user", "content": "tell me about cancer"},
                {
                    "role": "assistant",
                    "content": "Cancer is a disease involving uncontrolled abnormal cell growth.",
                    "metadata": {"route": {"tool_name": "web_search", "intent": "web_search"}},
                },
            ],
            last_tool="web_search",
            available_files=["Manish_Resume_Feb26.pdf"],
        ),
    )

    assert route.intent == "web_search"
    assert route.tool_name == "web_search"
    assert route.is_follow_up is True


def test_cough_medicine_question_uses_health_web_not_resume_pdf():
    route = IntentRouter().route(
        "what tablets are preferred for cough",
        SessionContext(
            previous_query="is it rainy today in bnglr",
            previous_answer="The current weather in Bengaluru is 30 C.",
            last_tool="weather_api",
            last_document_name="Manish_Resume_Feb26.pdf",
            available_files=["Manish_Resume_Feb26.pdf"],
        ),
    )

    assert route.intent == "web_search"
    assert route.tool_name == "web_search"
    assert route.needs_rag is False


def test_resume_person_follow_up_recovers_profile_after_unrelated_topics():
    messages = [
        {"role": "user", "content": "how good is he in gen ai"},
        {
            "role": "assistant",
            "content": (
                "Based on the resume, Manish has Generative AI, RAG, LangChain, "
                "LangGraph, and multi-agent finance assistant experience."
            ),
            "metadata": {
                "route": {"tool_name": "kb_retriever", "intent": "kb_qa"},
                "sources": [{"file_name": "Manish_Resume_Feb26.pdf"}],
            },
        },
        {"role": "user", "content": "is it rainy today in bnglr"},
        {
            "role": "assistant",
            "content": "The current weather in Bengaluru is 30 C.",
            "metadata": {"route": {"tool_name": "weather_api", "intent": "weather"}},
        },
        {"role": "user", "content": "what tablets are preferred for cough"},
        {
            "role": "assistant",
            "content": "Ask a doctor or pharmacist for cough medication advice.",
            "metadata": {"route": {"tool_name": "web_search", "intent": "web_search"}},
        },
    ]
    context = SessionContext(
        previous_query="what tablets are preferred for cough",
        previous_answer="Ask a doctor or pharmacist for cough medication advice.",
        recent_messages=messages,
        last_tool="web_search",
        last_document_name="Manish_Resume_Feb26.pdf",
        available_files=["Manish_Resume_Feb26.pdf"],
        memory_snippets=build_memory_snippets(messages),
    )

    router = IntentRouter()
    route = router.route("is he also good at cyber security?", context)
    rewritten = router.rewrite_follow_up("is he also good at cyber security?", context, route)

    assert route.intent == "kb_qa"
    assert route.tool_name == "kb_retriever"
    assert route.is_follow_up is True
    assert route.target_source == "Manish_Resume_Feb26.pdf"
    assert "Relevant earlier context" in rewritten
    assert "Manish_Resume_Feb26.pdf" in rewritten


def test_resume_rating_pronoun_follow_up_targets_current_resume():
    route = IntentRouter().route(
        "rate him out of 10",
        SessionContext(
            previous_query="how good is he in gen ai",
            previous_answer="The resume shows Manish has Generative AI and RAG experience.",
            last_tool="kb_retriever",
            last_document_name="Manish_Resume_Feb26.pdf",
            available_files=["Manish_Resume_Feb26.pdf"],
        ),
    )

    assert route.intent == "kb_qa"
    assert route.tool_name == "kb_retriever"
    assert route.is_follow_up is True
    assert route.target_source == "Manish_Resume_Feb26.pdf"


def test_resume_github_link_question_targets_profile_document():
    route = IntentRouter().route(
        "retrieve his github link",
        SessionContext(
            previous_query="is manish good at cloud based on his resume",
            previous_answer="The resume mentions AWS, Azure, Docker, and Kubernetes.",
            last_tool="kb_retriever",
            last_document_name="Manish_Resume_Feb26.pdf",
            available_files=["Manish_Resume_Feb26.pdf"],
        ),
    )

    assert route.intent == "kb_qa"
    assert route.tool_name == "kb_retriever"
    assert route.target_source == "Manish_Resume_Feb26.pdf"


def test_pdf_with_latest_web_routes_to_cross_source_planner():
    route = IntentRouter().route(
        "compare this PDF with latest web info",
        SessionContext(
            last_document_name="market_report.pdf",
            available_files=["market_report.pdf"],
        ),
    )

    assert route.intent == "cross_source"
    assert route.needs_rag is True
    assert route.needs_web is True
    assert route.sub_intents == ["kb", "web"]
    assert route.target_source == "market_report.pdf"


def test_excel_analysis_explained_from_pdf_routes_to_cross_source_planner():
    route = IntentRouter().route(
        "analyze the Excel revenue and explain it using the PDF",
        SessionContext(available_files=["revenue.xlsx", "strategy.pdf"]),
    )

    assert route.intent == "cross_source"
    assert "excel" in route.sub_intents
    assert "kb" in route.sub_intents


def test_multiple_docs_summary_routes_to_cross_source_planner():
    route = IntentRouter().route(
        "search multiple documents and summarize with citations",
        SessionContext(available_files=["one.pdf", "two.pdf"]),
    )

    assert route.intent == "cross_source"
    assert route.sub_intents == ["kb"]
    assert route.needs_rag is True
    assert route.needs_web is False


def test_parallel_acronym_follow_up_after_general_answer():
    route = IntentRouter().route(
        "dl?",
        SessionContext(
            previous_query="what is ml",
            previous_answer="ML stands for Machine Learning.",
            last_tool="general_llm",
        ),
    )

    assert route.intent == "general_qa"
    assert route.is_follow_up is True


def test_short_kb_fragment_follow_up_targets_last_document():
    route = IntentRouter().route(
        "graduated in?",
        SessionContext(
            previous_query="what's college name which is providing this bachelors certificate",
            previous_answer="The college is VNR Vignana Jyothi Institute of Engineering & Technology.",
            last_tool="kb_retriever",
            last_document_name="bachelor_certificate.pdf",
            available_files=["bachelor_certificate.pdf"],
        ),
    )

    assert route.intent == "kb_qa"
    assert route.tool_name == "kb_retriever"
    assert route.is_follow_up is True
    assert route.target_source == "bachelor_certificate.pdf"
