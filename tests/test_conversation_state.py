from core.conversation_state import ConversationState
from core.conversation_memory import build_memory_snippets, relevant_memory_snippets
from core.router import SessionContext


def test_conversation_state_extracts_kb_document_from_message_metadata():
    context = SessionContext(
        recent_messages=[
            {"role": "user", "content": "what is beta"},
            {
                "role": "assistant",
                "content": "Beta means this exam is in beta.",
                "metadata": {
                    "route": {"tool_name": "kb_retriever", "intent": "kb_qa"},
                    "sources": [{"file_name": "DP-800 Study Plan.pdf", "document_id": "dp_800"}],
                },
            },
        ]
    )

    state = ConversationState.from_context(context)

    assert state.latest_domain("kb").document_name == "DP-800 Study Plan.pdf"
    assert state.last_document_name == "DP-800 Study Plan.pdf"


def test_conversation_state_remembers_recent_weather_before_general_turn():
    context = SessionContext(
        recent_messages=[
            {"role": "user", "content": "weather in Bangalore"},
            {
                "role": "assistant",
                "content": "Weather for Bangalore, IN: 33 C.",
                "metadata": {"route": {"tool_name": "weather_api", "intent": "weather"}},
            },
            {"role": "user", "content": "what is c"},
            {
                "role": "assistant",
                "content": "C means Celsius.",
                "metadata": {"route": {"tool_name": "general_llm", "intent": "general_qa"}},
            },
        ]
    )

    state = ConversationState.from_context(context)

    assert state.latest.domain == "general"
    assert state.has_domain("weather", within=3)
    assert state.last_weather_location == "Bangalore,IN"


def test_weather_context_phrase_is_not_location():
    context = SessionContext(
        recent_messages=[
            {"role": "user", "content": "weather in Bangalore"},
            {
                "role": "assistant",
                "content": "Weather for Bangalore, IN: 33 C.",
                "metadata": {"route": {"tool_name": "weather_api", "intent": "weather"}},
            },
            {"role": "user", "content": "what is c"},
            {
                "role": "assistant",
                "content": "In weather context, C means Celsius.",
                "metadata": {"route": {"tool_name": "general_llm", "intent": "general_qa"}},
            },
        ]
    )

    state = ConversationState.from_context(context)

    assert state.latest.domain == "general"
    assert state.last_weather_location == "Bangalore,IN"


def test_memory_snippets_recover_goa_itinerary_after_topic_shift():
    messages = [
        {"role": "user", "content": "i'm planning a trip to goa"},
        {"role": "assistant", "content": "Goa works well as a beach and food trip."},
        {"role": "user", "content": "plan my 3 day itinerary"},
        {
            "role": "assistant",
            "content": (
                "Day 1: North Goa. Day 2: South Goa with Palolem, Old Goa, and a seafood dinner. "
                "Day 3: Dudhsagar and wellness."
            ),
        },
        {"role": "user", "content": "what's the weather today?"},
        {"role": "assistant", "content": "Weather for Bengaluru: 30 C."},
    ]

    snippets = build_memory_snippets(messages)
    relevant = relevant_memory_snippets("which restaurant do you recommend for day 2 evening", snippets)

    assert relevant
    assert relevant[0]["label"] == "travel"
    assert "Day 2" in relevant[0]["assistant_answer"]
