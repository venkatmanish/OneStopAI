from types import SimpleNamespace

from core.schemas import RetrievedChunk, SourceRef
from core.tools import ToolRouter


def test_weather_location_removes_today_and_normalizes_bangalore_typo():
    assert ToolRouter._extract_location("weather in banglore today") == "Bangalore,IN"


def test_weather_location_requires_city_for_country_only():
    assert ToolRouter._extract_location("i meant in india") is None


def test_weather_location_supports_follow_up_city_shorthand():
    assert ToolRouter._extract_location("Follow-up question: hyd") == "Hyderabad,IN"
    assert ToolRouter._extract_location("Follow-up question: bnglr") == "Bangalore,IN"


def test_general_llm_answers_rag_grounding_typo_directly():
    result = ToolRouter().general_llm("how are answers grounded inn rag")

    assert result.confidence == 0.9
    assert result.audit[0].stage == "direct_response"
    assert "Retrieval-Augmented Generation" in result.answer_context
    assert "retrieving relevant chunks" in result.answer_context


def test_weather_payload_builds_ui_ready_forecast():
    current = {
        "dt": 1714640400,
        "timezone": 19800,
        "name": "Bengaluru",
        "sys": {"country": "IN", "sunrise": 1714610400, "sunset": 1714657200},
        "main": {
            "temp": 32.4,
            "feels_like": 34.1,
            "humidity": 58,
            "pressure": 1011,
        },
        "weather": [{"main": "Clouds", "description": "scattered clouds", "icon": "03d"}],
        "wind": {"speed": 3.2},
        "visibility": 9000,
        "clouds": {"all": 42},
    }
    forecast = {
        "city": {"name": "Bengaluru", "country": "IN", "timezone": 19800},
        "list": [
            {
                "dt": 1714640400,
                "main": {"temp": 32.4, "temp_min": 31.0, "temp_max": 33.0, "humidity": 58},
                "weather": [{"main": "Clouds", "description": "scattered clouds", "icon": "03d"}],
                "wind": {"speed": 3.2},
                "pop": 0.2,
            },
            {
                "dt": 1714651200,
                "main": {"temp": 30.5, "temp_min": 30.0, "temp_max": 31.0, "humidity": 62},
                "weather": [{"main": "Rain", "description": "light rain", "icon": "10d"}],
                "wind": {"speed": 4.0},
                "pop": 0.7,
            },
        ],
    }

    payload = ToolRouter._weather_payload(current, forecast, "Bangalore,IN")
    context = ToolRouter._weather_context(payload)

    assert payload["type"] == "weather"
    assert payload["place"] == "Bengaluru, IN"
    assert payload["current"]["temp"] == 32.4
    assert payload["hourly"][0]["time"]
    assert payload["timeline"][0]["date"] == payload["daily"][0]["date"]
    assert payload["timeline"][0]["time_short"]
    assert payload["daily"][0]["temp_max"] == 33.0
    assert payload["daily"][0]["icon"] == "10d"
    assert payload["daily"][0]["main"] == "Rain"
    assert payload["daily"][0]["pop"] == 0.7
    assert "Forecast:" in context


def test_web_search_query_uses_current_follow_up_and_recent_context():
    query = """Use the conversation context to answer this follow-up.
Recent conversation:
User[kb]: when will the results get released for this
Assistant[kb]: The results for the DP-800 beta exam are typically released well after the beta exam ends.
Follow-up question: search online for it's release date"""

    search_query = ToolRouter._web_search_query(query)

    assert "release date" in search_query
    assert "DP-800 beta exam" in search_query
    assert "Use the conversation context" not in search_query
    assert "Assistant[kb]" not in search_query


def test_web_search_query_cleans_find_it_online_phrase():
    query = """Use the conversation context to answer this follow-up.
Recent conversation:
User[kb]: i meant the exam official results release date anything available like the month
Assistant[kb]: The results for the DP-800 beta exam are typically released well after the beta exam ends.
Follow-up question: find it online"""

    search_query = ToolRouter._web_search_query(query)

    assert "DP-800 beta exam" in search_query
    assert "find it online" not in search_query.lower()
    assert not search_query.lower().startswith("it ")
    assert "Use the conversation context" not in search_query


def test_kb_retriever_rejects_weak_unrelated_evidence():
    router = ToolRouter()
    router.retriever = SimpleNamespace(
        retrieve=lambda *args, **kwargs: [
            RetrievedChunk(
                chunk_id="weak",
                text="Offline reinforcement learning surveys discuss policy optimization.",
                score=0.82,
                source=SourceRef(file_name="gxs.pdf", page=1, chunk_id="weak"),
                metadata={"parent_context": "Offline reinforcement learning surveys discuss policy optimization."},
            )
        ]
    )

    result = router.kb_retriever("random unicorn banana unrelated")

    assert result.confidence == 0.35
    assert not result.sources
    assert "too weakly related" in result.answer_context
    assert result.audit[0].stage == "retrieval_quality"


def test_kb_retriever_accepts_relevant_parent_context():
    router = ToolRouter()
    router.retriever = SimpleNamespace(
        retrieve=lambda *args, **kwargs: [
            RetrievedChunk(
                chunk_id="strong",
                text="The message has four steps.",
                score=0.74,
                source=SourceRef(file_name="deck.pptx", page=3, chunk_id="strong"),
                metadata={
                    "parent_context": (
                        "GXS Nexus one empathic message uses four steps for outreach."
                    )
                },
            )
        ]
    )

    result = router.kb_retriever("gxs nexus one empathic message")

    assert result.confidence > 0.55
    assert result.sources[0].file_name == "deck.pptx"


def test_excel_calculation_result_format_does_not_require_tabulate():
    router = ToolRouter()
    router.llm = SimpleNamespace(complete=lambda *args, **kwargs: "SELECT * FROM orders")
    router.retriever = SimpleNamespace(
        retrieve=lambda *args, **kwargs: [
            SimpleNamespace(
                metadata={
                    "table": '[{"region":"South","revenue":1000},{"region":"North","revenue":800}]'
                },
                source=SourceRef(file_name="sales.xlsx", sheet="Orders", chunk_id="chunk-1"),
                chunk_id="chunk-1",
            )
        ]
    )

    result = router.excel_calculation("show rows")

    assert result.confidence == 0.84
    assert "Spreadsheet analysis SQL" in result.answer_context
    assert "South" in result.answer_context


def test_excel_calculation_repairs_weak_select_star_for_analysis_query():
    responses = iter(
        [
            "SELECT * FROM orders LIMIT 10",
            """
            SELECT region_norm, SUM(revenue_num) AS total_revenue
            FROM orders
            GROUP BY region_norm
            ORDER BY total_revenue DESC
            """,
        ]
    )
    router = ToolRouter()
    router.llm = SimpleNamespace(complete=lambda *args, **kwargs: next(responses))
    router.retriever = SimpleNamespace(
        retrieve=lambda *args, **kwargs: [
            SimpleNamespace(
                metadata={
                    "table": '[{"region":"South","revenue":"1,000"},{"region":"North","revenue":800}]'
                },
                source=SourceRef(file_name="sales.xlsx", sheet="Orders", chunk_id="chunk-1"),
                chunk_id="chunk-1",
            )
        ]
    )

    result = router.excel_calculation("calculate total revenue by region")

    assert result.confidence == 0.78
    assert "SUM(revenue_num)" in result.answer_context
    assert "SOUTH" in result.answer_context
    assert "1000.0" in result.answer_context


def test_spreadsheet_normalization_adds_helper_columns_generically():
    import pandas as pd

    frame = ToolRouter._normalize_spreadsheet_frame(
        pd.DataFrame([{"Region Code": " west ", "Units": "ten", "Revenue $": "1,250"}])
    )

    assert frame.loc[0, "region_code_norm"] == "WEST"
    assert frame.loc[0, "units_num"] == 10
    assert frame.loc[0, "revenue_num"] == 1250


def test_spreadsheet_safe_sql_ignores_comments():
    sql = """
    WITH values_cte AS (
        -- Create a target reference in prose only.
        SELECT 1 AS value
    )
    SELECT value FROM values_cte
    """

    assert ToolRouter._is_safe_select_sql(sql)
