from core.schemas import RouteDecision


def test_route_schema_shape():
    route = RouteDecision(intent="kb_qa", needs_rag=True, tool_name="kb_retriever", confidence=0.91)

    payload = route.model_dump()

    assert payload["intent"] == "kb_qa"
    assert payload["target_version"] == "latest_active"
    assert payload["needs_rag"] is True
