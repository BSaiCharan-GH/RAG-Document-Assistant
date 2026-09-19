from backend.agent import AgenticRAGService


def test_agent_service_has_graph_builder():
    service = AgenticRAGService.__new__(AgenticRAGService)
    assert hasattr(service, "build_graph")
    assert callable(service.build_graph)


def test_document_specific_question_routes_to_document():
    service = AgenticRAGService.__new__(AgenticRAGService)
    decision = service._decide_source_mode(
        "According to the uploaded document, explain the Transformer architecture.",
        has_documents=True,
    )
    assert decision["mode"] == "document"


def test_current_information_question_routes_to_web():
    service = AgenticRAGService.__new__(AgenticRAGService)
    decision = service._decide_source_mode(
        "What is the latest stable version of Python?",
        has_documents=False,
    )
    assert decision["mode"] == "web"


def test_hybrid_question_routes_to_hybrid():
    service = AgenticRAGService.__new__(AgenticRAGService)
    decision = service._decide_source_mode(
        "According to my uploaded document, explain this technology and tell me what the latest developments are.",
        has_documents=True,
    )
    assert decision["mode"] == "hybrid"


def test_external_market_question_routes_to_web():
    service = AgenticRAGService.__new__(AgenticRAGService)
    decision = service._decide_source_mode(
        "Tell me something about the Indian stock market.",
        has_documents=True,
    )
    assert decision["mode"] == "web"
