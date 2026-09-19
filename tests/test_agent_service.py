from backend.agent import AgenticRAGService


def test_agent_service_has_graph_builder():
    service = AgenticRAGService.__new__(AgenticRAGService)
    assert hasattr(service, "build_graph")
    assert callable(service.build_graph)
