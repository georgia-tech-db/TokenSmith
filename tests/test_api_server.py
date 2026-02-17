"""
Unit tests for API server endpoints and models.

These tests use mocks for LLM and artifacts so they can run fast in CI
without requiring actual model files or indexes.

Run with: pytest tests/test_api_server.py -v
Or: pytest -m unit
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json

# Mark all tests in this module as unit tests
pytestmark = pytest.mark.unit


# ====================== Pydantic Model Tests ======================

class TestPydanticModels:
    """Tests for Pydantic model validation."""
    def test_chat_request_minimal(self):
        """ChatRequest works with only required field."""
        from src.api_server import ChatRequest

        request = ChatRequest(query="What is a database?")

        assert request.query == "What is a database?"
        assert request.enable_chunks is None
        assert request.prompt_type is None
        assert request.max_chunks is None
        assert request.temperature is None
        assert request.top_k is None

    def test_chat_request_full(self):
        """ChatRequest accepts all optional fields."""
        from src.api_server import ChatRequest

        request = ChatRequest(
            query="What is ACID?",
            enable_chunks=True,
            prompt_type="tutor",
            max_chunks=5,
            temperature=0.7,
            top_k=10
        )

        assert request.query == "What is ACID?"
        assert request.enable_chunks is True
        assert request.prompt_type == "tutor"
        assert request.max_chunks == 5
        assert request.temperature == 0.7
        assert request.top_k == 10

    def test_chat_request_empty_query(self):
        """ChatRequest allows empty query (validation happens in endpoint)."""
        from src.api_server import ChatRequest

        # Pydantic allows empty string, endpoint validates
        request = ChatRequest(query="")
        assert request.query == ""

    def test_chat_response_structure(self):
        """ChatResponse has correct structure."""
        from src.api_server import ChatResponse, SourceItem

        sources = [SourceItem(page=1, text="source1")]
        response = ChatResponse(
            answer="This is the answer",
            sources=sources,
            chunks_used=[0, 1, 2],
            chunks_by_page={1: ["source1"]},
            query="What is a database?"
        )

        assert response.answer == "This is the answer"
        assert len(response.sources) == 1
        assert response.chunks_used == [0, 1, 2]
        assert response.chunks_by_page == {1: ["source1"]}
        assert response.query == "What is a database?"


# ====================== Helper Function Tests ======================

class TestHelperFunctions:
    """Tests for API server helper functions."""

    def test_resolve_config_path(self):
        """_resolve_config_path returns valid path."""
        from src.api_server import _resolve_config_path

        path = _resolve_config_path()

        assert isinstance(path, Path)
        assert path.name == "config.yaml"
        assert "config" in str(path)

    def test_ensure_initialized_raises_when_not_ready(self):
        """_ensure_initialized raises HTTPException when artifacts not loaded."""
        from src.api_server import _ensure_initialized
        from fastapi import HTTPException
        import src.api_server as api_module

        # Save original state
        orig_config = api_module._config
        orig_artifacts = api_module._artifacts

        try:
            # Set to None to simulate uninitialized state
            api_module._config = None
            api_module._artifacts = None

            with pytest.raises(HTTPException) as exc_info:
                _ensure_initialized()

            assert exc_info.value.status_code == 503
            assert "Artifacts not loaded" in exc_info.value.detail
        finally:
            # Restore original state
            api_module._config = orig_config
            api_module._artifacts = orig_artifacts


# ====================== FastAPI Endpoint Tests ======================

class TestHealthEndpoint:
    """Tests for /api/health endpoint."""

    def test_health_check(self):
        """Health endpoint returns ok status."""
        from fastapi.testclient import TestClient
        from src.api_server import app

        # Create client without lifespan to avoid loading artifacts
        with patch('src.api_server.lifespan'):
            client = TestClient(app, raise_server_exceptions=False)
            response = client.get("/api/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "TokenSmith" in data["message"]


class TestChatEndpoint:
    """Tests for /api/chat endpoint."""

    @pytest.fixture
    def mock_server_state(self):
        """Set up mock server state for testing."""
        import src.api_server as api_module

        # Create mock config
        mock_config = Mock()
        mock_config.disable_chunks = False
        mock_config.system_prompt_mode = "baseline"
        mock_config.top_k = 5
        mock_config.num_candidates = 60
        mock_config.temperature = 0.2
        mock_config.max_gen_tokens = 300
        mock_config.gen_model = "mock_model.gguf"

        # Create mock artifacts
        mock_artifacts = {
            "chunks": ["chunk0", "chunk1", "chunk2", "chunk3", "chunk4"],
            "sources": ["source0", "source1", "source2", "source3", "source4"],
            "meta": [
                {"page_number": 10},
                {"page_number": 20},
                {"page_number": 30},
                {"page_number": 40},
                {"page_number": 50},
            ],
        }

        # Create mock retrievers
        mock_faiss = Mock()
        mock_faiss.name = "faiss"
        mock_faiss.get_scores = Mock(return_value={0: 0.9, 1: 0.8, 2: 0.7})

        mock_bm25 = Mock()
        mock_bm25.name = "bm25"
        mock_bm25.get_scores = Mock(return_value={0: 0.7, 1: 0.9, 2: 0.6})

        mock_retrievers = [mock_faiss, mock_bm25]

        # Create mock ranker
        mock_ranker = Mock()
        mock_ranker.rank = Mock(return_value=([0, 1, 2, 3, 4], [0.1, 0.09, 0.08, 0.07, 0.06]))

        # Create mock logger
        mock_logger = Mock()

        # Save originals and set mocks
        originals = {
            "_config": api_module._config,
            "_artifacts": api_module._artifacts,
            "_retrievers": api_module._retrievers,
            "_ranker": api_module._ranker,
            "_logger": api_module._logger,
        }

        api_module._config = mock_config
        api_module._artifacts = mock_artifacts
        api_module._retrievers = mock_retrievers
        api_module._ranker = mock_ranker
        api_module._logger = mock_logger

        yield {
            "config": mock_config,
            "artifacts": mock_artifacts,
            "retrievers": mock_retrievers,
            "ranker": mock_ranker,
            "logger": mock_logger,
        }

        # Restore originals
        for key, value in originals.items():
            setattr(api_module, key, value)

    def test_chat_empty_query_rejected(self, mock_server_state):
        """Chat endpoint rejects empty query."""
        from fastapi.testclient import TestClient
        from src.api_server import app

        with patch('src.api_server.lifespan'):
            client = TestClient(app, raise_server_exceptions=False)
            response = client.post("/api/chat", json={"query": ""})

        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_chat_whitespace_query_rejected(self, mock_server_state):
        """Chat endpoint rejects whitespace-only query."""
        from fastapi.testclient import TestClient
        from src.api_server import app

        with patch('src.api_server.lifespan'):
            client = TestClient(app, raise_server_exceptions=False)
            response = client.post("/api/chat", json={"query": "   "})

        assert response.status_code == 400

    @patch('src.api_server.answer')
    def test_chat_success(self, mock_answer, mock_server_state):
        """Chat endpoint returns successful response."""
        from fastapi.testclient import TestClient
        from src.api_server import app

        # Mock the answer generator
        mock_answer.return_value = iter(["This ", "is ", "the ", "answer."])

        with patch('src.api_server.lifespan'):
            client = TestClient(app, raise_server_exceptions=False)
            response = client.post("/api/chat", json={"query": "What is a database?"})

        print("Response JSON:")
        print(response.json())
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "chunks_used" in data
        assert data["query"] == "What is a database?"

    @patch('src.api_server.answer')
    def test_chat_with_custom_params(self, mock_answer, mock_server_state):
        """Chat endpoint respects custom parameters."""
        from fastapi.testclient import TestClient
        from src.api_server import app

        mock_answer.return_value = iter(["Answer"])

        with patch('src.api_server.lifespan'):
            client = TestClient(app, raise_server_exceptions=False)
            response = client.post("/api/chat", json={
                "query": "Test query",
                "enable_chunks": True,
                "prompt_type": "tutor",
                "top_k": 3,
                "temperature": 0.5
            })

        assert response.status_code == 200
        # Verify answer was called with custom parameters
        mock_answer.assert_called_once()
        call_args = mock_answer.call_args
        assert call_args[1]["system_prompt_mode"] == "tutor"
        assert call_args[1]["temperature"] == 0.5

    @patch('src.api_server.answer')
    def test_chat_disable_chunks(self, mock_answer, mock_server_state):
        """Chat endpoint works with chunks disabled."""
        from fastapi.testclient import TestClient
        from src.api_server import app

        mock_answer.return_value = iter(["No context answer"])

        with patch('src.api_server.lifespan'):
            client = TestClient(app, raise_server_exceptions=False)
            response = client.post("/api/chat", json={
                "query": "What is a database?",
                "enable_chunks": False
            })

        assert response.status_code == 200
        data = response.json()
        # When chunks disabled, should have empty chunks_used
        assert data["chunks_used"] == []

    @patch('src.api_server.answer')
    def test_chat_generation_error_handled(self, mock_answer, mock_server_state):
        """Chat endpoint handles generation errors gracefully."""
        from fastapi.testclient import TestClient
        from src.api_server import app

        # Simulate generation error
        mock_answer.side_effect = Exception("Model loading failed")

        with patch('src.api_server.lifespan'):
            client = TestClient(app, raise_server_exceptions=False)
            response = client.post("/api/chat", json={"query": "What is ACID?"})

        assert response.status_code == 200
        data = response.json()
        # Should return error message instead of crashing
        assert "error" in data["answer"].lower() or "sorry" in data["answer"].lower()


class TestTestChatEndpoint:
    """Tests for /api/test-chat endpoint."""

    @pytest.fixture
    def mock_server_state(self):
        """Set up mock server state for testing."""
        import src.api_server as api_module

        mock_config = Mock()
        mock_config.disable_chunks = False
        mock_config.top_k = 5
        mock_config.num_candidates = 60

        mock_artifacts = {
            "chunks": ["chunk0", "chunk1", "chunk2"],
            "sources": ["source0", "source1", "source2"],
            "meta": [{"page_number": 1}, {"page_number": 2}, {"page_number": 3}],
        }

        mock_faiss = Mock()
        mock_faiss.name = "faiss"
        mock_faiss.get_scores = Mock(return_value={0: 0.9, 1: 0.8, 2: 0.7})

        mock_bm25 = Mock()
        mock_bm25.name = "bm25"
        mock_bm25.get_scores = Mock(return_value={0: 0.7, 1: 0.9, 2: 0.6})

        mock_ranker = Mock()
        mock_ranker.rank = Mock(return_value=([0, 1, 2], [0.1, 0.09, 0.08]))

        originals = {
            "_config": api_module._config,
            "_artifacts": api_module._artifacts,
            "_retrievers": api_module._retrievers,
            "_ranker": api_module._ranker,
            "_logger": api_module._logger,
        }

        api_module._config = mock_config
        api_module._artifacts = mock_artifacts
        api_module._retrievers = [mock_faiss, mock_bm25]
        api_module._ranker = mock_ranker
        api_module._logger = Mock()

        yield

        for key, value in originals.items():
            setattr(api_module, key, value)

    def test_test_chat_success(self, mock_server_state):
        """Test chat endpoint returns retrieval results."""
        from fastapi.testclient import TestClient
        from src.api_server import app

        with patch('src.api_server.lifespan'):
            client = TestClient(app, raise_server_exceptions=False)
            response = client.post("/api/test-chat", json={"query": "Test query"})

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "chunks_found" in data
        assert "top_chunks" in data

    def test_test_chat_empty_query(self, mock_server_state):
        """Test chat endpoint handles empty query."""
        from fastapi.testclient import TestClient
        from src.api_server import app

        with patch('src.api_server.lifespan'):
            client = TestClient(app, raise_server_exceptions=False)
            response = client.post("/api/test-chat", json={"query": ""})

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"

    def test_test_chat_chunks_disabled(self, mock_server_state):
        """Test chat endpoint reports error when chunks disabled."""
        from fastapi.testclient import TestClient
        from src.api_server import app

        with patch('src.api_server.lifespan'):
            client = TestClient(app, raise_server_exceptions=False)
            response = client.post("/api/test-chat", json={
                "query": "Test",
                "enable_chunks": False
            })

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"
        assert "disabled" in data["error"].lower()


class TestStreamingEndpoint:
    """Tests for /api/chat/stream endpoint."""

    @pytest.fixture
    def mock_server_state(self):
        """Set up mock server state for streaming tests."""
        import src.api_server as api_module

        mock_config = Mock()
        mock_config.disable_chunks = False
        mock_config.system_prompt_mode = "baseline"
        mock_config.top_k = 5
        mock_config.num_candidates = 60
        mock_config.temperature = 0.2
        mock_config.max_gen_tokens = 300
        mock_config.gen_model = "mock_model.gguf"

        mock_artifacts = {
            "chunks": ["chunk0", "chunk1", "chunk2"],
            "sources": ["source0", "source1", "source2"],
            "meta": [{"page_number": 10}, {"page_number": 20}, {"page_number": 30}],
        }

        mock_faiss = Mock()
        mock_faiss.name = "faiss"
        mock_faiss.get_scores = Mock(return_value={0: 0.9, 1: 0.8, 2: 0.7})

        mock_bm25 = Mock()
        mock_bm25.name = "bm25"
        mock_bm25.get_scores = Mock(return_value={0: 0.7, 1: 0.9, 2: 0.6})

        mock_ranker = Mock()
        mock_ranker.rank = Mock(return_value=([0, 1, 2], [0.1, 0.09, 0.08]))
        

        originals = {
            "_config": api_module._config,
            "_artifacts": api_module._artifacts,
            "_retrievers": api_module._retrievers,
            "_ranker": api_module._ranker,
            "_logger": api_module._logger,
        }

        api_module._config = mock_config
        api_module._artifacts = mock_artifacts
        api_module._retrievers = [mock_faiss, mock_bm25]
        api_module._ranker = mock_ranker
        api_module._logger = Mock()

        yield

        for key, value in originals.items():
            setattr(api_module, key, value)

    def test_stream_empty_query_rejected(self, mock_server_state):
        """Stream endpoint rejects empty query."""
        from fastapi.testclient import TestClient
        from src.api_server import app

        with patch('src.api_server.lifespan'):
            client = TestClient(app, raise_server_exceptions=False)
            response = client.post("/api/chat/stream", json={"query": ""})

        assert response.status_code == 400

    @patch('src.api_server.answer')
    @patch('src.api_server.get_page_numbers')
    def test_stream_returns_sse(self, mock_page_nums, mock_answer, mock_server_state):
        """Stream endpoint returns Server-Sent Events format."""
        from fastapi.testclient import TestClient
        from src.api_server import app

        mock_page_nums.return_value = {0: 10, 1: 20, 2: 30}
        mock_answer.return_value = iter(["Hello", " world"])

        with patch('src.api_server.lifespan'):
            client = TestClient(app, raise_server_exceptions=False)
            response = client.post("/api/chat/stream", json={"query": "Test"})

        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

        # Parse SSE events
        content = response.text
        assert "data:" in content


# ====================== Retrieve and Rank Tests ======================

class TestRetrieveAndRank:
    """Tests for _retrieve_and_rank function."""

    @pytest.fixture
    def mock_server_state(self):
        """Set up mock server state."""
        import src.api_server as api_module

        mock_config = Mock()
        mock_config.top_k = 5
        mock_config.num_candidates = 60

        mock_artifacts = {
            "chunks": ["c0", "c1", "c2", "c3", "c4"],
        }

        mock_faiss = Mock()
        mock_faiss.name = "faiss"
        mock_faiss.get_scores = Mock(return_value={0: 0.9, 1: 0.8, 2: 0.7, 3: 0.6, 4: 0.5})

        mock_bm25 = Mock()
        mock_bm25.name = "bm25"
        mock_bm25.get_scores = Mock(return_value={0: 0.5, 1: 0.6, 2: 0.9, 3: 0.7, 4: 0.8})

        mock_ranker = Mock()
        mock_ranker.rank = Mock(return_value=([2, 0, 1, 4, 3], [0.95, 0.9, 0.85, 0.8, 0.75]))

        originals = {
            "_config": api_module._config,
            "_artifacts": api_module._artifacts,
            "_retrievers": api_module._retrievers,
            "_ranker": api_module._ranker,
        }

        api_module._config = mock_config
        api_module._artifacts = mock_artifacts
        api_module._retrievers = [mock_faiss, mock_bm25]
        api_module._ranker = mock_ranker

        yield

        for key, value in originals.items():
            setattr(api_module, key, value)

    def test_retrieve_and_rank_returns_scores_and_indices(self, mock_server_state):
        """_retrieve_and_rank returns raw_scores and topk_idxs."""
        from src.api_server import _retrieve_and_rank

        topk_idxs, ordered_scores = _retrieve_and_rank("test query")

        assert isinstance(ordered_scores, list)
        assert isinstance(topk_idxs, list)

    def test_retrieve_and_rank_custom_top_k(self, mock_server_state):
        """_retrieve_and_rank respects custom top_k."""
        from src.api_server import _retrieve_and_rank
        import src.api_server as api_module

        topk_idxs, ordered_scores = _retrieve_and_rank("test query", top_k=3)

        # Should have called filter with top_k=3
        assert len(topk_idxs) <= 3


# ====================== Integration Tests ======================

class TestAPIIntegration:
    """Integration-style tests for API flows."""

    @pytest.fixture
    def full_mock_state(self):
        """Set up complete mock state for integration tests."""
        import src.api_server as api_module

        mock_config = Mock()
        mock_config.disable_chunks = False
        mock_config.system_prompt_mode = "baseline"
        mock_config.top_k = 3
        mock_config.num_candidates = 10
        mock_config.temperature = 0.2
        mock_config.max_gen_tokens = 100
        mock_config.gen_model = "test_model.gguf"

        mock_artifacts = {
            "chunks": [f"Chunk content {i}" for i in range(5)],
            "sources": [f"Source {i}" for i in range(5)],
            "meta": [{"page_number": i + 1} for i in range(5)],
        }

        mock_faiss = Mock()
        mock_faiss.name = "faiss"
        mock_faiss.get_scores = Mock(return_value={i: 1.0 - i * 0.1 for i in range(5)})

        mock_bm25 = Mock()
        mock_bm25.name = "bm25"
        mock_bm25.get_scores = Mock(return_value={i: 0.5 + i * 0.1 for i in range(5)})

        mock_ranker = Mock()
        mock_ranker.rank = Mock(return_value=([0, 1, 2, 3, 4], [0.9, 0.8, 0.7, 0.6, 0.5]))

        originals = {
            "_config": api_module._config,
            "_artifacts": api_module._artifacts,
            "_retrievers": api_module._retrievers,
            "_ranker": api_module._ranker,
            "_logger": api_module._logger,
        }

        api_module._config = mock_config
        api_module._artifacts = mock_artifacts
        api_module._retrievers = [mock_faiss, mock_bm25]
        api_module._ranker = mock_ranker
        api_module._logger = Mock()

        yield {
            "config": mock_config,
            "artifacts": mock_artifacts,
        }

        for key, value in originals.items():
            setattr(api_module, key, value)

    @patch('src.api_server.answer')
    def test_full_chat_flow(self, mock_answer, full_mock_state):
        """Test complete chat flow from request to response."""
        from fastapi.testclient import TestClient
        from src.api_server import app

        mock_answer.return_value = iter([
            "A database is ",
            "a structured collection ",
            "of data."
        ])

        with patch('src.api_server.lifespan'):
            client = TestClient(app, raise_server_exceptions=False)
            response = client.post("/api/chat", json={
                "query": "What is a database?",
                "enable_chunks": True,
                "top_k": 3
            })

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "answer" in data
        assert "sources" in data
        assert "chunks_used" in data
        assert "query" in data

        # Verify answer was assembled
        assert "database" in data["answer"].lower()

        # Verify query preserved
        assert data["query"] == "What is a database?"

    def test_health_to_chat_flow(self, full_mock_state):
        """Test health check followed by chat request."""
        from fastapi.testclient import TestClient
        from src.api_server import app

        with patch('src.api_server.lifespan'):
            client = TestClient(app, raise_server_exceptions=False)

            # Check health
            health_response = client.get("/api/health")
            assert health_response.status_code == 200
            assert health_response.json()["status"] == "ok"

            # Test retrieval
            test_response = client.post("/api/test-chat", json={
                "query": "Test query"
            })
            assert test_response.status_code == 200
            assert test_response.json()["status"] == "success"
