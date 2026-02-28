"""
Pytest fixtures for API tests.

Provides a mock recommender and test client that bypasses model loading.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.inference.serve_recommendations import RecommendationMetrics


@pytest.fixture
def mock_recommender():
    """
    Create a mock MonitoredRecommender that returns fixed recommendations.

    Mimics the interface of MonitoredRecommender: recommend(), pid_to_text, _last_metrics.
    """
    mock = MagicMock()
    mock.recommend.return_value = [
        ("13517", 0.76),
        ("34479", 0.71),
        ("48628", 0.70),
    ]
    mock.pid_to_text = {
        "13517": "Product: Whole Wheat Bread. Aisle: bread. Department: bakery.",
        "34479": "Product: Whole Wheat Walnut Bread. Aisle: bread. Department: bakery.",
        "48628": "Product: Organic Whole Wheat Bread. Aisle: bread. Department: bakery.",
    }
    mock.corpus_path = Path("/tmp/test/corpus.json")
    mock._last_metrics = RecommendationMetrics(
        user_id="test",
        query_embedding_time_ms=10.0,
        similarity_compute_time_ms=5.0,
        total_latency_ms=15.0,
        num_recommendations=3,
        top_score=0.76,
        avg_score=0.72,
        timestamp=1234567890.0,
    )
    return mock


@pytest.fixture
def client(mock_recommender, tmp_path):
    """
    Test client with mocked recommender and temp feedback DB.

    Patches load_monitored_recommender to avoid loading the real model.
    Uses a temporary directory for the feedback database.
    """
    # Use temp path for feedback DB so tests don't touch real data
    db_path = tmp_path / "feedback.db"
    os.environ["FEEDBACK_DB_PATH"] = str(db_path)

    with patch("src.api.main.load_monitored_recommender", return_value=mock_recommender):
        from src.api.main import app

        with TestClient(app) as c:
            yield c
