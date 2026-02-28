"""
API endpoint tests for the Instacart recommendation service.

Uses a mocked recommender to avoid loading the real model in CI.
Covers /health, /ready, /recommend, /feedback, and /metrics.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest


class TestHealthEndpoints:
    """Tests for liveness and readiness probes."""

    def test_health_returns_ok(self, client):
        """GET /health should always return status=ok when process is up."""
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_ready_returns_ready_when_model_loaded(self, client):
        """GET /ready should return status=ready when recommender is loaded."""
        resp = client.get("/ready")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ready"}

    def test_health_includes_x_request_id(self, client):
        """Responses should include X-Request-ID header for tracing."""
        resp = client.get("/health")
        assert "X-Request-ID" in resp.headers


class TestRecommendEndpoint:
    """Tests for POST /recommend."""

    def test_recommend_with_user_context_returns_200(self, client):
        """Valid user_context should return recommendations and request_id."""
        payload = {
            "user_context": "[+7d w4h14] Organic Milk, Whole Wheat Bread.",
            "top_k": 5,
        }
        resp = client.post("/recommend", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "request_id" in data
        assert "recommendations" in data
        assert len(data["recommendations"]) == 3  # mock returns 3
        assert data["recommendations"][0]["product_id"] == "13517"
        assert data["recommendations"][0]["score"] == 0.76

    def test_recommend_without_context_returns_400(self, client):
        """Missing user_context and user_id should return 400."""
        resp = client.post("/recommend", json={"top_k": 5})
        assert resp.status_code == 400
        assert "user_context" in resp.json()["detail"].lower() or "user_id" in resp.json()["detail"].lower()

    def test_recommend_validates_top_k_range(self, client):
        """top_k must be between 1 and 100."""
        # top_k=0 should fail validation
        resp = client.post(
            "/recommend",
            json={"user_context": "[+7d] Milk.", "top_k": 0},
        )
        assert resp.status_code == 422

        # top_k=101 should fail validation
        resp = client.post(
            "/recommend",
            json={"user_context": "[+7d] Milk.", "top_k": 101},
        )
        assert resp.status_code == 422

    def test_recommend_respects_exclude_product_ids(self, client, mock_recommender):
        """exclude_product_ids should be passed to recommender."""
        mock_recommender.recommend.return_value = [("999", 0.5)]  # Different result
        payload = {
            "user_context": "[+7d] Milk.",
            "top_k": 5,
            "exclude_product_ids": ["13517"],
        }
        resp = client.post("/recommend", json=payload)
        assert resp.status_code == 200
        # Verify exclude_product_ids was passed
        call_kwargs = mock_recommender.recommend.call_args[1]
        assert "exclude_product_ids" in call_kwargs


class TestApiKeyAuth:
    """Tests for API key authentication (when API_KEY env is set)."""

    def test_recommend_without_api_key_when_required_returns_401(self, mock_recommender, tmp_path):
        """When API_KEY is set, requests without key should return 401."""
        os.environ["FEEDBACK_DB_PATH"] = str(tmp_path / "feedback.db")
        os.environ["API_KEY"] = "secret-test-key"
        try:
            with patch("src.api.main.load_monitored_recommender", return_value=mock_recommender):
                from fastapi.testclient import TestClient
                from src.api.main import app

                with TestClient(app) as c:
                    resp = c.post(
                        "/recommend",
                        json={"user_context": "[+7d] Milk.", "top_k": 5},
                    )
                    assert resp.status_code == 401
        finally:
            os.environ.pop("API_KEY", None)

    def test_recommend_with_api_key_when_required_returns_200(self, mock_recommender, tmp_path):
        """When API_KEY is set, requests with valid key should succeed."""
        os.environ["FEEDBACK_DB_PATH"] = str(tmp_path / "feedback.db")
        os.environ["API_KEY"] = "secret-test-key"
        try:
            with patch("src.api.main.load_monitored_recommender", return_value=mock_recommender):
                from fastapi.testclient import TestClient
                from src.api.main import app

                with TestClient(app) as c:
                    resp = c.post(
                        "/recommend",
                        json={"user_context": "[+7d] Milk.", "top_k": 5},
                        headers={"X-API-Key": "secret-test-key"},
                    )
                    assert resp.status_code == 200
        finally:
            os.environ.pop("API_KEY", None)


class TestFeedbackEndpoint:
    """Tests for POST /feedback."""

    def test_feedback_single_event_returns_202(self, client):
        """Single feedback event should be accepted with 202."""
        payload = {
            "request_id": "test-req-123",
            "event_type": "impression",
            "product_id": "13517",
            "user_id": "user1",
        }
        resp = client.post("/feedback", json=payload)
        assert resp.status_code == 202
        assert resp.json() == {"status": "accepted", "count": 1}

    def test_feedback_batch_returns_202(self, client):
        """Batch of feedback events should be accepted."""
        payload = {
            "events": [
                {"request_id": "r1", "event_type": "impression", "product_id": "p1"},
                {"request_id": "r1", "event_type": "click", "product_id": "p1"},
            ]
        }
        resp = client.post("/feedback", json=payload)
        assert resp.status_code == 202
        assert resp.json() == {"status": "accepted", "count": 2}

    def test_feedback_empty_events_returns_400(self, client):
        """Empty events list should return 400."""
        resp = client.post("/feedback", json={"events": []})
        assert resp.status_code == 400

    def test_feedback_validates_event_type(self, client):
        """Invalid event_type should fail validation."""
        payload = {
            "request_id": "r1",
            "event_type": "invalid_type",
            "product_id": "p1",
        }
        resp = client.post("/feedback", json=payload)
        assert resp.status_code == 422


class TestMetricsEndpoint:
    """Tests for GET /metrics (Prometheus)."""

    def test_metrics_returns_200(self, client):
        """GET /metrics should return Prometheus format."""
        resp = client.get("/metrics")
        # Prometheus may mount at /metrics/ with redirect
        if resp.status_code == 307:
            resp = client.get("/metrics/")
        assert resp.status_code == 200

    def test_metrics_contains_expected_counters(self, client):
        """Metrics should include recommendation and feedback counters."""
        # Trigger a request to populate metrics
        client.post("/recommend", json={"user_context": "[+7d] Milk.", "top_k": 5})
        client.post(
            "/feedback",
            json={"request_id": "r1", "event_type": "impression", "product_id": "p1"},
        )

        resp = client.get("/metrics")
        if resp.status_code == 307:
            resp = client.get("/metrics/")
        text = resp.text

        # Should contain our custom metrics
        assert "recommendation_requests_total" in text or "recommendation_" in text
        assert "feedback_events_total" in text or "feedback_" in text
        assert "model_loaded" in text
