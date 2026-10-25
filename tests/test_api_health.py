"""Integration test for FastAPI /health endpoint."""

def test_health_check_endpoint(client):
    """Test health check returns HTTP 200 with complete system diagnostics."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "groq_configured" in data
    assert "embedding_model" in data
    assert "vector_db_ready" in data
    assert data["vector_db_ready"] is True
