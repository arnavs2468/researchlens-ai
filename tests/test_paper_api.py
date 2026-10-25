"""Integration tests for paper upload, retrieval, and API management."""

def test_paper_upload_and_delete_flow(client, sample_pdf_bytes):
    """Test full cycle: upload PDF, verify in list, search chunks, and delete."""
    # 1. Upload valid PDF
    files = {"files": ("attention_test.pdf", sample_pdf_bytes, "application/pdf")}
    response = client.post("/papers/upload", files=files)
    assert response.status_code == 200

    papers = response.json()
    assert len(papers) == 1
    doc_id = papers[0]["document_id"]
    assert "Attention Is All You Need" in papers[0]["paper_title"]
    assert papers[0]["total_pages"] == 2
    assert papers[0]["total_chunks"] >= 2

    # 2. Verify in list
    list_resp = client.get("/papers")
    assert list_resp.status_code == 200
    listed_ids = [p["document_id"] for p in list_resp.json()]
    assert doc_id in listed_ids

    # 3. Search indexed chunks
    search_payload = {
        "query": "What is the Transformer architecture?",
        "retrieval_mode": "hybrid",
        "top_k": 3,
        "document_ids": [doc_id],
    }
    search_resp = client.post("/papers/search", json=search_payload)
    assert search_resp.status_code == 200
    search_data = search_resp.json()
    assert search_data["total_results"] > 0
    assert any("Transformer" in c["text"] for c in search_data["chunks"])

    # 4. Clean up - Delete paper
    del_resp = client.delete(f"/papers/{doc_id}")
    assert del_resp.status_code == 200

    # 5. Verify deleted from list
    list_after = client.get("/papers")
    assert doc_id not in [p["document_id"] for p in list_after.json()]


def test_invalid_upload_handling(client):
    """Test that bad uploads (e.g. non-PDF) return HTTP 400."""
    files = {"files": ("corrupted.txt", b"Plain text not pdf", "application/pdf")}
    response = client.post("/papers/upload", files=files)
    assert response.status_code == 400
    assert "missing %PDF- header" in response.json()["detail"]
