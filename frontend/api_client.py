"""HTTP Client for communicating with the ResearchLens AI FastAPI backend."""

import os
from typing import Any, Dict, List, Optional
import requests


class APIClient:
    """Client for making structured calls to the FastAPI backend."""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

    def _url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"

    def get_health(self) -> Dict[str, Any]:
        """Check backend health and configuration status."""
        try:
            resp = requests.get(self._url("/health"), timeout=5.0)
            if resp.status_code == 200:
                return resp.json()
            return {"status": "unhealthy", "error": f"HTTP {resp.status_code}"}
        except requests.RequestException as e:
            return {"status": "offline", "error": str(e)}

    def list_papers(self) -> List[Dict[str, Any]]:
        """Fetch all indexed papers."""
        try:
            resp = requests.get(self._url("/papers"), timeout=10.0)
            if resp.status_code == 200:
                return resp.json()
            return []
        except requests.RequestException:
            return []

    def get_paper(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Fetch metadata for a single paper."""
        try:
            resp = requests.get(self._url(f"/papers/{document_id}"), timeout=10.0)
            if resp.status_code == 200:
                return resp.json()
            return None
        except requests.RequestException:
            return None

    def upload_pdfs(
        self,
        files: List[tuple],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Upload PDF files to be processed and indexed."""
        data = {}
        if chunk_size:
            data["chunk_size"] = chunk_size
        if chunk_overlap:
            data["chunk_overlap"] = chunk_overlap

        try:
            resp = requests.post(
                self._url("/papers/upload"),
                files=files,
                data=data,
                timeout=120.0,
            )
            if resp.status_code == 200:
                return {"success": True, "papers": resp.json()}
            return {"success": False, "error": resp.json().get("detail", resp.text)}
        except requests.RequestException as e:
            return {"success": False, "error": str(e)}

    def search_arxiv(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search papers on arXiv."""
        try:
            resp = requests.get(
                self._url("/papers/arxiv/search"),
                params={"query": query, "max_results": max_results},
                timeout=20.0,
            )
            if resp.status_code == 200:
                return resp.json()
            return []
        except requests.RequestException:
            return []

    def import_arxiv_paper(
        self,
        arxiv_id: str,
        title: Optional[str] = None,
        authors: Optional[List[str]] = None,
        pdf_url: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Import a paper from arXiv by ID."""
        payload = {
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": authors or [],
            "pdf_url": pdf_url,
        }
        params = {}
        if chunk_size:
            params["chunk_size"] = chunk_size
        if chunk_overlap:
            params["chunk_overlap"] = chunk_overlap

        try:
            resp = requests.post(
                self._url("/papers/import-arxiv"),
                json=payload,
                params=params,
                timeout=90.0,
            )
            if resp.status_code == 200:
                return {"success": True, "paper": resp.json()}
            return {"success": False, "error": resp.json().get("detail", resp.text)}
        except requests.RequestException as e:
            return {"success": False, "error": str(e)}

    def delete_paper(self, document_id: str) -> Dict[str, Any]:
        """Delete an indexed paper."""
        try:
            resp = requests.delete(self._url(f"/papers/{document_id}"), timeout=10.0)
            if resp.status_code == 200:
                return {"success": True, "data": resp.json()}
            return {"success": False, "error": resp.json().get("detail", resp.text)}
        except requests.RequestException as e:
            return {"success": False, "error": str(e)}

    def send_chat(
        self,
        question: str,
        document_ids: Optional[List[str]] = None,
        retrieval_mode: str = "hybrid",
        top_k: int = 5,
        enable_reranking: bool = False,
        chat_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Send question to RAG endpoint."""
        payload = {
            "question": question,
            "document_ids": document_ids,
            "retrieval_mode": retrieval_mode,
            "top_k": top_k,
            "enable_reranking": enable_reranking,
            "chat_history": chat_history or [],
        }
        try:
            resp = requests.post(self._url("/chat"), json=payload, timeout=45.0)
            if resp.status_code == 200:
                return resp.json()
            return {
                "question": question,
                "answer": f"Backend Error: {resp.json().get('detail', resp.text)}",
                "citations": [],
                "unvalidated_citations": [],
                "retrieved_chunks": [],
                "retrieval_mode": retrieval_mode,
                "is_grounded": False,
                "model_used": "Error",
                "warning": resp.text,
            }
        except requests.RequestException as e:
            return {
                "question": question,
                "answer": f"Connection Error: Could not reach backend server at {self.base_url}. Ensure it is running.",
                "citations": [],
                "unvalidated_citations": [],
                "retrieved_chunks": [],
                "retrieval_mode": retrieval_mode,
                "is_grounded": False,
                "model_used": "Error",
                "warning": str(e),
            }

    def compare_papers(
        self,
        document_ids: List[str],
        focus_aspects: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compare 2 to 5 papers."""
        payload = {"document_ids": document_ids, "focus_aspects": focus_aspects}
        try:
            resp = requests.post(self._url("/papers/compare"), json=payload, timeout=90.0)
            if resp.status_code == 200:
                return {"success": True, "data": resp.json()}
            return {"success": False, "error": resp.json().get("detail", resp.text)}
        except requests.RequestException as e:
            return {"success": False, "error": str(e)}

    def get_summary(self, document_id: str) -> Dict[str, Any]:
        """Fetch 8-section structured summary for a paper."""
        try:
            resp = requests.post(self._url(f"/papers/{document_id}/summary"), timeout=60.0)
            if resp.status_code == 200:
                return {"success": True, "data": resp.json()}
            return {"success": False, "error": resp.json().get("detail", resp.text)}
        except requests.RequestException as e:
            return {"success": False, "error": str(e)}

    def run_evaluation(
        self,
        retrieval_modes: Optional[List[str]] = None,
        k_values: Optional[List[int]] = None,
        selected_categories: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run benchmark evaluation."""
        payload = {
            "retrieval_modes": retrieval_modes or ["dense", "bm25", "hybrid"],
            "k_values": k_values or [3, 5, 10],
            "selected_categories": selected_categories,
        }
        try:
            resp = requests.post(self._url("/evaluation/run"), json=payload, timeout=120.0)
            if resp.status_code == 200:
                return {"success": True, "data": resp.json()}
            return {"success": False, "error": resp.json().get("detail", resp.text)}
        except requests.RequestException as e:
            return {"success": False, "error": str(e)}

    def get_evaluation_results(self) -> Optional[Dict[str, Any]]:
        """Fetch latest benchmark evaluation run."""
        try:
            resp = requests.get(self._url("/evaluation/results"), timeout=10.0)
            if resp.status_code == 200:
                return resp.json()
            return None
        except requests.RequestException:
            return None

    def get_evaluation_questions(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch benchmark questions list."""
        try:
            params = {"category": category} if category else {}
            resp = requests.get(self._url("/evaluation/questions"), params=params, timeout=10.0)
            if resp.status_code == 200:
                return resp.json()
            return []
        except requests.RequestException:
            return []


api_client = APIClient()
