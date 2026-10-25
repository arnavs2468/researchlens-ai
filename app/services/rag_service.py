"""Groq LLM RAG answer generation and citation orchestration."""

import logging
from typing import List, Optional
import groq
from app.config import settings
from app.prompts.answer_prompt import SYSTEM_PROMPT_RESEARCH_QA, USER_PROMPT_TEMPLATE
from app.schemas import ChatMessage, ChatRequest, ChatResponse, RetrievedChunk
from app.services.retrieval_service import retrieval_service
from app.utils.citations import format_evidence_block, validate_and_build_citations

logger = logging.getLogger(__name__)


class RAGService:
    """Orchestrates retrieval and Groq LLM grounded answering."""

    def __init__(self):
        self._groq_client: Optional[groq.Groq] = None

    def get_client(self) -> Optional[groq.Groq]:
        """Lazy load Groq client if key is configured."""
        if not settings.is_groq_configured:
            return None
        if self._groq_client is None:
            self._groq_client = groq.Groq(api_key=settings.groq_api_key)
        return self._groq_client

    def answer_question(self, request: ChatRequest) -> ChatResponse:
        """
        Execute grounded RAG pipeline:
        1. Retrieve relevant passages using requested mode (dense, bm25, or hybrid).
        2. Format evidence block with [S1], [S2] markers.
        3. Check Groq API key configuration.
        4. Generate response via Groq.
        5. Validate inline citations against retrieved sources.
        """
        # Step 1: Retrieve passages
        chunks: List[RetrievedChunk] = retrieval_service.retrieve(
            query=request.question,
            retrieval_mode=request.retrieval_mode or settings.default_retrieval_mode,
            top_k=request.top_k or settings.default_top_k,
            document_ids=request.document_ids,
            enable_reranking=request.enable_reranking or settings.enable_cross_encoder,
        )

        retrieval_mode_used = request.retrieval_mode or settings.default_retrieval_mode

        # Step 2: Handle empty retrieval
        if not chunks:
            return ChatResponse(
                question=request.question,
                answer=(
                    "No relevant passages were found in the indexed documents for your question. "
                    "Try broadening your search query or uploading/indexing papers related to this topic."
                ),
                citations=[],
                unvalidated_citations=[],
                retrieved_chunks=[],
                retrieval_mode=retrieval_mode_used,
                is_grounded=True,
                model_used="None (No chunks retrieved)",
            )

        evidence_block, citation_map = format_evidence_block(chunks)

        # Step 3: Check Groq API configuration
        client = self.get_client()
        if client is None:
            # Fallback when no Groq key is configured: show retrieved passages and prompt instructions
            fallback_answer = (
                "**[GROQ_API_KEY Not Configured]**\n\n"
                "The retrieval pipeline successfully identified **"
                f"{len(chunks)} relevant evidence chunks** from your indexed papers. "
                "However, LLM answer synthesis is paused because a Groq API key has not been configured.\n\n"
                "### How to enable AI Answer Generation:\n"
                "1. Get a free API key at [console.groq.com](https://console.groq.com).\n"
                "2. Add your key to `.env`: `GROQ_API_KEY=gsk_your_key_here`\n"
                "3. Restart the backend or update the Settings page.\n\n"
                "You can inspect all retrieved evidence passages below."
            )
            return ChatResponse(
                question=request.question,
                answer=fallback_answer,
                citations=[],
                unvalidated_citations=[],
                retrieved_chunks=chunks,
                retrieval_mode=retrieval_mode_used,
                is_grounded=True,
                model_used="None (Groq API Key Missing)",
                warning="GROQ_API_KEY is not configured. Showing retrieved passages only.",
            )

        # Step 4: Construct user prompt and conversation history
        user_prompt = USER_PROMPT_TEMPLATE.format(
            evidence_block=evidence_block,
            question=request.question,
        )

        messages = [{"role": "system", "content": SYSTEM_PROMPT_RESEARCH_QA}]

        # Inject brief recent history if provided
        if request.chat_history:
            for hist in request.chat_history[-4:]:  # last 4 turns
                if hist.role in ("user", "assistant"):
                    messages.append({"role": hist.role, "content": hist.content})

        messages.append({"role": "user", "content": user_prompt})

        # Step 5: Call Groq API
        model_name = settings.groq_model
        try:
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=model_name,
                temperature=0.2,  # Low temperature for strict factual grounding
                max_tokens=1500,
            )
            raw_answer = chat_completion.choices[0].message.content or ""
        except groq.RateLimitError as e:
            return ChatResponse(
                question=request.question,
                answer=(
                    "**[Groq API Rate Limit Reached]**\n\n"
                    "The Groq free-tier rate limit was encountered. "
                    "Please wait a moment before submitting your next question.\n\n"
                    f"Error details: {str(e)}"
                ),
                citations=[],
                unvalidated_citations=[],
                retrieved_chunks=chunks,
                retrieval_mode=retrieval_mode_used,
                is_grounded=False,
                model_used=model_name,
                warning="Rate limit exceeded on Groq free-tier.",
            )
        except groq.APIError as e:
            return ChatResponse(
                question=request.question,
                answer=f"**[Groq API Error]**: {str(e)}",
                citations=[],
                unvalidated_citations=[],
                retrieved_chunks=chunks,
                retrieval_mode=retrieval_mode_used,
                is_grounded=False,
                model_used=model_name,
                warning=f"Groq API call failed: {str(e)}",
            )
        except Exception as e:
            return ChatResponse(
                question=request.question,
                answer=f"An unexpected error occurred during answer generation: {str(e)}",
                citations=[],
                unvalidated_citations=[],
                retrieved_chunks=chunks,
                retrieval_mode=retrieval_mode_used,
                is_grounded=False,
                model_used=model_name,
                warning=str(e),
            )

        # Step 6: Validate citations
        validated_citations, unvalidated = validate_and_build_citations(
            raw_answer, citation_map=citation_map
        )

        warning_msg = None
        if unvalidated:
            warning_msg = (
                f"Note: The model cited {', '.join(unvalidated)} which were not present in the supplied evidence."
            )

        return ChatResponse(
            question=request.question,
            answer=raw_answer,
            citations=validated_citations,
            unvalidated_citations=unvalidated,
            retrieved_chunks=chunks,
            retrieval_mode=retrieval_mode_used,
            is_grounded=len(unvalidated) == 0,
            model_used=model_name,
            warning=warning_msg,
        )


rag_service = RAGService()
