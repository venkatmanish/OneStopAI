from core.db import Chunk
from core.retrieval import HybridRetriever


def test_search_query_for_follow_up_prefers_current_question_and_previous_answer():
    query = """Use the conversation context to answer this follow-up.
Recent conversation:
User[kb (document=bachelor_certificate.pdf)]: what's college name which is providing this bachelors certificate
Assistant[kb (document=bachelor_certificate.pdf)]: The college is VNR Vignana Jyothi Institute of Engineering & Technology.
Follow-up question: graduated in?"""

    search_query = HybridRetriever._search_query(query)

    assert "graduated in?" in search_query
    assert "VNR Vignana Jyothi" in search_query
    assert "Use the conversation context" not in search_query


def test_search_query_for_labeled_follow_up_includes_prior_answer():
    query = """Use the conversation context to answer this follow-up.
Recent conversation:
User[kb (document=DP-800 Study Plan.pdf)]: when will the results get released for this
Assistant[kb (document=DP-800 Study Plan.pdf)]: The results are typically released well after the beta exam ends.
Follow-up question: any estimated or expected date"""

    search_query = HybridRetriever._search_query(query)

    assert "any estimated or expected date" in search_query
    assert "released well after the beta exam ends" in search_query
    assert "Assistant[kb" not in search_query


def test_search_query_for_term_follow_up_uses_term_only():
    query = """Use the conversation context to answer this term follow-up.
Term to explain: beta
Task: explain what this term means here. Do not answer the previous question again.
Recent conversation:
User: this exam is being conducted by whom?
Assistant: This exam is being conducted by Microsoft.
Follow-up question: beta?"""

    search_query = HybridRetriever._search_query(query)

    assert search_query == "beta meaning definition context"
    assert "Microsoft" not in search_query
    assert "Use the conversation context" not in search_query


def test_merge_scores_ignores_vector_hits_outside_candidate_chunks():
    chunks = [
        Chunk(chunk_id="inside", text="target chunk"),
        Chunk(chunk_id="other", text="other target chunk"),
    ]

    merged = HybridRetriever._merge_scores(
        chunks,
        bm25_scores={"inside": 0.2, "other": 0.1},
        vector_scores={"inside": 0.9, "outside": 1.0},
    )

    assert [chunk_id for chunk_id, _ in merged] == ["inside", "other"]


def test_rerank_promotes_exact_term_coverage():
    retriever = HybridRetriever()
    relevant = Chunk(chunk_id="relevant", text="The student graduated in Computer Science.")
    weak = Chunk(chunk_id="weak", text="The college name is shown on the certificate.")

    reranked = retriever._rerank("graduated in", [(weak, 0.1), (relevant, 0.1)])

    assert reranked[0][0].chunk_id == "relevant"


def test_document_type_target_matches_only_requested_file_type():
    ppt_chunk = Chunk(chunk_id="ppt", text="slide text", extra={"file_name": "GXS Deck.pptx"})
    pdf_chunk = Chunk(chunk_id="pdf", text="pdf text", extra={"file_name": "GXS Report.pdf"})

    assert HybridRetriever._matches_target(ppt_chunk, "__document_type__:presentation") is True
    assert HybridRetriever._matches_target(pdf_chunk, "__document_type__:presentation") is False


def test_expanded_queries_add_task_specific_variants():
    queries = HybridRetriever._expanded_queries("gxs nexus one empathic message 4 steps")

    assert queries[0] == "gxs nexus one empathic message 4 steps"
    assert any("process sequence workflow" in query for query in queries)
    assert any("communication template outreach" in query for query in queries)


def test_parent_context_participates_in_rerank_and_excerpt():
    retriever = HybridRetriever()
    weak = Chunk(chunk_id="weak", text="generic slide text", extra={})
    relevant = Chunk(
        chunk_id="relevant",
        text="short child chunk",
        extra={"parent_context": "The slide describes one empathic message with four steps."},
    )

    reranked = retriever._rerank("one empathic message four steps", [(weak, 0.1), (relevant, 0.1)])
    retrieved = HybridRetriever._to_retrieved(relevant, 0.9, "empathic message")

    assert reranked[0][0].chunk_id == "relevant"
    assert "empathic message" in retrieved.source.excerpt


def test_mmr_select_promotes_source_diversity():
    retriever = HybridRetriever()
    first = Chunk(chunk_id="first", document_id="doc-a", text="alpha beta gamma shared content")
    duplicate = Chunk(chunk_id="duplicate", document_id="doc-a", text="alpha beta gamma shared content")
    diverse = Chunk(chunk_id="diverse", document_id="doc-b", text="alpha beta delta extra evidence")

    selected = retriever._mmr_select(
        "alpha beta evidence",
        [(first, 0.9), (duplicate, 0.89), (diverse, 0.86)],
        top_k=2,
    )

    assert [chunk.chunk_id for chunk, _ in selected] == ["first", "diverse"]
