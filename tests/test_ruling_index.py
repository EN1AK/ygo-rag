import json

import pytest

from rag_agent.adjudication import (
    RuleSection,
    YgocdbRulingIndex,
)
from rag_agent.cards import Card
from rag_agent.ruling_index import (
    RULING_INDEX_SCHEMA_VERSION,
    RulingCandidate,
    RulingDocument,
    RulingVectorStore,
    PersistentRulingRetriever,
    SparseRulingRetriever,
    build_persistent_ruling_index,
    build_rule_ruling_documents,
    build_ruling_llm_rerank_prompt,
    build_ygocdb_ruling_documents,
    create_manifest,
    hybrid_ruling_retrieval,
    parse_ruling_llm_rerank_response,
    rerank_ruling_candidates_locally,
    read_manifest,
    rerank_ruling_candidates_with_llm,
    validate_manifest,
    write_manifest,
)


def test_ygocdb_faq_and_supplement_documents_preserve_provenance():
    index = YgocdbRulingIndex.from_payloads(
        [
            {
                "id": 100,
                "faqs": [
                    {
                        "fid": "f1",
                        "title": "Destroy replacement",
                        "question": "Can it replace destruction?",
                        "answer": "Yes.",
                        "refer": {"200": True},
                    }
                ],
                "supplement": {"text": "Supplement text.", "refer": {"300": True}},
            }
        ],
        snapshot_at="snapshot",
    )
    cards = {
        100: Card(100, "Alpha", "Destroy.", category=0x2),
        200: Card(200, "Beta", "Replace.", category=0x4000000),
    }

    documents = build_ygocdb_ruling_documents(index, cards)
    by_id = {document.evidence_id: document for document in documents}

    faq = by_id["ygocdb:faq:f1"]
    assert faq.kind == "faq"
    assert faq.primary_card_ids == (100,)
    assert faq.refer_card_ids == (200,)
    assert faq.metadata["source_url"].endswith("/100?show=all")
    assert "Alpha" in faq.searchable_text
    assert "Beta" in faq.searchable_text

    supplement = by_id["ygocdb:supplement:100"]
    assert supplement.kind == "card_supplement"
    assert supplement.primary_card_ids == (100,)
    assert supplement.refer_card_ids == (300,)
    assert supplement.to_evidence().relationship == "card_supplement"


def test_rule_documents_preserve_anchor_metadata():
    documents = build_rule_ruling_documents(
        type(
            "RuleIndexLike",
            (),
            {
                "sections": [
                    RuleSection(
                        rule_id="ocgrules:c03/rule.xhtml#replace",
                        title="Replacement",
                        chapter_path=("Chapter",),
                        xhtml_path="c03/rule.xhtml",
                        anchor="replace",
                        text="Destroy replacement applies.",
                    )
                ]
            },
        )()
    )

    document = documents[0]

    assert document.evidence_id == "ocgrules:c03/rule.xhtml#replace"
    assert document.kind == "rule_passage"
    assert document.metadata["xhtml_path"] == "c03/rule.xhtml"
    assert document.to_evidence().relationship == "rule_document"


def test_manifest_write_read_and_validation(tmp_path):
    path = tmp_path / "manifest.json"
    manifest = create_manifest(
        index_kind="faq",
        persist_dir=tmp_path / "chroma",
        collection_name="ruling_faqs",
        source_paths=[tmp_path / "ygocdb"],
        source_count=2,
        document_count=3,
        embedding_model="model",
    )

    write_manifest(path, manifest)
    loaded = read_manifest(path)
    status = validate_manifest(
        path,
        index_kind="faq",
        collection_name="ruling_faqs",
        embedding_model="model",
    )
    stale = validate_manifest(
        path,
        index_kind="faq",
        collection_name="other",
        embedding_model="model",
    )

    assert loaded is not None
    assert loaded.schema_version == RULING_INDEX_SCHEMA_VERSION
    assert status.usable is True
    assert stale.usable is False
    assert stale.warnings


def test_sparse_retrieval_labels_direct_related_and_analogous():
    documents = [
        RulingDocument(
            evidence_id="ygocdb:faq:direct",
            kind="faq",
            source="ygocdb",
            title="Direct",
            text="Destroy replacement.",
            primary_card_ids=(100,),
            card_ids=(100,),
        ),
        RulingDocument(
            evidence_id="ygocdb:faq:related",
            kind="faq",
            source="ygocdb",
            title="Related",
            text="Destroy replacement.",
            refer_card_ids=(100,),
            card_ids=(100,),
        ),
        RulingDocument(
            evidence_id="ygocdb:faq:analogous",
            kind="faq",
            source="ygocdb",
            title="Analogous",
            text="Later special summon is not applied.",
        ),
    ]

    results = SparseRulingRetriever(documents).search(
        "Destroy replacement special summon",
        card_ids=(100,),
        top_k=3,
    )

    relationships = {candidate.evidence_id: candidate.relationship for candidate in results}
    assert relationships["ygocdb:faq:direct"] == "direct_card"
    assert relationships["ygocdb:faq:related"] == "related_card"
    assert relationships["ygocdb:faq:analogous"] == "analogous"


def test_sparse_rule_retrieval_filters_by_epub_chapter():
    documents = [
        RulingDocument(
            evidence_id="ocgrules:replacement",
            kind="rule_passage",
            source="ocgrules",
            title="代替破坏",
            text="破坏被代替后的处理。",
            metadata={"xhtml_path": "c03/resistance.xhtml"},
        ),
        RulingDocument(
            evidence_id="ocgrules:chain",
            kind="rule_passage",
            source="ocgrules",
            title="连锁处理",
            text="连锁中的破坏处理。",
            metadata={"xhtml_path": "c02/chain.xhtml"},
        ),
    ]

    results = SparseRulingRetriever(documents).search(
        "破坏 处理",
        chapter_ids=("c03/resistance.xhtml",),
        top_k=5,
    )

    assert [candidate.evidence_id for candidate in results] == [
        "ocgrules:replacement"
    ]


def test_hybrid_ruling_retrieval_deduplicates_by_evidence_id():
    doc = RulingDocument(
        evidence_id="ygocdb:faq:1",
        kind="faq",
        source="ygocdb",
        title="FAQ",
        text="Text",
    )

    results = hybrid_ruling_retrieval(
        "q",
        sparse_candidates=[
            RulingCandidate("ygocdb:faq:1", 10.0, "sparse", doc, "analogous")
        ],
        dense_candidates=[
            RulingCandidate("ygocdb:faq:1", 0.9, "dense", doc, "analogous")
        ],
    )

    assert [candidate.evidence_id for candidate in results] == ["ygocdb:faq:1"]
    assert results[0].source == "hybrid"


def test_ruling_llm_rerank_rejects_unknown_ids_and_preserves_unscored():
    doc1 = RulingDocument("a", "faq", "ygocdb", "A", "Text A")
    doc2 = RulingDocument("b", "faq", "ygocdb", "B", "Text B")
    candidates = [
        RulingCandidate("a", 0.1, "hybrid", doc1, "analogous"),
        RulingCandidate("b", 0.2, "hybrid", doc2, "analogous"),
    ]

    prompt = build_ruling_llm_rerank_prompt("query", candidates)
    parsed = parse_ruling_llm_rerank_response(
        json.dumps(
            {
                "results": [
                    {"evidence_id": "missing", "score": 1.0, "reason": "bad"},
                    {"evidence_id": "b", "score": 0.9, "reason": "best"},
                ]
            }
        ),
        allowed_ids={"a", "b"},
    )

    assert "evidence_id" in prompt
    assert [(item.evidence_id, item.score) for item in parsed] == [("b", 0.9)]


def test_local_rerank_maps_numeric_candidates_back_to_evidence_ids():
    class FakeLocalReranker:
        def rerank(self, query, candidates):
            return list(reversed(candidates))

    doc1 = RulingDocument("a", "faq", "ygocdb", "A", "Text A")
    doc2 = RulingDocument("b", "faq", "ygocdb", "B", "Text B")
    candidates = [
        RulingCandidate("a", 0.1, "hybrid", doc1, "analogous"),
        RulingCandidate("b", 0.2, "hybrid", doc2, "analogous"),
    ]

    results = rerank_ruling_candidates_locally(
        "query",
        candidates,
        FakeLocalReranker(),
    )

    assert [candidate.evidence_id for candidate in results] == ["b", "a"]
    assert all(candidate.source == "reranker" for candidate in results)


def test_ruling_llm_rerank_falls_back_on_malformed_output():
    class BadLlm:
        def invoke(self, prompt):
            return "not json"

    warnings = []
    doc1 = RulingDocument("a", "faq", "ygocdb", "A", "Text A")
    doc2 = RulingDocument("b", "faq", "ygocdb", "B", "Text B")
    candidates = [
        RulingCandidate("a", 0.1, "hybrid", doc1, "analogous"),
        RulingCandidate("b", 0.2, "hybrid", doc2, "analogous"),
    ]

    results = rerank_ruling_candidates_with_llm(
        "query",
        candidates,
        llm=BadLlm(),
        max_candidates=2,
        warning_callback=warnings.append,
    )

    assert [candidate.evidence_id for candidate in results] == ["a", "b"]
    assert warnings


def test_persistent_index_build_writes_to_vector_store_and_manifest(tmp_path):
    class FakeStore(RulingVectorStore):
        def __init__(self):
            super().__init__(tmp_path / "chroma", "ruling_faqs", object())
            self.added = []

        def build(self, documents, *, batch_size=128, reset=False, progress_callback=None):
            self.added.append((list(documents), batch_size, reset))
            if progress_callback:
                progress_callback(len(documents), len(documents))
            return object()

    fake_store = FakeStore()

    def fake_init(self, persist_dir, collection_name, embedding_function):
        self.persist_dir = fake_store.persist_dir
        self.collection_name = fake_store.collection_name
        self.embedding_function = fake_store.embedding_function

    original_build = RulingVectorStore.build
    try:
        RulingVectorStore.build = fake_store.build
        result = build_persistent_ruling_index(
            [
                RulingDocument(
                    evidence_id="ygocdb:faq:1",
                    kind="faq",
                    source="ygocdb",
                    title="FAQ",
                    text="Text",
                )
            ],
            index_kind="faq",
            persist_dir=tmp_path / "chroma",
            collection_name="ruling_faqs",
            manifest_path=tmp_path / "manifest.json",
            source_paths=[tmp_path / "ygocdb"],
            source_count=1,
            embedding_model="model",
            embedding_function=object(),
            batch_size=7,
            reset=True,
        )
    finally:
        RulingVectorStore.build = original_build

    assert result.document_count == 1
    assert result.source_count == 1
    assert read_manifest(tmp_path / "manifest.json") is not None


def test_persistent_ruling_retriever_uses_dense_and_fusion():
    doc_sparse = RulingDocument("sparse", "faq", "ygocdb", "Sparse", "Destroy replacement")
    doc_dense = RulingDocument("dense", "faq", "ygocdb", "Dense", "Unrelated dense text")

    class FakeDense:
        def search(self, query, documents_by_id, *, top_k, card_ids=(), relationship="analogous"):
            return [RulingCandidate("dense", 0.9, "dense", doc_dense, "analogous")]

    retriever = PersistentRulingRetriever(
        [doc_sparse, doc_dense],
        dense_store=FakeDense(),
    )

    result = retriever.search(
        "Destroy replacement",
        sparse_top_k=1,
        dense_top_k=1,
        fused_top_k=2,
        final_top_k=2,
    )

    assert {candidate.evidence_id for candidate in result.candidates} == {"sparse", "dense"}


def test_persistent_ruling_retriever_records_unusable_manifest_warning():
    doc = RulingDocument("a", "faq", "ygocdb", "A", "Destroy")

    class FakeDense:
        def search(self, *args, **kwargs):
            raise AssertionError("dense search should be skipped")

    retriever = PersistentRulingRetriever(
        [doc],
        dense_store=FakeDense(),
        manifest_status=type(
            "Status",
            (),
            {"usable": False, "warnings": ("stale",)},
        )(),
    )

    result = retriever.search("Destroy", semantic=True)

    assert "stale" in result.warnings
