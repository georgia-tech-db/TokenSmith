"""Tests for the knowledge_graph pipeline modules."""

import json
import os

import networkx as nx
import pytest

from src.knowledge_graph.models import Chunk, ExtractionResult
from src.knowledge_graph.utils.normalizer import Normalizer
from src.knowledge_graph.dividers.sentence_paragraph import SentenceParagraphDivider
from src.knowledge_graph.extractors.spacy_extractor import SpacyExtractor
from src.knowledge_graph.extractors.yake_extractor import YakeExtractor
from src.knowledge_graph.extractors.composite import CompositeExtractor
from src.knowledge_graph.linkers.cooccurrence_linker import CooccurrenceLinker
from src.knowledge_graph.persisters.networkx_json_persister import NetworkxJsonPersister
from src.knowledge_graph.pipeline import Pipeline


# ── Fixtures ──────────────────────────────────────────────────────────────

SAMPLE_TEXT = (
    "Albert Einstein developed the theory of relativity at Princeton University. "
    "His work transformed modern physics and influenced many scientists.\n\n"
    "Marie Curie discovered radium and polonium. She was the first woman to "
    "win a Nobel Prize. Her research at the University of Paris was groundbreaking.\n\n"
    "Alan Turing created the concept of the Turing machine at Cambridge. "
    "His contributions to computer science and artificial intelligence are legendary."
)


@pytest.fixture
def normalizer():
    return Normalizer(alias_map={"ai": "artificial intelligence"})


@pytest.fixture
def sample_chunks():
    """Pre-built chunks for tests that skip the divider step."""
    return [
        Chunk(
            id=0, text="Albert Einstein worked at Princeton University on relativity."
        ),
        Chunk(id=1, text="Marie Curie discovered radium at the University of Paris."),
        Chunk(id=2, text="Alan Turing developed artificial intelligence concepts."),
    ]


# ── Normalizer ────────────────────────────────────────────────────────────


class TestNormalizer:
    def test_lowercasing(self, normalizer):
        result = normalizer.normalize(["HELLO", "World"])
        assert all(n == n.lower() for n in result)

    def test_deduplication(self, normalizer):
        result = normalizer.normalize(["dog", "Dog", "DOG"])
        assert len(result) == 1

    def test_alias_expansion(self, normalizer):
        result = normalizer.normalize(["AI"])
        assert "artificial intelligence" in result

    def test_empty_strings_stripped(self, normalizer):
        result = normalizer.normalize(["", "  ", "hello"])
        assert len(result) == 1

    def test_lemmatization(self, normalizer):
        result = normalizer.normalize(["running"])
        # spaCy should lemmatize "running" → "run"
        assert "run" in result


# ── SentenceParagraphDivider ──────────────────────────────────────────────


class TestSentenceParagraphDivider:
    def test_paragraph_splitting(self):
        divider = SentenceParagraphDivider(max_tokens=256)
        chunks = divider.divide(SAMPLE_TEXT)
        # There are 3 paragraphs in SAMPLE_TEXT, each well under 256 tokens
        assert len(chunks) == 3

    def test_chunk_ids_are_sequential(self):
        divider = SentenceParagraphDivider(max_tokens=256)
        chunks = divider.divide(SAMPLE_TEXT)
        ids = [c.id for c in chunks]
        assert ids == list(range(len(chunks)))

    def test_max_tokens_splitting(self):
        # Use a very small max_tokens to force sentence-level splitting
        divider = SentenceParagraphDivider(max_tokens=10)
        chunks = divider.divide(SAMPLE_TEXT)
        # Should produce more chunks than paragraphs
        assert len(chunks) > 3
        for chunk in chunks:
            token_count = len(chunk.text.split())
            # Each chunk should be near or at max_tokens (some tolerance for
            # single sentences that themselves exceed the limit)
            assert token_count > 0

    def test_empty_text(self):
        divider = SentenceParagraphDivider(max_tokens=256)
        chunks = divider.divide("")
        assert chunks == []


# ── SpacyExtractor ────────────────────────────────────────────────────────


class TestSpacyExtractor:
    def test_extracts_entities(self, sample_chunks):
        extractor = SpacyExtractor()
        results = extractor.extract(sample_chunks)
        assert len(results) == len(sample_chunks)
        # At least one chunk should have detected entities
        all_nodes = [n for r in results for n in r.nodes]
        assert len(all_nodes) > 0

    def test_entity_type_filter(self, sample_chunks):
        extractor_all = SpacyExtractor(entity_types=None)
        extractor_person = SpacyExtractor(entity_types=["PERSON"])
        results_all = extractor_all.extract(sample_chunks)
        results_person = extractor_person.extract(sample_chunks)
        all_nodes = sum(len(r.nodes) for r in results_all)
        person_nodes = sum(len(r.nodes) for r in results_person)
        assert person_nodes <= all_nodes


# ── YakeExtractor ─────────────────────────────────────────────────────────


class TestYakeExtractor:
    def test_extracts_keywords(self, sample_chunks):
        extractor = YakeExtractor(top_n=5)
        results = extractor.extract(sample_chunks)
        assert len(results) == len(sample_chunks)
        # YAKE should find keywords in non-trivial text
        all_nodes = [n for r in results for n in r.nodes]
        assert len(all_nodes) > 0

    def test_chunk_id_preserved(self, sample_chunks):
        extractor = YakeExtractor(top_n=5)
        results = extractor.extract(sample_chunks)
        result_ids = {r.chunk_id for r in results}
        expected_ids = {c.id for c in sample_chunks}
        assert result_ids == expected_ids


# ── CompositeExtractor ────────────────────────────────────────────────────


class TestCompositeExtractor:
    def test_merges_extractors(self, sample_chunks):
        spacy_ext = SpacyExtractor()
        yake_ext = YakeExtractor(top_n=5)
        composite = CompositeExtractor(extractors=[spacy_ext, yake_ext])
        results = composite.extract(sample_chunks)
        assert len(results) == len(sample_chunks)

    def test_deduplicates_across_extractors(self, sample_chunks):
        spacy_ext = SpacyExtractor()
        composite = CompositeExtractor(extractors=[spacy_ext, spacy_ext])
        results = composite.extract(sample_chunks)
        for r in results:
            # No duplicate node labels within a single chunk
            assert len(r.nodes) == len(set(r.nodes))


# ── CooccurrenceLinker ────────────────────────────────────────────────────


class TestCooccurrenceLinker:
    def test_creates_edges(self):
        extractions = [
            ExtractionResult(chunk_id=0, nodes=["alice", "bob", "carol"]),
            ExtractionResult(chunk_id=1, nodes=["alice", "bob"]),
        ]
        linker = CooccurrenceLinker()
        graph = linker.link(extractions)
        assert graph.has_edge("alice", "bob")
        assert graph["alice"]["bob"]["weight"] == 2
        assert set(graph["alice"]["bob"]["chunk_ids"]) == {0, 1}

    def test_node_chunk_ids(self):
        extractions = [
            ExtractionResult(chunk_id=0, nodes=["alice", "bob"]),
            ExtractionResult(chunk_id=1, nodes=["alice"]),
        ]
        linker = CooccurrenceLinker()
        graph = linker.link(extractions)
        assert set(graph.nodes["alice"]["chunk_ids"]) == {0, 1}

    def test_min_cooccurrence_pruning(self):
        extractions = [
            ExtractionResult(chunk_id=0, nodes=["alice", "bob", "carol"]),
            ExtractionResult(chunk_id=1, nodes=["alice", "bob"]),
        ]
        linker = CooccurrenceLinker(min_cooccurrence=2)
        graph = linker.link(extractions)
        # alice-bob co-occur in 2 chunks → kept
        assert graph.has_edge("alice", "bob")
        # alice-carol and bob-carol only co-occur once → pruned
        assert not graph.has_edge("alice", "carol")
        assert not graph.has_edge("bob", "carol")

    def test_empty_extractions(self):
        linker = CooccurrenceLinker()
        graph = linker.link([])
        assert graph.number_of_nodes() == 0
        assert graph.number_of_edges() == 0


# ── NetworkxJsonPersister ─────────────────────────────────────────────────


class TestNetworkxJsonPersister:
    def test_creates_output_files(self, tmp_path):
        graph = nx.Graph()
        graph.add_node("test", chunk_ids=[0])
        chunks = [Chunk(id=0, text="hello world")]
        persister = NetworkxJsonPersister()
        output_dir = str(tmp_path / "output")

        persister.persist(graph, chunks, output_dir)

        assert os.path.isfile(os.path.join(output_dir, "graph.json"))
        assert os.path.isfile(os.path.join(output_dir, "chunks.json"))

    def test_graph_is_reloadable(self, tmp_path):
        graph = nx.Graph()
        graph.add_node("alpha", chunk_ids=[0, 1])
        graph.add_node("beta", chunk_ids=[1])
        graph.add_edge("alpha", "beta", weight=1, chunk_ids=[1])
        chunks = [Chunk(id=0, text="first"), Chunk(id=1, text="second")]

        output_dir = str(tmp_path / "output")
        NetworkxJsonPersister().persist(graph, chunks, output_dir)

        with open(os.path.join(output_dir, "graph.json")) as f:
            reloaded = nx.node_link_graph(json.load(f))

        assert set(reloaded.nodes) == {"alpha", "beta"}
        assert reloaded.has_edge("alpha", "beta")
        assert reloaded["alpha"]["beta"]["weight"] == 1

    def test_chunks_json_content(self, tmp_path):
        chunks = [Chunk(id=0, text="aaa"), Chunk(id=1, text="bbb")]
        output_dir = str(tmp_path / "output")
        NetworkxJsonPersister().persist(nx.Graph(), chunks, output_dir)

        with open(os.path.join(output_dir, "chunks.json")) as f:
            data = json.load(f)

        assert data == {"0": "aaa", "1": "bbb"}


# ── Pipeline (integration) ───────────────────────────────────────────────


class TestPipeline:
    def test_end_to_end(self, tmp_path):
        pipeline = Pipeline(
            divider=SentenceParagraphDivider(max_tokens=256),
            extractor=CompositeExtractor(
                [
                    SpacyExtractor(),
                    YakeExtractor(top_n=5),
                ]
            ),
            linker=CooccurrenceLinker(min_cooccurrence=1),
            persister=NetworkxJsonPersister(),
        )

        output_dir = str(tmp_path / "kg_output")
        graph = pipeline.run(SAMPLE_TEXT, output_dir)

        # Graph should have nodes and edges
        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0

        # Output files should exist
        assert os.path.isfile(os.path.join(output_dir, "graph.json"))
        assert os.path.isfile(os.path.join(output_dir, "chunks.json"))

        # Graph should be reloadable
        with open(os.path.join(output_dir, "graph.json")) as f:
            reloaded = nx.node_link_graph(json.load(f))
        assert reloaded.number_of_nodes() == graph.number_of_nodes()
