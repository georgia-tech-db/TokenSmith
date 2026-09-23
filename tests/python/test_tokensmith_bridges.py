"""Bridge-chunk contracts; retrieval quality is measured separately on real documents."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from python_engine import tokensmith_bridges as bridges
from python_engine import tokensmith_engine as engine
from python_engine import tokensmith_preparation as prep
from python_engine import tokensmith_store as store

DIMENSION = 32
# A vocabulary gap: this phrasing shares no words with the eviction section, but does with the
# fillers, so the section falls below the top-4 window.
GAP_PHRASING = "what happens when there is no room left in storage"
BOOK = [
    "# Buffer pool eviction\n" + "When the pool is full the evict method removes the least "
    "recently used page and writes it back to disk if the dirty bit is set. " * 6,
    "# Dirty bit\n" + "The dirty bit is a flag on each frame recording whether the page was "
    "modified since it was loaded from disk. " * 6,
    "# Trivial\nToo short to matter.",
] + [
    f"# Filler {n}\n" + f"Unrelated topic {n} about indexing, joins, parsing and storage layout. " * 6
    for n in range(6)
]


def embed(texts):
    """Deterministic bag-of-words embedding: texts sharing words land near each other."""
    matrix = np.zeros((len(texts), DIMENSION), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in bridges.tokens(text):
            digest = hashlib.sha256(word.encode()).digest()
            matrix[row][int.from_bytes(digest[:4], "little") % DIMENSION] += 1.0
    return bridges.normalize_rows(matrix)


def fake_gen(prompt, _temperature, _num_predict):
    if "casually ask" in prompt:
        if "evict" in prompt:
            return f"{GAP_PHRASING}\nhow does it pick which page to kick out\n"
        return "what does the dirty bit flag mean\nwhy track modified pages\n"
    return ("When the pool is full, evict removes the least recently used page and writes it to "
            "disk if its dirty bit is set.")


class BridgeTests(unittest.TestCase):
    def test_inventory_prefers_source_units_over_headers(self):
        units = ["u1", "u1", None] + [None] * 6
        inventory = bridges.build_inventory(BOOK, unit_ids=units)
        self.assertIn("unit:u1", inventory)
        self.assertNotIn("Buffer pool eviction", inventory)
        # without units it falls back to headers, and trivial sections are skipped
        by_header = bridges.build_inventory(BOOK)
        self.assertIn("Buffer pool eviction", by_header)
        self.assertNotIn("Trivial", by_header)

    def test_only_missing_phrasings_are_bridged_and_the_budget_caps_growth(self):
        result, stats = bridges.generate_bridges(
            BOOK, embed(BOOK), embed_fn=embed, gen_fn=fake_gen, budget_fraction=1.0
        )
        texts = [item["text"] for item in result]
        self.assertTrue(any(text.startswith(f"Question: {GAP_PHRASING}?") for text in texts))
        for item in result:
            self.assertEqual(item["tokensmithChunkKind"], bridges.BRIDGE_KIND)
            self.assertTrue(item["sectionHeader"].startswith(bridges.BRIDGE_HEADER_PREFIX))
            self.assertIn("least recently used", item["text"])  # carries the mechanism, not just the term
        self.assertLessEqual(stats["selected"], stats["budget"])
        self.assertLessEqual(len(result), stats["selected"] * bridges.N_PHRASINGS)

    def test_meta_answers_are_rejected_and_exercises_are_not_bridged(self):
        def meta_gen(prompt, _t, _n):
            if "casually ask" in prompt:
                return f"{GAP_PHRASING}\n"
            return "This passage describes a mechanism where the least recently used page is evicted."
        result, stats = bridges.generate_bridges(BOOK, embed(BOOK), embed_fn=embed, gen_fn=meta_gen, budget_fraction=1.0)
        self.assertEqual(result, [])  # commentary about "the passage" is never indexed as content
        self.assertGreater(stats["selected"], 0)  # the concept was selected; only its answer failed
        quiz = ["# 5.4.5 Check Your Understanding\n" + "Propose a set of changes to add a dirty bit. " * 8]
        self.assertEqual(bridges.build_inventory(quiz), {})

    def test_prepared_materials_store_bridges_without_a_source_unit(self):
        """The AI-preparation path persists in one transaction; bridges must ride along and must
        not be given a parent_id, or source-unit expansion would try to merge them."""
        with tempfile.TemporaryDirectory() as root:
            path = (Path(root) / "study.txt").resolve()
            path.write_text("Opening\n" + "Buffer pool eviction explained.\n" * 170 + "Conclusion\n")
            chunks = prep.materialize(prep.source_blocks([{"page": 1, "text": path.read_text()}]),
                                      {"title": "Unit", "kind": "source unit", "reason": "One unit"})
            row = {"text": "Question: what if memory is full?\nThe pool evicts the least recently used page.",
                   "sectionHeader": "Bridge: Unit", "tokensmithChunkKind": "bridge",
                   "path": str(path), "documentTitle": "study", "embedding": [0.0, 1.0]}
            with patch("python_engine.tokensmith_preparation_job.prepare_blocks", return_value=chunks), \
                 patch.object(engine, "resolve_embedding_provider_from_spec", return_value=("test", lambda _: [1.0, 0.0], None)), \
                 patch.object(engine, "bridge_chunk_rows", return_value=([row], {"bridges": 1})):
                engine.index_material({"path": str(path), "userDataPath": root, "materialId": "m",
                                       "preparation": {"mode": "ai"}, "preparationModel": {"engine": "ollama"},
                                       "model": {}, "generateBridges": True})
            with store.connect(root) as conn:
                stored = [dict(r) for r in conn.execute("SELECT chunk_kind, parent_id FROM chunks")]
            self.assertEqual([r for r in stored if r["chunk_kind"] == "bridge"], [{"chunk_kind": "bridge", "parent_id": None}])
            self.assertTrue(all(r["parent_id"] for r in stored if r["chunk_kind"] != "bridge"))

    def test_bridge_rows_carry_the_source_document_and_an_embedding(self):
        chunks = [{"text": text, "embedding": vec.tolist(), "sectionHeader": None, "parentId": f"u{i}",
                   "path": f"/book/{i}.md", "documentTitle": f"doc{i}"}
                  for i, (text, vec) in enumerate(zip(BOOK, embed(BOOK)))]
        with patch.object(engine, "generator_from_spec", return_value=fake_gen):
            rows, stats = engine.bridge_chunk_rows({"generatorModel": {}}, chunks, lambda t: embed([t])[0].tolist())
        self.assertGreater(len(rows), 0)
        for row in rows:
            self.assertEqual(row["path"], chunks[row["sourceChunk"]]["path"])
            self.assertEqual(len(row["embedding"]), DIMENSION)
            self.assertNotIn("parentId", row)

    def test_generator_is_ollama_only(self):
        self.assertIsNone(bridges.generator_from_spec(None))
        self.assertIsNone(bridges.generator_from_spec({"engine": "python", "path": "/x.gguf"}))
        self.assertIsNotNone(bridges.generator_from_spec({"engine": "ollama", "ollamaModelName": "llama3"}))

    def test_keyword_arm_excludes_bridges(self):
        sql, params = store._active_chunks_filter(["1"])
        self.assertIn("<> 'bridge'", sql)
        self.assertEqual(params, ["1"])

    def test_reserved_slot_fusion_keeps_the_book_window_intact(self):
        book = [{"rowid": i} for i in range(1, 6)]
        bridge = {"rowid": 99, "chunk_kind": "bridge"}
        # a bridge that beats the best book chunk leads; the window never grows
        self.assertEqual([r["rowid"] for r in bridges.splice_bridge_row(book, bridge, 0.80, 0.70, 4)],
                         [99, 1, 2, 3])
        # otherwise it takes the last slot
        self.assertEqual([r["rowid"] for r in bridges.splice_bridge_row(book, bridge, 0.60, 0.70, 4)],
                         [1, 2, 3, 99])
        # below the threshold none is admitted and the book window is untouched
        self.assertEqual([r["rowid"] for r in bridges.splice_bridge_row(book, bridge, 0.40, 0.70, 4)],
                         [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
