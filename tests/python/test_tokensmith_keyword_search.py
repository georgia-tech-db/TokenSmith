import tempfile
import unittest

from python_engine import tokensmith_store as store


class KeywordSearchTests(unittest.TestCase):
    def insert_material(self, conn, material_id, chunks):
        conn.execute(
            "INSERT INTO collections(id, name) VALUES (?, ?)",
            (material_id, f"Material {material_id}"),
        )
        conn.execute(
            "INSERT INTO folders(id, path) VALUES (?, ?)",
            (material_id, f"/folder/{material_id}"),
        )
        conn.execute("INSERT INTO collection_items VALUES (?, ?)", (material_id, material_id))
        conn.execute(
            "INSERT INTO tokensmith_collection_state"
            "(collection_id, status, is_active, added_at) VALUES (?, ?, ?, ?)",
            (material_id, "ready", 1, store.now_iso()),
        )
        conn.execute(
            "INSERT INTO documents(id, folder_id, document_time, document_path) VALUES (?, ?, 0, ?)",
            (material_id, material_id, f"/folder/{material_id}/paper.pdf"),
        )
        rowids = []
        for text in chunks:
            cursor = conn.execute(
                "INSERT INTO chunks(document_id, chunk_text, file) VALUES (?, ?, ?)",
                (material_id, text, "paper.pdf"),
            )
            rowids.append(cursor.lastrowid)
        return rowids

    def test_filters_materials_before_limit(self):
        with tempfile.TemporaryDirectory() as user_data_path:
            store.init_db(user_data_path)
            with store.connect(user_data_path) as conn:
                for material_id, status, is_active in [
                    (1, "ready", 1),
                    (2, "ready", 1),
                    (3, "ready", 0),
                    (4, "indexing", 1),
                ]:
                    conn.execute("INSERT INTO collections(id, name) VALUES (?, ?)",
                                 (material_id, f"Material {material_id}"))
                    conn.execute("INSERT INTO folders(id, path) VALUES (?, ?)",
                                 (material_id, f"/folder/{material_id}"))
                    conn.execute("INSERT INTO collection_items VALUES (?, ?)",
                                 (material_id, material_id))
                    conn.execute(
                        "INSERT INTO tokensmith_collection_state"
                        "(collection_id, status, is_active, added_at) VALUES (?, ?, ?, ?)",
                        (material_id, status, is_active, store.now_iso()),
                    )
                    conn.execute(
                        "INSERT INTO documents(id, folder_id, document_time, document_path)"
                        " VALUES (?, ?, 0, ?)",
                        (material_id, material_id, f"/folder/{material_id}/paper.pdf"),
                    )
                # Each excluded material alone can fill the old limit * 8 window.
                for material_id in (2, 3, 4):
                    conn.executemany(
                        "INSERT INTO chunks(document_id, chunk_text, file) VALUES (?, ?, ?)",
                        [(material_id, "components", "paper.pdf")] * 20,
                    )
                expected = []
                for text in ("components " + "background " * 30,
                             "components " + "background " * 60):
                    cursor = conn.execute(
                        "INSERT INTO chunks(document_id, chunk_text, file) VALUES (1, ?, ?)",
                        (text, "paper.pdf"),
                    )
                    expected.append(cursor.lastrowid)
                # The same folder belongs to two selected collections.
                conn.execute("INSERT INTO collection_items VALUES (2, 1)")

            for selected in (["1"], ["1", "3", "4"]):
                with self.subTest(selected=selected):
                    hits = store.keyword_search(user_data_path, "components", selected, 2)
                    self.assertEqual([rowid for rowid, _ in hits], expected)
                    self.assertGreater(hits[0][1], hits[1][1])
            with store.connect(user_data_path) as conn:
                conn.execute("DELETE FROM chunks WHERE document_id = 2")
            hits = store.keyword_search(user_data_path, "components", ["1", "2"], 2)
            self.assertEqual([rowid for rowid, _ in hits], expected)
            self.assertEqual(store.keyword_search(user_data_path, "components", ["1"], 1)[0][0], expected[0])
            self.assertEqual(store.keyword_search(user_data_path, "components", [], 2), [])
            self.assertEqual(store.keyword_search(user_data_path, "?!", ["1"], 2), [])
            self.assertEqual(store.keyword_search(user_data_path, "unmatched", ["1"], 2), [])

    def test_keyword_search_uses_rare_terms_instead_of_question_filler(self):
        with tempfile.TemporaryDirectory() as user_data_path:
            store.init_db(user_data_path)
            with store.connect(user_data_path) as conn:
                chunks = [
                    "Why systems do not just use a generic cache.",
                    "Alpha keeps the important frame resident while a worker is actively using it.",
                ]
                chunks.extend(
                    f"Beta replacement chooses a victim frame in cache simulation example {index}."
                    for index in range(20)
                )
                rowids = self.insert_material(
                    conn,
                    1,
                    chunks,
                )

            query = "why cant we just do beta. what is the need for alpha"
            terms = store.keyword_terms_for_query(user_data_path, query, ["1"])

            self.assertEqual(terms[:2], ["alpha", "beta"])
            self.assertNotIn("why", terms)
            self.assertNotIn("the", terms)

            hits = store.keyword_search(user_data_path, query, ["1"], 2)
            self.assertEqual(hits[0][0], rowids[1])

    def test_keyword_search_prefers_chunks_that_match_all_meaningful_terms(self):
        with tempfile.TemporaryDirectory() as user_data_path:
            store.init_db(user_data_path)
            with store.connect(user_data_path) as conn:
                rowids = self.insert_material(
                    conn,
                    1,
                    [
                        "Beta simulation question asks for an access pattern where beta beats gamma.",
                        "Alpha prevents the beta policy from evicting a frame while a worker is using it.",
                        "Alpha keeps active frames resident in the buffer pool.",
                    ],
                )

            query = "why can't we just do beta, what is the need for alpha?"
            hits = store.keyword_search(user_data_path, query, ["1"], 3)

            self.assertEqual(hits[0][0], rowids[1])
            self.assertIn(rowids[2], [rowid for rowid, _score in hits])


if __name__ == "__main__":
    unittest.main()
