-- Example analytics for TokenSmith SQLite workload DB (sqlite3 path/to.db < this file)
-- Tables: queries, retrieval_hits

-- Recent queries
-- SELECT id, created_at, substr(query_text, 1, 80) AS q FROM queries ORDER BY id DESC LIMIT 20;

-- Chunks retrieved most often (by FAISS row id)
-- SELECT chunk_idx, COUNT(*) AS hits
-- FROM retrieval_hits
-- GROUP BY chunk_idx
-- ORDER BY hits DESC
-- LIMIT 20;

-- Average hits per query (sanity)
-- SELECT AVG(cnt) FROM (SELECT query_id, COUNT(*) AS cnt FROM retrieval_hits GROUP BY query_id);
