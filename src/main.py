import argparse, yaml
from src.preprocess import build_index
from src.ranking.ensemble import EnsembleRanker
from src.ranking.rankers import FaissSimilarityRanker, BM25Ranker, TfIDFRanker
from src.retriever import retrieve, get_candidates, apply_seg_filter
from src.ranker import rerank
from src.generator  import answer

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["index", "chat"])
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--pdf_dir", default="data/chapters/")
    p.add_argument("--index_prefix", default="textbook_index")
    p.add_argument("--model_path", default="models/qwen2.5-0.5b-instruct-q5_k_m.gguf")

    # Extra indexing knobs
    p.add_argument("--pdf_range", type=str, default=None, help="e.g., 27-33")
    p.add_argument("--chunk_mode", choices=["tokens", "chars", "sections","sliding-tokens"], default="sliding-tokens")
    p.add_argument("--chunk_tokens", type=int, default=500)
    p.add_argument("--chunk_size_char", type=int, default=20000)
    p.add_argument("--keep_tables", action="store_true")
    p.add_argument("--visualize", action="store_true")

    return p.parse_args()

def main():
    args = parse_args()
    cfg  = yaml.safe_load(open(args.config))

    if args.mode == "index":
        # Optional range filtering
        if args.pdf_range:
            start, end = map(int, args.pdf_range.split("-"))
            pdf_paths = [f"{i}.pdf" for i in range(start, end)]
        else:
            pdf_paths = None

        build_index(
            pdf_dir=args.pdf_dir,
            out_prefix=args.index_prefix,
            model_name=cfg.get("embed_model", args.model_path),
            chunk_size_char=args.chunk_size_char,
            chunk_mode=args.chunk_mode,
            chunk_tokens=args.chunk_tokens,
            keep_tables=args.keep_tables,
            pdf_files=pdf_paths,
            do_visualize=args.visualize
        )
        print("Index built ✓")

    elif args.mode == "chat":
        from src.retriever import load_artifacts
        index, chunks, sources, vectorizer, chunk_tags = load_artifacts(args.index_prefix)

        print("📚 Ready. Type 'exit' to quit.")
        while True:
            q = input("\nAsk > ").strip()
            if q.lower() in {"exit","quit"}:
                break

            pool_n = max(cfg.get("pool_size", 60), cfg["top_k"] + 10)
            cand_idxs, faiss_dists = get_candidates(
                q, pool_n, index, chunks,
                embed_model=cfg.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2"),
            )

            # 2) shared context for various rankers
            context = {
                "faiss_distances": faiss_dists, # for FaissSimilarityRanker
                "vectorizer": vectorizer,  # for TfIDFRanker
                "chunk_tags": chunk_tags,  # for TfIDFRanker
            }

            # 3) build rankers + ensemble (using weights from config)
            rankers = [
                FaissSimilarityRanker(),
                BM25Ranker(),
                TfIDFRanker(),
            ]
            weights = cfg.get("ranker_weights", {"faiss": 0.6, "bm25": 0.25, "tf-idf": 0.15})
            method = cfg.get("ensemble_method", "linear")
            rrf_k = int(cfg.get("rrf_k", 60)) # unused as rrf method is not used for fusing

            ensemble = EnsembleRanker(method, rankers, weights, rrf_k=rrf_k)
            ordered = ensemble.rank(query=q, chunks=chunks, cand_idxs=cand_idxs, context=context)

            topk_idxs = apply_seg_filter(cfg, chunks, ordered)

            # 4) materialize indices into text and continue
            ranked_chunks = [chunks[i] for i in topk_idxs]

            # HALO Stub (NO OP for now)
            ranked_chunks = rerank(q, ranked_chunks, mode=cfg.get("halo_mode", "none"))

            # cands  = retrieve(
            #     q, cfg["top_k"], index, chunks,
            #     embed_model=cfg.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2"),
            #     seg_filter=cfg.get("seg_filter"),
            #     preview=True,                      # hide 100-char previews
            #     sources=sources,
            #     vectorizer=vectorizer,
            #     chunk_tags=chunk_tags,
            # )
            # ranked = rerank(q, cands, mode=cfg.get("halo_mode", "none"))

            ans = answer(
                q, ranked_chunks, args.model_path,
                max_tokens=cfg.get("max_gen_tokens", 400),
            )
            print("\n=== ANSWER =========================================\n")
            print(ans if ans.strip() else "(no output)")
            print("\n====================================================\n")

if __name__ == "__main__":
    main()