import argparse, yaml
from src.preprocess import build_index
from src.retriever  import retrieve
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

            # Get retrieval configuration from config.yaml
            retrieval_method = cfg.get("retrieval_method", "hybrid")
            faiss_weight = cfg.get("faiss_weight", 0.5)
            bm25_weight = cfg.get("bm25_weight", 0.3)
            tag_weight = cfg.get("tag_weight", 0.2)
            
            # Normalize weights if they don't sum to 1.0
            total_weight = faiss_weight + bm25_weight + tag_weight
            if total_weight > 0:
                faiss_weight = faiss_weight / total_weight
                bm25_weight = bm25_weight / total_weight
                tag_weight = tag_weight / total_weight

            cands  = retrieve(
                q, cfg.get("top_k", 5), index, chunks,
                embed_model=cfg.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2"),
                seg_filter=cfg.get("seg_filter"),
                preview=True,                      # hide 100-char previews
                sources=sources,
                vectorizer=vectorizer,
                chunk_tags=chunk_tags,
                bm25_weight=bm25_weight,
                tag_weight=tag_weight,
            )
            ranked = rerank(q, cands, mode=cfg.get("halo_mode", "none"))

            # Use generator_model from config if available, fallback to args.model_path
            model_path = cfg.get("generator_model", args.model_path)
            
            ans = answer(
                q, ranked, model_path,
                max_tokens=cfg.get("max_gen_tokens", 400),
                system_prompt_mode=cfg.get("system_prompt_mode", "tutor"),
            )
            print("\n=== ANSWER =========================================\n")
            print(ans if ans.strip() else "(no output)")
            print("\n====================================================\n")

if __name__ == "__main__":
    main()