import argparse, yaml, pathlib
from preprocess import build_index
from retriever  import retrieve
from ranker import rerank
from generator  import answer

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["index", "chat"])
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--pdf_dir", default="chapters/")
    p.add_argument("--index_prefix", default="textbook_index")
    p.add_argument("--model_path", default="./qwen2.5-0.5b-instruct-q5_k_m.gguf")

    # Extra indexing knobs
    p.add_argument("--pdf_range", type=str, default=None, help="e.g., 27-33")
    p.add_argument("--chunk_mode", choices=["tokens", "chars"], default="chars")
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
        import faiss, pickle
        index  = faiss.read_index(f"{args.index_prefix}.faiss")
        chunks = pickle.load(open(f"{args.index_prefix}_chunks.pkl","rb"))

        print("📚 Ready. Type 'exit' to quit.")
        while True:
            q = input("\nAsk > ").strip()
            if q.lower() in {"exit","quit"}:
                break

            cands  = retrieve(
                q, cfg["top_k"], index, chunks,
                embed_model=cfg.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2"),
                seg_filter=cfg.get("seg_filter"),
                preview=False,                      # hide 100-char previews
            )
            ranked = rerank(q, cands, mode=cfg.get("halo_mode", "none"))

            ans = answer(
                q, ranked, args.model_path,
                max_tokens=cfg.get("max_gen_tokens", 400),
            )
            print("\n=== ANSWER =========================================\n")
            print(ans if ans.strip() else "(no output)")
            print("\n====================================================\n")

if __name__ == "__main__":
    main()