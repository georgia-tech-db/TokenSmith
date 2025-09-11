import argparse, yaml, pathlib
from typing import Optional
from src.preprocess import build_index
from src.retriever  import retrieve
from src.ranker import rerank
from src.generator  import answer

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["index", "chat"])
    p.add_argument("--config", default=None, required=False)
    p.add_argument("--pdf_dir", default="data/chapters/")
    p.add_argument("--index_prefix", default="textbook_index")
    p.add_argument("--model_path", default="build/models/qwen2.5-0.5b-instruct-q5_k_m.gguf")

    # Extra indexing knobs
    p.add_argument("--pdf_range", type=str, default=None, help="e.g., 27-33")
    p.add_argument("--chunk_mode", choices=["tokens", "chars", "sections","sliding-tokens"], default="sliding-tokens")
    p.add_argument("--chunk_tokens", type=int, default=500)
    p.add_argument("--chunk_size_char", type=int, default=20000)
    p.add_argument("--keep_tables", action="store_true")
    p.add_argument("--visualize", action="store_true")

    return p.parse_args()

def load_correct_fallback_config_file() -> Optional[any]:
    user_config = pathlib.Path("~/.config/tokensmith/config.yaml")
    user_config_alt = pathlib.Path("~/.config/tokensmith/config.yml")
    default_config = pathlib.Path("config/config.yaml")
    
    if user_config.exists():
        with user_config.open("r") as f:
            return yaml.safe_load(f)
    
    if user_config_alt.exists():
        with user_config_alt.open("r") as f:
            return yaml.safe_load(f)
    
    if default_config.exists():
        with default_config.open("r") as f:
            return yaml.safe_load(f)
    
    return None

def main():
    args = parse_args()
    
    # load config file from argument. If none provided, open fallback
    cfg = None
    if args.config is not None:
        cfg  = yaml.safe_load(open(args.config))
    else:
        cfg = load_correct_fallback_config_file()
    
    if cfg is None:
        raise ValueError("Default config file not found. Expected at config/config.yaml")

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

            cands  = retrieve(
                q, cfg["top_k"], index, chunks,
                embed_model=cfg.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2"),
                seg_filter=cfg.get("seg_filter"),
                preview=True,                      # hide 100-char previews
                sources=sources,
                vectorizer=vectorizer,
                chunk_tags=chunk_tags,
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