import argparse, yaml, uuid
from src.preprocess import build_index
from src.retriever  import retrieve
from src.ranker import rerank
from src.generator  import answer
from src.feedback_db import FeedbackDB, FeedbackEntry
from src.prompt_selector import select_prompt_style_for_query
from src.feedback_analyzer import FeedbackAnalyzer
from src.system_improver import SystemImprover
import json

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
        # Initialize feedback system
        feedback_db = FeedbackDB()
        session_id = str(uuid.uuid4())

        from src.retriever import load_artifacts
        index, chunks, sources, vectorizer, chunk_tags = load_artifacts(args.index_prefix)

        print("📚 Ready. Type 'exit' to quit, 'stats' for feedback stats, 'improve' for improvement suggestions, 'apply' to apply improvements, 'dashboard' for full analytics, 'prompt' to test prompt styles.")
        while True:
            q = input("\nAsk > ").strip()
            if q.lower() in {"exit","quit"}:
                break
            elif q.lower() == "stats":
                _show_feedback_stats(feedback_db)
                continue
            elif q.lower() == "improve":
                _show_improvement_suggestions(feedback_db)
                continue
            elif q.lower() == "apply":
                _apply_improvements_simple(feedback_db)
                continue
            elif q.lower() == "dashboard":
                _launch_dashboard(feedback_db)
                continue
            elif q.lower() == "prompt":
                _test_prompt_styles(feedback_db, index, chunks, cfg, args.model_path)
                continue

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


            prompt_style = select_prompt_style_for_query(q)
            print(f"Using prompt style: {prompt_style}")

            ans = answer(
                q, ranked, args.model_path,
                max_tokens=cfg.get("max_gen_tokens", 400),
                prompt_style=prompt_style,
            )
            print("\n=== ANSWER =========================================\n")
            print(ans if ans.strip() else "(no output)")
            print("\n====================================================\n")
            
            _collect_feedback(feedback_db, q, ans, ranked, session_id, prompt_style)

def _collect_feedback(feedback_db: FeedbackDB, query: str, answer: str, 
                     retrieved_chunks: list, session_id: str, 
                     prompt_style: str = "default"):
    print("\nHow was this answer? (Press Enter to skip)")
    
    thumbs_feedback = input("Thumbs up (y) or thumbs down (n): ").strip().lower()
    thumbs_up = None
    if thumbs_feedback in ['y', 'yes']:
        thumbs_up = True
    elif thumbs_feedback in ['n', 'no']:
        thumbs_up = False
    

    rating = None
    if thumbs_up is not None:
        try:
            rating_input = input("Rating (1-5, Enter to skip): ").strip()
            if rating_input:
                rating = int(rating_input)
                if not 1 <= rating <= 5:
                    rating = None
        except ValueError:
            pass
    

    comment = input("Any comments or suggestions? (Enter to skip): ").strip()
    

    if thumbs_up is not None or comment:
        feedback = FeedbackEntry(
            query=query,
            answer=answer,
            retrieved_chunks=json.dumps(retrieved_chunks),
            thumbs_up=thumbs_up,
            rating=rating,
            comment=comment,
            session_id=session_id,
            prompt_style=prompt_style
        )
        feedback_id = feedback_db.add_feedback(feedback)
        print(f"Feedback recorded (ID: {feedback_id})")
    else:
        print("Feedback skipped")

def _show_feedback_stats(feedback_db: FeedbackDB):
    stats = feedback_db.get_feedback_stats()
    print("\nFEEDBACK STATISTICS")
    print("=" * 50)
    print(f"Total feedback entries: {stats['total_feedback']}")
    print(f"Thumbs up rate: {stats['thumbs_up_rate']:.1%}")
    print(f"Average rating: {stats['avg_rating']:.1f}/5.0")
    print(f"Comments received: {stats['comments_count']}")
    

    negative_feedback = feedback_db.get_negative_feedback()
    if negative_feedback:
        print(f"\nRecent issues ({len(negative_feedback)} entries):")
        for fb in negative_feedback[:3]:
            print(f"  • Query: {fb['query'][:50]}...")
            if fb['comment']:
                print(f"    Comment: {fb['comment'][:100]}...")
            print()

def _show_improvement_suggestions(feedback_db: FeedbackDB):
    print("\nIMPROVEMENT SUGGESTIONS")
    print("=" * 50)
    

    analyzer = FeedbackAnalyzer(feedback_db)
    improver = SystemImprover(feedback_db)
    

    analysis = analyzer.analyze_feedback()
    improvements = improver.analyze_and_improve()
    

    if analysis.query_patterns:
        print("Query patterns needing attention:")
        for pattern in analysis.query_patterns[:3]:
            status = "NEEDS_ATTENTION" if pattern['needs_attention'] else "GOOD"
            print(f"  {status} '{pattern['query'][:40]}...' (freq: {pattern['frequency']}, success: {pattern['success_rate']:.1%})")
    
    if analysis.common_issues:
        print(f"\nFocus areas (from {len(analysis.common_issues)} issue types):")
        for issue in analysis.common_issues[:3]:
            print(f"  • {issue['type'].replace('_', ' ').title()}: {issue['count']} occurrences")
    

    if improvements:
        print(f"\nSystem improvements available ({len(improvements)} suggestions):")
        for i, imp in enumerate(improvements[:3], 1):
            confidence_level = "HIGH" if imp.confidence >= 0.7 else "MEDIUM" if imp.confidence >= 0.6 else "LOW"
            print(f"  {i}. {confidence_level} {imp.description}")
            print(f"     {imp.parameter}: {imp.old_value} → {imp.new_value} (Confidence: {imp.confidence:.1%})")
    else:
        print("\nNo system improvements needed based on current feedback.")
    
    if analysis.priority_actions:
        print(f"\nPriority actions:")
        for i, action in enumerate(analysis.priority_actions, 1):
            print(f"  {i}. {action}")

def _apply_improvements_simple(feedback_db: FeedbackDB):
    print("\nAPPLY IMPROVEMENTS")
    print("-" * 25)
    
    improver = SystemImprover(feedback_db)
    improvements = improver.analyze_and_improve()
    
    if not improvements:
        print("No improvements available to apply.")
        return
    high_confidence = [imp for imp in improvements if imp.confidence >= 0.7]
    
    if high_confidence:
        print(f"High-confidence improvements available ({len(high_confidence)}):")
        for i, imp in enumerate(high_confidence, 1):
            print(f"  {i}. {imp.description}")
            print(f"     {imp.parameter}: {imp.old_value} → {imp.new_value}")
        
        apply_choice = input(f"\nApply all {len(high_confidence)} high-confidence improvements? (y/n): ").strip().lower()
        
        if apply_choice in ['y', 'yes']:
            results = improver.apply_improvements(high_confidence, dry_run=False)
            
            print(f"\nRESULTS:")
            print(f"Applied: {len(results['applied'])} improvements")
            if results['skipped']:
                print(f"Skipped: {len(results['skipped'])} improvements")
            if results['errors']:
                print(f"Errors: {len(results['errors'])}")
            
            print("\nSystem improvements applied! Restart chat to see changes.")
        else:
            print("Improvements not applied.")
    else:
        print("No high-confidence improvements available.")
        print("Use 'improve' command to see all suggestions, or 'dashboard' for full control.")

def _launch_dashboard(feedback_db: FeedbackDB):
    from feedback_dashboard import FeedbackDashboard
    dashboard = FeedbackDashboard(feedback_db)
    dashboard.show_main_dashboard()

def _test_prompt_styles(feedback_db: FeedbackDB, index, chunks, cfg, model_path):
    print("\nPROMPT STYLE TESTING")
    print("-" * 30)
    
    test_query = input("Enter a test query: ").strip()
    if not test_query:
        print("No query provided")
        return
    

    cands = retrieve(
        test_query, cfg["top_k"], index, chunks,
        embed_model=cfg.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2"),
        seg_filter=cfg.get("seg_filter"),
        preview=False,
    )
    ranked = rerank(test_query, cands, mode=cfg.get("halo_mode", "none"))
    

    styles = ["default", "detailed", "simple", "focused"]
    
    for style in styles:
        print(f"\nTesting '{style}' style:")
        print("=" * 50)
        
        ans = answer(
            test_query, ranked, model_path,
            max_tokens=cfg.get("max_gen_tokens", 400),
            prompt_style=style,
        )
        print(f"\n{ans if ans.strip() else '(no output)'}")
        print("=" * 50)
        

        feedback = input(f"Rate this '{style}' answer (1-5, Enter to skip): ").strip()
        if feedback:
            try:
                rating = int(feedback)
                if 1 <= rating <= 5:

                    test_feedback = FeedbackEntry(
                        query=test_query,
                        answer=ans,
                        retrieved_chunks=json.dumps(ranked),
                        rating=rating,
                        session_id="prompt_test",
                        prompt_style=style
                    )
                    feedback_db.add_feedback(test_feedback)
                    print(f"Test feedback recorded for '{style}' style")
            except ValueError:
                pass

if __name__ == "__main__":
    main()