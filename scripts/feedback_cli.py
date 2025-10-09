import argparse
from feedback_db import FeedbackDB


def list_feedback(limit: int) -> None:
    db = FeedbackDB()
    rows = db.get_recent_feedback(limit=limit)
    if not rows:
        print("No feedback found.")
        return
    for i, fb in enumerate(rows, 1):
        if fb.get("thumbs_up") is True:
            thumbs = "THUMBS_UP"
        elif fb.get("thumbs_up") is False:
            thumbs = "THUMBS_DOWN"
        else:
            thumbs = "NO_FEEDBACK"
        rating = f"RATING_{fb.get('rating')}" if fb.get("rating") else "NO_RATING"
        print(f"{i}. {thumbs} {rating} | {fb.get('timestamp','')[:19]} | {fb.get('query','')[:80]}")
        if fb.get("comment"):
            print(f"   COMMENT: {fb['comment'][:120]}")


def show_stats() -> None:
    db = FeedbackDB()
    stats = db.get_feedback_stats()
    print("Total Interactions:", stats.get("total_feedback", 0))
    print("Success Rate:", f"{(stats.get('thumbs_up_rate') or 0)*100:.1f}%")
    print("Average Rating:", f"{(stats.get('avg_rating') or 0):.2f}/5.00")
    print("Comments Count:", stats.get("comments_count", 0))


def main() -> None:
    parser = argparse.ArgumentParser(description="TokenSmith Feedback CLI")
    sub = parser.add_subparsers(dest="command")

    p_list = sub.add_parser("list", help="List recent feedback")
    p_list.add_argument("--limit", type=int, default=20, help="Number of entries to list")

    sub.add_parser("stats", help="Show aggregate feedback stats")

    args = parser.parse_args()
    if args.command == "list":
        list_feedback(limit=args.limit)
    elif args.command == "stats":
        show_stats()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()


