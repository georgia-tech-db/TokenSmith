import json
import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import requests

logger = logging.getLogger(__name__)


class LocalScheduler:
    """Background scheduler that forwards logged queries to the remote server."""

    def __init__(
        self,
        remote_url: str = "http://localhost:8001",
        logs_dir: str = "logs",
        poll_interval: float = 60.0,
    ):
        self.remote_url = remote_url.rstrip("/")
        self.logs_dir = Path(logs_dir)
        self.poll_interval = poll_interval

        self._processed_files: Set[str] = set()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ #
    #  Log reading                                                        #
    # ------------------------------------------------------------------ #

    def _read_new_logs(self) -> List[Dict]:
        """Read unprocessed log files and return list of {filename, query}."""
        if not self.logs_dir.exists():
            return []

        new_entries = []
        for log_file in sorted(self.logs_dir.glob("chat_*.json")):
            if log_file.name in self._processed_files:
                continue

            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                query = data.get("query", "").strip()
                if query:
                    new_entries.append({"filename": log_file.name, "query": query})

                # Mark as processed regardless of whether we extracted a query
                self._processed_files.add(log_file.name)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to read log file %s: %s", log_file.name, e)

        return new_entries

    # ------------------------------------------------------------------ #
    #  Remote communication                                               #
    # ------------------------------------------------------------------ #

    def _send_to_remote(self, queries: List[str]) -> List[Dict]:
        """POST a batch of queries to the remote scheduler. Returns list of results."""
        url = f"{self.remote_url}/scheduler/run"

        try:
            response = requests.post(
                url,
                json={"queries": queries},
                timeout=300,  # generous timeout for large-model inference
            )
            response.raise_for_status()
            return response.json().get("results", [])
        except requests.RequestException as e:
            logger.error("Failed to reach remote scheduler at %s: %s", url, e)
            return []

    def _check_remote_health(self) -> bool:
        """Check if the remote scheduler is reachable."""
        url = f"{self.remote_url}/scheduler/health"
        try:
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    # ------------------------------------------------------------------ #
    #  Cache stub                                                         #
    # ------------------------------------------------------------------ #

    def _update_cache(self, results: List[Dict]) -> None:
        """
        Stub: store remote results in the local cache.

        TODO: The cache team will replace this with actual cache writes.
        For now, just log what would be cached.
        """
        for result in results:
            query = result.get("query", "")
            answer = result.get("answer", "")
            logger.info(
                "[cache stub] Would cache answer for query: '%s' (%d chars)",
                query[:80],
                len(answer),
            )

    # ------------------------------------------------------------------ #
    #  Main loop                                                          #
    # ------------------------------------------------------------------ #

    def _poll_once(self) -> int:
        """Run one poll cycle. Returns number of queries processed."""
        new_logs = self._read_new_logs()
        if not new_logs:
            return 0

        queries = [entry["query"] for entry in new_logs]
        logger.info("Sending %d queries to remote scheduler", len(queries))

        results = self._send_to_remote(queries)

        if results:
            self._update_cache(results)
            logger.info("Received %d results from remote", len(results))
        else:
            logger.warning("No results received from remote for %d queries", len(queries))

        return len(queries)

    def run(self) -> None:
        """Main loop: poll logs → send to remote → update cache → sleep."""
        logger.info(
            "Local scheduler started (remote=%s, poll_interval=%.0fs)",
            self.remote_url,
            self.poll_interval,
        )

        while not self._stop_event.is_set():
            try:
                self._poll_once()
            except Exception as e:
                logger.error("Unexpected error in poll cycle: %s", e)

            self._stop_event.wait(timeout=self.poll_interval)

        logger.info("Local scheduler stopped")

    # ------------------------------------------------------------------ #
    #  Thread management                                                  #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Start the scheduler in a background daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Scheduler is already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self.run, daemon=True, name="local-scheduler")
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        """Signal the scheduler to stop and wait for the thread to finish."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="TokenSmith Local Scheduler")
    parser.add_argument(
        "--remote-url",
        default="http://localhost:8001",
        help="URL of the remote scheduler (default: http://localhost:8001)",
    )
    parser.add_argument(
        "--logs-dir",
        default="logs",
        help="Directory containing query log files (default: logs)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=60.0,
        help="Seconds between poll cycles (default: 60)",
    )
    args = parser.parse_args()

    scheduler = LocalScheduler(
        remote_url=args.remote_url,
        logs_dir=args.logs_dir,
        poll_interval=args.poll_interval,
    )

    try:
        scheduler.run()  # run in foreground when invoked directly
    except KeyboardInterrupt:
        print("\nShutting down...")
        scheduler.stop()
