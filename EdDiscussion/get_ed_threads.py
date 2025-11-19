import os
import json
import requests

def request(method, url, **kwargs):
    status = ''
    body = ''

    try:
        r = requests.request(method, url, **kwargs)
        status = r.status_code
        body = r.text

        if 200 <= r.status_code <= 299:
            return r
        else:
            raise Exception('Bad status code: ', status)

    except Exception as e:
        print('Request failed')
        print(f'Status: {status}')
        print(f'URL: {url}')
        print(f'Body: {body}')
        raise e


def get_all_threads(base_url, token, limit=30, max_threads=None):
    all_threads = []
    offset = 0

    headers = {"Authorization": "Bearer " + token}

    while True:
        if max_threads is not None and len(all_threads) >= max_threads:
            break

        url = f"{base_url}?limit={limit}&offset={offset}"
        r = request("GET", url, headers=headers)
        data = r.json()

        threads = data.get("threads", [])
        if not threads:
            break

        for t in threads:
            if max_threads is not None and len(all_threads) >= max_threads:
                break
            all_threads.append(t)

        offset += limit

    return all_threads


def get_thread_details(thread_id, token):
    url = f"https://us.edstem.org/api/threads/{thread_id}?view=1"
    headers = {"Authorization": "Bearer " + token}
    r = request("GET", url, headers=headers)
    return r.json()


def extract_combined_text(thread_json):
    t = thread_json["thread"]

    parts = []

    # Base fields
    parts.append(str(t.get("title", "")))
    parts.append(str(t.get("content", "")))
    parts.append(str(t.get("document", "")))

    # Thread-level comments
    for c in t.get("comments", []):
        parts.append(str(c.get("content", "")))
        parts.append(str(c.get("document", "")))

    # Answers and their comments
    for ans in t.get("answers", []):
        parts.append(str(ans.get("content", "")))
        parts.append(str(ans.get("document", "")))

        for c in ans.get("comments", []):
            parts.append(str(c.get("content", "")))
            parts.append(str(c.get("document", "")))

    combined = "\n".join([p for p in parts if p.strip() != ""])
    return {
        "id": t["id"],
        "combined_text": combined
    }


def main():
    token = os.getenv("ED_API")
    if not token:
        raise Exception("Environment variable ED_API is not set")

    base_url = "https://us.edstem.org/api/courses/82583/threads"
    MAX_THREADS = 300

    # Fetch thread list (ids only)
    threads = get_all_threads(
        base_url,
        token,
        limit=30,
        max_threads=MAX_THREADS
    )

    output = []

    for t in threads:
        thread_id = t.get("id")
        if thread_id is None:
            continue

        full = get_thread_details(thread_id, token)
        reduced = extract_combined_text(full)
        output.append(reduced)

    with open("threads_reduced.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved {len(output)} threads to threads_reduced.json")


if __name__ == '__main__':
    main()
