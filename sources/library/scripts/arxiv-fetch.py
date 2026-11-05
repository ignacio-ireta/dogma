#!/usr/bin/env python3
"""
arXiv Paper Fetcher for the Library project.
Downloads papers from arXiv with polite rate limiting.

Usage:
    python3 arxiv-fetch.py "search query" [--max 5]
    python3 arxiv-fetch.py --id 1111.6088
    python3 arxiv-fetch.py --queue  # process the download queue
"""

import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import time
import os
import json
import argparse
import re
from pathlib import Path

LIBRARY_ROOT = Path(__file__).resolve().parents[1]  # sources/library in this repo
PAPERS_DIR = Path(__file__).resolve().parents[3] / "corpus" / "raw" / "library-papers"
QUEUE_FILE = LIBRARY_ROOT / "scripts" / "download-queue.json"
WAIT_SECONDS = 3  # polite rate limiting

ARXIV_API = "http://export.arxiv.org/api/query"
ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"


def search_arxiv(query, max_results=5):
    """Search arXiv and return parsed entries."""
    params = urllib.parse.urlencode({
        "search_query": query,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending"
    })
    url = f"{ARXIV_API}?{params}"
    print(f"Querying: {url}")
    
    req = urllib.request.Request(url, headers={"User-Agent": "deepml-arxiv-fetch/1.0"})
    with urllib.request.urlopen(req) as response:
        data = response.read()
    
    root = ET.fromstring(data)
    entries = []
    for entry in root.findall(f"{ATOM_NS}entry"):
        arxiv_id = entry.find(f"{ATOM_NS}id").text.split("/abs/")[-1]
        title = entry.find(f"{ATOM_NS}title").text.strip().replace("\n", " ")
        summary = entry.find(f"{ATOM_NS}summary").text.strip()[:300]
        authors = [a.find(f"{ATOM_NS}name").text for a in entry.findall(f"{ATOM_NS}author")]
        published = entry.find(f"{ATOM_NS}published").text[:10]
        
        # Find PDF link
        pdf_url = None
        for link in entry.findall(f"{ATOM_NS}link"):
            if link.get("title") == "pdf":
                pdf_url = link.get("href")
        
        entries.append({
            "id": arxiv_id,
            "title": title,
            "authors": authors,
            "published": published,
            "summary": summary,
            "pdf_url": pdf_url or f"https://arxiv.org/pdf/{arxiv_id}"
        })
    
    return entries


def download_pdf(arxiv_id, pdf_url=None):
    """Download a paper PDF from arXiv."""
    if pdf_url is None:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    
    safe_id = arxiv_id.replace("/", "_")
    pdf_path = PAPERS_DIR / f"{safe_id}.pdf"
    
    if pdf_path.exists():
        print(f"Already downloaded: {pdf_path}")
        return pdf_path
    
    print(f"Downloading {arxiv_id} from {pdf_url}...")
    req = urllib.request.Request(pdf_url, headers={"User-Agent": "deepml-arxiv-fetch/1.0"})
    with urllib.request.urlopen(req) as response:
        PAPERS_DIR.mkdir(parents=True, exist_ok=True)
        with open(pdf_path, "wb") as f:
            f.write(response.read())
    
    print(f"Saved: {pdf_path}")
    return pdf_path


def load_queue():
    """Load the download queue."""
    if QUEUE_FILE.exists():
        with open(QUEUE_FILE) as f:
            return json.load(f)
    return []


def save_queue(queue):
    """Save the download queue."""
    QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(QUEUE_FILE, "w") as f:
        json.dump(queue, f, indent=2)


def add_to_queue(arxiv_id, title="", context=""):
    """Add a paper to the download queue."""
    queue = load_queue()
    if any(item["id"] == arxiv_id for item in queue):
        print(f"Already in queue: {arxiv_id}")
        return
    queue.append({
        "id": arxiv_id,
        "title": title,
        "context": context,
        "added": time.strftime("%Y-%m-%d")
    })
    save_queue(queue)
    print(f"Added to queue: {arxiv_id} — {title}")


def process_queue():
    """Download all papers in the queue."""
    queue = load_queue()
    if not queue:
        print("Queue is empty!")
        return
    
    downloaded = []
    for item in queue:
        try:
            download_pdf(item["id"])
            downloaded.append(item["id"])
            time.sleep(WAIT_SECONDS)
        except Exception as e:
            print(f"Error downloading {item['id']}: {e}")
    
    # Remove downloaded items from queue
    queue = [item for item in queue if item["id"] not in downloaded]
    save_queue(queue)
    print(f"\nDownloaded {len(downloaded)} papers. {len(queue)} remaining in queue.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="arXiv Paper Fetcher")
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("--id", help="Download specific arXiv ID")
    parser.add_argument("--max", type=int, default=5, help="Max search results")
    parser.add_argument("--queue", action="store_true", help="Process download queue")
    parser.add_argument("--add", help="Add arXiv ID to queue")
    parser.add_argument("--title", default="", help="Title for queue item")
    
    args = parser.parse_args()
    
    if args.queue:
        process_queue()
    elif args.add:
        add_to_queue(args.add, args.title)
    elif args.id:
        download_pdf(args.id)
    elif args.query:
        entries = search_arxiv(args.query, args.max)
        for e in entries:
            print(f"\n{'='*60}")
            print(f"ID: {e['id']}")
            print(f"Title: {e['title']}")
            print(f"Authors: {', '.join(e['authors'])}")
            print(f"Published: {e['published']}")
            print(f"Summary: {e['summary']}...")
    else:
        parser.print_help()
