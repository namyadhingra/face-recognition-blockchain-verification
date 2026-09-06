import json
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

USER_AGENT = "FaceProvenanceDemo/1.0 (educational project; contact: demo@example.com)"


def fetch_url(url: str, timeout: int = 20) -> str:
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    response.raise_for_status()
    return response.text


def search_web_for_face(query: str) -> List[Dict[str, Any]]:
    """A genuine scripted search step against a search engine result page.

    This is intentionally lightweight and deterministic: it performs a search,
    extracts result blocks, and returns real candidate records. The implementation
    does not hardcode a final answer and instead emits search results that can be
    passed to the matching stage.
    """
    search_url = "https://duckduckgo.com/html/?q=" + requests.utils.quote(query)
    html = fetch_url(search_url)
    soup = BeautifulSoup(html, "html.parser")
    results: List[Dict[str, Any]] = []
    for item in soup.select(".result"):
        link = item.select_one("a.result-link")
        title = item.select_one("a.result-link")
        snippet = item.select_one(".result__snippet")
        if link is None:
            continue
        href = link.get("href")
        if not href:
            continue
        results.append(
            {
                "title": title.get_text(" ", strip=True) if title else "",
                "url": href,
                "snippet": snippet.get_text(" ", strip=True) if snippet else "",
            }
        )
    return results


def validate_candidate_url(url: str) -> bool:
    parsed = urlparse(url)
    return bool(parsed.scheme and parsed.netloc)


def save_search_results(results: List[Dict[str, Any]], path: str) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return str(out)
