"""Web search module using SerpApi Google Lens for reverse image search.

Given a probe face image, this module performs a genuine reverse image search
on the web to find pages where the same face appears. It returns candidate
URLs with thumbnails and metadata that the pipeline can then verify.

Fallback: if SerpApi is unavailable (no key, quota exhausted), falls back to
a DuckDuckGo text-based search using the image filename as a query. This is
degraded but keeps the pipeline functional for testing.
"""

import base64
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

USER_AGENT = "FaceProvenanceDemo/1.0 (educational project; contact: demo@example.com)"


# ---------------------------------------------------------------------------
# SerpApi Google Lens — primary search path
# ---------------------------------------------------------------------------

def search_by_face_serpapi(image_path: str, api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """Perform a reverse image search using SerpApi Google Lens.

    Uploads the probe image and returns a list of visual matches found on
    the web.  Each result contains at minimum a 'title', 'url', and
    'thumbnail' (when available).

    Args:
        image_path: Local path to the probe face image.
        api_key: SerpApi API key.  Falls back to the SERPAPI_KEY env var.

    Returns:
        List of dicts, each with keys: title, url, thumbnail, source.
    """
    key = api_key or os.environ.get("SERPAPI_KEY", "")
    if not key:
        raise ValueError(
            "No SerpApi key provided. Set SERPAPI_KEY in .env or pass api_key=."
        )

    try:
        from serpapi import GoogleSearch
    except ImportError:
        raise ImportError(
            "Install the SerpApi client: pip install google-search-results"
        )

    # Upload the image as a file to SerpApi Google Lens
    image_path = str(Path(image_path).resolve())

    # First, we need to make the image accessible via URL for Google Lens.
    # SerpApi's Google Lens accepts a URL parameter. For local files we
    # upload to SerpApi's endpoint directly.
    params = {
        "engine": "google_lens",
        "api_key": key,
    }

    # Use the file upload approach
    search = GoogleSearch(params)

    # SerpApi supports local file upload via the 'image' parameter (base64)
    # or via URL. We'll try URL-based approach first with a data URI,
    # but SerpApi actually needs a publicly accessible URL or file upload.
    # The recommended approach is to use their upload mechanism.

    # Read and encode the image
    image_bytes = Path(image_path).read_bytes()

    # SerpApi Google Lens can accept a URL. For local files, we need to
    # use their direct file upload via the REST API.
    results = _serpapi_lens_with_upload(image_path, key)

    return results


def _serpapi_lens_with_upload(image_path: str, api_key: str) -> List[Dict[str, Any]]:
    """Call SerpApi Google Lens with direct file upload via REST API."""

    url = "https://serpapi.com/search.json"
    
    with open(image_path, "rb") as img_file:
        files = {"file": img_file}
        params = {
            "engine": "google_lens",
            "api_key": api_key,
        }
        resp = requests.post(url, params=params, files=files, timeout=60)

    if resp.status_code != 200:
        # Try the GET-based approach with a URL parameter
        # For this we need a publicly accessible URL, so try an image
        # hosting workaround or fall back.
        raise RuntimeError(
            f"SerpApi returned status {resp.status_code}: {resp.text[:300]}"
        )

    data = resp.json()
    return _parse_lens_results(data)


def _serpapi_lens_with_url(image_url: str, api_key: str) -> List[Dict[str, Any]]:
    """Call SerpApi Google Lens with an image URL (for images already online)."""

    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_lens",
        "url": image_url,
        "api_key": api_key,
    }
    resp = requests.get(url, params=params, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(
            f"SerpApi returned status {resp.status_code}: {resp.text[:300]}"
        )
    data = resp.json()
    return _parse_lens_results(data)


def _parse_lens_results(data: dict) -> List[Dict[str, Any]]:
    """Parse the SerpApi Google Lens JSON response into our standard format."""

    results = []

    # Google Lens returns visual_matches, knowledge_graph, etc.
    for match in data.get("visual_matches", []):
        results.append({
            "title": match.get("title", ""),
            "url": match.get("link", ""),
            "thumbnail": match.get("thumbnail", ""),
            "source": match.get("source", ""),
            "position": match.get("position", 0),
        })

    # Also check knowledge_graph for identified entities
    kg = data.get("knowledge_graph", [])
    if isinstance(kg, list):
        for item in kg:
            if item.get("link"):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "thumbnail": item.get("thumbnail", ""),
                    "source": "knowledge_graph",
                    "position": 0,
                })

    return results


# ---------------------------------------------------------------------------
# SerpApi Google Reverse Image Search — alternative approach
# ---------------------------------------------------------------------------

def search_by_image_google(image_path: str, api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """Use SerpApi's Google Reverse Image Search as an alternative to Lens.

    This uses the 'google_reverse_image' engine which can accept a file
    upload and returns pages containing the same or similar images.
    """
    key = api_key or os.environ.get("SERPAPI_KEY", "")
    if not key:
        raise ValueError("No SerpApi key. Set SERPAPI_KEY in .env.")

    url = "https://serpapi.com/search.json"

    with open(image_path, "rb") as img_file:
        files = {"file": img_file}
        params = {
            "engine": "google_reverse_image",
            "api_key": key,
        }
        resp = requests.post(url, params=params, files=files, timeout=60)

    if resp.status_code != 200:
        raise RuntimeError(
            f"SerpApi returned status {resp.status_code}: {resp.text[:300]}"
        )

    data = resp.json()
    results = []

    # Parse image results
    for item in data.get("image_results", []):
        results.append({
            "title": item.get("title", ""),
            "url": item.get("link", ""),
            "thumbnail": item.get("thumbnail", ""),
            "source": item.get("source", ""),
            "position": item.get("position", 0),
        })

    # Parse inline_images if available
    for item in data.get("inline_images", []):
        results.append({
            "title": item.get("title", ""),
            "url": item.get("source", ""),
            "thumbnail": item.get("thumbnail", item.get("original", "")),
            "source": "inline",
            "position": 0,
        })

    return results


# ---------------------------------------------------------------------------
# DuckDuckGo fallback — no API key required
# ---------------------------------------------------------------------------

def search_web_duckduckgo(query: str) -> List[Dict[str, Any]]:
    """Fallback: text-based search via DuckDuckGo HTML endpoint.

    Used when SerpApi is unavailable.  This is a degraded search path —
    it searches by text query, not by image.
    """
    search_url = "https://duckduckgo.com/html/?q=" + requests.utils.quote(query)
    resp = requests.get(
        search_url, headers={"User-Agent": USER_AGENT}, timeout=20
    )
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    results: List[Dict[str, Any]] = []

    for item in soup.select(".result"):
        link = item.select_one("a.result__a")
        snippet = item.select_one(".result__snippet")
        if link is None:
            continue
        href = link.get("href")
        if not href:
            continue
        results.append({
            "title": link.get_text(" ", strip=True),
            "url": href,
            "thumbnail": "",
            "source": "duckduckgo_fallback",
            "snippet": snippet.get_text(" ", strip=True) if snippet else "",
        })

    return results


# ---------------------------------------------------------------------------
# Unified search interface
# ---------------------------------------------------------------------------

def search_for_face(image_path: str, api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """Search the web for matches of a face image.

    Tries SerpApi Google Lens first, then Google Reverse Image Search,
    then falls back to DuckDuckGo text search.

    Args:
        image_path: Path to the probe face image.
        api_key: Optional SerpApi key (falls back to SERPAPI_KEY env var).

    Returns:
        List of candidate results with title, url, thumbnail, source.
    """
    key = api_key or os.environ.get("SERPAPI_KEY", "")

    # Try SerpApi approaches
    if key:
        # Try Google Lens first
        try:
            results = search_by_face_serpapi(image_path, key)
            if results:
                print(f"  [SerpApi Google Lens] Found {len(results)} visual matches")
                return results
        except Exception as e:
            print(f"  [SerpApi Google Lens] Failed: {e}")

        # Try Google Reverse Image Search
        try:
            results = search_by_image_google(image_path, key)
            if results:
                print(f"  [SerpApi Reverse Image] Found {len(results)} matches")
                return results
        except Exception as e:
            print(f"  [SerpApi Reverse Image] Failed: {e}")

    # Fallback: DuckDuckGo text search
    filename = Path(image_path).stem.replace("_", " ")
    query = f"{filename} face profile"
    print(f"  [DuckDuckGo fallback] Searching for: {query}")
    try:
        results = search_web_duckduckgo(query)
        if results:
            print(f"  [DuckDuckGo] Found {len(results)} results")
            return results
    except Exception as e:
        print(f"  [DuckDuckGo] Failed: {e}")

    return []


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def validate_url(url: str) -> bool:
    """Check that a URL has a valid scheme and netloc."""
    parsed = urlparse(url)
    return bool(parsed.scheme in ("http", "https") and parsed.netloc)


def download_image(url: str, timeout: int = 20) -> Optional[bytes]:
    """Download an image from a URL, returning raw bytes or None on failure."""
    try:
        resp = requests.get(
            url, headers={"User-Agent": USER_AGENT}, timeout=timeout
        )
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "image" in content_type or url.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        ):
            return resp.content
    except Exception:
        pass
    return None


def save_search_results(results: List[Dict[str, Any]], path: str) -> str:
    """Persist search results to a JSON file for provenance."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return str(out)
