"""Latest baseball web/news fallback search tool."""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from typing import Dict, List

import httpx

logger = logging.getLogger(__name__)


class LatestBaseballSearchTool:
    """Search latest baseball information from a lightweight public news feed."""

    NEWS_RSS_URL = "https://news.google.com/rss/search"

    @staticmethod
    def _build_query(query: str) -> str:
        query_lower = query.lower()
        if any(token in query_lower for token in ("kbo", "야구", "프로야구", "baseball")):
            return query
        return f"KBO 프로야구 {query}"

    @staticmethod
    def _strip_html(text: str) -> str:
        cleaned = re.sub(r"<[^>]+>", " ", text or "")
        cleaned = re.sub(r"\s+", " ", unescape(cleaned)).strip()
        return cleaned

    @staticmethod
    def _parse_published_at(raw_value: str) -> str | None:
        if not raw_value:
            return None
        try:
            published = parsedate_to_datetime(raw_value)
            if published.tzinfo is None:
                published = published.replace(tzinfo=timezone.utc)
            return published.isoformat()
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _extract_source_name(item: ET.Element, title: str) -> str:
        source_tag = item.find("source")
        if source_tag is not None and source_tag.text:
            return source_tag.text.strip()
        if " - " in title:
            return title.rsplit(" - ", 1)[-1].strip()
        return "latest_web"

    def search_latest_baseball(
        self,
        query: str,
        limit: int = 5,
    ) -> Dict[str, object]:
        result: Dict[str, object] = {
            "query": query,
            "results": [],
            "found": False,
            "error": None,
            "source": "web_search",
            "as_of_date": datetime.now(timezone.utc).date().isoformat(),
        }

        enriched_query = self._build_query(query)
        params = {
            "q": enriched_query,
            "hl": "ko",
            "gl": "KR",
            "ceid": "KR:ko",
        }

        try:
            with httpx.Client(timeout=8.0, follow_redirects=True) as client:
                response = client.get(
                    self.NEWS_RSS_URL,
                    params=params,
                    headers={"User-Agent": "BEGA-AI/1.0"},
                )
                response.raise_for_status()

            root = ET.fromstring(response.text)
            items = root.findall(".//item")
            seen: set[str] = set()
            rows: List[Dict[str, object]] = []

            for item in items:
                title = self._strip_html(item.findtext("title", ""))
                link = (item.findtext("link", "") or "").strip()
                description = self._strip_html(item.findtext("description", ""))
                source_name = self._extract_source_name(item, title)
                published_at = self._parse_published_at(item.findtext("pubDate", ""))
                snippet = description or title
                dedupe_key = link or title
                if not dedupe_key or dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                rows.append(
                    {
                        "title": title,
                        "snippet": snippet,
                        "url": link,
                        "source_name": source_name,
                        "published_at": published_at,
                    }
                )
                if len(rows) >= limit:
                    break

            result["results"] = rows
            result["found"] = len(rows) > 0
        except Exception as exc:  # noqa: BLE001
            logger.warning("[LatestBaseballSearchTool] latest search failed: %s", exc)
            result["error"] = f"latest_search_failed:{type(exc).__name__}"

        return result
