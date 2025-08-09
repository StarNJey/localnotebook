# ───────────────────────────────────── crawler.py
from __future__ import annotations
import re, time, hashlib
import requests
from dataclasses import dataclass
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin, quote_plus
from typing import List, Dict, Any, Optional

@dataclass
class CrawledDocument:
    url: str
    domain: str
    title: str
    text: str
    crawl_time: str                     # ISO-8601
    source: str                         # e.g. "web:https://…"
    page: int = 0                       # Dummy to reuse existing fields
    paragraph: int = 0
    chunk_id: str = ""

class WebCrawler:
    """간단한 GET + BeautifulSoup 기반 HTML 크롤러 (JS 미실행)"""
    USER_AGENT = (
        "Mozilla/5.0 (compatible; DeepResearchBot/1.0; "
        "+https://example.com/bot)"
    )

    def __init__(self, timeout: int = 8):
        self.timeout = timeout

    # 1) 검색(duckduckgo lite) → 상위 N개 URL 반환
    def search(self, query: str, top_k: int = 6) -> List[str]:
        qs = quote_plus(query)
        url = f"https://duckduckgo.com/html/?q={qs}&kl=kr-ko"
        res = requests.get(url, headers={"User-Agent": self.USER_AGENT}, timeout=8)
        soup = BeautifulSoup(res.text, "html.parser")
        links = [
            a["href"]
            for a in soup.select("a.result__a[href]")
            if a["href"].startswith("http")
        ]
        return links[:top_k]

    # 2) 단일 페이지 크롤링 → CrawledDocument
    def crawl_page(self, url: str) -> Optional[CrawledDocument]:
        try:
            r = requests.get(
                url, headers={"User-Agent": self.USER_AGENT}, timeout=self.timeout
            )
            if "text/html" not in r.headers.get("Content-Type", ""):
                return None
            soup = BeautifulSoup(r.text, "html.parser")

            # 제목 & 본문 추출
            title = soup.title.string.strip() if soup.title else url
            texts = soup.get_text(" ", strip=True)
            clean = re.sub(r"\s{2,}", " ", texts)
            domain = urlparse(url).netloc

            doc = CrawledDocument(
                url=url,
                domain=domain,
                title=title[:200],
                text=clean[:15_000],            # LLM 토큰 폭 방지
                crawl_time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                source=url,
            )
            # hash는 기존 chunk 체계 재사용
            hid = hashlib.md5((url + doc.text[:200]).encode()).hexdigest()[:8]
            doc.chunk_id = hid
            return doc
        except Exception:
            return None

    # 3) end-to-end: 검색 → 페이지 크롤링
    def crawl(self, query: str, top_k: int = 6) -> List[CrawledDocument]:
        docs: List[CrawledDocument] = []
        for link in self.search(query, top_k=top_k):
            page = self.crawl_page(link)
            if page and len(page.text) > 300:   # 노이즈 필터
                docs.append(page)
        return docs
