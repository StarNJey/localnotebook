# app_hybrid.py
import os, re, time, hashlib, math, json
import numpy as np
import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus, urlparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import CrossEncoder

# =========================
# Utilities
# =========================
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def mean_pooling(output, attention_mask):
    token_embeddings = output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# =========================
# Web Crawler (from app4 idea, expanded)
# =========================
@dataclass
class WebDoc:
    url: str
    domain: str
    title: str
    text: str
    crawl_time: str
    source: str
    page: int = 0
    paragraph: int = 0
    chunk_id: str = ""
    similarity: float = 0.0
    rerank_score: float = 0.0

class WebCrawler:
    USER_AGENT = "Mozilla/5.0 (compatible; HybridResearchBot/1.0)"
    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def search(self, query: str, max_results: int = 12) -> List[str]:
        # DuckDuckGo HTML endpoint
        q = quote_plus(query)
        url = f"https://duckduckgo.com/html/?q={q}&kl=kr-ko"
        try:
            r = requests.get(url, headers={"User-Agent": self.USER_AGENT}, timeout=self.timeout)
            soup = BeautifulSoup(r.text, "html.parser")
            links = [a["href"] for a in soup.select("a.result__a[href]") if a["href"].startswith("http")]
            return links[:max_results]
        except Exception:
            return []

    def crawl_page(self, url: str) -> Optional[WebDoc]:
        try:
            r = requests.get(url, headers={"User-Agent": self.USER_AGENT}, timeout=self.timeout)
            ctype = r.headers.get("Content-Type", "")
            if "text/html" not in ctype:
                return None
            soup = BeautifulSoup(r.text, "html.parser")
            title = soup.title.string.strip() if soup.title else url
            text = soup.get_text(" ", strip=True)
            clean = re.sub(r"\s{2,}", " ", text)[:20000]
            if len(clean) < 300:
                return None
            domain = urlparse(url).netloc
            tstamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            h = hashlib.md5((url + clean[:200]).encode()).hexdigest()[:10]
            return WebDoc(
                url=url, domain=domain, title=title[:200], text=clean,
                crawl_time=tstamp, source=url, chunk_id=h
            )
        except Exception:
            return None

    def search_and_crawl(self, query: str, max_results: int = 12) -> List[WebDoc]:
        out: List[WebDoc] = []
        for u in self.search(query, max_results=max_results):
            d = self.crawl_page(u)
            if d:
                out.append(d)
        return out

# =========================
# Analyzer & Validator (from app3 idea, restored/extended)
# =========================
class AnalyzerAgent:
    """
    - 각 문서(PDF/웹) 관련성 점수(0~1)와 핵심 인사이트 한 줄 추출
    - 상위 문서들 간 교차 검증(합치/모순) 요약
    """
    def __init__(self, tok, model, device):
        self.tok, self.model, self.device = tok, model, device

    def _gen(self, prompt: str, max_new_tokens=256) -> str:
        inp = self.tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False,
                                      eos_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def analyse_per_doc(self, query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for d in docs:
            snippet = d["text"][:500]
            p = (
                f"Question: {query}\n"
                f"Snippet: {snippet}\n"
                "Rate relevance 1-10 and give one-line key insight.\n"
                "Format: score: <int>, insight: <text>"
            )
            resp = self._gen(p, 128)
            m = re.search(r"score\s*:\s*(10|[1-9])", resp, re.I)
            score = int(m.group(1)) if m else 6
            insight = ""
            mi = re.search(r"insight\s*:\s*(.+)", resp, re.I)
            if mi: insight = mi.group(1).strip()[:160]
            d["analysis_score"] = round(score / 10.0, 2)
            d["key_insight"] = insight
        return docs

    def cross_validate(self, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(docs) < 2:
            return {"consistency": 1.0, "consensus": [], "conflicts": []}
        bundle = "\n\n".join(f"[{i}] {d['text'][:300]}..." for i, d in enumerate(docs[:8], 1))
        p = (
            "Review the following snippets and:\n"
            "1) List consistent points (prefix with ✓)\n"
            "2) List conflicts (prefix with ✗)\n"
            f"{bundle}\n"
        )
        resp = self._gen(p, 256)
        consensus = re.findall(r"[✓]\s*(.+)", resp)
        conflicts = re.findall(r"[✗]\s*(.+)", resp)
        ratio = round(len(consensus) / (len(consensus) + len(conflicts) + 1e-6), 2)
        return {"consistency": ratio, "consensus": consensus, "conflicts": conflicts}

class ValidatorAgent:
    """
    - 최종 답변이 근거(문서 스니펫)에 연결되는지(grounding)
    - 사실 일치도(LLM 기반 quick check)
    - 종합 신뢰도 점수
    """
    def __init__(self, tok, model, device):
        self.tok, self.model, self.device = tok, model, device

    def _gen(self, prompt: str, max_new_tokens=128) -> str:
        inp = self.tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False,
                                      eos_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def evaluate(self, query: str, answer: str, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 근거 연결: 간단히 source/도메인/파일명이 답변 내 언급되는 비율
        def key_from(d):
            s = d.get("source", "")
            dom = urlparse(s).netloc if s.startswith("http") else os.path.basename(s)
            return (dom or s)[:80]
        grounding_hits = sum(1 for d in docs if key_from(d) and (key_from(d) in answer))
        grounding = grounding_hits / max(len(docs), 1)

        ctx = " ".join(d["text"][:300] for d in docs[:8])
        p = (
            "Rate factual alignment of the Answer with the Context from 0.0 to 1.0. "
            "Return only a float like 0.82.\n"
            f"Question: {query}\nContext: {ctx}\nAnswer: {answer[:1200]}"
        )
        resp = self._gen(p, 64)
        m = re.search(r"(0\.\d+|1\.0)", resp)
        factual = float(m.group(1)) if m else 0.6
        confidence = round(0.4 * grounding + 0.6 * factual, 3)
        return {"grounding": round(grounding, 3), "factual": round(factual, 3), "confidence": confidence}

# =========================
# Embedding / Reranker backbone (shared by app3/app4)
# =========================
class EmbeddingBackend:
    def __init__(self, embed_model_id: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(embed_model_id)
        self.model = AutoModel.from_pretrained(embed_model_id).to(self.device).eval()

    def encode(self, texts: List[str], batch: int = 32) -> np.ndarray:
        vecs = []
        for i in range(0, len(texts), batch):
            batch_texts = texts[i:i+batch]
            enc = self.tok(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512).to(self.device)
            with torch.no_grad():
                out = self.model(**enc)
                pooled = mean_pooling(out, enc["attention_mask"])
            vecs.append(pooled.cpu().numpy())
        return np.vstack(vecs) if vecs else np.zeros((0, 384), dtype=np.float32)

# =========================
# Retrieval & Re-ranking (PDF + Web)
# =========================
class HybridRetriever:
    def __init__(self, embedder: EmbeddingBackend, reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.embedder = embedder
        self.reranker = CrossEncoder(reranker_model)

    def _embed_one(self, text: str) -> np.ndarray:
        return self.embedder.encode([text])[0]

    def retrieve_pdf(self, query: str, pdf_docs: List[Dict[str, Any]], pdf_embs: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        if len(pdf_docs) == 0 or pdf_embs is None or len(pdf_embs) == 0:
            return []
        qv = self._embed_one(query)
        sims = np.array([cosine_sim(qv, e) for e in pdf_embs])
        idx = np.argsort(-sims)[: max(top_k, 0)]
        out = []
        for i in idx:
            d = dict(pdf_docs[i])
            d["similarity"] = float(sims[i])
            d["search_category"] = d.get("search_category", "PDF")
            out.append(d)
        return out

    def filter_and_rerank_web(self, query: str, web_docs: List[WebDoc], keep_top: int = 20) -> List[Dict[str, Any]]:
        if not web_docs:
            return []
        # 1) 임베딩 1차 컷
        qv = self._embed_one(query)
        kept = []
        for w in web_docs:
            sv = self._embed_one(w.text[:2000])
            sim = cosine_sim(qv, sv)
            if sim >= 0.25:
                w.similarity = sim
                kept.append(w)
        if not kept:
            return []
        # 2) Cross-Encoder 재랭킹
        pairs = [(query, w.text[:1200]) for w in kept]
        scores = self.reranker.predict(pairs)
        for w, s in zip(kept, scores):
            w.rerank_score = float(s)
        kept.sort(key=lambda x: x.rerank_score, reverse=True)
        kept = kept[:keep_top]
        # dict 변환
        out: List[Dict[str, Any]] = []
        for w in kept:
            out.append({
                "text": w.text,
                "page": 0,
                "paragraph": 0,
                "source": w.url,
                "domain": w.domain,
                "title": w.title,
                "crawl_time": w.crawl_time,
                "chunk_id": w.chunk_id,
                "search_category": "WEB",
                "similarity": w.similarity,
                "rerank_score": w.rerank_score,
            })
        return out

# =========================
# Synthesizer (answers with source citation policy)
# =========================
class Synthesizer:
    """
    - PDF와 WEB 소스를 구분해 출처를 반드시 표기
    - 웹 소스가 답변 논거로 쓰일 경우: [도메인, URL, 크롤링시각] 포함
    """
    def __init__(self, tok, model, device):
        self.tok, self.model, self.device = tok, model, device

    def _gen(self, prompt: str, max_new_tokens=512) -> str:
        inp = self.tok(prompt, return_tensors="pt", truncation=True, max_length=1600).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False,
                                      eos_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def build_context(self, docs: List[Dict[str, Any]], max_chars: int = 5000) -> Tuple[str, List[str]]:
        buf = []
        cites = []
        used = 0
        for d in docs:
            txt = d["text"][:800]
            used += len(txt)
            if used > max_chars:
                break
            if d.get("search_category") == "WEB":
                cite = f"[{d.get('domain','')}, {d.get('source','')}, {d.get('crawl_time','')}]"
            else:
                srcname = os.path.basename(d.get("source", ""))
                cite = f"[{srcname}]"
            cites.append(cite)
            buf.append(f"{txt}\nSOURCE: {cite}")
        return "\n\n".join(buf), cites

    def synthesize(self, query: str, docs: List[Dict[str, Any]], cross: Optional[Dict[str, Any]] = None) -> str:
        context, cites = self.build_context(docs)
        cross_blk = ""
        if cross:
            cross_blk = (
                f"\nCross-Validation: consistency={cross.get('consistency',1.0)}\n"
                f"Consensus: {', '.join(cross.get('consensus', [])[:5])}\n"
                f"Conflicts: {', '.join(cross.get('conflicts', [])[:5])}\n"
            )
        p = (
            "You are a careful research assistant. Answer in Korean. "
            "Use only the Context for factual claims. Always include citations inline.\n"
            f"Question: {query}\n\n"
            f"Context:\n{context}\n"
            f"{cross_blk}\n"
            "Instructions:\n"
            "- 각 주장 뒤에 근거 SOURCE를 대괄호로 표기하세요.\n"
            "- WEB 근거를 사용하면 반드시 [도메인, URL, 크롤링시각] 형식으로 인용하세요.\n"
            "- PDF 근거는 [파일명] 형식으로 인용하세요.\n"
            "- 불확실하면 명시하고, 근거가 없는 주장은 피하세요.\n"
        )
        ans = self._gen(p, 512)
        # 안전장치: 웹 근거를 쓴 흔적이 없으면, 웹 문서가 포함되어도 답변에 넣도록 후처리
        if any(d.get("search_category") == "WEB" for d in docs):
            if not re.search(r"\[(.+?),\s*https?://", ans):
                # 웹 출처 중 1~2개를 참고문헌으로 강제 덧붙임
                web_cites = []
                for d in docs:
                    if d.get("search_category") == "WEB":
                        web_cites.append(f"[{d.get('domain')}, {d.get('source')}, {d.get('crawl_time')}]")
                web_cites = list(dict.fromkeys(web_cites))[:2]
                if web_cites:
                    ans += "\n\n참고(웹 출처): " + " ; ".join(web_cites)
        return ans

# =========================
# End-to-end Orchestrator
# =========================
class HybridOrchestrator:
    def __init__(self, llm_model_id: str = "microsoft/phi-2", embed_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
        self.llm = AutoModel.from_pretrained(llm_model_id).to(self.device).eval()

        self.embedder = EmbeddingBackend(embed_model_id, self.device)
        self.retriever = HybridRetriever(self.embedder)
        self.synth = Synthesizer(self.tokenizer, self.llm, self.device)
        self.analyzer = AnalyzerAgent(self.tokenizer, self.llm, self.device)
        self.validator = ValidatorAgent(self.tokenizer, self.llm, self.device)
        self.crawler = WebCrawler()

        # Stores
        self.pdf_docs: List[Dict[str, Any]] = []
        self.pdf_embs: Optional[np.ndarray] = None

    # PDF ingestion API expected in app3/app4
    def add_pdf_chunks(self, chunks: List[Dict[str, Any]]):
        # chunks fields: text, source(file path), page, paragraph, chunk_id, ...
        self.pdf_docs.extend(chunks)
        texts = [c["text"] for c in chunks]
        embs = self.embedder.encode(texts, batch=32)
        if self.pdf_embs is None or len(self.pdf_embs) == 0:
            self.pdf_embs = embs
        else:
            self.pdf_embs = np.vstack([self.pdf_embs, embs])

    def deep_research(self, query: str, enable_web: bool = True, web_per_query: int = 12) -> Dict[str, Any]:
        # 1) PDF retrieval
        pdf_hits = self.retriever.retrieve_pdf(query, self.pdf_docs, self.pdf_embs, top_k=12)

        # 2) WEB crawling + 2-stage filtering
        web_hits: List[Dict[str, Any]] = []
        if enable_web:
            crawled = self.crawler.search_and_crawl(query, max_results=web_per_query)
            web_hits = self.retriever.filter_and_rerank_web(query, crawled, keep_top=20)

        # 3) Merge and analyze
        merged = pdf_hits + web_hits
        # 분석(문서별 관련성/인사이트) + 교차검증
        if merged:
            analysed = self.analyzer.analyse_per_doc(query, merged)
        else:
            analysed = merged
        cross = self.analyzer.cross_validate(analysed[:12]) if len(analysed) >= 2 else None

        # 4) Compose answer with strict citation rule
        answer = self.synth.synthesize(query, analysed[:12], cross)

        # 5) Validate final answer
        validation = self.validator.evaluate(query, answer, analysed[:12])

        return {
            "query": query,
            "pdf_hits": pdf_hits,
            "web_hits": web_hits,
            "documents": analysed[:12],
            "cross": cross,
            "answer": answer,
            "validation": validation,
        }

# =========================
# Streamlit UI
# =========================
def main():
    st.set_page_config(page_title="Hybrid Deep Research (PDF + Web + Cross-Validation)", layout="wide")
    st.title("Hybrid Deep Research (PDF + Web + Cross-Validation)")

    # Sidebar settings
    with st.sidebar:
        st.header("Settings")
        llm_id = st.text_input("LLM model id", "microsoft/phi-2")
        embed_id = st.text_input("Embedding model id", "sentence-transformers/all-MiniLM-L6-v2")
        enable_web = st.checkbox("Enable Web Crawling", value=True)
        web_per_query = st.slider("Max web results per query", 3, 20, 12)
        st.caption("웹 문서는 임베딩 컷(0.25) → 크로스인코더 재랭킹(상위20) → Analyzer 교차검증 순으로 필터링됩니다.")
        if st.button("Re-initialize models"):
            st.session_state.pop("orch", None)
            st.success("Models reset. They will be reloaded on next run.")

    # Orchestrator init (cache)
    if "orch" not in st.session_state:
        orch = HybridOrchestrator(llm_model_id=llm_id, embed_model_id=embed_id)
        st.session_state["orch"] = orch
    orch: HybridOrchestrator = st.session_state["orch"]

    st.subheader("PDF 업로드")
    uploaded_files = st.file_uploader("여러 개의 PDF를 업로드하세요", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        # NOTE: 실제 app3/app4의 PDF 파서/청킹을 사용하세요.
        # 여기서는 예시용으로 간단 청킹
        import pdfminer.high_level
        for uf in uploaded_files:
            content = pdfminer.high_level.extract_text(uf)
            # 단순 청킹
            parts = [content[i:i+1200] for i in range(0, len(content), 1200)]
            chunks = []
            for idx, t in enumerate(parts):
                chunks.append({
                    "text": t,
                    "source": uf.name,
                    "page": idx // 3,
                    "paragraph": idx,
                    "chunk_id": hashlib.md5((uf.name+str(idx)).encode()).hexdigest()[:10],
                    "search_category": "PDF",
                })
            if chunks:
                orch.add_pdf_chunks(chunks)
        st.success("PDF 임베딩 완료")

    st.subheader("질문")
    q = st.text_input("질문을 입력하세요", value="문서 내용과 최신 관련 정보를 종합해 요약해줘.")
    if st.button("Deep Research 실행"):
        with st.spinner("연구 중..."):
            result = orch.deep_research(q, enable_web=enable_web, web_per_query=web_per_query)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("### 답변")
            st.write(result["answer"])
            st.markdown("### 교차 검증")
            if result["cross"]:
                st.write(f"- 일관성: {result['cross'].get('consistency')}")
                cons = result["cross"].get("consensus", [])[:5]
                conf = result["cross"].get("conflicts", [])[:5]
                if cons:
                    st.write("합치:", cons)
                if conf:
                    st.write("모순:", conf)
        with col2:
            st.markdown("### 검증 지표")
            val = result["validation"]
            st.metric("근거 연결", f"{val['grounding']:.2f}")
            st.metric("사실 일치", f"{val['factual']:.2f}")
            st.metric("신뢰도", f"{val['confidence']:.2f}")
            st.markdown("### 문서 사용(상위)")
            for d in result["documents"]:
                if d.get("search_category") == "WEB":
                    st.write(f"WEB | {d.get('domain')} | sim {d.get('similarity',0):.2f} | rerank {d.get('rerank_score',0):.2f}")
                else:
                    st.write(f"PDF | {os.path.basename(d.get('source',''))} | sim {d.get('similarity',0):.2f}")
                if d.get("key_insight"):
                    st.caption(f"- insight: {d['key_insight']}")

        st.markdown("### 참고 문서(웹)")
        for d in result["web_hits"][:10]:
            st.write(f"- {d.get('title','')} [{d.get('domain')}] {d.get('source')} @ {d.get('crawl_time')}")

        st.markdown("### 참고 문서(PDF)")
        for d in result["pdf_hits"][:10]:
            st.write(f"- {os.path.basename(d.get('source',''))} (sim {d.get('similarity',0):.2f})")

if __name__ == "__main__":
    main()
