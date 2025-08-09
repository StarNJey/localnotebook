# 웹크롤링 기능이 추가된 Deep Research Chatbot

# 기존 PDF 처리 기능을 유지하면서 웹 검색 및 크롤링 기능을 추가

from __future__ import annotations
import os
import re
import json
import hashlib
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlparse, quote_plus

import streamlit as st
import numpy as np
import torch, gc
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from sentence_transformers.cross_encoder import CrossEncoder
import warnings

# 웹크롤링을 위한 새로운 라이브러리
try:
    import requests
    from bs4 import BeautifulSoup
    WEB_CRAWLING_AVAILABLE = True
except ImportError:
    WEB_CRAWLING_AVAILABLE = False

# ⚠️ 수정사항 1: st.set_page_config를 맨 앞으로 이동
st.set_page_config(page_title="Deep Research Chatbot with Web Crawling", layout="wide")

# PDF 라이브러리들
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

# 한국어 문장 분리 라이브러리들
try:
    from kiwipiepy import Kiwi
    KIWI_AVAILABLE = True
except ImportError:
    KIWI_AVAILABLE = False

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# ────────────────────────────────────────────────────────────
# 1. 웹크롤링 관련 클래스 (새로 추가)
# ────────────────────────────────────────────────────────────

@dataclass
class WebDocument:
    """크롤링된 웹 문서 데이터 구조"""
    url: str
    title: str
    text: str
    domain: str
    crawl_time: str
    source: str
    page: int = 0
    paragraph: int = 0
    chunk_id: str = ""

class WebCrawler:
    """웹 검색 및 크롤링을 담당하는 클래스"""
    def __init__(self, timeout: int = 10, max_content_length: int = 50000):
        self.timeout = timeout
        self.max_content_length = max_content_length
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0'
        })

    def search_web(self, query: str, max_results: int = 15) -> List[str]:
        """DuckDuckGo를 통한 웹 검색"""
        try:
            search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
            response = self.session.get(search_url, timeout=self.timeout)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            urls: List[str] = []
            for link in soup.find_all('a', class_='result__a')[:max_results]:
                href = link.get('href')
                if href and href.startswith('http'):
                    urls.append(href)
            return urls
        except Exception as e:
            st.warning(f"웹 검색 실패: {e}")
            return []

    def crawl_page(self, url: str) -> Optional[WebDocument]:
        """단일 웹페이지 크롤링"""
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            content_type = response.headers.get('content-type', '')
            if 'text/html' not in content_type.lower():
                return None

            soup = BeautifulSoup(response.text, 'html.parser')
            title_tag = soup.find('title')
            title = title_tag.get_text().strip() if title_tag else "제목 없음"

            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            if len(text) > self.max_content_length:
                text = text[:self.max_content_length] + "..."
            if len(text) < 100:
                return None

            domain = urlparse(url).netloc
            crawl_time = datetime.now().isoformat()
            source = f"web:{url}"
            chunk_id = hashlib.md5(f"{url}_{text[:100]}".encode()).hexdigest()[:8]

            return WebDocument(
                url=url, title=title[:200], text=text,
                domain=domain, crawl_time=crawl_time,
                source=source, chunk_id=chunk_id
            )
        except Exception as e:
            st.warning(f"페이지 크롤링 실패 ({url}): {e}")
            return None

    def crawl_multiple_pages(self, urls: List[str]) -> List[WebDocument]:
        """여러 페이지 크롤링"""
        web_docs: List[WebDocument] = []
        progress_bar = st.progress(0)
        for i, url in enumerate(urls):
            doc = self.crawl_page(url)
            if doc:
                web_docs.append(doc)
            progress_bar.progress((i + 1) / len(urls))
            time.sleep(0.5)
        return web_docs

    def search_and_crawl(self, query: str, max_results: int = 15) -> List[WebDocument]:
        """검색 + 크롤링 통합 메서드"""
        urls = self.search_web(query, max_results)
        if not urls:
            return []
        return self.crawl_multiple_pages(urls)

# ────────────────────────────────────────────────────────────
# 2. 기존 데이터 구조 정의 (유지)
# ────────────────────────────────────────────────────────────

class ResearchPhase(Enum):
    PLANNING = "planning"
    INITIAL_RETRIEVAL = "initial_retrieval"
    WEB_CRAWLING = "web_crawling"
    DEEP_ANALYSIS = "deep_analysis"
    CROSS_VALIDATION = "cross_validation"
    SYNTHESIS = "synthesis"
    FINAL_VALIDATION = "final_validation"

@dataclass
class ResearchState:
    """연구 진행 상태를 추적하는 클래스"""
    phase: ResearchPhase
    query: str
    sub_queries: List[str]
    retrieved_docs: List[Dict[str, Any]]
    web_docs: List[WebDocument]
    confidence_history: List[float]
    insights: List[str]
    gaps: List[str]
    cycle_count: int = 0
    max_cycles: int = 3

@dataclass
class SearchQuery:
    """검색 쿼리 정보"""
    text: str
    priority: float
    category: str
    reason: str
    search_web: bool = True

# ────────────────────────────────────────────────────────────
# 3. 기존 유틸리티 함수들 (유지)
# ────────────────────────────────────────────────────────────

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)

# ────────────────────────────────────────────────────────────
# 4. 기존 유틸리티 클래스들 (유지)
# ────────────────────────────────────────────────────────────

class ImprovedKoreanSentenceChunker:
    def __init__(self, min_chunk_length=50, max_chunk_length=300, sentences_per_chunk=2):
        self.min_chunk_length = min_chunk_length
        self.max_chunk_length = max_chunk_length
        self.sentences_per_chunk = sentences_per_chunk
        self.kiwi = Kiwi() if KIWI_AVAILABLE else None

    def chunk_text(self, text: str) -> List[str]:
        if not text.strip():
            return []
        sentences = self._split_sentences(text)
        if not sentences:
            return []
        sentences = self._postprocess_sentences(sentences)
        return self._create_chunks(sentences)

    def _split_sentences(self, text: str) -> List[str]:
        if KIWI_AVAILABLE and self.kiwi:
            try:
                kiwi_result = self.kiwi.split_into_sents(text.strip())
                sents = [sent.text.strip() for sent in kiwi_result if sent.text.strip()]
                if len(sents) > 1:
                    return sents
            except:
                pass
        return self._regex_sentence_split(text)

    def _regex_sentence_split(self, text: str) -> List[str]:
        patterns = [
            r'[.!?]+\s+', r'[다가나니까요래습니다]\s*[.!?]*\s+',
            r'[니다했다습니다였다았다]\s*[.!?]*\s+', r'\n\s*\n', r'\.\s*\n',
        ]
        combined = '|'.join(f'({p})' for p in patterns)
        parts = re.split(combined, text)
        return [p.strip() for p in parts if p and not re.match(r'^[\s.!?\n]+$', p)]

    def _postprocess_sentences(self, sents: List[str]) -> List[str]:
        processed, current = [], ""
        for sent in sents:
            sent = sent.strip()
            if not sent: continue
            if len(current) < self.min_chunk_length:
                current = f"{current} {sent}".strip() if current else sent
            else:
                processed.append(current)
                current = sent
        if current:
            if processed and len(current) < self.min_chunk_length:
                processed[-1] += f" {current}"
            else:
                processed.append(current)
        return processed

    def _create_chunks(self, sents: List[str]) -> List[str]:
        chunks, curr, cnt = [], "", 0
        for sent in sents:
            test = f"{curr} {sent}".strip()
            if len(test) <= self.max_chunk_length and cnt < self.sentences_per_chunk:
                curr, cnt = test, cnt + 1
            else:
                if curr: chunks.append(curr.strip())
                curr, cnt = sent, 1
        if curr: chunks.append(curr.strip())
        return chunks

class ImprovedPDFExtractor:
    def __init__(self):
        self.available_methods = ["pdfplumber"] if PDFPLUMBER_AVAILABLE else []

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        if PDFPLUMBER_AVAILABLE:
            try:
                text = self._extract_with_pdfplumber(pdf_path)
                if text.strip():
                    return text
            except Exception as e:
                st.warning(f"pdfplumber 추출 실패: {e}")
        return ""

    def _extract_with_pdfplumber(self, pdf_path: str) -> str:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                if page.extract_text():
                    text += page.extract_text() + "\n"
        return text

class DeviceConfig:
    def __init__(self):
        self.device, self.info = self._detect()
        self.config = self._get_config()

    def _detect(self):
        if torch.cuda.is_available():
            return "cuda", {
                "type": "GPU",
                "name": torch.cuda.get_device_name(0),
                "memory": torch.cuda.get_device_properties(0).total_memory // (1024**3),
                "count": torch.cuda.device_count()
            }
        import psutil
        return "cpu", {
            "type": "CPU",
            "name": "CPU",
            "cores": psutil.cpu_count(),
            "memory": psutil.virtual_memory().total // (1024**3)
        }

    def _get_config(self):
        if self.device == "cuda":
            mem = self.info["memory"]
            if mem >= 24:
                return {"torch_dtype": torch.bfloat16, "max_new_tokens": 8000, "top_k": 12,
                        "embedding_batch_size": 64, "do_sample": True, "temperature": 0.1, "top_p": 0.9, "sim_threshold": 0.3}
            if mem >= 12:
                return {"torch_dtype": torch.bfloat16, "max_new_tokens": 6000, "top_k": 12,
                        "embedding_batch_size": 32, "do_sample": True, "temperature": 0.1, "top_p": 0.9, "sim_threshold": 0.3}
            if mem >= 8:
                return {"torch_dtype": torch.bfloat16, "max_new_tokens": 4000, "top_k": 10,
                        "embedding_batch_size": 16, "do_sample": True, "temperature": 0.1, "top_p": 0.9, "sim_threshold": 0.3}
            return {"torch_dtype": torch.float16, "max_new_tokens": 3000, "top_k": 10,
                    "embedding_batch_size": 8, "do_sample": False, "temperature": 0.1, "sim_threshold": 0.3}
        return {"torch_dtype": torch.float32, "max_new_tokens": 2000, "top_k": 10,
                "embedding_batch_size": 4, "do_sample": False, "temperature": 0.1, "sim_threshold": 0.3}

    def get_adaptive_config(self, complexity_level: float) -> Dict[str, Any]:
        base = self.config.copy()
        if complexity_level > 0.8:
            base["max_new_tokens"] = int(base["max_new_tokens"] * 1.5)
            base["top_k"] = min(base["top_k"] + 5, 20)
            base["temperature"] = min(base.get("temperature", 0.1) + 0.1, 0.3)
        elif complexity_level < 0.3:
            base["max_new_tokens"] = int(base["max_new_tokens"] * 0.7)
            base["top_k"] = max(base["top_k"] - 2, 5)
            base["do_sample"] = False
        return base

# ────────────────────────────────────────────────────────────
# 5. Agent 클래스들 (웹크롤링 기능 추가)
# ────────────────────────────────────────────────────────────

class ResearchPlannerAgent:
    """연구 계획 및 전략 수립 에이전트"""
    def __init__(self, llm_tokenizer, llm_model, device_config):
        self.tokenizer = llm_tokenizer
        self.model = llm_model
        self.device_config = device_config

    def analyze_query_complexity(self, query: str) -> float:
        indicators = ["비교","분석","평가","검토","연관","관계","영향",
                      "원인","결과","어떻게","왜","언제","어디서","누가",
                      "무엇을","상세","구체적","최신","현재","동향","트렌드"]
        score = 0.3
        if len(query) > 50: score += 0.2
        cnt = sum(1 for i in indicators if i in query)
        score += min(cnt * 0.1, 0.3)
        if "?" in query: score += 0.1
        if any(w in query for w in ["그리고","또한","하지만","그러나"]): score += 0.1
        return min(score,1.0)

    def generate_research_plan(self, query: str, state: ResearchState) -> List[SearchQuery]:
        prompt = f"""다음 질문에 대한 체계적인 연구 계획을 수립하세요:
질문: {query}
다음 단계로 구성된 검색 계획을 생성하세요:
1. 핵심 키워드 추출
2. 하위 질문 분해
3. 우선순위 설정
4. 웹 검색 필요성 판단
각 검색 쿼리는 다음 형식으로 생성:
- 검색어: [구체적 검색어]
- 우선순위: [1-10점]
- 카테고리: [주요개념/세부사항/배경지식/비교분석/최신정보]
- 이유: [왜 이 검색이 필요한지]
- 웹검색: [YES/NO]
최대 5개의 검색 쿼리를 생성하세요."""
        messages = [
            {"role":"system","content":"당신은 연구 계획을 수립하는 전문가입니다. PDF 문서뿐만 아니라 웹 검색도 고려하세요."},
            {"role":"user","content":prompt}
        ]
        response = self._generate_llm_response(messages, max_tokens=1000)
        return self._parse_search_queries(response, state)

    def _parse_search_queries(self, response: str, state: ResearchState) -> List[SearchQuery]:
        queries, curr = [], {}
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("- 검색어:"):
                curr["text"] = line.replace("- 검색어:","").strip()
            elif line.startswith("- 우선순위:"):
                try:
                    curr["priority"] = float(re.findall(r"\d+", line)[0]) / 10.0
                except:
                    curr["priority"] = 0.5
            elif line.startswith("- 카테고리:"):
                curr["category"] = line.replace("- 카테고리:","").strip()
            elif line.startswith("- 이유:"):
                curr["reason"] = line.replace("- 이유:","").strip()
            elif line.startswith("- 웹검색:"):
                val = line.replace("- 웹검색:","").strip().upper()
                curr["search_web"] = "YES" in val
            if all(k in curr for k in ["text","priority","category","reason"]):
                if "search_web" not in curr:
                    curr["search_web"] = True
                queries.append(SearchQuery(**curr))
                curr = {}
        if not queries:
            queries.append(SearchQuery(text=state.query, priority=1.0, category="주요개념", reason="기본 검색", search_web=True))
        return queries[:10]

    def identify_knowledge_gaps(self, state: ResearchState) -> List[str]:
        if not state.retrieved_docs and not state.web_docs:
            return ["기초 정보 부족"]
        texts = ""
        if state.retrieved_docs:
            texts += "\n".join(d.get("text","")[:200] for d in state.retrieved_docs[:3])
        if state.web_docs:
            texts += "\n".join(d.text[:200] for d in state.web_docs[:3])
        prompt = f"""다음 연구 결과를 분석하여 부족한 정보를 식별하세요:
원본 질문: {state.query}
현재까지 수집된 정보 (PDF + 웹):
{texts[:1000]}...
부족한 정보나 추가 조사가 필요한 영역을 최대 3개까지 나열하세요."""
        messages = [
            {"role":"system","content":"당신은 연구 격차를 분석하는 전문가입니다."},
            {"role":"user","content":prompt}
        ]
        resp = self._generate_llm_response(messages, max_tokens=500)
        gaps = [g.strip() for g in resp.split("\n") if g.strip()]
        return gaps[:3]

    def _generate_llm_response(self, messages: List[Dict], max_tokens: int) -> str:
        try:
            input_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
            if self.device_config.device == "cuda":
                input_ids = input_ids.to("cuda")
            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "do_sample": False,
                "repetition_penalty":1.1,
                "pad_token_id":self.tokenizer.eos_token_id,
                "eos_token_id":self.tokenizer.eos_token_id
            }
            with torch.no_grad():
                output = self.model.generate(input_ids, **gen_kwargs)
            return self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        except Exception as e:
            st.warning(f"LLM 응답 생성 실패: {e}")
            return ""

class RetrieverAgent:
    """문서 검색 전문 에이전트 (PDF + 웹 통합)"""
    def __init__(self, embed_tokenizer, embed_model, reranker, device_config, web_crawler):
        self.embed_tokenizer = embed_tokenizer
        self.embed_model = embed_model
        self.reranker = reranker
        self.device_config = device_config
        self.web_crawler = web_crawler

    def multi_query_retrieval(self, search_queries: List[SearchQuery],
                              documents: List[Dict], embeddings: np.ndarray,
                              orchestrator=None) -> List[Dict]:
        all_results, web_docs_collected = [], []

        # PDF 검색
        for sq in search_queries:
            pdf_res = self._single_query_retrieval(
                sq.text, documents, embeddings,
                top_k=max(5, int(10 * sq.priority))
            )
            for r in pdf_res:
                r.update({
                    "search_priority": sq.priority,
                    "search_category": sq.category,
                    "search_reason": sq.reason,
                    "source_type": "PDF"
                })
            all_results.extend(pdf_res)

        # 웹 크롤링
        if WEB_CRAWLING_AVAILABLE:
            web_qs = [sq for sq in search_queries if sq.search_web]
            if web_qs:
                st.info(f"🌐 웹 검색 수행 중... ({len(web_qs)}개)")
                for wq in web_qs:
                    try:
                        crawled = self.web_crawler.search_and_crawl(wq.text, max_results=8)
                        web_docs_collected.extend(crawled)
                    except Exception as e:
                        st.warning(f"웹 크롤링 실패: {e}")
            if web_docs_collected:
                # 💡 변경점 1: _process_web_docs 호출 시 orchestrator 인자 제거
                web_results = self._process_web_docs(web_docs_collected)
                all_results.extend(web_results)

        unique = self._deduplicate_results(all_results)
        # 💡 변경점 3: _rerank_results가 웹/PDF 점수를 다르게 계산
        return self._rerank_results(search_queries[0].text if search_queries else "", unique)

    # 💡 변경점 2: _process_web_docs에서 하드코딩된 점수 제거
    def _process_web_docs(self, web_docs: List[WebDocument]) -> List[Dict]:
        """크롤링된 웹 문서를 표준 형식으로 변환 (점수 계산 없음)"""
        web_results = []
        for wd in web_docs:
            web_results.append({
                "text": wd.text,
                "page": 0, "paragraph": 0,
                "source": wd.url,
                "chunk_id": wd.chunk_id,
                "search_category": "웹정보",
                "source_type": "WEB",
                "web_title": wd.title,
                "web_domain": wd.domain,
                "web_crawl_time": wd.crawl_time
                # "similarity" 와 "final_score" 는 여기서 설정하지 않음
            })
        return web_results

    def _single_query_retrieval(self, query: str, documents: List[Dict],
                                embeddings: np.ndarray, top_k: int) -> List[Dict]:
        if embeddings is None or not documents:
            return []
        enc = self.embed_tokenizer([query], padding=True, truncation=True, return_tensors="pt")
        enc = enc.to(self.device_config.device)
        with torch.no_grad():
            out = self.embed_model(**enc)
        q_emb = mean_pooling(out, enc["attention_mask"]).cpu().numpy()
        sims = cosine_similarity(q_emb, embeddings)[0]
        idxs = np.argsort(sims)[::-1][:top_k * 3]
        threshold = self.device_config.config["sim_threshold"]
        res = []
        for idx in idxs:
            if idx < len(documents) and sims[idx] >= threshold:
                doc = documents[idx].copy()
                doc["similarity"] = float(sims[idx])
                res.append(doc)
        return res[:top_k]

    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        seen, unique = set(), []
        for r in results:
            # chunk_id가 있는 경우 우선 사용, 없으면 텍스트 해시 사용
            identifier = r.get("chunk_id")
            if not identifier:
                 identifier = hashlib.md5(r["text"].encode()).hexdigest()

            if identifier not in seen:
                seen.add(identifier)
                unique.append(r)
        return unique

    # 💡 변경점 4: _rerank_results에서 웹/PDF 점수 계산 로직 분리
    def _rerank_results(self, main_query: str, results: List[Dict], top_k: int = 15) -> List[Dict]:
        """Cross-Encoder로 결과 재평가 (웹/PDF 점수 계산 분리)"""
        if not results:
            return []
        
        texts = [r["text"] for r in results]
        # CrossEncoder는 정규화되지 않은 점수를 반환하므로 시그모이드 함수로 0-1 사이 값으로 변환
        scores = torch.sigmoid(torch.tensor(self.reranker.predict([(main_query, t) for t in texts]))).tolist()

        for r, sc in zip(results, scores):
            r["rerank_score"] = float(sc)
            
            # PDF는 임베딩 유사도와 재평가 점수를 결합
            if r.get("source_type") == "PDF":
                sim = r.get("similarity", 0.0)
                r["final_score"] = sim * 0.4 + r["rerank_score"] * 0.6
            # WEB은 재평가 점수를 최종 점수로 사용
            else: 
                r["similarity"] = 0.0 # 초기 유사도 없음
                r["final_score"] = r["rerank_score"]

        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results[:top_k]


class SynthesizerAgent:
    """정보 통합 및 최종 답변 생성 에이전트 (웹 출처 표시 기능 강화)"""
    def __init__(self, llm_tokenizer, llm_model, device_config):
        self.tokenizer = llm_tokenizer
        self.model = llm_model
        self.device_config = device_config

    def synthesize_comprehensive_answer(self, query: str, state: ResearchState,
                                       analysis_results: Dict) -> str:
        total_docs = len(state.retrieved_docs) + len(state.web_docs)
        complexity = total_docs * 0.1 + len(state.insights) * 0.2
        cfg = self.device_config.get_adaptive_config(complexity)
        prompt = self._build_synthesis_prompt(query, state, analysis_results)
        messages = [
            {"role":"system","content":self._get_synthesis_system_prompt()},
            {"role":"user","content":prompt}
        ]
        return self._generate_final_answer(messages, cfg)
    def _build_synthesis_prompt(self, query: str, state: ResearchState, analysis_results: Dict) -> str:
        # PDF 문서 요약에 인용 태그 자동 할당
        pdf_docs = [d for d in state.retrieved_docs if d.get("source_type") == "PDF"][:5]
        pdf_summaries = []
        for idx, d in enumerate(pdf_docs, 1):
            tag = f"[{idx}]"
            pdf_summaries.append(
                f"{tag} 파일: {d['source']}, 페이지:{d['page']}, 단락:{d['paragraph']}, 관련도:{d['final_score']:.2f}\n"
                f"내용: {d['text'][:200]}...\n"
            )
    
        # 웹 문서 요약에 인용 태그 자동 할당
        web_docs = [d for d in state.retrieved_docs if d.get("source_type") == "WEB"][:5]
        web_summaries = []
        offset = len(pdf_summaries)
        for j, d in enumerate(web_docs, 1):
            tag = f"[{offset + j}]"
            web_summaries.append(
                f"{tag} 도메인: {d['web_domain']}, URL: {d['source']}, 크롤링시간:{d['web_crawl_time']}, 관련도:{d['final_score']:.2f}\n"
                f"내용: {d['text'][:200]}...\n"
            )
    
        # 교차검증 결과 요약
        analysis_section = (
            f"교차 검증 결과:\n"
            f"- 정보 일관성: {analysis_results.get('consistency', 0.5):.2f}\n"
            f"- 상충 정보: {len(analysis_results.get('conflicts', []))}건\n"
            f"- 공통 정보: {len(analysis_results.get('consensus', []))}건\n"
        )
    
        # 연구 진행 현황
        status_section = (
            f"연구 진행 현황:\n"
            f"- 탐색 사이클: {state.cycle_count + 1}/{state.max_cycles}\n"
            f"- 발견된 인사이트: {len(state.insights)}개\n"
            f"- 식별된 지식 격차: {len(state.gaps)}개\n"
            f"- PDF 문서: {len(pdf_docs)}개\n"
            f"- 웹 문서: {len(web_docs)}개\n"
        )
    
        pdf_block = "".join(pdf_summaries) if pdf_summaries else "PDF 문서 없음\n"
        web_block = "".join(web_summaries) if web_summaries else "웹 문서 없음\n"
    
        return f"""다음 연구 결과를 바탕으로 질문에 대한 종합적인 답변을 작성하세요:
    
    원본 질문: {query}
    
    === PDF 문서 요약 ===
    {pdf_block}
    
    === 웹 문서 요약 ===
    {web_block}
    
    === 분석 결과 ===
    {analysis_section}
    {status_section}
    
    === 출처 및 인용 태그 안내 ===
    - 위 요약 블록에서 정의된 인용 태그([1], [2], …)를 문장 끝에 반드시 재사용하세요.
    - PDF 인용 형식: [파일명, 페이지:단락, 관련도: X.XX]
    - 웹 인용 형식: [도메인, URL, 크롤링시간, 관련도: X.XX]
    - 예시:
      > “AI 시장은 연평균 20% 성장할 전망입니다.” [example.pdf, p.3:2, 0.87]
    
    === 작성 지침 ===
    1. 각 문장마다 해당 인용 태그를 붙여 근거를 명확히 표시하세요.
    2. 상충되는 정보가 있으면 태그와 함께 객관적으로 제시하세요.
    3. 부족한 정보가 있으면 솔직히 언급하세요.
    4. 답변 말미에 종합 신뢰도를 제시하세요.
    
    답변:
    """

    def _get_synthesis_system_prompt(self) -> str:
        return """당신은 다중 소스 정보를 종합하여 정확하고 포괄적인 답변을 생성하는 연구 전문가입니다.
핵심 원칙:
1. 제공된 PDF 및 웹 문서 정보만 사용
2. 모든 주장에 대한 정확한 출처 명시 필수
3. PDF와 웹 출처를 구분하여 반드시 표시
4. 불확실한 정보는 신뢰도와 함께 제시
5. 상충되는 정보는 객관적으로 제시
6. 지식 격차는 솔직하게 인정
7. 논리적이고 체계적인 구조로 답변
출처 표시 형식:

- PDF: [파일명, 페이지:단락, 관련도: 0.XX]
- 웹: [도메인, URL, 크롤링시간, 관련도: 0.XX]

본문 작성 지시:
- 앞서 요약 블록에서 정의된 인용 태그([1], , …)를 **문장 끝에 반드시 재사용**.
- 예시:
  > “AI 시장은 연평균 20% 성장할 전망입니다.” [example.pdf, p.3:2, 0.87]
답변 구조(출처 반드시 표시):
- 핵심 답변 (요약)
- 상세 설명 (근거와 출처 함께)
- 추가 고려사항 (한계점 포함)
- 신뢰도 평가
웹 출처의 경우 반드시 URL과 크롤링 시간을 포함하여 독자가 정보의 출처와 시점을 확인할 수 있도록 하세요."""
    def _generate_final_answer(self, messages: List[Dict], config: Dict) -> str:
        try:
            input_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
            if self.device_config.device == "cuda":
                input_ids = input_ids.to("cuda")
            gen_kwargs = {
                "max_new_tokens": config["max_new_tokens"],
                "do_sample": config.get("do_sample", False),
                "repetition_penalty":1.1,
                "pad_token_id":self.tokenizer.eos_token_id,
                "eos_token_id":self.tokenizer.eos_token_id
            }
            if config.get("do_sample"):
                gen_kwargs.update({"temperature":config.get("temperature",0.1),"top_p":config.get("top_p",0.9)})
            with torch.no_grad():
                output = self.model.generate(input_ids, **gen_kwargs)
            return self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        except Exception as e:
            st.error(f"답변 생성 실패: {e}")
            return "답변 생성 중 오류가 발생했습니다."

class DeepResearchOrchestrator:
    """Deep Research 스타일의 다중 에이전트 협업 시스템 (웹크롤링 통합)"""
    def __init__(
        self,
        model_name: str = "LGAI-EXAONE/EXAONE-4.0-1.2B",
        embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        reranker_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        min_chunk_length: int = 50,
        max_chunk_length: int = 300,
        sentences_per_chunk: int = 2,
    ):
        st.info("🚀 Deep Research 시스템 초기화 중... (웹크롤링 기능 포함)")
        if not WEB_CRAWLING_AVAILABLE:
            st.warning("⚠️ 웹크롤링 라이브러리가 없습니다. pip install requests beautifulsoup4를 실행하세요.")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_config = DeviceConfig()
        self.pdf_extractor = ImprovedPDFExtractor()
        self.chunker = ImprovedKoreanSentenceChunker(
            min_chunk_length=min_chunk_length,
            max_chunk_length=max_chunk_length,
            sentences_per_chunk=sentences_per_chunk
        )
        self.web_crawler = WebCrawler() if WEB_CRAWLING_AVAILABLE else None
        self._load_models()
        self._initialize_agents()
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.chunk_hashes: Dict[str, Dict] = {}
        self.loaded_pdfs: List[str] = []
        st.success("✅ Deep Research 시스템 준비 완료! (웹크롤링 기능 포함)")

    def _load_models(self):
        st.info("▶ 임베딩 모델 로딩...")
        self.embed_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.embed_model = AutoModel.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2").to(self.device_config.device)
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=self.device_config.device)
        st.info("▶ LLM 로딩...")
        self.tokenizer = AutoTokenizer.from_pretrained("LGAI-EXAONE/EXAONE-4.0-1.2B", trust_remote_code=True)
        map_dev = "auto" if self.device_config.device == "cuda" else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            "LGAI-EXAONE/EXAONE-4.0-1.2B",
            torch_dtype=self.device_config.config["torch_dtype"],
            device_map=map_dev,
            max_memory={0: "14GB"},
            trust_remote_code=True
        ).to(self.device)
        torch.cuda.empty_cache(); gc.collect()

    def _initialize_agents(self):
        st.info("▶ 에이전트 시스템 초기화...")
        self.planner = ResearchPlannerAgent(self.tokenizer, self.model, self.device_config)
        self.retriever = RetrieverAgent(self.embed_tokenizer, self.embed_model, self.reranker, self.device_config, self.web_crawler)
        self.synthesizer = SynthesizerAgent(self.tokenizer, self.model, self.device_config)

    def load_pdf_documents(self, pdf_paths: List[str]) -> None:
        st.info(f"📚 PDF 문서 처리 중... ({len(pdf_paths)}개)")
        all_docs: List[Dict[str, Any]] = []
        progress_bar = st.progress(0)
        for i, pdf_path in enumerate(pdf_paths):
            if not os.path.exists(pdf_path):
                st.warning(f"❌ 파일을 찾을 수 없음: {pdf_path}")
                continue
            full_text = self.pdf_extractor.extract_text_from_pdf(pdf_path)
            if not full_text.strip():
                st.warning(f"⚠️ 텍스트를 추출할 수 없음: {pdf_path}")
                continue
            chunks = self.chunker.chunk_text(full_text)
            for idx, chunk in enumerate(chunks,1):
                if chunk.strip():
                    all_docs.append({
                        "text": chunk.strip(),
                        "page": 1,
                        "paragraph": idx,
                        "source": os.path.basename(pdf_path),
                        "full_path": pdf_path,
                        "source_type": "PDF"
                    })
            self.loaded_pdfs.append(pdf_path)
            progress_bar.progress((i+1)/len(pdf_paths))
        if not all_docs:
            st.error("❌ 처리할 수 있는 문서가 없습니다.")
            return
        self.documents = all_docs
        st.info("🧮 임베딩 생성 중...")
        self._generate_embeddings()
        for d in self.documents:
            cid = hashlib.md5(f"{d['source']}_{d['page']}_{d['paragraph']}_{d['text']}".encode()).hexdigest()[:8]
            d["chunk_id"] = cid
            self.chunk_hashes[cid] = d
        st.success(f"🎉 처리 완료! 총 {len(self.documents)}개 청크 생성")

    def _generate_embeddings(self):
        texts = [d["text"] for d in self.documents]
        bs = self.device_config.config["embedding_batch_size"]
        embs = []
        progress_bar = st.progress(0)
        for i in range(0, len(texts), bs):
            batch = texts[i:i+bs]
            enc = self.embed_tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device_config.device)
            with torch.no_grad():
                out = self.embed_model(**enc)
            pooled = mean_pooling(out, enc["attention_mask"])
            embs.append(pooled.cpu().numpy())
            progress_bar.progress(min(1.0, (i+bs)/len(texts)))
        self.embeddings = np.vstack(embs)

    def deep_research(self, query: str) -> Dict[str, Any]:
        st.info("🔍 Deep Research 프로세스 시작... (PDF + 웹)")
        complexity = self.planner.analyze_query_complexity(query)
        max_cycles = 2 if complexity < 0.5 else 3
        state = ResearchState(
            phase=ResearchPhase.PLANNING,
            query=query,
            sub_queries=[],
            retrieved_docs=[],
            web_docs=[],
            confidence_history=[],
            insights=[],
            gaps=[],
            max_cycles=max_cycles
        )
        research_log, search_queries = [], []
        for cycle in range(state.max_cycles):
            state.cycle_count = cycle
            research_log.append(f"=== 사이클 {cycle+1} ===")
            # Phase1: 계획
            if cycle == 0:
                st.info(f"📋 연구 계획 수립 중... (복잡도: {complexity:.2f})")
                state.phase = ResearchPhase.PLANNING
                search_queries = self.planner.generate_research_plan(query, state)
                state.sub_queries = [sq.text for sq in search_queries]
                web_cnt = sum(1 for sq in search_queries if sq.search_web)
                research_log.append(f"생성된 검색 쿼리: {len(search_queries)}개 (웹검색: {web_cnt}개)")
            # Phase2: 검색
            st.info(f"🔎 다중 쿼리 검색 수행 중... (쿼리 {len(search_queries)}개)")
            state.phase = ResearchPhase.INITIAL_RETRIEVAL if cycle==0 else ResearchPhase.DEEP_ANALYSIS
            retrieved = self.retriever.multi_query_retrieval(search_queries, self.documents, self.embeddings, self)
            existing = {d.get("chunk_id","") for d in state.retrieved_docs}
            new_docs = [d for d in retrieved if d.get("chunk_id","") not in existing]
            state.retrieved_docs.extend(new_docs)
            web_docs = [d for d in new_docs if d.get("source_type")=="WEB"]
            pdf_docs = [d for d in new_docs if d.get("source_type")=="PDF"]
            research_log.append(f"검색된 문서: PDF {len(pdf_docs)}개, 웹 {len(web_docs)}개")
            # 중간 신뢰도
            conf = self._calculate_interim_confidence(state)
            state.confidence_history.append(conf)
            if conf > 0.85 and len(state.retrieved_docs)>=5:
                research_log.append(f"높은 신뢰도 달성 ({conf:.2f}) - 조기 종료")
                break

        # Phase2 종료 → 실제 교차검증 수행
        st.info("✅ 교차검증 수행 중... (PDF + 웹)")
        state.phase = ResearchPhase.CROSS_VALIDATION
        analysis_results = self.analyzer.cross_validate_information(state.retrieved_docs[:10])

# Phase3: 종합답변
        st.info("📝 종합 답변 생성 중... (PDF + 웹)")
        state.phase = ResearchPhase.SYNTHESIS
        answer = self.synthesizer.synthesize_comprehensive_answer(query, state, analysis_results)


        # 결과 포맷
        sources = []
        for d in state.retrieved_docs[:15]:
            info = {
                "page": d.get("page",0),
                "paragraph": d.get("paragraph",0),
                "chunk_id": d.get("chunk_id",""),
                "source_file": d.get("source","Unknown"),
                "source_type": d.get("source_type","PDF"),
                "preview": d["text"][:400]+"..." if len(d["text"])>400 else d["text"],
                "similarity": float(d.get("final_score",d.get("similarity",0))),
                "chunk_size": len(d["text"]),
                "search_category": d.get("search_category","일반"),
                "key_insight": d.get("key_insight","")
            }
            if info["source_type"]=="WEB":
                info.update({
                    "web_title": d.get("web_title","제목 없음"),
                    "web_domain": d.get("web_domain","Unknown"),
                    "web_crawl_time": d.get("web_crawl_time","Unknown")
                })
            sources.append(info)

        return {
            "answer": answer,
            "confidence": state.confidence_history[-1] if state.confidence_history else 0.0,
            "warnings": [],
            "sources": sources,
            "research_metadata": {
                "cycles_completed": state.cycle_count+1,
                "total_documents_analyzed": len(state.retrieved_docs),
                "pdf_documents": sum(1 for s in sources if s["source_type"]=="PDF"),
                "web_documents": sum(1 for s in sources if s["source_type"]=="WEB"),
                "insights_discovered": len(state.insights),
                "knowledge_gaps_identified": len(state.gaps),
                "query_complexity": complexity,
                "confidence_progression": state.confidence_history,
                "research_log": research_log,
                "web_crawling_enabled": WEB_CRAWLING_AVAILABLE
            }
        }

    def _calculate_interim_confidence(self, state: ResearchState) -> float:
        if not state.retrieved_docs:
            return 0.0
        scores = [d.get("final_score", d.get("similarity",0.5)) for d in state.retrieved_docs]
        avg = np.mean(scores) if scores else 0.5
        pdf_src = len(set(d["source"] for d in state.retrieved_docs if d["source_type"]=="PDF"))
        web_src = len(set(d["source"] for d in state.retrieved_docs if d["source_type"]=="WEB"))
        div = min((pdf_src + web_src) / 5.0, 1.0)
        ins = min(len(state.insights)/5.0,1.0)
        return avg*0.5 + div*0.3 + ins*0.2

# ────────────────────────────────────────────────────────────
# 6. Streamlit UI (웹크롤링 기능 추가)
# ────────────────────────────────────────────────────────────

st.write("CUDA available:", torch.cuda.is_available())
st.write("웹크롤링 기능:", "✅ 사용 가능" if WEB_CRAWLING_AVAILABLE else "❌ 라이브러리 설치 필요")
st.title("🧠 Deep Research Chatbot with Web Crawling by C.H.PARK")
st.markdown("### AI 다중 에이전트 협업 시스템 + 실시간 웹크롤링으로 구현한 하이브리드 RAG")

st.sidebar.header("⚙️ 설정")
st.sidebar.subheader("청킹 설정")
min_chunk_length = st.sidebar.slider("최소 청크 길이", 30, 500, 50)
max_chunk_length = st.sidebar.slider("최대 청크 길이", 200, 3000, 300)
sentences_per_chunk = st.sidebar.slider("청크당 최대 문장 수", 1, 10, 2)

st.sidebar.subheader("🌐 웹크롤링 설정")
enable_web_crawling = st.sidebar.checkbox("웹크롤링 활성화", value=WEB_CRAWLING_AVAILABLE)

st.sidebar.subheader("📁 파일 업로드")
uploaded_files = st.sidebar.file_uploader(
    "PDF 파일 업로드",
    type="pdf",
    accept_multiple_files=True,
    help="여러 PDF 파일을 동시에 업로드할 수 있습니다"
)

if st.sidebar.button("🔄 Deep Research 시스템 시작", type="primary"):
    if not uploaded_files:
        st.sidebar.error("❌ 최소 하나의 PDF 파일을 업로드해주세요.")
    else:
        pdf_paths = []
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        for f in uploaded_files:
            path = os.path.join(temp_dir, f.name)
            with open(path, "wb") as wf:
                wf.write(f.getbuffer())
            pdf_paths.append(path)
        with st.spinner("🔄 시스템 초기화 중..."):
            try:
                st.session_state.research_bot = DeepResearchOrchestrator(
                    min_chunk_length=min_chunk_length,
                    max_chunk_length=max_chunk_length,
                    sentences_per_chunk=sentences_per_chunk
                )
                st.session_state.research_bot.load_pdf_documents(pdf_paths)
                st.sidebar.success("✅ 시스템 준비 완료! (PDF + 웹)")
                st.sidebar.info(f"📊 총 {len(st.session_state.research_bot.documents)}청크 생성됨")
                st.sidebar.info(f"📁 {len(st.session_state.research_bot.loaded_pdfs)}파일 로딩됨")
                st.sidebar.info(f"🌐 웹크롤링: {'ON' if WEB_CRAWLING_AVAILABLE else 'OFF'}")
            except Exception as e:
                st.sidebar.error(f"❌ 초기화 실패: {e}")

if "research_bot" in st.session_state:
    bot = st.session_state.research_bot
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📁 로딩된 파일", len(bot.loaded_pdfs))
    with col2:
        st.metric("📄 PDF 청크", len(bot.documents))
    with col3:
        avg_size = np.mean([len(d["text"]) for d in bot.documents]) if bot.documents else 0
        st.metric("📏 평균 청크 크기", f"{avg_size:.0f}자")
    with col4:
        st.metric("🤖 활성 에이전트", "5개")
    with col5:
        st.metric("웹크롤링", "🌐 ON" if WEB_CRAWLING_AVAILABLE else "🌐 OFF")

    st.divider()
    st.subheader("💬 하이브리드 Deep Research 질문")
    st.info("💡 PDF 문서 분석과 최신 웹 정보를 함께 활용하여 답변합니다.")
    query = st.text_input("심층 연구할 주제를 입력하세요:", key="hybrid_query_input")

    if st.button("🧠 하이브리드 Research 시작", type="primary") and query:
        with st.spinner("🧠 연구 수행 중..."):
            start = time.time()
            result = bot.deep_research(query)
            elapsed = time.time() - start
            torch.cuda.empty_cache(); gc.collect()
            if "research_history" not in st.session_state:
                st.session_state.research_history = []
            st.session_state.research_history.append({
                "query": query,
                "result": result,
                "timestamp": datetime.now(),
                "elapsed_time": elapsed
            })

        st.subheader("🎯 하이브리드 Deep Research 결과")
        st.write(result["answer"])
        md = result["research_metadata"]
        c1,c2,c3,c4,c5 = st.columns(5)
        with c1: st.metric("신뢰도", f"{result['confidence']:.3f}")
        with c2: st.metric("사이클", f"{md['cycles_completed']}/{md.get('max_cycles',3)}")
        with c3: st.metric("PDF 문서", md.get('pdf_documents',0))
        with c4: st.metric("웹 문서", md.get('web_documents',0))
        with c5: st.metric("소요 시간", f"{elapsed:.1f}초")

        if result["sources"]:
            pdf_srcs = [s for s in result["sources"] if s["source_type"]=="PDF"]
            web_srcs = [s for s in result["sources"] if s["source_type"]=="WEB"]
            colp, colw = st.columns(2)
            with colp:
                if pdf_srcs:
                    st.subheader("📄 PDF 문서 소스")
                    for i, s in enumerate(pdf_srcs[:5],1):
                        with st.expander(f"PDF {i}: {s['source_file']} (점수: {s['similarity']:.3f})"):
                            st.write(f"**파일**: {s['source_file']}")
                            st.write(f"**페이지**: {s['page']}, **단락**: {s['paragraph']}")
                            st.write(f"**미리보기**: {s['preview'][:200]}...")
                            if s.get("key_insight"):
                                st.write(f"💡 **인사이트**: {s['key_insight']}")
            with colw:
                if web_srcs:
                    st.subheader("🌐 웹 문서 소스")
                    for i, s in enumerate(web_srcs[:5],1):
                        with st.expander(f"웹 {i}: {s['web_title']} (점수: {s['similarity']:.3f})"):
                            st.write(f"**제목**: {s['web_title']}")
                            st.write(f"**URL**: {s['source_file']}")
                            st.write(f"**도메인**: {s['web_domain']}")
                            st.write(f"**크롤링 시간**: {s['web_crawl_time']}")
                            st.write(f"**미리보기**: {s['preview'][:200]}...")
        with st.expander("🔍 상세 연구 로그"):
            for entry in md.get("research_log", []):
                st.write(entry)

        if "research_history" in st.session_state:
            st.divider()
            st.subheader("📜 하이브리드 연구 기록")
            for rec in reversed(st.session_state.research_history[-3:]):
                with st.expander(f"연구: {rec['query'][:50]}..."):
                    st.write(f"**질문**: {rec['query']}")
                    st.write(f"**답변 일부**: {rec['result']['answer'][:300]}...")
                    st.write(f"**시간**: {rec['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write(f"**신뢰도**: {rec['result']['confidence']:.3f}")
    else:
        st.info("👆 왼쪽에서 PDF 업로드 후 시스템 시작을 눌러주세요.")

    st.subheader("🆕 하이브리드 Deep Research의 새로운 기능")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**📄 PDF 문서 분석**
- 업로드된 문서 심층 분석
- 청킹 및 임베딩
- 출처: [파일명, 관련도]
""")
    with c2:
        st.markdown("""
**🌐 실시간 웹크롤링**
- 최신 정보 자동 검색
- 웹페이지 크롤링
- 출처: [제목, URL, 크롤링시간, 관련도]
""")
    with st.expander("🎓 사용 가이드"):
        st.markdown("""
### 🧠 하이브리드 Deep Research 시스템
**다중 에이전트 + 웹크롤링**  
- Research Planner: PDF/웹 계획 수립  
- Retriever Agent: PDF+웹 통합 검색  
- Synthesizer Agent: 출처별 종합 답변  
- Validator Agent: 최종 검증  

**출처 표시**  
- PDF: [파일명, 관련도: 0.XX]  
- 웹: [제목/도메인, URL, 크롤링시간, 관련도: 0.XX]  
""")
    st.info("💡 PDF의 정확성과 웹의 최신성을 결합해드립니다.")
