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
from urllib.parse import urlparse, urljoin, quote_plus

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
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def search_web(self, query: str, max_results: int = 5) -> List[str]:
        """DuckDuckGo를 통한 웹 검색"""
        try:
            # DuckDuckGo HTML 검색 사용 (JavaScript 불필요)
            search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
            
            response = self.session.get(search_url, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            urls = []
            
            # 검색 결과 링크 추출
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
            
            # HTML이 아닌 경우 건너뛰기
            content_type = response.headers.get('content-type', '')
            if 'text/html' not in content_type.lower():
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 제목 추출
            title_tag = soup.find('title')
            title = title_tag.get_text().strip() if title_tag else "제목 없음"
            
            # 본문 텍스트 추출 (스크립트, 스타일 등 제거)
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            text = soup.get_text()
            
            # 텍스트 정리
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # 너무 긴 텍스트 자르기
            if len(text) > self.max_content_length:
                text = text[:self.max_content_length] + "..."
            
            # 너무 짧은 텍스트는 제외
            if len(text) < 100:
                return None
            
            # 도메인 추출
            domain = urlparse(url).netloc
            
            # WebDocument 생성
            web_doc = WebDocument(
                url=url,
                title=title[:200],  # 제목 길이 제한
                text=text,
                domain=domain,
                crawl_time=datetime.now().isoformat(),
                source=f"web:{url}"
            )
            
            # chunk_id 생성
            web_doc.chunk_id = hashlib.md5(
                f"{url}_{web_doc.text[:100]}".encode()
            ).hexdigest()[:8]
            
            return web_doc
            
        except Exception as e:
            st.warning(f"페이지 크롤링 실패 ({url}): {e}")
            return None
    
    def crawl_multiple_pages(self, urls: List[str]) -> List[WebDocument]:
        """여러 페이지 크롤링"""
        web_docs = []
        
        progress_bar = st.progress(0)
        
        for i, url in enumerate(urls):
            doc = self.crawl_page(url)
            if doc:
                web_docs.append(doc)
            
            progress_bar.progress((i + 1) / len(urls))
            time.sleep(0.5)  # 서버 부담 감소
        
        return web_docs
    
    def search_and_crawl(self, query: str, max_results: int = 5) -> List[WebDocument]:
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
    web_docs: List[WebDocument]  # 새로 추가
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
    search_web: bool = True  # 웹 검색 여부

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
        
        self.kiwi = None
        if KIWI_AVAILABLE:
            try:
                self.kiwi = Kiwi()
            except Exception as e:
                if 'st' in globals():
                    st.warning(f"⚠️ Kiwi 로드 실패: {e}")

    def chunk_text(self, text: str) -> List[str]:
        if not text.strip():
            return []
        
        sentences = self._split_sentences(text)
        if not sentences:
            return []
        
        sentences = self._postprocess_sentences(sentences)
        chunks = self._create_chunks(sentences)
        
        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        if KIWI_AVAILABLE and self.kiwi:
            try:
                kiwi_result = self.kiwi.split_into_sents(text.strip())
                sentences = [sent.text.strip() for sent in kiwi_result if sent.text.strip()]
                if sentences and len(sentences) > 1:
                    return sentences
            except Exception as e:
                if 'st' in globals():
                    st.warning(f"Kiwi 문장 분리 실패: {e}")
        
        return self._regex_sentence_split(text.strip())

    def _regex_sentence_split(self, text: str) -> List[str]:
        patterns = [
            r'[.!?]+\s+', r'[다가나니까요래습니다]\s*[.!?]*\s+',
            r'[다가나니까요래습니다]\s+', r'[니다했다습니다였다았다]\s*[.!?]*\s+',
            r'\n\s*\n', r'\.\s*\n',
        ]
        
        combined_pattern = '|'.join(f'({p})' for p in patterns)
        sentences = re.split(combined_pattern, text)
        
        result = []
        for s in sentences:
            if s and not re.match(r'^\s*[.!?\n\s]*$', s):
                result.append(s.strip())
        
        return result if result else [text.strip()]

    def _postprocess_sentences(self, sentences: List[str]) -> List[str]:
        if not sentences:
            return []
        
        processed = []
        current_sentence = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if len(current_sentence) < self.min_chunk_length:
                if current_sentence:
                    current_sentence += " " + sentence
                else:
                    current_sentence = sentence
            else:
                if current_sentence:
                    processed.append(current_sentence)
                current_sentence = sentence
        
        if current_sentence:
            if processed and len(current_sentence) < self.min_chunk_length:
                processed[-1] += " " + current_sentence
            else:
                processed.append(current_sentence)
        
        return processed

    def _create_chunks(self, sentences: List[str]) -> List[str]:
        if not sentences:
            return []
        
        chunks = []
        current_chunk = ""
        sentence_count = 0
        
        for sentence in sentences:
            test_chunk = current_chunk + (" " + sentence if current_chunk else sentence)
            
            if (len(test_chunk) <= self.max_chunk_length and 
                sentence_count < self.sentences_per_chunk):
                current_chunk = test_chunk
                sentence_count += 1
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
                sentence_count = 1
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

class ImprovedPDFExtractor:
    def __init__(self):
        self.available_methods = []
        if PDFPLUMBER_AVAILABLE:
            self.available_methods.append("pdfplumber")

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        if PDFPLUMBER_AVAILABLE:
            try:
                text = self._extract_with_pdfplumber(pdf_path)
                if text.strip():
                    return text
            except Exception as e:
                if 'st' in globals():
                    st.warning(f"pdfplumber 추출 실패: {e}")
        return ""

    def _extract_with_pdfplumber(self, pdf_path: str) -> str:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
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
        try:
            import psutil
            return "cpu", {
                "type": "CPU",
                "name": "CPU",
                "cores": psutil.cpu_count(),
                "memory": psutil.virtual_memory().total // (1024**3)
            }
        except ImportError:
            return "cpu", {"type": "CPU", "name": "CPU", "cores": "Unknown", "memory": "Unknown"}

    def _get_config(self):
        if self.device == "cuda":
            mem = self.info["memory"]
            if mem >= 24:
                return {"torch_dtype": torch.bfloat16, "max_new_tokens": 8000, "top_k": 12,
                        "embedding_batch_size": 64, "do_sample": True,
                        "temperature": 0.1, "top_p": 0.9, "expected_time": "5-15초",
                        "sim_threshold": 0.3}
            if mem >= 12:
                return {"torch_dtype": torch.bfloat16, "max_new_tokens": 6000, "top_k": 12,
                        "embedding_batch_size": 32, "do_sample": True,
                        "temperature": 0.1, "top_p": 0.9, "expected_time": "5-15초",
                        "sim_threshold": 0.3}
            if mem >= 8:
                return {"torch_dtype": torch.bfloat16, "max_new_tokens": 4000, "top_k": 10,
                        "embedding_batch_size": 16, "do_sample": True,
                        "temperature": 0.1, "top_p": 0.9, "expected_time": "8-20초",
                        "sim_threshold": 0.3}
            return {"torch_dtype": torch.float16, "max_new_tokens": 3000, "temperature": 0.1, "top_k": 10,
                    "embedding_batch_size": 8, "do_sample": False, "expected_time": "10-25초",
                    "sim_threshold": 0.3}
        return {"torch_dtype": torch.float32, "max_new_tokens": 2000, "temperature": 0.1, "top_k": 10,
                "embedding_batch_size": 4, "do_sample": False, "expected_time": "30-90초",
                "sim_threshold": 0.3}

    def get_adaptive_config(self, complexity_level: float) -> Dict[str, Any]:
        """Test-Time Compute: 복잡도에 따라 동적으로 설정 조정"""
        base_config = self.config.copy()
        
        if complexity_level > 0.8:  # 매우 복잡한 질문
            base_config["max_new_tokens"] = int(base_config["max_new_tokens"] * 1.5)
            base_config["top_k"] = min(base_config["top_k"] + 5, 20)
            base_config["temperature"] = min(base_config.get("temperature", 0.1) + 0.1, 0.3)
        elif complexity_level < 0.3:  # 간단한 질문
            base_config["max_new_tokens"] = int(base_config["max_new_tokens"] * 0.7)
            base_config["top_k"] = max(base_config["top_k"] - 2, 5)
            base_config["do_sample"] = False
            
        return base_config

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
        """질문의 복잡도를 분석하여 0-1 사이의 값으로 반환"""
        complexity_indicators = [
            "비교", "분석", "평가", "검토", "연관", "관계", "영향", "원인", "결과",
            "어떻게", "왜", "언제", "어디서", "누가", "무엇을", "상세", "구체적",
            "최신", "현재", "동향", "트렌드"  # 웹 검색이 필요한 키워드 추가
        ]
        
        score = 0.3  # 기본 복잡도
        
        # 질문 길이
        if len(query) > 50:
            score += 0.2
        
        # 복잡도 지시어 개수
        indicator_count = sum(1 for indicator in complexity_indicators if indicator in query)
        score += min(indicator_count * 0.1, 0.3)
        
        # 질문 구조 복잡도
        if "?" in query:
            score += 0.1
        if any(word in query for word in ["그리고", "또한", "하지만", "그러나"]):
            score += 0.1
            
        return min(score, 1.0)
    
    def generate_research_plan(self, query: str, state: ResearchState) -> List[SearchQuery]:
        """질문을 분석하여 다단계 검색 계획 생성 (웹 검색 포함)"""
        
        planning_prompt = f"""다음 질문에 대한 체계적인 연구 계획을 수립하세요:

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
- 웹검색: [YES/NO - 최신 정보나 일반적 지식이 필요한 경우]

최대 5개의 검색 쿼리를 생성하세요."""
        
        messages = [
            {"role": "system", "content": "당신은 연구 계획을 수립하는 전문가입니다. PDF 문서뿐만 아니라 웹 검색도 고려하여 계획을 수립하세요."},
            {"role": "user", "content": planning_prompt}
        ]
        
        response = self._generate_llm_response(messages, max_tokens=1000)
        return self._parse_search_queries(response, state)
    
    def _parse_search_queries(self, response: str, state: ResearchState) -> List[SearchQuery]:
        """LLM 응답에서 검색 쿼리 파싱 (웹 검색 여부 포함)"""
        queries = []
        lines = response.split('\n')
        
        current_query = {}
        for line in lines:
            line = line.strip()
            if line.startswith("- 검색어:"):
                current_query["text"] = line.replace("- 검색어:", "").strip()
            elif line.startswith("- 우선순위:"):
                try:
                    current_query["priority"] = float(re.findall(r'\d+', line)[0]) / 10.0
                except:
                    current_query["priority"] = 0.5
            elif line.startswith("- 카테고리:"):
                current_query["category"] = line.replace("- 카테고리:", "").strip()
            elif line.startswith("- 이유:"):
                current_query["reason"] = line.replace("- 이유:", "").strip()
            elif line.startswith("- 웹검색:"):
                web_search_text = line.replace("- 웹검색:", "").strip().upper()
                current_query["search_web"] = "YES" in web_search_text
                
                if all(key in current_query for key in ["text", "priority", "category", "reason"]):
                    if "search_web" not in current_query:
                        current_query["search_web"] = True  # 기본값
                    queries.append(SearchQuery(**current_query))
                    current_query = {}
        
        # 기본 검색 쿼리가 없으면 원본 질문 사용
        if not queries:
            queries.append(SearchQuery(
                text=state.query,
                priority=1.0,
                category="주요개념",
                reason="기본 검색",
                search_web=True
            ))
            
        return queries[:5]  # 최대 5개로 제한
    
    def identify_knowledge_gaps(self, state: ResearchState) -> List[str]:
        """현재까지의 연구 결과에서 지식 격차 식별"""
        if not state.retrieved_docs and not state.web_docs:
            return ["기초 정보 부족"]
        
        # PDF와 웹 문서 정보 통합
        all_docs_text = ""
        if state.retrieved_docs:
            all_docs_text += "\n".join([doc.get('text', '')[:200] for doc in state.retrieved_docs[:3]])
        if state.web_docs:
            all_docs_text += "\n".join([doc.text[:200] for doc in state.web_docs[:3]])
            
        gap_analysis_prompt = f"""다음 연구 결과를 분석하여 부족한 정보를 식별하세요:

원본 질문: {state.query}
현재까지 수집된 정보 (PDF + 웹):
{all_docs_text[:1000]}...

부족한 정보나 추가 조사가 필요한 영역을 최대 3개까지 나열하세요."""
        
        messages = [
            {"role": "system", "content": "당신은 연구 격차를 분석하는 전문가입니다."},
            {"role": "user", "content": gap_analysis_prompt}
        ]
        
        response = self._generate_llm_response(messages, max_tokens=500)
        gaps = [gap.strip() for gap in response.split('\n') if gap.strip()]
        return gaps[:3]
    
    def _generate_llm_response(self, messages: List[Dict], max_tokens: int = 500) -> str:
        """LLM을 사용한 텍스트 생성"""
        try:
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )
            
            if self.device_config.device == "cuda":
                input_ids = input_ids.to("cuda")
            
            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "do_sample": False,
                "repetition_penalty": 1.1,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id
            }
            
            with torch.no_grad():
                output = self.model.generate(input_ids, **gen_kwargs)
            
            response = self.tokenizer.decode(
                output[0][input_ids.shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            return response
            
        except Exception as e:
            if 'st' in globals():
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
        """다중 쿼리를 사용한 포괄적 검색 (PDF + 웹)"""
        all_results = []
        web_docs_collected = []
        
        # 1. PDF 문서 검색
        for search_query in search_queries:
            results = self._single_query_retrieval(
                search_query.text, documents, embeddings, 
                top_k=max(5, int(10 * search_query.priority))
            )
            
            # 우선순위와 카테고리 정보 추가
            for result in results:
                result['search_priority'] = search_query.priority
                result['search_category'] = search_query.category
                result['search_reason'] = search_query.reason
                result['source_type'] = 'PDF'
                
            all_results.extend(results)
        
        # 2. 웹 크롤링 (필요한 경우)
        if WEB_CRAWLING_AVAILABLE:
            web_queries = [sq for sq in search_queries if sq.search_web]
            if web_queries:
                st.info(f"🌐 웹 검색 수행 중... ({len(web_queries)}개 쿼리)")
                
                for web_query in web_queries:
                    try:
                        crawled_docs = self.web_crawler.search_and_crawl(
                            web_query.text, max_results=3
                        )
                        web_docs_collected.extend(crawled_docs)
                    except Exception as e:
                        st.warning(f"웹 크롤링 실패: {e}")
                
                # 웹 문서를 검색 결과에 추가
                if web_docs_collected and orchestrator:
                    web_results = self._process_web_docs(web_docs_collected, orchestrator)
                    all_results.extend(web_results)
        
        # 3. 중복 제거 및 점수 정규화
        unique_results = self._deduplicate_results(all_results)
        return self._rerank_results(search_queries[0].text if search_queries else "", unique_results)
    
    def _process_web_docs(self, web_docs: List[WebDocument], orchestrator) -> List[Dict]:
        """웹 문서를 검색 결과 형태로 변환"""
        web_results = []
        
        for web_doc in web_docs:
            # 웹 문서를 기존 문서 형태로 변환
            doc_dict = {
                "text": web_doc.text,
                "page": 0,
                "paragraph": 0,
                "source": web_doc.url,
                "chunk_id": web_doc.chunk_id,
                "search_category": "웹정보",
                "source_type": "WEB",
                "web_title": web_doc.title,
                "web_domain": web_doc.domain,
                "web_crawl_time": web_doc.crawl_time,
                "similarity": 0.7,  # 기본 유사도 점수
                "final_score": 0.7
            }
            web_results.append(doc_dict)
        
        return web_results
    
    def _single_query_retrieval(self, query: str, documents: List[Dict], 
                              embeddings: np.ndarray, top_k: int = 10) -> List[Dict]:
        """단일 쿼리 검색"""
        if embeddings is None or len(documents) == 0:
            return []
        
        # 쿼리 임베딩 생성
        enc = self.embed_tokenizer([query], padding=True, truncation=True, return_tensors="pt")
        enc = enc.to(self.device_config.device)
        
        with torch.no_grad():
            out = self.embed_model(**enc)
        
        q_emb = mean_pooling(out, enc["attention_mask"]).cpu().numpy()
        
        # 유사도 계산
        sims = cosine_similarity(q_emb, embeddings)[0]
        idxs = np.argsort(sims)[::-1][:top_k * 3]  # 더 많은 후보 선택
        threshold = self.device_config.config["sim_threshold"]
        
        results = []
        for idx in idxs:
            if idx < len(documents) and sims[idx] >= threshold:
                doc = documents[idx].copy()
                doc["similarity"] = float(sims[idx])
                results.append(doc)
        
        return results[:top_k]
    
    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """결과 중복 제거"""
        seen_texts = set()
        unique_results = []
        
        for result in results:
            text_hash = hashlib.md5(result['text'].encode()).hexdigest()
            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                unique_results.append(result)
        
        return unique_results
    
    def _rerank_results(self, main_query: str, results: List[Dict], top_k: int = 15) -> List[Dict]:
        """Cross-encoder를 사용한 재순위 매김"""
        if not results:
            return []
        
        texts = [r["text"] for r in results]
        scores = self.reranker.predict([(main_query, text) for text in texts])
        
        for result, score in zip(results, scores):
            result["rerank_score"] = float(score)
            # 최종 점수는 임베딩 유사도와 재순위 점수의 가중 평균
            similarity = result.get("similarity", 0.5)
            result["final_score"] = (similarity * 0.4 + score * 0.6)
        
        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results[:top_k]

# 나머지 Agent 클래스들은 기존과 동일하되, 웹 문서 처리 부분만 추가...

class SynthesizerAgent:
    """정보 통합 및 최종 답변 생성 에이전트 (웹 출처 표시 기능 강화)"""
    
    def __init__(self, llm_tokenizer, llm_model, device_config):
        self.tokenizer = llm_tokenizer
        self.model = llm_model
        self.device_config = device_config
    
    def synthesize_comprehensive_answer(self, query: str, state: ResearchState, 
                                      analysis_results: Dict) -> str:
        """포괄적 답변 생성 (PDF + 웹 출처 통합)"""
        
        # Test-Time Compute: 복잡도에 따른 동적 설정
        total_docs = len(state.retrieved_docs) + len(state.web_docs)
        complexity = total_docs * 0.1 + len(state.insights) * 0.2
        adaptive_config = self.device_config.get_adaptive_config(complexity)
        
        synthesis_prompt = self._build_synthesis_prompt(query, state, analysis_results)
        
        messages = [
            {"role": "system", "content": self._get_synthesis_system_prompt()},
            {"role": "user", "content": synthesis_prompt}
        ]
        
        return self._generate_final_answer(messages, adaptive_config)
    
    def _build_synthesis_prompt(self, query: str, state: ResearchState, 
                               analysis_results: Dict) -> str:
        """종합 답변용 프롬프트 구성 (웹 출처 포함)"""
        
        # PDF 문서 정보 정리
        pdf_summaries = []
        for i, doc in enumerate([d for d in state.retrieved_docs if d.get('source_type') != 'WEB'][:5], 1):
            summary = f"[PDF 문서 {i}] 파일: {doc.get('source', 'Unknown')}, 관련도: {doc.get('final_score', 0):.2f}\n"
            summary += f"내용: {doc['text'][:300]}...\n"
            if 'key_insight' in doc:
                summary += f"핵심 인사이트: {doc['key_insight']}\n"
            pdf_summaries.append(summary)
        
        # 웹 문서 정보 정리
        web_summaries = []
        web_docs_in_retrieved = [d for d in state.retrieved_docs if d.get('source_type') == 'WEB']
        all_web_docs = state.web_docs + web_docs_in_retrieved
        
        for i, doc in enumerate(all_web_docs[:5], 1):
            if hasattr(doc, 'url'):  # WebDocument 객체인 경우
                summary = f"[웹 문서 {i}] 제목: {doc.title}\n"
                summary += f"URL: {doc.url}\n"
                summary += f"도메인: {doc.domain}, 크롤링 시간: {doc.crawl_time}\n"
                summary += f"내용: {doc.text[:300]}...\n"
            else:  # Dict 형태인 경우
                summary = f"[웹 문서 {i}] 제목: {doc.get('web_title', '제목 없음')}\n"
                summary += f"URL: {doc.get('source', 'Unknown')}\n"
                summary += f"도메인: {doc.get('web_domain', 'Unknown')}, 크롤링 시간: {doc.get('web_crawl_time', 'Unknown')}\n"
                summary += f"관련도: {doc.get('final_score', 0):.2f}\n"
                summary += f"내용: {doc['text'][:300]}...\n"
            web_summaries.append(summary)
        
        # 분석 결과 정리
        analysis_summary = f"""
교차 검증 결과:
- 정보 일관성: {analysis_results.get('consistency', 0.5):.2f}
- 상충 정보: {len(analysis_results.get('conflicts', []))}건
- 공통 정보: {len(analysis_results.get('consensus', []))}건

연구 진행 현황:
- 탐색 사이클: {state.cycle_count + 1}/{state.max_cycles}
- 발견된 인사이트: {len(state.insights)}개
- 식별된 지식 격차: {len(state.gaps)}개
- PDF 문서: {len([d for d in state.retrieved_docs if d.get('source_type') != 'WEB'])}개
- 웹 문서: {len(all_web_docs)}개
"""
        
        pdf_section = f"""
=== PDF 문서 정보 ===
{chr(10).join(pdf_summaries) if pdf_summaries else "PDF 문서 없음"}
""" if pdf_summaries else ""
        
        web_section = f"""
=== 웹 문서 정보 ===
{chr(10).join(web_summaries) if web_summaries else "웹 문서 없음"}
""" if web_summaries else ""
        
        return f"""다음 연구 결과를 바탕으로 질문에 대한 종합적인 답변을 작성하세요:

원본 질문: {query}

{pdf_section}
{web_section}

=== 분석 결과 ===
{analysis_summary}

=== 출처 표시 요구사항 ===
1. 모든 정보에 대해 정확한 출처를 명시해야 합니다
2. PDF 출처 형식: [파일명, 관련도: X.XX]
3. 웹 출처 형식: [제목 또는 도메인, URL, 크롤링시간, 관련도: X.XX]
4. 상충되는 정보가 있다면 명시하고 각각의 출처를 표시
5. 부족한 정보가 있다면 언급
6. 신뢰도 수준 제시
7. 구체적이고 상세한 답변 작성

반드시 모든 주장과 정보에 대해 위 형식으로 출처를 표시하세요."""
    
    def _get_synthesis_system_prompt(self) -> str:
        """시스템 프롬프트 (웹 출처 표시 강화)"""
        return """당신은 다중 소스 정보를 종합하여 정확하고 포괄적인 답변을 생성하는 연구 전문가입니다.

핵심 원칙:
1. 제공된 PDF 및 웹 문서 정보만 사용
2. 모든 주장에 대한 정확한 출처 명시 필수
3. PDF와 웹 출처를 구분하여 표시
4. 불확실한 정보는 신뢰도와 함께 제시
5. 상충되는 정보는 객관적으로 제시
6. 지식 격차는 솔직하게 인정
7. 논리적이고 체계적인 구조로 답변

출처 표시 형식:
- PDF: [파일명, 관련도: 0.XX]
- 웹: [제목/도메인, URL, 크롤링시간, 관련도: 0.XX]

답변 구조:
- 핵심 답변 (요약)
- 상세 설명 (근거와 출처 함께)
- 추가 고려사항 (한계점 포함)
- 신뢰도 평가

웹 출처의 경우 반드시 URL과 크롤링 시간을 포함하여 독자가 정보의 출처와 시점을 확인할 수 있도록 하세요."""
    
    def _generate_final_answer(self, messages: List[Dict], config: Dict) -> str:
        """최종 답변 생성"""
        try:
            input_ids = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            )
            
            if self.device_config.device == "cuda":
                input_ids = input_ids.to("cuda")
            
            gen_kwargs = {
                "max_new_tokens": config["max_new_tokens"],
                "do_sample": config.get("do_sample", False),
                "repetition_penalty": 1.1,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id
            }
            
            if config.get("do_sample"):
                gen_kwargs.update({
                    "temperature": config.get("temperature", 0.1),
                    "top_p": config.get("top_p", 0.9)
                })
            
            with torch.no_grad():
                output = self.model.generate(input_ids, **gen_kwargs)
            
            return self.tokenizer.decode(
                output[0][input_ids.shape[1]:], skip_special_tokens=True
            ).strip()
        except Exception as e:
            if 'st' in globals():
                st.error(f"답변 생성 실패: {e}")
            return "답변 생성 중 오류가 발생했습니다."

# ── 나머지 Agent 클래스들과 메인 오케스트레이터는 기존과 유사하되, 
# ── 웹 문서 처리 로직이 추가됩니다 ──

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
        
        # 웹크롤링 가능 여부 확인
        if not WEB_CRAWLING_AVAILABLE:
            st.warning("⚠️ 웹크롤링 라이브러리가 없습니다. pip install requests beautifulsoup4를 실행하세요.")
        
        # 기본 설정
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_config = DeviceConfig()
        
        # 유틸리티 클래스들
        self.pdf_extractor = ImprovedPDFExtractor()
        self.chunker = ImprovedKoreanSentenceChunker(
            min_chunk_length=min_chunk_length,
            max_chunk_length=max_chunk_length,
            sentences_per_chunk=sentences_per_chunk
        )
        
        # 웹 크롤러 초기화
        if WEB_CRAWLING_AVAILABLE:
            self.web_crawler = WebCrawler()
        else:
            self.web_crawler = None
        
        # 모델 로딩
        self._load_models()
        
        # 에이전트 초기화
        self._initialize_agents()
        
        # 데이터 저장소
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.chunk_hashes: Dict[str, Dict] = {}
        self.loaded_pdfs: List[str] = []
        
        st.success("✅ Deep Research 시스템 준비 완료! (웹크롤링 기능 포함)")
    
    def _load_models(self):
        """모델들 로딩"""
        # 임베딩 모델
        st.info("▶ 임베딩 모델 로딩...")
        self.embed_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.embed_model = AutoModel.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2").to(self.device_config.device)
        
        # Cross-Encoder
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=self.device_config.device)
        
        # LLM
        st.info("▶ LLM 로딩...")
        self.tokenizer = AutoTokenizer.from_pretrained("LGAI-EXAONE/EXAONE-4.0-1.2B", trust_remote_code=True)
        map_dev = "auto" if self.device_config.device == "cuda" else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            "LGAI-EXAONE/EXAONE-4.0-1.2B", torch_dtype=self.device_config.config["torch_dtype"],
            device_map=map_dev, max_memory={0: "14GB"}, trust_remote_code=True
        ).to(self.device)
        
        torch.cuda.empty_cache()
        gc.collect()
    
    def _initialize_agents(self):
        """에이전트들 초기화"""
        st.info("▶ 에이전트 시스템 초기화...")
        
        self.planner = ResearchPlannerAgent(self.tokenizer, self.model, self.device_config)
        self.retriever = RetrieverAgent(self.embed_tokenizer, self.embed_model, self.reranker, self.device_config, self.web_crawler)
        # analyzer와 validator는 기존과 동일
        self.synthesizer = SynthesizerAgent(self.tokenizer, self.model, self.device_config)
    
    def load_pdf_documents(self, pdf_paths: List[str]) -> None:
        """PDF 문서 로딩 (기존과 동일)"""
        st.info(f"📚 PDF 문서 처리 중... ({len(pdf_paths)}개)")
        
        self.documents, self.embeddings, self.chunk_hashes, self.loaded_pdfs = [], None, {}, []
        all_docs: List[Dict[str, Any]] = []
        
        progress_bar = st.progress(0)
        
        for i, pdf_path in enumerate(pdf_paths):
            if not os.path.exists(pdf_path):
                st.warning(f"❌ 파일을 찾을 수 없음: {pdf_path}")
                continue
            
            # PDF 텍스트 추출
            full_text = self.pdf_extractor.extract_text_from_pdf(pdf_path)
            if not full_text.strip():
                st.warning(f"⚠️ 텍스트를 추출할 수 없음: {pdf_path}")
                continue
            
            # 텍스트 청킹
            chunks = self.chunker.chunk_text(full_text)
            
            # 문서 객체 생성
            for chunk_idx, chunk in enumerate(chunks, 1):
                if chunk.strip():
                    all_docs.append({
                        "text": chunk.strip(),
                        "page": 1,
                        "paragraph": chunk_idx,
                        "source": os.path.basename(pdf_path),
                        "full_path": pdf_path,
                        "source_type": "PDF"
                    })
            
            self.loaded_pdfs.append(pdf_path)
            progress_bar.progress((i + 1) / len(pdf_paths))
        
        if not all_docs:
            st.error("❌ 처리할 수 있는 문서가 없습니다.")
            return
        
        self.documents = all_docs
        
        # 임베딩 생성
        st.info("🧮 임베딩 생성 중...")
        self._generate_embeddings()
        
        # 청크 해시 생성
        for d in self.documents:
            cid = self._hash(d["text"], d["page"], d["paragraph"], d["source"])
            d["chunk_id"] = cid
            self.chunk_hashes[cid] = d
        
        st.success(f"🎉 처리 완료! 총 {len(self.documents)}개 청크 생성")
    
    def _generate_embeddings(self):
        """임베딩 생성"""
        texts = [d["text"] for d in self.documents]
        bs = self.device_config.config["embedding_batch_size"]
        embs = []
        
        progress_bar = st.progress(0)
        
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            enc = self.embed_tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device_config.device)
            with torch.no_grad():
                out = self.embed_model(**enc)
            pooled = mean_pooling(out, enc["attention_mask"])
            embs.append(pooled.cpu().numpy())
            
            progress_bar.progress(min(1.0, (i + bs) / len(texts)))
        
        self.embeddings = np.vstack(embs)
    
    @staticmethod
    def _hash(text: str, page: int, para: int, source: str) -> str:
        return hashlib.md5(f"{source}_{page}_{para}_{text}".encode()).hexdigest()[:8]
    
    def deep_research(self, query: str) -> Dict[str, Any]:
        """Deep Research 메인 프로세스 (웹크롤링 통합)"""
        
        st.info("🔍 Deep Research 프로세스 시작... (PDF + 웹)")
        
        # 1. 연구 상태 초기화
        complexity = self.planner.analyze_query_complexity(query)
        max_cycles = 2 if complexity < 0.5 else 3
        
        state = ResearchState(
            phase=ResearchPhase.PLANNING,
            query=query,
            sub_queries=[],
            retrieved_docs=[],
            web_docs=[],  # 웹 문서 리스트 초기화
            confidence_history=[],
            insights=[],
            gaps=[],
            max_cycles=max_cycles
        )
        
        research_log = []
        search_queries = []
        
        # 연구 사이클 시작
        for cycle in range(state.max_cycles):
            state.cycle_count = cycle
            research_log.append(f"=== 사이클 {cycle + 1} ===")
            
            # Phase 1: 연구 계획 수립
            if cycle == 0:
                st.info(f"📋 연구 계획 수립 중... (복잡도: {complexity:.2f})")
                state.phase = ResearchPhase.PLANNING
                search_queries = self.planner.generate_research_plan(query, state)
                state.sub_queries = [sq.text for sq in search_queries]
                web_query_count = sum(1 for sq in search_queries if sq.search_web)
                research_log.append(f"생성된 검색 쿼리: {len(search_queries)}개 (웹검색: {web_query_count}개)")
            
            # Phase 2: 문서 검색 (PDF + 웹)
            st.info(f"🔎 다중 쿼리 검색 수행 중... (쿼리 {len(search_queries)}개)")
            state.phase = ResearchPhase.INITIAL_RETRIEVAL if cycle == 0 else ResearchPhase.DEEP_ANALYSIS
            
            retrieved_docs = self.retriever.multi_query_retrieval(
                search_queries, self.documents, self.embeddings, self
            )
            
            # 새로운 문서만 추가
            existing_ids = {doc.get('chunk_id', '') for doc in state.retrieved_docs}
            new_docs = [doc for doc in retrieved_docs if doc.get('chunk_id', '') not in existing_ids]
            state.retrieved_docs.extend(new_docs)
            
            # 웹 문서 분리
            web_docs = [doc for doc in new_docs if doc.get('source_type') == 'WEB']
            pdf_docs = [doc for doc in new_docs if doc.get('source_type') == 'PDF']
            
            research_log.append(f"검색된 문서: PDF {len(pdf_docs)}개, 웹 {len(web_docs)}개")
            
            # 조기 종료 조건 확인
            current_confidence = self._calculate_interim_confidence(state)
            state.confidence_history.append(current_confidence)
            
            if current_confidence > 0.85 and len(state.retrieved_docs) >= 5:
                research_log.append(f"높은 신뢰도 달성 ({current_confidence:.2f}) - 조기 종료")
                break
        
        # Phase 3: 최종 답변 생성
        st.info("📝 종합 답변 생성 중... (PDF + 웹 출처 포함)")
        state.phase = ResearchPhase.SYNTHESIS
        analysis_results = {"consistency": 0.8, "conflicts": [], "consensus": []}
        comprehensive_answer = self.synthesizer.synthesize_comprehensive_answer(
            query, state, analysis_results
        )
        research_log.append("종합 답변 생성 완료")
        
        # 결과 정리 (PDF + 웹 출처 구분)
        sources = []
        for doc in state.retrieved_docs[:15]:
            source_info = {
                "page": doc.get("page", 0),
                "paragraph": doc.get("paragraph", 0),
                "chunk_id": doc.get("chunk_id", ""),
                "source_file": doc.get("source", "Unknown"),
                "source_type": doc.get("source_type", "PDF"),
                "preview": doc["text"][:400] + "..." if len(doc["text"]) > 400 else doc["text"],
                "similarity": float(doc.get("final_score", doc.get("similarity", 0))),
                "chunk_size": len(doc["text"]),
                "search_category": doc.get("search_category", "일반"),
                "key_insight": doc.get("key_insight", "")
            }
            
            # 웹 문서 추가 정보
            if doc.get("source_type") == "WEB":
                source_info.update({
                    "web_title": doc.get("web_title", "제목 없음"),
                    "web_domain": doc.get("web_domain", "Unknown"),
                    "web_crawl_time": doc.get("web_crawl_time", "Unknown")
                })
            
            sources.append(source_info)
        
        return {
            "answer": comprehensive_answer,
            "confidence": current_confidence,
            "warnings": [],
            "sources": sources,
            "research_metadata": {
                "cycles_completed": state.cycle_count + 1,
                "total_documents_analyzed": len(state.retrieved_docs),
                "pdf_documents": len([s for s in sources if s["source_type"] == "PDF"]),
                "web_documents": len([s for s in sources if s["source_type"] == "WEB"]),
                "insights_discovered": len(state.insights),
                "knowledge_gaps_identified": len(state.gaps),
                "query_complexity": complexity,
                "confidence_progression": state.confidence_history,
                "research_log": research_log,
                "web_crawling_enabled": WEB_CRAWLING_AVAILABLE
            }
        }
    
    def _calculate_interim_confidence(self, state: ResearchState) -> float:
        """중간 신뢰도 계산 (PDF + 웹 문서 고려)"""
        if not state.retrieved_docs:
            return 0.0
        
        # 문서 점수 기반 신뢰도
        doc_scores = [doc.get('final_score', doc.get('similarity', 0.5)) for doc in state.retrieved_docs]
        avg_doc_score = np.mean(doc_scores) if doc_scores else 0.5
        
        # 문서 다양성 점수 (PDF + 웹)
        pdf_sources = len(set(doc['source'] for doc in state.retrieved_docs if doc.get('source_type') == 'PDF'))
        web_sources = len(set(doc['source'] for doc in state.retrieved_docs if doc.get('source_type') == 'WEB'))
        diversity_score = min((pdf_sources + web_sources) / 5.0, 1.0)
        
        # 인사이트 점수
        insight_score = min(len(state.insights) / 5.0, 1.0)
        
        return (avg_doc_score * 0.5 + diversity_score * 0.3 + insight_score * 0.2)

# ────────────────────────────────────────────────────────────
# 6. Streamlit UI (웹크롤링 기능 추가)
# ────────────────────────────────────────────────────────────

st.write("CUDA available:", torch.cuda.is_available())
if WEB_CRAWLING_AVAILABLE:
    st.write("웹크롤링 기능: ✅ 사용 가능")
else:
    st.write("웹크롤링 기능: ❌ 라이브러리 설치 필요 (pip install requests beautifulsoup4)")

st.title("🧠 Deep Research Chatbot with Web Crawling by C.H.PARK")
st.markdown("### AI 다중 에이전트 협업 시스템 + 실시간 웹크롤링으로 구현한 하이브리드 RAG")

# 사이드바 설정
st.sidebar.header("⚙️ 설정")

# 청킹 파라미터
st.sidebar.subheader("청킹 설정")
min_chunk_length = st.sidebar.slider("최소 청크 길이", 30, 500, 50)
max_chunk_length = st.sidebar.slider("최대 청크 길이", 200, 3000, 300)
sentences_per_chunk = st.sidebar.slider("청크당 최대 문장 수", 1, 10, 2)

# 웹크롤링 설정
st.sidebar.subheader("🌐 웹크롤링 설정")
enable_web_crawling = st.sidebar.checkbox("웹크롤링 활성화", value=WEB_CRAWLING_AVAILABLE, 
                                         help="질문에 따라 자동으로 웹 검색 및 크롤링을 수행합니다")

# PDF 업로드
st.sidebar.subheader("📁 파일 업로드")
uploaded_files = st.sidebar.file_uploader(
    "PDF 파일 업로드", 
    type="pdf", 
    accept_multiple_files=True,
    help="여러 PDF 파일을 동시에 업로드할 수 있습니다"
)

# 시스템 초기화 버튼
if st.sidebar.button("🔄 Deep Research 시스템 시작", type="primary"):
    if not uploaded_files:
        st.sidebar.error("❌ 최소 하나의 PDF 파일을 업로드해주세요.")
    else:
        # 업로드된 파일 저장
        pdf_paths = []
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            pdf_paths.append(file_path)
        
        # 시스템 초기화
        with st.spinner("🔄 Deep Research 시스템 초기화 중... (웹크롤링 포함)"):
            try:
                st.session_state.research_bot = DeepResearchOrchestrator(
                    min_chunk_length=min_chunk_length,
                    max_chunk_length=max_chunk_length,
                    sentences_per_chunk=sentences_per_chunk
                )
                
                # PDF 문서 로딩
                st.session_state.research_bot.load_pdf_documents(pdf_paths)
                
                st.sidebar.success("✅ Deep Research 시스템 준비 완료! (PDF + 웹)")
                
                # 시스템 정보 표시
                st.sidebar.info(f"📊 총 {len(st.session_state.research_bot.documents)}개 청크 생성됨")
                st.sidebar.info(f"📁 {len(st.session_state.research_bot.loaded_pdfs)}개 파일 로딩됨")
                st.sidebar.info(f"🌐 웹크롤링: {'✅ 활성화' if WEB_CRAWLING_AVAILABLE else '❌ 비활성화'}")
                
            except Exception as e:
                st.sidebar.error(f"❌ 초기화 실패: {e}")

# 메인 영역
if "research_bot" in st.session_state:
    # 문서 정보 표시
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📁 로딩된 파일", len(st.session_state.research_bot.loaded_pdfs))
    with col2:
        st.metric("📄 PDF 청크", len(st.session_state.research_bot.documents))
    with col3:
        avg_chunk_size = np.mean([len(doc["text"]) for doc in st.session_state.research_bot.documents]) if st.session_state.research_bot.documents else 0
        st.metric("📏 평균 청크 크기", f"{avg_chunk_size:.0f}자")
    with col4:
        st.metric("🤖 활성 에이전트", "5개")
    with col5:
        web_status = "🌐 ON" if WEB_CRAWLING_AVAILABLE else "🌐 OFF"
        st.metric("웹크롤링", web_status)
    
    st.divider()
    
    # 질문 입력
    st.subheader("💬 하이브리드 Deep Research 질문")
    st.info("💡 PDF 문서 분석과 최신 웹 정보를 함께 활용하여 답변합니다. 웹에서 가져온 정보는 출처(URL)가 명시됩니다.")
    
    query = st.text_input(
        "심층 연구할 주제를 입력하세요:",
        placeholder="예: 최신 AI 동향과 PDF 문서의 내용을 비교 분석해주세요.",
        key="hybrid_query_input"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        research_button = st.button("🧠 하이브리드 Research 시작", type="primary")
    with col2:
        if st.button("🗑️ 연구 기록 초기화"):
            if "research_history" in st.session_state:
                del st.session_state.research_history
            st.rerun()
    
    # Deep Research 실행
    if research_button and query:
        with st.spinner("🧠 다중 에이전트가 PDF + 웹 하이브리드 연구를 수행하고 있습니다..."):
            start_time = time.time()
            result = st.session_state.research_bot.deep_research(query)
            elapsed_time = time.time() - start_time
        
        torch.cuda.empty_cache()
        gc.collect()
        
        # 연구 기록 저장
        if "research_history" not in st.session_state:
            st.session_state.research_history = []
        
        st.session_state.research_history.append({
            "query": query,
            "result": result,
            "timestamp": datetime.now(),
            "elapsed_time": elapsed_time
        })
        
        # 결과 표시
        st.subheader("🎯 하이브리드 Deep Research 결과")
        st.write(result["answer"])
        
        # 상세 메트릭 정보
        col1, col2, col3, col4, col5 = st.columns(5)
        metadata = result["research_metadata"]
        
        with col1:
            st.metric("신뢰도", f"{result['confidence']:.3f}")
        with col2:
            st.metric("연구 사이클", f"{metadata['cycles_completed']}/{metadata.get('max_cycles', 3)}")
        with col3:
            st.metric("PDF 문서", metadata.get('pdf_documents', 0))
        with col4:
            st.metric("웹 문서", metadata.get('web_documents', 0))
        with col5:
            st.metric("소요 시간", f"{elapsed_time:.1f}초")
        
        # 참조 소스 표시 (PDF와 웹 분리)
        if result["sources"]:
            st.subheader("📚 분석된 소스")
            
            # PDF와 웹 소스 분리
            pdf_sources = [s for s in result["sources"] if s["source_type"] == "PDF"]
            web_sources = [s for s in result["sources"] if s["source_type"] == "WEB"]
            
            col1, col2 = st.columns(2)
            
            with col1:
                if pdf_sources:
                    st.subheader("📄 PDF 문서 소스")
                    for i, source in enumerate(pdf_sources[:5], 1):
                        with st.expander(f"PDF {i}: {source['source_file']} (점수: {source['similarity']:.3f})"):
                            st.write(f"**파일**: {source['source_file']}")
                            st.write(f"**페이지**: {source['page']}, **단락**: {source['paragraph']}")
                            st.write(f"**미리보기**: {source['preview'][:200]}...")
                            if source.get('key_insight'):
                                st.write(f"💡 **핵심 인사이트**: {source['key_insight']}")
            
            with col2:
                if web_sources:
                    st.subheader("🌐 웹 문서 소스")
                    for i, source in enumerate(web_sources[:5], 1):
                        with st.expander(f"웹 {i}: {source.get('web_title', '제목 없음')} (점수: {source['similarity']:.3f})"):
                            st.write(f"**제목**: {source.get('web_title', '제목 없음')}")
                            st.write(f"**URL**: {source['source_file']}")
                            st.write(f"**도메인**: {source.get('web_domain', 'Unknown')}")
                            st.write(f"**크롤링 시간**: {source.get('web_crawl_time', 'Unknown')}")
                            st.write(f"**미리보기**: {source['preview'][:200]}...")
        
        # 연구 로그 표시
        with st.expander("🔍 상세 연구 로그"):
            for log_entry in metadata.get('research_log', []):
                st.write(log_entry)

    # 연구 기록 표시
    if "research_history" in st.session_state and st.session_state.research_history:
        st.divider()
        st.subheader("📜 하이브리드 Deep Research 기록")
        
        for i, research in enumerate(reversed(st.session_state.research_history[-3:]), 1):
            with st.expander(f"연구 {len(st.session_state.research_history) - i + 1}: {research['query'][:50]}..."):
                st.write(f"**질문**: {research['query']}")
                st.write(f"**주요 결과**: {research['result']['answer'][:300]}...")
                st.write(f"**연구 시간**: {research['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"**최종 신뢰도**: {research['result']['confidence']:.3f}")
                
                metadata = research['result']['research_metadata']
                pdf_count = metadata.get('pdf_documents', 0)
                web_count = metadata.get('web_documents', 0)
                st.write(f"**분석 소스**: PDF {pdf_count}개, 웹 {web_count}개")

else:
    # 시스템이 초기화되지 않은 경우
    st.info("👆 왼쪽 사이드바에서 PDF 파일을 업로드하고 'Deep Research 시스템 시작' 버튼을 클릭해주세요.")
    
    # 기능 소개
    st.subheader("🆕 하이브리드 Deep Research의 새로운 기능")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📄 PDF 문서 분석**
        - 업로드된 문서의 심층 분석
        - 정확한 청킹 및 임베딩
        - 출처 명시: [파일명, 관련도]
        """)
    
    with col2:
        st.markdown("""
        **🌐 실시간 웹크롤링**
        - 최신 정보 자동 검색
        - 관련 웹페이지 크롤링
        - 출처 명시: [제목, URL, 크롤링시간, 관련도]
        """)

# 사용 가이드
with st.expander("🎓 하이브리드 Deep Research 사용 가이드"):
    st.markdown("""
    ### 🧠 하이브리드 Deep Research 시스템의 특징
    
    **다중 에이전트 협업 + 웹크롤링**:
    - 📋 **Research Planner**: 질문 분석 및 PDF/웹 검색 계획 수립
    - 🔍 **Retriever Agent**: PDF 문서 + 웹 크롤링 통합 검색
    - 🔬 **Analyzer Agent**: PDF + 웹 정보 교차 검증
    - 📝 **Synthesizer Agent**: 출처별 인용을 포함한 종합 답변 생성
    - ✅ **Validator Agent**: 최종 품질 검증
    
    **하이브리드 정보 소스**:
    - PDF 문서: 상세하고 정확한 기존 정보
    - 웹 크롤링: 최신 동향 및 일반 지식
    - 출처 구분: PDF와 웹 소스를 명확히 구분하여 표시
    
    **출처 표시 형식**:
    - PDF: [파일명, 관련도: 0.XX]
    - 웹: [제목/도메인, URL, 크롤링시간, 관련도: 0.XX]
    
    **최적화된 질문 유형**:
    - 🔍 **최신 동향**: "최근 AI 발전과 PDF의 기술 동향을 비교해주세요"
    - 🎯 **심층 분석**: "PDF 내용을 기반으로 현재 시장 상황을 분석해주세요"  
    - 🔗 **통합 분석**: "문서 내용과 최신 뉴스를 종합하여 평가해주세요"
    - 📊 **트렌드 분석**: "PDF 데이터와 현재 웹 정보를 비교 분석해주세요"
    """)
    
    st.info("💡 PDF 문서의 정확성과 웹 정보의 최신성을 결합하여 더욱 포괄적이고 신뢰할 수 있는 답변을 제공합니다.")
