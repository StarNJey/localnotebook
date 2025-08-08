from __future__ import annotations

import os
import re
import hashlib
import time
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

import streamlit as st
import numpy as np
import torch, gc
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from sentence_transformers.cross_encoder import CrossEncoder
import warnings

# Streamlit 페이지 설정
st.set_page_config(page_title="Deep Research Chatbot", layout="wide")
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# PDF 라이브러리
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

# 한국어 문장 분리 라이브러리
try:
    from kiwipiepy import Kiwi
    KIWI_AVAILABLE = True
except ImportError:
    KIWI_AVAILABLE = False

# ────────────────────────────────────────────────────────────
# 1. 데이터 구조 정의
# ────────────────────────────────────────────────────────────

class ResearchPhase(Enum):
    PLANNING = "planning"
    INITIAL_RETRIEVAL = "initial_retrieval"
    DEEP_ANALYSIS = "deep_analysis"
    CROSS_VALIDATION = "cross_validation"
    SYNTHESIS = "synthesis"
    FINAL_VALIDATION = "final_validation"

@dataclass
class ResearchState:
    phase: ResearchPhase
    query: str
    sub_queries: List[str]
    retrieved_docs: List[Dict[str, Any]]
    confidence_history: List[float]
    insights: List[str]
    gaps: List[str]
    cycle_count: int = 0
    max_cycles: int = 3

@dataclass
class SearchQuery:
    text: str
    priority: float
    category: str
    reason: str

# ────────────────────────────────────────────────────────────
# 2. 유틸리티 함수
# ────────────────────────────────────────────────────────────

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)

# ────────────────────────────────────────────────────────────
# 3. 청킹 및 추출기
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
        sentences = self._postprocess_sentences(sentences)
        return self._create_chunks(sentences)

    def _split_sentences(self, text: str) -> List[str]:
        if self.kiwi:
            try:
                kiwi_result = self.kiwi.split_into_sents(text)
                sents = [sent.text.strip() for sent in kiwi_result if sent.text.strip()]
                if len(sents) > 1:
                    return sents
            except:
                pass
        return self._regex_sentence_split(text)

    def _regex_sentence_split(self, text: str) -> List[str]:
        patterns = [
            r'[.!?]+\s+', r'\n\s*\n', r'\.\s*\n',
            r'[다가나니까요래습니다]\s*[.!?]*\s+',
            r'[니다했다습니다였다았다]\s*[.!?]*\s+',
        ]
        combined = '|'.join(f'({p})' for p in patterns)
        parts = re.split(combined, text)
        return [p.strip() for p in parts if p and not re.match(r'^\s*$', p)]

    def _postprocess_sentences(self, sentences: List[str]) -> List[str]:
        processed, buf = [], ""
        for s in sentences:
            if len(buf) < self.min_chunk_length:
                buf = f"{buf} {s}".strip()
            else:
                processed.append(buf)
                buf = s
        if buf:
            processed.append(buf)
        return processed

    def _create_chunks(self, sentences: List[str]) -> List[str]:
        chunks, buf, count = [], "", 0
        for s in sentences:
            candidate = f"{buf} {s}".strip() if buf else s
            if len(candidate) <= self.max_chunk_length and count < self.sentences_per_chunk:
                buf, count = candidate, count + 1
            else:
                if buf:
                    chunks.append(buf)
                buf, count = s, 1
        if buf:
            chunks.append(buf)
        return chunks

class ImprovedPDFExtractor:
    def extract_text_from_pdf(self, path: str) -> str:
        if not PDFPLUMBER_AVAILABLE:
            return ""
        try:
            text = ""
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    ptxt = page.extract_text() or ""
                    text += ptxt + "\n"
            return text
        except:
            return ""

class CrawlerAgent:
    """웹 크롤링 및 청크 생성 에이전트"""
    def __init__(self, chunker: ImprovedKoreanSentenceChunker):
        self.chunker = chunker

    def crawl_and_chunk(self, url: str) -> List[Dict[str, Any]]:
        try:
            resp = requests.get(url, timeout=10, headers={"User-Agent": "DeepResearchBot/1.0"})
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            paras = [p.get_text(strip=True) for p in soup.find_all("p") if p.get_text(strip=True)]
            text = "\n".join(paras)
        except:
            return []
        chunks = self.chunker.chunk_text(text)
        return [
            {"text": c, "page": 0, "paragraph": i+1, "source": url, "full_path": url}
            for i, c in enumerate(chunks)
        ]

# ────────────────────────────────────────────────────────────
# 4. 디바이스 및 에이전트 설정
# ────────────────────────────────────────────────────────────

class DeviceConfig:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = self._get_config()

    def _get_config(self) -> Dict[str, Any]:
        if self.device == "cuda":
            return {"torch_dtype": torch.bfloat16, "embedding_batch_size": 32, "sim_threshold": 0.3}
        else:
            return {"torch_dtype": torch.float32, "embedding_batch_size": 4, "sim_threshold": 0.3}

# Placeholder agent classes for completeness; implement as needed
class ResearchPlannerAgent: ...
class RetrieverAgent: ...
class AnalyzerAgent: ...
class SynthesizerAgent: ...
class ValidatorAgent: ...

# ────────────────────────────────────────────────────────────
# 5. 메인 오케스트레이터 클래스
# ────────────────────────────────────────────────────────────

class DeepResearchOrchestrator:
    def __init__(
        self,
        model_name: str,
        embed_model: str,
        reranker_name: str,
        min_chunk_length: int,
        max_chunk_length: int,
        sentences_per_chunk: int,
    ):
        st.info("🚀 시스템 초기화 중...")
        self.device_config = DeviceConfig()

        # 유틸리티
        self.chunker = ImprovedKoreanSentenceChunker(min_chunk_length, max_chunk_length, sentences_per_chunk)
        self.pdf_extractor = ImprovedPDFExtractor()
        self.crawler = CrawlerAgent(self.chunker)

        # 모델 로딩
        self.embed_tokenizer = AutoTokenizer.from_pretrained(embed_model)
        self.embed_model = AutoModel.from_pretrained(embed_model).to(self.device_config.device)
        self.reranker = CrossEncoder(reranker_name, device=self.device_config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=self.device_config.config["torch_dtype"],
            device_map="auto", trust_remote_code=True
        ).to(self.device_config.device)
        torch.cuda.empty_cache(); gc.collect()

        # 에이전트 초기화
        self.planner = ResearchPlannerAgent(self.tokenizer, self.model, self.device_config)
        self.retriever = RetrieverAgent(self.embed_tokenizer, self.embed_model, self.reranker, self.device_config)
        self.analyzer = AnalyzerAgent(self.tokenizer, self.model, self.device_config)
        self.synthesizer = SynthesizerAgent(self.tokenizer, self.model, self.device_config)
        self.validator = ValidatorAgent(self.tokenizer, self.model, self.device_config)

        # 데이터 저장소
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.chunk_hashes: Dict[str, Dict] = {}
        self.loaded_pdfs: List[str] = []
        self.loaded_urls: List[str] = []
        st.success("✅ 시스템 준비 완료!")

    def _hash(self, text: str, page: int, para: int, source: str) -> str:
        return hashlib.md5(f"{source}_{page}_{para}_{text}".encode()).hexdigest()[:8]

    def _generate_embeddings(self):
        texts = [d["text"] for d in self.documents]
        bs = self.device_config.config["embedding_batch_size"]
        embs = []
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            enc = self.embed_tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device_config.device)
            with torch.no_grad():
                out = self.embed_model(**enc)
            embs.append(mean_pooling(out, enc["attention_mask"]).cpu().numpy())
        self.embeddings = np.vstack(embs) if embs else np.array([])

    def load_documents(self, pdf_paths: List[str], crawl_urls: List[str]) -> None:
        st.info(f"📚 PDF 처리 중 ({len(pdf_paths)}개)")
        all_docs: List[Dict[str, Any]] = []

        # PDF 로딩
        for path in pdf_paths:
            if not os.path.exists(path):
                st.warning(f"❌ 파일 못 찾음: {path}")
                continue
            text = self.pdf_extractor.extract_text_from_pdf(path)
            if not text.strip():
                st.warning(f"⚠️ PDF 텍스트 없음: {path}")
                continue
            self.loaded_pdfs.append(path)
            for idx, chunk in enumerate(self.chunker.chunk_text(text), start=1):
                all_docs.append({"text": chunk, "page": 1, "paragraph": idx,
                                 "source": os.path.basename(path), "full_path": path})

        # 웹 크롤링
        urls = [u.strip() for u in crawl_urls if u.strip()]
        if urls:
            st.info(f"🌐 크롤링 중 ({len(urls)}개)")
            for url in urls:
                crawled = self.crawler.crawl_and_chunk(url)
                if crawled:
                    all_docs.extend(crawled)
                    self.loaded_urls.append(url)
                else:
                    st.warning(f"⚠️ 크롤링 실패: {url}")

        if not all_docs:
            st.error("❌ 처리할 문서 없음")
            return

        # 청크 ID 생성
        self.documents = all_docs
        self.chunk_hashes = {}
        for d in self.documents:
            cid = self._hash(d["text"], d["page"], d["paragraph"], d["source"])
            d["chunk_id"] = cid
            self.chunk_hashes[cid] = d

        # 임베딩
        st.info("🧮 임베딩 생성 중...")
        self._generate_embeddings()
        st.success(f"🎉 총 {len(self.documents)}개 청크 생성됨")

    def deep_research(self, query: str) -> Dict[str, Any]:
        # 실제 구현 필요
        return {
            "answer": "답변 생성 로직이 구현되어야 합니다.",
            "confidence": 0.0,
            "warnings": [],
            "sources": [],
            "research_metadata": {
                "cycles_completed": 0,
                "max_cycles": 0,
                "total_documents_analyzed": 0,
                "confidence_progression": [],
                "research_log": []
            }
        }

# ────────────────────────────────────────────────────────────
# 6. Streamlit UI
# ────────────────────────────────────────────────────────────

st.title("🧠 Deep Research Chatbot")
st.sidebar.header("⚙️ 설정")

# 청킹 파라미터
min_chunk_length = st.sidebar.slider("최소 청크 길이", 30, 500, 50)
max_chunk_length = st.sidebar.slider("최대 청크 길이", 200, 3000, 300)
sentences_per_chunk = st.sidebar.slider("문장 수/청크", 1, 10, 2)

# 파일 업로드 & URL 입력
uploaded_files = st.sidebar.file_uploader("PDF 업로드", type="pdf", accept_multiple_files=True)
crawl_input = st.sidebar.text_area("크롤링할 URL (줄바꿈)", "").splitlines()

if st.sidebar.button("🔄 시스템 시작"):
    if not uploaded_files:
        st.sidebar.error("PDF 파일 최소 1개 필요")
    else:
        pdf_paths = []
        tmp = "temp_uploads"
        os.makedirs(tmp, exist_ok=True)
        for f in uploaded_files:
            p = os.path.join(tmp, f.name)
            with open(p, "wb") as fp:
                fp.write(f.getbuffer())
            pdf_paths.append(p)
        bot = DeepResearchOrchestrator(
            model_name="LGAI-EXAONE/EXAONE-4.0-1.2B",
            embed_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            reranker_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            min_chunk_length=min_chunk_length,
            max_chunk_length=max_chunk_length,
            sentences_per_chunk=sentences_per_chunk
        )
        bot.load_documents(pdf_paths, crawl_input)
        st.session_state.research_bot = bot
        st.sidebar.success("✅ 초기화 완료")

if "research_bot" in st.session_state:
    bot = st.session_state.research_bot

    # 질문 입력 및 실행
    st.subheader("💬 Deep Research 질문")
    query = st.text_input("심층 연구할 주제를 입력하세요:", key="deep_query_input")

    if st.button("🧠 Deep Research 시작") and query:
        with st.spinner("🧠 다중 에이전트가 심층 연구 중..."):
            start_time = time.time()
            result = bot.deep_research(query)
            elapsed = time.time() - start_time

        st.subheader("🎯 결과")
        st.write(result["answer"])

        meta = result["research_metadata"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("신뢰도", f"{result['confidence']:.3f}")
        col2.metric("사이클", f"{meta['cycles_completed']}/{meta['max_cycles']}")
        col3.metric("문서 분석", meta["total_documents_analyzed"])
        col4.metric("소요 시간", f"{elapsed:.1f}초")

        if meta.get("confidence_progression"):
            st.subheader("📈 신뢰도 변화")
            st.line_chart({"신뢰도": meta["confidence_progression"]})

        if result.get("sources"):
            st.subheader("📚 참조 소스")
            for src in result["sources"]:
                st.markdown(f"- **{src['source_file']}** ({src['search_category']}, 점수 {src['similarity']:.3f})")
                if src.get("key_insight"):
                    st.markdown(f"  - 인사이트: {src['key_insight']}")

        if result.get("warnings"):
            st.warning("⚠️ " + " | ".join(result["warnings"]))

        if meta.get("research_log"):
            with st.expander("🔍 연구 로그"):
                for entry in meta["research_log"]:
                    st.write(entry)

        # 연구 기록 저장
        if "research_history" not in st.session_state:
            st.session_state.research_history = []
        st.session_state.research_history.append({
            "query": query,
            "result": result,
            "timestamp": datetime.now(),
            "elapsed_time": elapsed
        })

    # 이전 연구 기록 표시
    if "research_history" in st.session_state and st.session_state.research_history:
        st.divider()
        st.subheader("📜 이전 Deep Research 기록")
        for idx, record in enumerate(reversed(st.session_state.research_history[-3:]), 1):
            with st.expander(f"{len(st.session_state.research_history)-idx+1}. {record['query'][:30]}..."):
                st.write(f"**질문:** {record['query']}")
                st.write(f"**결과 요약:** {record['result']['answer'][:200]}...")
                st.write(f"**완료 시각:** {record['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"**소요 시간:** {record['elapsed_time']:.1f}초")
                st.write(f"**최종 신뢰도:** {record['result']['confidence']:.3f}")

    # 사용 가이드
    st.divider()
    with st.expander("🎓 Deep Research 사용 가이드"):
        st.markdown("""
        ### 🧠 Deep Research 시스템의 특징
        
        **다중 에이전트 협업**:
        - Research Planner, Retriever Agent, Analyzer Agent, Synthesizer Agent, Validator Agent
        **지능형 연구 과정**:
        - 적응형 연구 사이클, 지식 격차 보완, 신뢰도 평가 및 조기 종료
        """)
        st.info("💡 더욱 정확한 답변을 위해 다수 에이전트를 활용합니다.")
