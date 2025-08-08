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
        if not url.startswith("http"):
            return []
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
# 4. 에이전트 정의
# ────────────────────────────────────────────────────────────

class ResearchPlannerAgent:
    def __init__(self, tokenizer, model, device_config):
        self.tokenizer = tokenizer
        self.model = model
        self.device_config = device_config

    def analyze_query_complexity(self, query: str) -> float:
        score = 0.3
        if len(query) > 50: score += 0.2
        indicators = ["비교","분석","평가","왜","어떻게","언제"]
        score += min(sum(1 for w in indicators if w in query)*0.1, 0.3)
        if "?" in query: score += 0.1
        return min(score,1.0)

    def generate_research_plan(self, query: str, state: ResearchState) -> List[SearchQuery]:
        return [SearchQuery(text=query, priority=1.0, category="주요개념", reason="기본")]

class RetrieverAgent:
    def __init__(self, etok, emodel, reranker, dconfig):
        self.etok, self.emodel, self.reranker, self.dconfig = etok, emodel, reranker, dconfig

    def multi_query_retrieval(self, sqs, docs, embs):
        return docs[:5]

class AnalyzerAgent:
    def __init__(self, tokenizer, model, dconfig):
        pass

    def cross_validate_information(self, docs):
        return {"consistency":1.0,"conflicts":[],"consensus":[]}

class SynthesizerAgent:
    def __init__(self, tokenizer, model, dconfig):
        pass

    def synthesize_comprehensive_answer(self, q, state, ar):
        return "아직 구현되지 않음"

class ValidatorAgent:
    def __init__(self, tokenizer, model, dconfig):
        pass

    def comprehensive_validation(self, q, ans, state):
        return {"confidence":1.0,"warnings":[]}

# ────────────────────────────────────────────────────────────
# 5. Orchestrator
# ────────────────────────────────────────────────────────────

class DeepResearchOrchestrator:
    def __init__(self, model_name, embed_model, reranker_name,
                 min_chunk_length, max_chunk_length, sentences_per_chunk):
        self.device_config = DeviceConfig()
        self.chunker = ImprovedKoreanSentenceChunker(min_chunk_length, max_chunk_length, sentences_per_chunk)
        self.pdf_extractor = ImprovedPDFExtractor()
        self.crawler = CrawlerAgent(self.chunker)

        # load models
        self.embed_tokenizer = AutoTokenizer.from_pretrained(embed_model)
        self.embed_model = AutoModel.from_pretrained(embed_model).to(self.device_config.device)
        self.reranker = CrossEncoder(reranker_name, device=self.device_config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=self.device_config.config["torch_dtype"],
            device_map="auto", trust_remote_code=True
        ).to(self.device_config.device)
        torch.cuda.empty_cache(); gc.collect()

        # agents
        self.planner = ResearchPlannerAgent(self.tokenizer, self.model, self.device_config)
        self.retriever = RetrieverAgent(self.embed_tokenizer, self.embed_model, self.reranker, self.device_config)
        self.analyzer = AnalyzerAgent(self.tokenizer, self.model, self.device_config)
        self.synthesizer = SynthesizerAgent(self.tokenizer, self.model, self.device_config)
        self.validator = ValidatorAgent(self.tokenizer, self.model, self.device_config)

        self.documents: List[Dict[str,Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.chunk_hashes: Dict[str,Dict] = {}
        self.loaded_pdfs: List[str] = []
        self.loaded_urls: List[str] = []

    def _hash(self, text, page, para, source):
        return hashlib.md5(f"{source}_{page}_{para}_{text}".encode()).hexdigest()[:8]

    def _generate_embeddings(self):
        if not self.documents:
            self.embeddings = np.array([])
            return
        texts = [d["text"] for d in self.documents]
        bs = self.device_config.config["embedding_batch_size"]
        embs = []
        for i in range(0,len(texts),bs):
            batch = texts[i:i+bs]
            enc = self.embed_tokenizer(batch,padding=True,truncation=True,return_tensors="pt").to(self.device_config.device)
            with torch.no_grad():
                out = self.embed_model(**enc)
            embs.append(mean_pooling(out,enc["attention_mask"]).cpu().numpy())
        self.embeddings = np.vstack(embs)

    def load_documents(self, pdf_paths: List[str], crawl_urls: List[str]):
        all_docs=[]
        for path in pdf_paths:
            if not os.path.exists(path): continue
            txt=self.pdf_extractor.extract_text_from_pdf(path)
            if not txt.strip(): continue
            self.loaded_pdfs.append(path)
            for idx,chunk in enumerate(self.chunker.chunk_text(txt),1):
                all_docs.append({"text":chunk,"page":1,"paragraph":idx,
                                 "source":os.path.basename(path),"full_path":path})
        urls=[u.strip() for u in crawl_urls if u.strip().startswith("http")]
        for url in urls:
            crawled=self.crawler.crawl_and_chunk(url)
            if crawled:
                all_docs.extend(crawled)
                self.loaded_urls.append(url)
        self.documents=all_docs
        self.chunk_hashes={}
        for d in self.documents:
            cid=self._hash(d["text"],d["page"],d["paragraph"],d["source"])
            d["chunk_id"]=cid
            self.chunk_hashes[cid]=d
        self._generate_embeddings()

    def deep_research(self, query:str)->Dict[str,Any]:
        state=ResearchState(ResearchPhase.PLANNING,query,[],[],[],[],[])
        plan=self.planner.generate_research_plan(query,state)
        state.sub_queries=[q.text for q in plan]
        retrieved=self.retriever.multi_query_retrieval(plan,self.documents,self.embeddings)
        state.retrieved_docs=retrieved
        ar=self.analyzer.cross_validate_information(retrieved)
        ans=self.synthesizer.synthesize_comprehensive_answer(query,state,ar)
        val=self.validator.comprehensive_validation(query,ans,state)
        return {
            "answer": ans,
            "confidence": val["confidence"],
            "warnings": val["warnings"],
            "sources":[{"source_file":d["source"],"search_category":"N/A","similarity":0.0,"key_insight":""} for d in retrieved],
            "research_metadata":{
                "cycles_completed":1,
                "max_cycles":state.max_cycles,
                "total_documents_analyzed":len(retrieved),
                "confidence_progression":[val["confidence"]],
                "research_log":["Completed"],
                "cross_validation":ar
            }
        }

# ────────────────────────────────────────────────────────────
# 6. Streamlit UI
# ────────────────────────────────────────────────────────────

st.title("🧠 Deep Research Chatbot")
st.sidebar.header("⚙️ 설정")

min_chunk_length = st.sidebar.slider("최소 청크 길이", 30, 500, 50)
max_chunk_length = st.sidebar.slider("최대 청크 길이", 200, 3000, 300)
sentences_per_chunk = st.sidebar.slider("문장 수/청크", 1, 10, 2)

uploaded_files = st.sidebar.file_uploader("PDF 업로드",type="pdf",accept_multiple_files=True)
crawl_input = st.sidebar.text_area("크롤링할 URL (줄바꿈)", "").splitlines()

if st.sidebar.button("🔄 시스템 시작"):
    if not uploaded_files:
        st.sidebar.error("PDF 파일 최소 1개 필요")
    else:
        pdf_paths=[]
        tmp="temp_uploads";os.makedirs(tmp,exist_ok=True)
        for f in uploaded_files:
            p=os.path.join(tmp,f.name)
            with open(p,"wb") as fp:fp.write(f.getbuffer())
            pdf_paths.append(p)
        bot=DeepResearchOrchestrator(
            model_name="LGAI-EXAONE/EXAONE-4.0-1.2B",
            embed_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            reranker_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            min_chunk_length=min_chunk_length,
            max_chunk_length=max_chunk_length,
            sentences_per_chunk=sentences_per_chunk
        )
        bot.load_documents(pdf_paths,crawl_input)
        st.session_state.research_bot=bot
        st.sidebar.success("✅ 초기화 완료")

if "research_bot" in st.session_state:
    bot=st.session_state.research_bot
    st.subheader("💬 Deep Research 질문")
    query=st.text_input("질문 입력:",key="deep_query_input")
    if st.button("🧠 시작") and query:
        with st.spinner("연구 중..."):
            start=time.time()
            res=bot.deep_research(query)
            elapsed=time.time()-start
        st.subheader("🎯 결과")
        st.write(res["answer"])
        meta=res["research_metadata"]
        c1,c2,c3,c4=st.columns(4)
        c1.metric("신뢰도",f"{res['confidence']:.3f}")
        c2.metric("사이클",f"{meta['cycles_completed']}/{meta['max_cycles']}")
        c3.metric("문서 수",meta["total_documents_analyzed"])
        c4.metric("소요 시간",f"{elapsed:.1f}초")
        if meta.get("confidence_progression"):
            st.line_chart({"신뢰도":meta["confidence_progression"]})
        if res.get("sources"):
            st.subheader("📚 소스")
            for src in res["sources"]:
                st.write(f"{src['source_file']} ({src['search_category']})")
        if res.get("warnings"):
            st.warning("⚠️ "+" | ".join(res["warnings"]))
        if meta.get("research_log"):
            with st.expander("로그"):
                for e in meta["research_log"]:
                    st.write(e)
        if "research_history" not in st.session_state:
            st.session_state.research_history=[]
        st.session_state.research_history.append({
            "query":query,"result":res,
            "timestamp":datetime.now(),"elapsed_time":elapsed
        })
    if "research_history" in st.session_state and st.session_state.research_history:
        st.divider();st.subheader("📜 기록")
        for idx,rec in enumerate(reversed(st.session_state.research_history[-3:]),1):
            with st.expander(f"{len(st.session_state.research_history)-idx+1}. {rec['query'][:30]}..."):
                st.write(f"**질문:** {rec['query']}")
                st.write(f"**답변:** {rec['result']['answer'][:200]}...")
                st.write(f"**시간:** {rec['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"**신뢰도:** {rec['result']['confidence']:.3f}")

    st.divider()
    with st.expander("🎓 사용 가이드"):
        st.markdown("""
        - 다중 에이전트 기반 심층 연구
        - PDF 및 웹 크롤링 지원
        """)
