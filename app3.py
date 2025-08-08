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

import streamlit as st
import numpy as np
import torch, gc
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from sentence_transformers.cross_encoder import CrossEncoder
import warnings

# ⚠️ 수정사항 1: st.set_page_config를 맨 앞으로 이동
st.set_page_config(page_title="Deep Research Chatbot", layout="wide")

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
    """연구 진행 상태를 추적하는 클래스"""
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
    """검색 쿼리 정보"""
    text: str
    priority: float
    category: str
    reason: str

# ────────────────────────────────────────────────────────────
# 2. 유틸리티 함수들
# ────────────────────────────────────────────────────────────

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)

# ────────────────────────────────────────────────────────────
# 3. 기존 유틸리티 클래스들
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
# 4. Agent 클래스들
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
            "어떻게", "왜", "언제", "어디서", "누가", "무엇을", "상세", "구체적"
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
        """질문을 분석하여 다단계 검색 계획 생성"""
        
        planning_prompt = f"""다음 질문에 대한 체계적인 연구 계획을 수립하세요:

질문: {query}

다음 단계로 구성된 검색 계획을 생성하세요:
1. 핵심 키워드 추출
2. 하위 질문 분해
3. 우선순위 설정

각 검색 쿼리는 다음 형식으로 생성:
- 검색어: [구체적 검색어]
- 우선순위: [1-10점]
- 카테고리: [주요개념/세부사항/배경지식/비교분석]
- 이유: [왜 이 검색이 필요한지]

최대 5개의 검색 쿼리를 생성하세요."""
        
        messages = [
            {"role": "system", "content": "당신은 연구 계획을 수립하는 전문가입니다."},
            {"role": "user", "content": planning_prompt}
        ]
        
        response = self._generate_llm_response(messages, max_tokens=1000)
        return self._parse_search_queries(response, state)  # ✅ 수정: state 매개변수 추가
    
    def _parse_search_queries(self, response: str, state: ResearchState) -> List[SearchQuery]:  # ✅ 수정: state 매개변수 추가
        """LLM 응답에서 검색 쿼리 파싱"""
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
                
                if all(key in current_query for key in ["text", "priority", "category", "reason"]):
                    queries.append(SearchQuery(**current_query))
                    current_query = {}
        
        # 기본 검색 쿼리가 없으면 원본 질문 사용
        if not queries:
            queries.append(SearchQuery(
                text=state.query,  # ✅ 수정: 이제 state를 사용할 수 있음
                priority=1.0,
                category="주요개념",
                reason="기본 검색"
            ))
            
        return queries[:5]  # 최대 5개로 제한
    
    def identify_knowledge_gaps(self, state: ResearchState) -> List[str]:
        """현재까지의 연구 결과에서 지식 격차 식별"""
        if not state.retrieved_docs:
            return ["기초 정보 부족"]
            
        gap_analysis_prompt = f"""다음 연구 결과를 분석하여 부족한 정보를 식별하세요:

원본 질문: {state.query}
현재까지 수집된 정보:
{chr(10).join([doc.get('text', '')[:200] + '...' for doc in state.retrieved_docs[:3]])}

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
    """문서 검색 전문 에이전트"""
    
    def __init__(self, embed_tokenizer, embed_model, reranker, device_config):
        self.embed_tokenizer = embed_tokenizer
        self.embed_model = embed_model
        self.reranker = reranker
        self.device_config = device_config
        
    def multi_query_retrieval(self, search_queries: List[SearchQuery], 
                            documents: List[Dict], embeddings: np.ndarray) -> List[Dict]:
        """다중 쿼리를 사용한 포괄적 검색"""
        all_results = []
        
        for search_query in search_queries:
            # 각 검색 쿼리에 대해 검색 수행
            results = self._single_query_retrieval(
                search_query.text, documents, embeddings, 
                top_k=max(5, int(10 * search_query.priority))
            )
            
            # 우선순위와 카테고리 정보 추가
            for result in results:
                result['search_priority'] = search_query.priority
                result['search_category'] = search_query.category
                result['search_reason'] = search_query.reason
                
            all_results.extend(results)
        
        # 중복 제거 및 점수 정규화
        unique_results = self._deduplicate_results(all_results)
        return self._rerank_results(search_queries[0].text if search_queries else "", unique_results)
    
    def _single_query_retrieval(self, query: str, documents: List[Dict], 
                              embeddings: np.ndarray, top_k: int = 10) -> List[Dict]:
        """단일 쿼리 검색"""
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
            if sims[idx] >= threshold:
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
            result["final_score"] = (result["similarity"] * 0.4 + score * 0.6)
        
        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results[:top_k]

class AnalyzerAgent:
    """문서 분석 및 검증 에이전트"""
    
    def __init__(self, llm_tokenizer, llm_model, device_config):
        self.tokenizer = llm_tokenizer
        self.model = llm_model
        self.device_config = device_config
    
    def analyze_document_relevance(self, query: str, documents: List[Dict]) -> List[Dict]:
        """문서 관련성 심화 분석"""
        analyzed_docs = []
        
        for doc in documents:
            analysis = self._deep_analyze_single_doc(query, doc)
            doc.update(analysis)
            analyzed_docs.append(doc)
        
        return analyzed_docs
    
    def _deep_analyze_single_doc(self, query: str, doc: Dict) -> Dict[str, Any]:
        """개별 문서 심화 분석"""
        analysis_prompt = f"""다음 문서가 질문에 얼마나 관련있는지 분석하세요:

질문: {query}

문서 내용:
{doc['text'][:500]}...

다음 항목을 평가하세요:
1. 직접적 관련성 (1-10점)
2. 핵심 정보 포함 여부
3. 신뢰성 수준
4. 주요 인사이트 (한 줄로)

간단히 답변하세요."""
        
        messages = [
            {"role": "system", "content": "당신은 문서 관련성을 분석하는 전문가입니다."},
            {"role": "user", "content": analysis_prompt}
        ]
        
        response = self._generate_llm_response(messages, max_tokens=200)
        
        # 응답 파싱
        relevance_score = self._extract_score(response)
        key_insight = self._extract_insight(response)
        
        return {
            "analysis_score": relevance_score,
            "key_insight": key_insight,
            "analyzed": True
        }
    
    def cross_validate_information(self, documents: List[Dict]) -> Dict[str, Any]:
        """여러 문서 간 정보 교차 검증"""
        if len(documents) < 2:
            return {"consistency": 1.0, "conflicts": [], "consensus": []}
        
        validation_prompt = f"""다음 문서들의 정보를 교차 검증하세요:

{chr(10).join([f"문서 {i+1}: {doc['text'][:200]}..." for i, doc in enumerate(documents[:3])])}

1. 일관된 정보
2. 상충되는 정보
3. 신뢰도 평가

간단히 정리하세요."""
        
        messages = [
            {"role": "system", "content": "당신은 정보 검증 전문가입니다."},
            {"role": "user", "content": validation_prompt}
        ]
        
        response = self._generate_llm_response(messages, max_tokens=300)
        
        return {
            "consistency": self._calculate_consistency(documents),
            "conflicts": self._identify_conflicts(response),
            "consensus": self._identify_consensus(response)
        }
    
    def _generate_llm_response(self, messages: List[Dict], max_tokens: int = 300) -> str:
        """LLM 응답 생성"""
        try:
            input_ids = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
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
            
            return self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        except:
            return ""
    
    def _extract_score(self, text: str) -> float:
        """텍스트에서 점수 추출"""
        scores = re.findall(r'(\d+)점', text)
        if scores:
            return float(scores[0]) / 10.0
        return 0.5
    
    def _extract_insight(self, text: str) -> str:
        """텍스트에서 핵심 인사이트 추출"""
        lines = text.split('\n')
        for line in lines:
            if '인사이트' in line or '핵심' in line:
                return line.strip()
        return "추가 분석 필요"
    
    def _calculate_consistency(self, documents: List[Dict]) -> float:
        """문서 간 일관성 계산"""
        if len(documents) < 2:
            return 1.0
        
        # 간단한 텍스트 유사도 기반 일관성 측정
        texts = [doc['text'] for doc in documents]
        total_similarity = 0
        pairs = 0
        
        for i in range(len(texts)):
            for j in range(i+1, len(texts)):
                words1 = set(texts[i].lower().split())
                words2 = set(texts[j].lower().split())
                similarity = len(words1 & words2) / len(words1 | words2) if words1 | words2 else 0
                total_similarity += similarity
                pairs += 1
        
        return total_similarity / pairs if pairs > 0 else 0.5
    
    def _identify_conflicts(self, text: str) -> List[str]:
        """상충 정보 식별"""
        conflicts = []
        if '상충' in text or '모순' in text or '다르' in text:
            lines = text.split('\n')
            for line in lines:
                if any(word in line for word in ['상충', '모순', '다르']):
                    conflicts.append(line.strip())
        return conflicts
    
    def _identify_consensus(self, text: str) -> List[str]:
        """일관된 정보 식별"""
        consensus = []
        if '일관' in text or '공통' in text or '같' in text:
            lines = text.split('\n')
            for line in lines:
                if any(word in line for word in ['일관', '공통', '같']):
                    consensus.append(line.strip())
        return consensus

class SynthesizerAgent:
    """정보 통합 및 최종 답변 생성 에이전트"""
    
    def __init__(self, llm_tokenizer, llm_model, device_config):
        self.tokenizer = llm_tokenizer
        self.model = llm_model
        self.device_config = device_config
    
    def synthesize_comprehensive_answer(self, query: str, state: ResearchState, 
                                      analysis_results: Dict) -> str:
        """포괄적 답변 생성"""
        
        # Test-Time Compute: 복잡도에 따른 동적 설정
        complexity = len(state.retrieved_docs) * 0.1 + len(state.insights) * 0.2
        adaptive_config = self.device_config.get_adaptive_config(complexity)
        
        synthesis_prompt = self._build_synthesis_prompt(query, state, analysis_results)
        
        messages = [
            {"role": "system", "content": self._get_synthesis_system_prompt()},
            {"role": "user", "content": synthesis_prompt}
        ]
        
        return self._generate_final_answer(messages, adaptive_config)
    
    def _build_synthesis_prompt(self, query: str, state: ResearchState, 
                               analysis_results: Dict) -> str:
        """종합 답변용 프롬프트 구성"""
        
        # 문서 정보 정리
        doc_summaries = []
        for i, doc in enumerate(state.retrieved_docs[:8], 1):  # 최대 8개 문서
            summary = f"[문서 {i}] 출처: {doc['source']}, 관련도: {doc.get('final_score', 0):.2f}\n"
            summary += f"내용: {doc['text'][:300]}...\n"
            if 'key_insight' in doc:
                summary += f"핵심 인사이트: {doc['key_insight']}\n"
            doc_summaries.append(summary)
        
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
"""
        
        return f"""다음 연구 결과를 바탕으로 질문에 대한 종합적인 답변을 작성하세요:

원본 질문: {query}

=== 수집된 문서 정보 ===
{chr(10).join(doc_summaries)}

=== 분석 결과 ===
{analysis_summary}

=== 요구사항 ===
1. 문서에서 찾은 정보만을 기반으로 답변
2. 각 주장에 대해 출처 명시 (파일명, 관련도 점수)
3. 상충되는 정보가 있다면 명시
4. 부족한 정보가 있다면 언급
5. 신뢰도 수준 제시
6. 구체적이고 상세한 답변 작성"""
    
    def _get_synthesis_system_prompt(self) -> str:
        """시스템 프롬프트"""
        return """당신은 다중 소스 정보를 종합하여 정확하고 포괄적인 답변을 생성하는 연구 전문가입니다.

핵심 원칙:
1. 제공된 문서 정보만 사용
2. 모든 주장에 대한 출처 명시
3. 불확실한 정보는 신뢰도와 함께 제시
4. 상충되는 정보는 객관적으로 제시
5. 지식 격차는 솔직하게 인정
6. 논리적이고 체계적인 구조로 답변

답변 구조:
- 핵심 답변 (요약)
- 상세 설명 (근거와 함께)
- 추가 고려사항 (한계점 포함)
- 신뢰도 평가"""
    
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

class ValidatorAgent:
    """최종 검증 에이전트"""
    
    def __init__(self, llm_tokenizer, llm_model, device_config):
        self.tokenizer = llm_tokenizer
        self.model = llm_model
        self.device_config = device_config
        self.forbidden_phrases = {
            "알려지지 않은", "확실하지 않은", "아마도", "추측하건대", 
            "일반적으로", "보통", "대부분의 경우"
        }
    
    def comprehensive_validation(self, query: str, answer: str, 
                               state: ResearchState) -> Dict[str, Any]:
        """포괄적 답변 검증"""
        
        validations = {
            "source_grounding": self._validate_source_grounding(answer, state.retrieved_docs),
            "factual_consistency": self._validate_factual_consistency(answer, state.retrieved_docs),
            "completeness": self._validate_completeness(query, answer, state),
            "confidence_assessment": self._assess_confidence(answer, state),
            "forbidden_phrases": self._check_forbidden_phrases(answer)
        }
        
        # 종합 신뢰도 계산
        overall_confidence = self._calculate_overall_confidence(validations)
        warnings = self._generate_warnings(validations)
        
        return {
            "confidence": overall_confidence,
            "warnings": warnings,
            "detailed_validation": validations
        }
    
    def _validate_source_grounding(self, answer: str, documents: List[Dict]) -> Dict[str, Any]:
        """소스 기반 검증"""
        source_files = set(doc['source'] for doc in documents)
        mentioned_sources = set()
        
        for source in source_files:
            if source in answer:
                mentioned_sources.add(source)
        
        grounding_score = len(mentioned_sources) / len(source_files) if source_files else 0
        
        return {
            "score": grounding_score,
            "total_sources": len(source_files),
            "mentioned_sources": len(mentioned_sources),
            "missing_sources": source_files - mentioned_sources
        }
    
    def _validate_factual_consistency(self, answer: str, documents: List[Dict]) -> Dict[str, Any]:
        """사실 일관성 검증"""
        if not documents:
            return {"score": 0.0, "issues": ["참조 문서 없음"]}
        
        context_text = " ".join([doc['text'] for doc in documents])
        answer_words = set(answer.lower().split())
        context_words = set(context_text.lower().split())
        
        overlap_ratio = len(answer_words & context_words) / max(len(answer_words), 1)
        
        issues = []
        if overlap_ratio < 0.3:
            issues.append("문서 내용과 연관성 낮음")
        if overlap_ratio < 0.1:
            issues.append("문서 외부 정보 사용 의심")
            
        return {
            "score": overlap_ratio,
            "overlap_ratio": overlap_ratio,
            "issues": issues
        }
    
    def _validate_completeness(self, query: str, answer: str, state: ResearchState) -> Dict[str, Any]:
        """답변 완성도 검증"""
        completeness_prompt = f"""다음 답변이 질문을 얼마나 완전히 답했는지 평가하세요:

질문: {query}
답변: {answer[:1000]}...

평가 항목:
1. 질문의 모든 측면을 다뤘는가?
2. 충분한 세부사항을 제공했는가?
3. 명확하고 이해하기 쉬운가?

점수 (1-10): """
        
        messages = [
            {"role": "system", "content": "당신은 답변 완성도를 평가하는 전문가입니다."},
            {"role": "user", "content": completeness_prompt}
        ]
        
        response = self._generate_llm_response(messages, max_tokens=100)
        score = self._extract_score_from_response(response)
        
        return {
            "score": score,
            "evaluation": response
        }
    
    def _assess_confidence(self, answer: str, state: ResearchState) -> Dict[str, float]:  # ✅ 수정: Dict[str, float] 반환
        """신뢰도 평가"""
        base_confidence = 0.5
        
        # 문서 품질에 따른 신뢰도
        if state.retrieved_docs:
            doc_scores = [doc.get('final_score', 0.5) for doc in state.retrieved_docs]
            avg_doc_score = np.mean(doc_scores)
            base_confidence += avg_doc_score * 0.3
        
        # 연구 깊이에 따른 신뢰도
        research_depth = min(state.cycle_count / state.max_cycles, 1.0)
        base_confidence += research_depth * 0.2
        
        # 인사이트 수에 따른 신뢰도
        insight_bonus = min(len(state.insights) * 0.05, 0.2)
        base_confidence += insight_bonus
        
        return {"score": min(base_confidence, 1.0)}  # ✅ 수정: dict로 반환
    
    def _check_forbidden_phrases(self, answer: str) -> List[str]:
        """금지 표현 검사"""
        found_phrases = []
        for phrase in self.forbidden_phrases:
            if phrase in answer:
                found_phrases.append(phrase)
        return found_phrases
    
    def _calculate_overall_confidence(self, validations: Dict) -> float:
        """종합 신뢰도 계산"""
        weights = {
            "source_grounding": 0.3,
            "factual_consistency": 0.3,
            "completeness": 0.2,
            "confidence_assessment": 0.2
        }
        
        total_score = 0
        for key, weight in weights.items():
            if key in validations:
                value = validations[key]
                # ✅ 수정: 타입 체크 추가
                if isinstance(value, dict):
                    score = value.get('score', 0.5)
                elif isinstance(value, (int, float)):
                    score = float(value)
                else:
                    score = 0.5
                total_score += score * weight
        
        # 금지 표현 페널티
        if validations.get("forbidden_phrases"):
            total_score *= 0.8
            
        return min(total_score, 1.0)
    
    def _generate_warnings(self, validations: Dict) -> List[str]:
        """경고 메시지 생성"""
        warnings = []
        
        if validations["source_grounding"]["score"] < 0.5:
            warnings.append("문서 출처 명시 부족")
            
        if validations["factual_consistency"]["score"] < 0.3:
            warnings.append("문서 내용과 연관성 낮음")
            
        if validations["completeness"]["score"] < 0.6:
            warnings.append("답변 완성도 부족")
            
        if validations["forbidden_phrases"]:
            warnings.append(f"불확실 표현 사용: {', '.join(validations['forbidden_phrases'])}")
            
        return warnings
    
    def _generate_llm_response(self, messages: List[Dict], max_tokens: int = 200) -> str:
        """LLM 응답 생성"""
        try:
            input_ids = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
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
            
            return self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        except:
            return ""
    
    def _extract_score_from_response(self, response: str) -> float:
        """응답에서 점수 추출"""
        scores = re.findall(r'(\d+)', response)
        if scores:
            return float(scores[0]) / 10.0
        return 0.5

# ────────────────────────────────────────────────────────────
# 5. 메인 오케스트레이터 클래스 (Deep Research 스타일)
# ────────────────────────────────────────────────────────────

class DeepResearchOrchestrator:
    """Deep Research 스타일의 다중 에이전트 협업 시스템"""
    
    def __init__(
        self,
        model_name: str = "LGAI-EXAONE/EXAONE-4.0-1.2B",
        embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        reranker_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        min_chunk_length: int = 50,
        max_chunk_length: int = 300,
        sentences_per_chunk: int = 2,
    ):
        st.info("🚀 Deep Research 시스템 초기화 중...")
        
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
        
        # 모델 로딩
        self._load_models()
        
        # 에이전트 초기화
        self._initialize_agents()
        
        # 데이터 저장소
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.chunk_hashes: Dict[str, Dict] = {}
        self.loaded_pdfs: List[str] = []
        
        st.success("✅ Deep Research 시스템 준비 완료!")
    
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
        self.retriever = RetrieverAgent(self.embed_tokenizer, self.embed_model, self.reranker, self.device_config)
        self.analyzer = AnalyzerAgent(self.tokenizer, self.model, self.device_config)
        self.synthesizer = SynthesizerAgent(self.tokenizer, self.model, self.device_config)
        self.validator = ValidatorAgent(self.tokenizer, self.model, self.device_config)
    
    def load_pdf_documents(self, pdf_paths: List[str]) -> None:
        """PDF 문서 로딩"""
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
                        "full_path": pdf_path
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
        """Deep Research 메인 프로세스"""
        
        st.info("🔍 Deep Research 프로세스 시작...")
        
        # 1. 연구 상태 초기화
        complexity = self.planner.analyze_query_complexity(query)
        max_cycles = 2 if complexity < 0.5 else 3  # 복잡도에 따른 사이클 수 조정
        
        state = ResearchState(
            phase=ResearchPhase.PLANNING,
            query=query,
            sub_queries=[],
            retrieved_docs=[],
            confidence_history=[],
            insights=[],
            gaps=[],
            max_cycles=max_cycles
        )
        
        research_log = []  # 연구 과정 로그
        search_queries = []  # ⚠️ 수정사항 2: search_queries 변수 미리 정의
        
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
                research_log.append(f"생성된 검색 쿼리: {len(search_queries)}개")
            else:
                # 후속 사이클에서는 지식 격차 기반으로 추가 쿼리 생성
                gaps = self.planner.identify_knowledge_gaps(state)
                state.gaps = gaps
                if gaps:
                    additional_queries = [SearchQuery(
                        text=gap, priority=0.8, category="격차보완", reason="지식격차해결"
                    ) for gap in gaps[:2]]
                    search_queries.extend(additional_queries)
                    research_log.append(f"지식 격차 보완 쿼리: {len(additional_queries)}개")
            
            # Phase 2: 문서 검색
            st.info(f"🔎 다중 쿼리 검색 수행 중... (쿼리 {len(search_queries)}개)")
            state.phase = ResearchPhase.INITIAL_RETRIEVAL if cycle == 0 else ResearchPhase.DEEP_ANALYSIS
            
            retrieved_docs = self.retriever.multi_query_retrieval(
                search_queries, self.documents, self.embeddings
            )
            
            # 새로운 문서만 추가 (중복 제거)
            existing_ids = {doc.get('chunk_id', '') for doc in state.retrieved_docs}
            new_docs = [doc for doc in retrieved_docs if doc.get('chunk_id', '') not in existing_ids]
            state.retrieved_docs.extend(new_docs)
            
            research_log.append(f"검색된 문서: {len(retrieved_docs)}개 (신규: {len(new_docs)}개)")
            
            # Phase 3: 문서 분석
            if state.retrieved_docs:
                st.info("🔬 문서 심화 분석 중...")
                state.phase = ResearchPhase.CROSS_VALIDATION
                
                analyzed_docs = self.analyzer.analyze_document_relevance(query, state.retrieved_docs[-10:])
                
                # 인사이트 추출
                new_insights = [doc.get('key_insight', '') for doc in analyzed_docs if doc.get('key_insight')]
                state.insights.extend([insight for insight in new_insights if insight not in state.insights])
                
                research_log.append(f"분석 완료: {len(analyzed_docs)}개 문서, 인사이트: {len(new_insights)}개")
            
            # 조기 종료 조건 확인
            current_confidence = self._calculate_interim_confidence(state)
            state.confidence_history.append(current_confidence)
            
            if current_confidence > 0.85 and len(state.retrieved_docs) >= 5:
                research_log.append(f"높은 신뢰도 달성 ({current_confidence:.2f}) - 조기 종료")
                break
            
            if cycle < state.max_cycles - 1:
                time.sleep(0.5)  # UI 응답성을 위한 짧은 대기
        
        # Phase 4: 교차 검증
        st.info("✅ 정보 교차 검증 중...")
        state.phase = ResearchPhase.CROSS_VALIDATION
        analysis_results = self.analyzer.cross_validate_information(state.retrieved_docs[:10])
        research_log.append(f"교차 검증 완료 - 일관성: {analysis_results['consistency']:.2f}")
        
        # Phase 5: 종합 답변 생성
        st.info("📝 종합 답변 생성 중...")
        state.phase = ResearchPhase.SYNTHESIS
        comprehensive_answer = self.synthesizer.synthesize_comprehensive_answer(
            query, state, analysis_results
        )
        research_log.append("종합 답변 생성 완료")
        
        # Phase 6: 최종 검증
        st.info("🔍 최종 답변 검증 중...")
        state.phase = ResearchPhase.FINAL_VALIDATION
        validation_results = self.validator.comprehensive_validation(
            query, comprehensive_answer, state
        )
        research_log.append(f"최종 검증 완료 - 신뢰도: {validation_results['confidence']:.2f}")
        
        # 결과 정리
        sources = []
        for doc in state.retrieved_docs[:15]:  # 상위 15개 문서만
            source_info = {
                "page": doc["page"],
                "paragraph": doc["paragraph"],
                "chunk_id": doc["chunk_id"],
                "source_file": doc["source"],
                "preview": doc["text"][:400] + "..." if len(doc["text"]) > 400 else doc["text"],
                "similarity": float(doc.get("final_score", doc.get("similarity", 0))),
                "chunk_size": len(doc["text"]),
                "search_category": doc.get("search_category", "일반"),
                "key_insight": doc.get("key_insight", "")
            }
            sources.append(source_info)
        
        return {
            "answer": comprehensive_answer,
            "confidence": validation_results["confidence"],
            "warnings": validation_results["warnings"],
            "sources": sources,
            "research_metadata": {
                "cycles_completed": state.cycle_count + 1,
                "total_documents_analyzed": len(state.retrieved_docs),
                "insights_discovered": len(state.insights),
                "knowledge_gaps_identified": len(state.gaps),
                "query_complexity": complexity,
                "confidence_progression": state.confidence_history,
                "research_log": research_log,
                "cross_validation": analysis_results
            }
        }
    
    def _calculate_interim_confidence(self, state: ResearchState) -> float:
        """중간 신뢰도 계산"""
        if not state.retrieved_docs:
            return 0.0
        
        # 문서 점수 기반 신뢰도
        doc_scores = [doc.get('final_score', doc.get('similarity', 0.5)) for doc in state.retrieved_docs]
        avg_doc_score = np.mean(doc_scores) if doc_scores else 0.5
        
        # 문서 다양성 점수
        unique_sources = len(set(doc['source'] for doc in state.retrieved_docs))
        diversity_score = min(unique_sources / 3.0, 1.0)  # 3개 이상 소스면 만점
        
        # 인사이트 점수
        insight_score = min(len(state.insights) / 5.0, 1.0)  # 5개 이상 인사이트면 만점
        
        return (avg_doc_score * 0.5 + diversity_score * 0.3 + insight_score * 0.2)

# ────────────────────────────────────────────────────────────
# 6. Streamlit UI (Deep Research 스타일로 업데이트)
# ────────────────────────────────────────────────────────────

st.write("CUDA available:", torch.cuda.is_available())

st.title("🧠 Deep Research Chatbot by C.H.PARK")
st.markdown("### AI 다중 에이전트 협업 시스템으로 구현한 심층 연구 기능")

# 사이드바 설정
st.sidebar.header("⚙️ 설정")

# 청킹 파라미터
st.sidebar.subheader("청킹 설정")
min_chunk_length = st.sidebar.slider("최소 청크 길이", 30, 500, 50)
max_chunk_length = st.sidebar.slider("최대 청크 길이", 200, 3000, 300)
sentences_per_chunk = st.sidebar.slider("청크당 최대 문장 수", 1, 10, 2)

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
        with st.spinner("🔄 Deep Research 시스템 초기화 중..."):
            try:
                st.session_state.research_bot = DeepResearchOrchestrator(
                    min_chunk_length=min_chunk_length,
                    max_chunk_length=max_chunk_length,
                    sentences_per_chunk=sentences_per_chunk
                )
                
                # PDF 문서 로딩
                st.session_state.research_bot.load_pdf_documents(pdf_paths)
                
                st.sidebar.success("✅ Deep Research 시스템 준비 완료!")
                
                # 시스템 정보 표시
                st.sidebar.info(f"📊 총 {len(st.session_state.research_bot.documents)}개 청크 생성됨")
                st.sidebar.info(f"📁 {len(st.session_state.research_bot.loaded_pdfs)}개 파일 로딩됨")
                
            except Exception as e:
                st.sidebar.error(f"❌ 초기화 실패: {e}")

# 메인 영역
if "research_bot" in st.session_state:
    # 문서 정보 표시
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📁 로딩된 파일", len(st.session_state.research_bot.loaded_pdfs))
    with col2:
        st.metric("📄 생성된 청크", len(st.session_state.research_bot.documents))
    with col3:
        avg_chunk_size = np.mean([len(doc["text"]) for doc in st.session_state.research_bot.documents]) if st.session_state.research_bot.documents else 0
        st.metric("📏 평균 청크 크기", f"{avg_chunk_size:.0f}자")
    with col4:
        st.metric("🤖 활성 에이전트", "5개")
    
    st.divider()
    
    # 질문 입력
    st.subheader("💬 Deep Research 질문")
    query = st.text_input(
        "심층 연구할 주제를 입력하세요:",
        placeholder="예: 첫 번째 문서와 두 번째 문서의 주장을 비교 분석해주세요.",
        key="deep_query_input"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        research_button = st.button("🧠 Deep Research 시작", type="primary")
    with col2:
        if st.button("🗑️ 연구 기록 초기화"):
            if "research_history" in st.session_state:
                del st.session_state.research_history
            st.rerun()
    
    # Deep Research 실행
    if research_button and query:
        with st.spinner("🧠 다중 에이전트가 심층 연구를 수행하고 있습니다..."):
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
        st.subheader("🎯 Deep Research 결과")
        st.write(result["answer"])
        
        # 상세 메트릭 정보
        col1, col2, col3, col4 = st.columns(4)
        metadata = result["research_metadata"]
        
        with col1:
            st.metric("신뢰도", f"{result['confidence']:.3f}")
        with col2:
            st.metric("연구 사이클", f"{metadata['cycles_completed']}/{metadata.get('max_cycles', 3)}")
        with col3:
            st.metric("분석 문서", metadata['total_documents_analyzed'])
        with col4:
            st.metric("소요 시간", f"{elapsed_time:.1f}초")
        
        # 연구 과정 시각화
        if metadata.get('confidence_progression'):
            st.subheader("📈 연구 진행 과정")
            col1, col2 = st.columns(2)
            
            with col1:
                st.line_chart({
                    "신뢰도 변화": metadata['confidence_progression']
                })
            
            with col2:
                research_metrics = {
                    "발견한 인사이트": metadata['insights_discovered'],
                    "식별한 지식 격차": metadata['knowledge_gaps_identified'],
                    "질문 복잡도": f"{metadata['query_complexity']:.2f}",
                }
                for metric, value in research_metrics.items():
                    st.write(f"**{metric}**: {value}")
        
        # 경고 표시
        if result["warnings"]:
            st.warning("⚠️ " + " | ".join(result["warnings"]))
        
        # 교차 검증 결과
        if "cross_validation" in metadata:
            cv = metadata["cross_validation"]
            st.subheader("🔍 교차 검증 결과")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("정보 일관성", f"{cv['consistency']:.2f}")
            with col2:
                st.metric("상충 정보", len(cv.get('conflicts', [])))
            with col3:
                st.metric("공통 정보", len(cv.get('consensus', [])))
        
        # 참조 소스 표시 (카테고리별 분류)
        if result["sources"]:
            st.subheader("📚 분석된 문서 소스")
            
            # 카테고리별 분류
            categories = {}
            for source in result["sources"]:
                category = source.get('search_category', '일반')
                if category not in categories:
                    categories[category] = []
                categories[category].append(source)
            
            for category, sources in categories.items():
                with st.expander(f"📂 {category} 카테고리 ({len(sources)}개 문서)"):
                    for i, source in enumerate(sources, 1):
                        st.write(f"**문서 {i}: {source['source_file']}** (점수: {source['similarity']:.3f})")
                        if source.get('key_insight'):
                            st.write(f"💡 **핵심 인사이트**: {source['key_insight']}")
                        st.write(f"📄 **내용 미리보기**: {source['preview'][:200]}...")
                        st.write("---")
        
        # 연구 로그 표시
        with st.expander("🔍 상세 연구 로그"):
            for log_entry in metadata.get('research_log', []):
                st.write(log_entry)

    # 연구 기록 표시
    if "research_history" in st.session_state and st.session_state.research_history:
        st.divider()
        st.subheader("📜 Deep Research 기록")
        
        for i, research in enumerate(reversed(st.session_state.research_history[-3:]), 1):
            with st.expander(f"연구 {len(st.session_state.research_history) - i + 1}: {research['query'][:50]}..."):
                st.write(f"**질문**: {research['query']}")
                st.write(f"**주요 결과**: {research['result']['answer'][:300]}...")
                st.write(f"**연구 시간**: {research['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"**최종 신뢰도**: {research['result']['confidence']:.3f}")
                
                metadata = research['result']['research_metadata']
                st.write(f"**연구 깊이**: {metadata['cycles_completed']}사이클, {metadata['total_documents_analyzed']}개 문서 분석")

else:
    # 시스템이 초기화되지 않은 경우
    st.info("👆 왼쪽 사이드바에서 PDF 파일을 업로드하고 'Deep Research 시스템 시작' 버튼을 클릭해주세요.")

# 사용 가이드
with st.expander("🎓 Deep Research 사용 가이드"):
    st.markdown("""
    ### 🧠 Deep Research 시스템의 특징
    
    **다중 에이전트 협업**:
    - 📋 **Research Planner**: 질문 분석 및 연구 계획 수립
    - 🔍 **Retriever Agent**: 다중 쿼리 기반 포괄적 검색
    - 🔬 **Analyzer Agent**: 문서 분석 및 교차 검증
    - 📝 **Synthesizer Agent**: 종합적 답변 생성
    - ✅ **Validator Agent**: 최종 품질 검증
    
    **지능형 연구 과정**:
    - 질문 복잡도에 따른 적응형 연구 사이클 (2-3회)
    - 지식 격차 자동 식별 및 보완 검색
    - 실시간 신뢰도 평가 및 조기 종료
    - 다각도 정보 수집 및 교차 검증
    
    **최적화된 질문 유형**:
    - 🔍 **비교 분석**: "A와 B의 차이점을 분석해주세요"
    - 🎯 **심층 탐구**: "X의 원인과 결과를 상세히 설명해주세요"  
    - 🔗 **관계 분석**: "Y와 Z 사이의 연관성을 찾아주세요"
    - 📊 **종합 평가**: "여러 관점에서 W를 평가해주세요"
    """)
    
    st.info("💡 일반 RAG 대비 더 정확하고 포괄적인 답변을 제공하지만, 처리 시간이 더 오래 걸릴 수 있습니다.")
