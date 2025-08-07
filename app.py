from __future__ import annotations

import os
import re
import json
import hashlib
import time
from datetime import datetime
from typing import List, Dict, Any

import streamlit as st
import numpy as np
import torch, gc
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from sentence_transformers.cross_encoder import CrossEncoder
import warnings

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
# 1. Mean Pooling 함수 정의
# ────────────────────────────────────────────────────────────
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)

# ────────────────────────────────────────────────────────────
# 2. 개선된 한국어 문장 분리 클래스
# ────────────────────────────────────────────────────────────
class ImprovedKoreanSentenceChunker:
    def __init__(self, min_chunk_length=50, max_chunk_length=300, sentences_per_chunk=2):
        self.min_chunk_length = min_chunk_length
        self.max_chunk_length = max_chunk_length
        self.sentences_per_chunk = sentences_per_chunk
        
        # Kiwi 초기화 (문장 분리용)
        self.kiwi = None
        if KIWI_AVAILABLE:
            try:
                self.kiwi = Kiwi()
                # st.info("✅ Kiwi 한국어 형태소 분석기가 로드되었습니다.")
            except Exception as e:
                st.warning(f"⚠️ Kiwi 로드 실패: {e}")
        
        self.available_methods = []
        if KIWI_AVAILABLE and self.kiwi:
            self.available_methods.append("Kiwi")
        self.available_methods.append("Regex")
        
        # st.info(f"사용 가능한 문장 분리 방법: {', '.join(self.available_methods)}")

    def chunk_text(self, text: str) -> List[str]:
        if not text.strip():
            return []
        
        # 1단계: 문장 분리
        sentences = self._split_sentences(text)
        if not sentences:
            return []
        
        # 2단계: 문장 후처리 (너무 짧은 문장 병합)
        sentences = self._postprocess_sentences(sentences)
        
        # 3단계: 청킹
        chunks = self._create_chunks(sentences)
        
        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """다양한 방법으로 문장 분리 시도"""
                
        # 방법 2: Kiwi 사용
        if KIWI_AVAILABLE and self.kiwi:
            try:
                kiwi_result = self.kiwi.split_into_sents(text.strip())
                sentences = [sent.text.strip() for sent in kiwi_result if sent.text.strip()]
                if sentences and len(sentences) > 1:
                    return sentences
            except Exception as e:
                st.warning(f"Kiwi 문장 분리 실패: {e}")
        
        # 방법 3: 개선된 정규식 사용 (fallback)
        return self._regex_sentence_split(text.strip())

    def _regex_sentence_split(self, text: str) -> List[str]:
        """개선된 정규식 기반 문장 분리"""
        # 한국어 문장 종결 패턴들
        patterns = [
            r'[.!?]+\s+',  # 기본 문장 부호 + 공백
            r'[다가나니까요래습니다]\s*[.!?]*\s+',  # 한국어 종결어미 + 문장부호
            r'[다가나니까요래습니다]\s+',  # 한국어 종결어미 + 공백
            r'[니다했다습니다였다았다]\s*[.!?]*\s+',  # 추가 종결어미
            r'\n\s*\n',  # 빈 줄
            r'\.\s*\n',  # 마침표 + 줄바꿈
        ]
        
        combined_pattern = '|'.join(f'({p})' for p in patterns)
        sentences = re.split(combined_pattern, text)
        
        # 빈 문자열과 구분자 제거
        result = []
        for s in sentences:
            if s and not re.match(r'^\s*[.!?\n\s]*$', s):
                result.append(s.strip())
        
        return result if result else [text.strip()]

    def _postprocess_sentences(self, sentences: List[str]) -> List[str]:
        """문장 후처리: 너무 짧은 문장 병합"""
        if not sentences:
            return []
        
        processed = []
        current_sentence = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 현재 문장이 너무 짧으면 다음 문장과 병합
            if len(current_sentence) < self.min_chunk_length:
                if current_sentence:
                    current_sentence += " " + sentence
                else:
                    current_sentence = sentence
            else:
                # 현재 문장이 충분히 길면 저장하고 새 문장 시작
                if current_sentence:
                    processed.append(current_sentence)
                current_sentence = sentence
        
        # 마지막 문장 처리
        if current_sentence:
            if processed and len(current_sentence) < self.min_chunk_length:
                processed[-1] += " " + current_sentence
            else:
                processed.append(current_sentence)
        
        return processed

    def _create_chunks(self, sentences: List[str]) -> List[str]:
        """문장들을 청크로 그룹화"""
        if not sentences:
            return []
        
        chunks = []
        current_chunk = ""
        sentence_count = 0
        
        for sentence in sentences:
            # 청크에 문장 추가 가능한지 확인
            test_chunk = current_chunk + (" " + sentence if current_chunk else sentence)
            
            if (len(test_chunk) <= self.max_chunk_length and 
                sentence_count < self.sentences_per_chunk):
                current_chunk = test_chunk
                sentence_count += 1
            else:
                # 현재 청크 완성하고 새 청크 시작
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
                sentence_count = 1
        
        # 마지막 청크 추가
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

# ────────────────────────────────────────────────────────────
# 3. 개선된 PDF 텍스트 추출 클래스
# ────────────────────────────────────────────────────────────
class ImprovedPDFExtractor:
    def __init__(self):
        self.available_methods = []
        if PDFPLUMBER_AVAILABLE:
            self.available_methods.append("pdfplumber")
    
        
        # st.info(f"사용 가능한 PDF 추출 방법: {', '.join(self.available_methods)}")

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """PDF에서 텍스트 추출 (여러 방법 시도)"""
        
        # 방법 1: pdfplumber (가장 정확한 한글 처리)
        if PDFPLUMBER_AVAILABLE:
            try:
                text = self._extract_with_pdfplumber(pdf_path)
                if text.strip():
                    # st.success("✅ pdfplumber로 텍스트 추출 성공")
                    return text
            except Exception as e:
                st.warning(f"pdfplumber 추출 실패: {e}")

    

    def _extract_with_pdfplumber(self, pdf_path: str) -> str:
        """pdfplumber를 사용한 텍스트 추출"""
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text

    
# ────────────────────────────────────────────────────────────
# 4. 디바이스 설정 클래스
# ────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────
# 5. 개선된 메인 챗봇 클래스
# ────────────────────────────────────────────────────────────
class ImprovedPDFNotebookLM:
    _FORBIDDEN_PHRASES = {
        "알려지지 않은", "확실하지 않은", "아마도",
        "추측하건대", "일반적으로", "보통", "대부분의 경우"
    }

    def __init__(
        self,
        model_name: str = "LGAI-EXAONE/EXAONE-4.0-1.2B",
        embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        reranker_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        min_chunk_length: int = 50,
        max_chunk_length: int = 300,
        sentences_per_chunk: int = 2,
    ):
        # st.info("🚀 PDF 분석 중...")
        
        # Device & Components
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_config = DeviceConfig()
        self.pdf_extractor = ImprovedPDFExtractor()
        self.chunker = ImprovedKoreanSentenceChunker(
            min_chunk_length=min_chunk_length,
            max_chunk_length=max_chunk_length,
            sentences_per_chunk=sentences_per_chunk
        )


        # Embedding Model
        st.info("▶ 임베딩 모델 로딩...")
        self.embed_tokenizer = AutoTokenizer.from_pretrained(embed_model)
        self.embed_model = AutoModel.from_pretrained(embed_model).to(self.device_config.device)

        # Cross-Encoder
        # st.info("▶ Cross-Encoder 로딩...")
        self.reranker = CrossEncoder(reranker_name, device=self.device_config.device)

        # Containers
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: np.ndarray | None = None
        self.chunk_hashes: Dict[str, Dict] = {}
        self.loaded_pdfs: List[str] = []
        torch.cuda.empty_cache()
        gc.collect()
        # st.success("✅ PDF 분석 완료!")

                # LLM
        st.info("▶ LLM 로딩...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
      #  self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=self.device_config.config["torch_dtype"], max_memory={0: "14GB"}, trust_remote_code=True).to(self.device)

        map_dev = "auto" if self.device_config.device == "cuda" else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=self.device_config.config["torch_dtype"],
            device_map=map_dev, max_memory={0: "14GB"}, trust_remote_code=True
        ).to(self.device)

    def load_pdf_documents(self, pdf_paths: List[str]) -> None:
        """개선된 PDF 문서 로딩"""
        st.info(f"📚 PDF 문서 처리 중... ({len(pdf_paths)}개)")
        
        self.documents, self.embeddings, self.chunk_hashes, self.loaded_pdfs = [], None, {}, []
        all_docs: List[Dict[str, Any]] = []
        
        progress_bar = st.progress(0)
        
        for i, pdf_path in enumerate(pdf_paths):
            if not os.path.exists(pdf_path):
                st.warning(f"❌ 파일을 찾을 수 없음: {pdf_path}")
                continue
                
            # st.info(f"처리 중: {os.path.basename(pdf_path)}")
            
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
                        "page": 1,  # 실제 페이지 번호는 복잡하므로 임시로 1 사용
                        "paragraph": chunk_idx,
                        "source": os.path.basename(pdf_path),
                        "full_path": pdf_path
                    })
            
            self.loaded_pdfs.append(pdf_path)
            progress_bar.progress((i + 1) / len(pdf_paths))
            
            # st.success(f"✅ {os.path.basename(pdf_path)}: {len(chunks)}개 청크 생성")

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

    def generate_answer(self, query: str) -> Dict[str, Any]:
        top_k = self.device_config.config["top_k"]
        chunks = self._retrieve(query, top_k)
        
        if not chunks:
            return {"answer": "문서에서 관련 정보를 찾을 수 없습니다.", "sources": [], "confidence": 0.0, "warnings": []}

        prompt = self._build_prompt(query, chunks)
        answer = self._llm_infer(prompt, chunks)
        # answer = self._llm_infer(prompt)
        confidence, warns = self._validate(answer, chunks)
        
        if confidence < 0.5:
            answer = self._conservative_reply(query, chunks)
            warns.append("자동 보수적 답변 모드 적용")

        sources = [
            {
                "page": c["page"],
                "paragraph": c["paragraph"],
                "chunk_id": c["chunk_id"],
                "source_file": c["source"],
                "preview": c["text"][:400] + "..." if len(c["text"]) > 400 else c["text"],
                "similarity": float(c["similarity"]),
                "chunk_size": len(c["text"])
            }
            for c in chunks
        ]

        return {"answer": answer, "confidence": confidence, "warnings": warns, "sources": sources}

    def _retrieve(self, query: str, top_k: int) -> List[Dict]:
        enc = self.embed_tokenizer([query], padding=True, truncation=True, return_tensors="pt").to(self.device_config.device)
        with torch.no_grad():
            out = self.embed_model(**enc)
        q_emb = mean_pooling(out, enc["attention_mask"]).cpu().numpy()
        
        sims = cosine_similarity(q_emb, self.embeddings)[0]
        idxs = np.argsort(sims)[::-1][: top_k * 6]
        threshold = self.device_config.config["sim_threshold"]
        
        candidates = []
        for idx in idxs:
            if sims[idx] < threshold:
                continue
            doc = self.documents[idx].copy()
            doc["similarity"] = sims[idx]
            candidates.append(doc)

        if not candidates:
            return []

        texts = [c["text"] for c in candidates]
        scores = self.reranker.predict([(query, t) for t in texts])
        for c, s in zip(candidates, scores):
            c["similarity"] = float(s)
        
        candidates.sort(key=lambda x: x["similarity"], reverse=True)
        return candidates[:top_k]
    
    def _build_prompt(self, query: str, chunks: List[Dict]) -> List[Dict]:
        context = ""
        for i, c in enumerate(chunks, 1):
            context += f"[문서 {i}] 출처: {c['source']}, {c['page']}페이지\n{c['text']}\n\n"
    
        sys = """당신은 오직 제공된 문서 내용만을 기반으로 답변하는 전문 분석 AI입니다.

핵심 규칙:
1. 제공된 문서에 명시된 정보만 사용하여 답변하세요.
2. 문서에 없는 정보는 절대 추가하지 마세요.
3. 불확실하거나 문서에서 찾을 수 없는 내용은 "문서에서 해당 정보를 찾을 수 없습니다"라고 명시하세요.
4. 답변할 때 반드시 참조한 문서의 출처(파일명, 페이지)를 명시하세요.
5. 문서 내용과 모순되는 답변을 하지 마세요.
6. 답변할 때 결론에 도달하게 된 이유를 반드시 구체적으로 설명하세요.

답변 형식:
- 문서에서 찾은 정보를 바탕으로 정확하게 답변
- 각 정보의 출처를 (파일명, 페이지번호)로 명시
- 문서에 없는 내용은 추측하지 않음"""

        messages = [
            {"role": "system", "content": sys},
        {"role": "user", "content": f"다음 문서들을 참조하여 질문에 답해주세요.\n\n=== 참조 문서 ===\n{context}\n=== 질문 ===\n{query}"}
    ]
    
        return messages


    # def _build_prompt(self, query: str, chunks: List[Dict]) -> str:
    #     context = ""
    #     for i, c in enumerate(chunks, 1):
    #         context += f"[청크 {i}] {c['source']}, {c['page']}페이지:\n{c['text']}\n\n"
        
    #     sys = ("""당신은 프롬프트로 주어진 context만 기준으로 삼아 답변하는 판사 AI입니다. 
    #            프롬프트로 주어진 context에 없는 정보는 절대 추가하지 말고 정확도가 높은 context를 찾을 수 없다면 '해당 질문과 관련된 내용을 문서에서 찾을 수 없습니다'라고 먼저 답한 후에 나머지 답변을 하세요. 
    #            프롬프트로 주어진 context에 있는 정보에 기초하여 완성된 문장 형식으로 답변해야 합니다. 답변은 길게 해야 합니다. 
    #            참조한 context의 (파일명, 페이지)를 반드시 명시하여 구체적이고, 명확하게 생각의 과정과 이유를 설명해야 합니다."""
    #     )
    #     return f"<system>\n{sys}\n</system>\n<context>\n{context}</context>\n<user>\n{query}\n</user>\n<assistant>"
    
    def _llm_infer(self, query: str, chunks: List[Dict]) -> str:
        messages = self._build_prompt(query, chunks)
    
    # EXAONE 4.0 chat template 사용
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            # return_dict=True # 이 부분 추가 안 하면 어텐션 없다는 경고 나옴. 이 부분 추가하면 리스트 > 딕셔너리가 되어서 다른 부분도 바꿔야 함
    )
    
        if self.device_config.device == "cuda":
            input_ids = input_ids.to("cuda")
    
    # 더 보수적인 생성 파라미터
        gen_kwargs = {
            "max_new_tokens": self.device_config.config["max_new_tokens"],
            "do_sample": False,  # deterministic 생성
            # "temperature": 0.1,  # 낮은 temperature
            "repetition_penalty": 1.1,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id
    }
    
        out = self.model.generate(input_ids, **gen_kwargs) # return_dict=True면 **input_ids
        return self.tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip() # return_dict=True를 넣지 않을 때 - 어텐션 정보 없다는 경고 나옴
        # input_len = input_ids["input_ids"].shape[1]
        # return self.tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()


    # def _llm_infer(self, prompt: str) -> str:
    #     cfg = self.device_config.config
    #     ids = self.tokenizer(prompt, return_tensors="pt")
    #     if self.device_config.device == "cuda":
    #         ids = {k: v.to("cuda") for k, v in ids.items()}
        
    #     gen_kwargs = {
    #         "max_new_tokens": cfg["max_new_tokens"],
    #         "do_sample": cfg["do_sample"],
    #         "repetition_penalty": 1.05,
    #         "pad_token_id": self.tokenizer.eos_token_id,
    #         "eos_token_id": self.tokenizer.eos_token_id
    #     }
        
    #     if cfg["do_sample"]:
    #         gen_kwargs.update({"temperature": cfg["temperature"], "top_p": cfg["top_p"]})
        
    #     out = self.model.generate(**ids, **gen_kwargs)
    #     return self.tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def _validate(self, answer: str, chunks: List[Dict]) -> tuple[float, List[str]]:
        """Context 준수율을 더 엄격하게 검증"""
        confidence = 0.0
        warnings = []
    
        if not chunks:
            return 0.0, ["참조할 문서가 없습니다"]
    
    # 1. 문서 출처 명시 확인
        source_mentioned = False
        for chunk in chunks:
            if chunk['source'] in answer:
                source_mentioned = True
                break
    
        if not source_mentioned:
            warnings.append("주의! 문서 출처가 보다 구체적으로 명시되지 않았습니다")
            confidence *= 0.7
    
    # 2. Context 외부 정보 사용 탐지
        context_text = " ".join([c['text'] for c in chunks])
        answer_words = set(answer.split())
        context_words = set(context_text.split())
    
        overlap_ratio = len(answer_words & context_words) / max(len(answer_words), 1)
        if overlap_ratio < 0.3:
            warnings.append("주의! 문서 내용과 연관성이 낮아 답변 내용을 믿을 수 없습니다.")
            confidence *= 0.6
    
    # 3. 일반적 지식 사용 탐지
        general_knowledge_indicators = [
        "일반적으로", "보통", "대부분", "알려진 바에 따르면",
        "전문가들은", "연구에 따르면", "통상적으로"
    ]
    
        for indicator in general_knowledge_indicators:
            if indicator in answer:
                warnings.append(f"일반적 지식 사용 탐지: '{indicator}'")
                confidence *= 0.5
    
    # 4. 기본 신뢰도 계산
        base_confidence = np.mean([c['similarity'] for c in chunks])
        confidence += base_confidence
    
        return min(confidence, 1.0), warnings


    # def _validate(self, answer: str, chunks: List[Dict]) -> tuple[float, List[str]]:
    #     confidence = float(np.mean([c["similarity"] for c in chunks])) if chunks else 0.0
    #     warns: List[str] = []
        
    #     for p in self._FORBIDDEN_PHRASES:
    #         if p in answer:
    #             warns.append(f"불확실 표현 발견: '{p}'")
    #             confidence *= 0.8
        
    #     if chunks:
    #         src = " ".join(c["text"] for c in chunks)
    #         overlap = len(set(answer.split()) & set(src.split()))
    #         ratio = overlap / max(1, len(answer.split()))
    #         if ratio < 0.3:
    #             warns.append("출처-답변 단어 중복률 낮음")
    #             confidence *= 0.6
        
    #     return round(confidence, 3), warns

    @staticmethod
    def _conservative_reply(query: str, chunks: List[Dict]) -> str:
        lines = ["문서에서 다음 관련 청크들을 찾았습니다:\n"]
        for i, c in enumerate(chunks, 1):
            lines.append(f"{i}. ({c['source']}, {c['page']}페이지)\n   \"{c['text'][:150]}...\"")
        lines.append("\n더 구체적인 질문을 해주시면 정확한 답변을 드릴 수 있습니다.")
        return "\n".join(lines)

# ────────────────────────────────────────────────────────────
# 6. Streamlit UI
# ────────────────────────────────────────────────────────────

st.write("CUDA available:", torch.cuda.is_available(),
         " | GPU count:", torch.cuda.device_count(),
         " | Current device:", torch.cuda.current_device())
st.set_page_config(page_title="PDF Chatbot", layout="wide")

st.title("🚀 Chatbot 우주 by C.H.PARK")
st.markdown("### 인터넷 연결 없이 LG EXAONE으로 NotebookLM 기능 구현해보기")

# 사이드바 설정
st.sidebar.header("⚙️ 설정")

# 청킹 파라미터
st.sidebar.subheader("청킹 설정")
min_chunk_length = st.sidebar.slider("최소 청크 길이", 30, 500, 50, help="너무 짧은 문장들을 병합하기 위해")
max_chunk_length = st.sidebar.slider("최대 청크 길이", 200, 3000, 300, help="모델 성능이 낮으면 최대 청크 길이를 길게 할 수 없고, 청크 길이와 결과의 정확도가 비례하지 않아 조정 필요")
sentences_per_chunk = st.sidebar.slider("청크당 최대 문장 수", 1, 10, 2, help="PDF마다 최대 성능을 끌어내는 청크당 최대 문장 수가 다르므로")

# PDF 업로드
st.sidebar.subheader("📁 파일 업로드")
uploaded_files = st.sidebar.file_uploader(
    "PDF 파일 업로드", 
    type="pdf", 
    accept_multiple_files=True,
    help="여러 PDF 파일을 동시에 업로드할 수 있습니다"
)

# 시스템 초기화 버튼
if st.sidebar.button("🔄 PDF 분석 시작", type="primary"):
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
        with st.spinner("🔄 PDF 분석 중... 잠시만 기다려주세요."):
            try:
                st.session_state.bot = ImprovedPDFNotebookLM(
                    min_chunk_length=min_chunk_length,
                    max_chunk_length=max_chunk_length,
                    sentences_per_chunk=sentences_per_chunk
                )
                
                # PDF 문서 로딩
                st.session_state.bot.load_pdf_documents(pdf_paths)
                
                st.sidebar.success("✅ PDF 분석 완료!")
                
                # 시스템 정보 표시
                st.sidebar.info(f"📊 총 {len(st.session_state.bot.documents)}개 청크 생성됨")
                st.sidebar.info(f"📁 {len(st.session_state.bot.loaded_pdfs)}개 파일 로딩됨")
                
            except Exception as e:
                st.sidebar.error(f"❌ 초기화 실패: {e}")

# 메인 영역
if "bot" in st.session_state:
    # 문서 정보 표시
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📁 로딩된 파일", len(st.session_state.bot.loaded_pdfs))
    with col2:
        st.metric("📄 생성된 청크", len(st.session_state.bot.documents))
    with col3:
        avg_chunk_size = np.mean([len(doc["text"]) for doc in st.session_state.bot.documents]) if st.session_state.bot.documents else 0
        st.metric("📏 평균 청크 크기", f"{avg_chunk_size:.0f}자")
    
    st.divider()
    
    # 질문 입력
    st.subheader("💬 질문하기")
    query = st.text_input(
        "궁금한 내용을 질문해주세요:",
        placeholder="예: 첫 번째 문서의 주요 내용은 무엇인가요?",
        key="query_input"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        ask_button = st.button("🔍 답변 생성", type="primary")
    with col2:
        if st.button("🗑️ 대화 초기화"):
            if "conversation_history" in st.session_state:
                del st.session_state.conversation_history
            st.rerun()
    
    # 답변 생성
    if ask_button and query:
        with st.spinner("🤔 답변을 생성하고 있습니다..."):
            start_time = time.time()
            result = st.session_state.bot.generate_answer(query)
            elapsed_time = time.time() - start_time
            torch.cuda.empty_cache()
            gc.collect()
        
        # 대화 기록 저장
        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = []
        
        st.session_state.conversation_history.append({
            "query": query,
            "result": result,
            "timestamp": datetime.now(),
            "elapsed_time": elapsed_time
        })
        
        # 결과 표시
        st.subheader("💡 답변")
        st.write(result["answer"])
        
        # 메트릭 정보
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("신뢰도", f"{result['confidence']:.3f}")
        with col2:
            st.metric("응답 시간", f"{elapsed_time:.2f}초")
        with col3:
            st.metric("참조 소스", len(result["sources"]))
        
        # 경고 표시
        if result["warnings"]:
            st.warning("⚠️ " + " | ".join(result["warnings"]))
        
        # 참조 소스 표시
        if result["sources"]:
            st.subheader("📚 참조된 문서 청크")
            for i, source in enumerate(result["sources"], 1):
                with st.expander(f"청크 {i}: {source['source_file']} (유사도: {source['similarity']:.3f})"):
                    st.write(f"**파일:** {source['source_file']}")
                    st.write(f"**페이지:** {source['page']}")
                    st.write(f"**청크 크기:** {source['chunk_size']}자")
                    st.write(f"**내용 미리보기:**")
                    st.write(source["preview"])

    # 대화 기록 표시
    if "conversation_history" in st.session_state and st.session_state.conversation_history:
        st.divider()
        st.subheader("📜 대화 기록")
        
        for i, conversation in enumerate(reversed(st.session_state.conversation_history[-5:]), 1):
            with st.expander(f"질문 {len(st.session_state.conversation_history) - i + 1}: {conversation['query'][:50]}..."):
                st.write(f"**질문:** {conversation['query']}")
                st.write(f"**답변:** {conversation['result']['answer']}")
                st.write(f"**시간:** {conversation['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"**신뢰도:** {conversation['result']['confidence']:.3f}")

else:
    # 시스템이 초기화되지 않은 경우
    st.info("👆 왼쪽 사이드바에서 PDF 파일을 업로드하고 'PDF 분석 시작' 버튼을 클릭해주세요.")
    
# 설치 가이드
with st.expander("📦 질문 가이드"):
    st.code("""
# 문장 형식으로 질문해주세요.
# 명확하고 간단한 문장으로 질문해야 정확한 답변이 나옵니다.
# 상호 관련성이 높은 PDF를 함께 로드해야 답변이 정확해집니다. 
# 모순된 내용의 PDF를 함께 로드하면 참조한 부분에 따라 다른 답변을 할 가능성이 높습니다.  
    """)
    

    st.info("💡관련성이 낮더라도 일단 답변하도록 설정하였습니다. 답변 신뢰도 점수가 낮은 때에는 틀린 답변일 가능성이 높아 확인이 필요합니다.")


