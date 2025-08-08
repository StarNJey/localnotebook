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

# 웹 크롤링용 라이브러리
import requests
from bs4 import BeautifulSoup

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
# 2. 유틸리티 함수들
# ────────────────────────────────────────────────────────────

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask_expanded, 1) / torch.clamp(mask_exp
