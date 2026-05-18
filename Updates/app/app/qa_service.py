import json
import os
import random
import re
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote, urlparse

import google.auth
from fastapi import APIRouter, HTTPException
from google.auth.transport.requests import AuthorizedSession
from google.cloud import aiplatform_v1, storage
from pydantic import BaseModel, Field

# ------------------------------------------------------------
# ENV / CONFIG
# ------------------------------------------------------------
PROJECT_ID = os.getenv("PROJECT_ID", "slidesenhanced")
PROJECT_NUMBER = os.getenv("PROJECT_NUMBER", "")
LOCATION = os.getenv("LOCATION", "europe-west6")
GEN_LOCATION = os.getenv("GEN_LOCATION", "global")

VECTOR_API_ENDPOINT = os.getenv("VECTOR_API_ENDPOINT", "")
INDEX_ENDPOINT = os.getenv("INDEX_ENDPOINT", "")
DEPLOYED_INDEX_ID = os.getenv("DEPLOYED_INDEX_ID", "")

BUCKET = os.getenv("BUCKET", "slidesenhanced-medslides")
ID_MAP_OBJECT = os.getenv("ID_MAP_OBJECT", "vector_data/pilot-new/mapping/id_map.json")
ID_MAP_TTL_SECONDS = int(os.getenv("ID_MAP_TTL_SECONDS", "300"))

# UPDATED: Use text-embedding-004 instead of multimodal
EMBED_DIMENSION = int(os.getenv("EMBED_DIMENSION", "768"))  # Changed from 1408
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-004")  # Changed from multimodalembedding@001

# Gemini model
GEN_MODEL = os.getenv("GEN_MODEL", "gemini-2.0-flash-exp")

# UPDATED: Much larger context sizes to handle full text + image descriptions
MAX_SOURCES_TO_SEND = int(os.getenv("MAX_SOURCES_TO_SEND", "30"))  # Up from 10
MAX_TEXT_CHARS_PER_SOURCE = int(os.getenv("MAX_TEXT_CHARS_PER_SOURCE", "100000"))  # Up from 2500

REQUIRE_CITATIONS = os.getenv("REQUIRE_CITATIONS", "true").lower() == "true"

# Robust parsing / retry
GEMINI_MAX_PARSE_RETRIES = int(os.getenv("GEMINI_MAX_PARSE_RETRIES", "2"))
GEMINI_RETRY_BACKOFF_S = float(os.getenv("GEMINI_RETRY_BACKOFF_S", "0.6"))
GEMINI_RETRY_JITTER_S = float(os.getenv("GEMINI_RETRY_JITTER_S", "0.25"))

# Generation defaults
GEN_TEMPERATURE = float(os.getenv("GEN_TEMPERATURE", "0.1"))
GEN_TOP_P = float(os.getenv("GEN_TOP_P", "0.95"))
GEN_MAX_OUTPUT_TOKENS = int(os.getenv("GEN_MAX_OUTPUT_TOKENS", "65536"))

# ------------------------------------------------------------
# UPDATED: Reasoning Depth Configuration with larger values
# ------------------------------------------------------------
REASONING_DEPTH_MAP = {
    "fast": {
        "top_k": 30,  # Up from 20
        "expand_neighbors": 0,
        "max_sources": 10,  # Up from 5
        "description": "Quick lookup with minimal sources"
    },
    "balanced": {
        "top_k": 50,  # Up from 40
        "expand_neighbors": 1,
        "max_sources": 20,  # Up from 10
        "description": "Standard retrieval with context"
    },
    "deep": {
        "top_k": 100,  # Up from 50
        "expand_neighbors": 2,
        "max_sources": 30,  # Up from 12
        "description": "Maximum reasoning with extended sources"
    }
}

# ------------------------------------------------------------
# Web allowlist
# ------------------------------------------------------------
_DEFAULT_APPROVED_WEB_DOMAINS = [
    "flexikon.doccheck.com",
    "next.amboss.com",
    "uptodate.com",
    "msdmanuals.com",
    "medix.ch",
    "smartermedicine.ch",
    "sci-hub.se",
    "medsurf.iml.unibe.ch",
    "rmsetudiants.ch",
    "moodle.unil.ch",
    "doccom.iml.unibe.ch",
    "fevertravel.ch",
    "who.int",
    "epha.health",
    "opimeter.usz.ch",
    "embryotox.de",
    "abreviationsmedicales.ch",
    "altmeyers.org",
    "dermacompass.net",
    "euromelanoma.eu",
    "radiopaedia.org",
    "radiologyassistant.nl",
    "info-radiologie.ch",
    "orthorad.de",
    "earthslab.com",
    "mdcalc.com",
    "pap-pediatrie.fr",
    "escardio.org",
    "litfl.com",
    "orthopediedocteurrenard.blogspot.com",
    "eyewiki.org",
    "richmondeye.com",
    "imaios.com",
    "wwwfbm.unil.ch",
    "app.sop-easy.de",
    "nejm.org",
]

APPROVED_WEB_DOMAINS = [
    d.strip().lower()
    for d in os.getenv("APPROVED_WEB_DOMAINS", ",".join(_DEFAULT_APPROVED_WEB_DOMAINS)).split(",")
    if d.strip()
]

LANG_CACHE_TTL_SECONDS = int(os.getenv("LANG_CACHE_TTL_SECONDS", "3600"))

# ------------------------------------------------------------
# Router
# ------------------------------------------------------------
router = APIRouter(tags=["qa"])


class QAMode(str, Enum):
    LECTURES = "lectures"
    WEB = "web"


class ReasoningDepth(str, Enum):
    FAST = "fast"
    BALANCED = "balanced"
    DEEP = "deep"


class QARequest(BaseModel):
    question: str = Field(..., min_length=1)
    mode: QAMode = QAMode.LECTURES
    reasoning_depth: ReasoningDepth = ReasoningDepth.BALANCED
    
    # Legacy parameters (optional overrides)
    top_k: Optional[int] = Field(None, ge=1, le=100)
    expand_neighbors: Optional[int] = Field(None, ge=0, le=5)
    max_sources: Optional[int] = Field(None, ge=1, le=50)


class Citation(BaseModel):
    pdf_name: str
    page_number: int
    source_pdf: Optional[str] = None
    image_uri: Optional[str] = None
    text_uri: Optional[str] = None
    doc_id: Optional[str] = None
    url: Optional[str] = None
    kind: Optional[str] = None
    # NEW: Chunk information (if using chunked embeddings)
    chunk_index: Optional[int] = None
    total_chunks: Optional[int] = None


class QASource(BaseModel):
    id: str
    doc_id: str
    page_number: int
    image_uri: str
    text_uri: str
    source_pdf: Optional[str] = None
    text_excerpt: Optional[str] = None
    score: Optional[float] = None
    # NEW: Content type and chunk info
    content_type: Optional[str] = None
    has_images: Optional[bool] = None
    chunk_index: Optional[int] = None
    total_chunks: Optional[int] = None


class WebSource(BaseModel):
    url: str
    quote: str


class QAResponse(BaseModel):
    language_hint: Optional[str] = None
    answer: str
    citations: List[Citation]
    sources: List[QASource] = []
    used_sources: List[QASource] = []
    web_sources: List[WebSource] = []
    retrieval_config: Optional[Dict[str, Any]] = None


# ------------------------------------------------------------
# Globals
# ------------------------------------------------------------
creds = None
authed_session: Optional[AuthorizedSession] = None
storage_client: Optional[storage.Client] = None
match_client: Optional[aiplatform_v1.MatchServiceClient] = None

_id_map_cache: Dict[str, Any] = {}
_id_map_cache_ts: float = 0.0
_lang_cache: Dict[str, Tuple[str, float]] = {}


# ------------------------------------------------------------
# Core infra
# ------------------------------------------------------------
def _require_env():
    missing = []
    if not PROJECT_NUMBER:
        missing.append("PROJECT_NUMBER")
    if not VECTOR_API_ENDPOINT:
        missing.append("VECTOR_API_ENDPOINT")
    if not INDEX_ENDPOINT:
        missing.append("INDEX_ENDPOINT")
    if not DEPLOYED_INDEX_ID:
        missing.append("DEPLOYED_INDEX_ID")
    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")


def init_clients():
    global creds, authed_session, storage_client, match_client
    if authed_session and storage_client and match_client:
        return

    _require_env()
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    authed_session = AuthorizedSession(creds)
    storage_client = storage.Client()
    match_client = aiplatform_v1.MatchServiceClient(client_options={"api_endpoint": VECTOR_API_ENDPOINT})


def _gcs_read_text(bucket_name: str, obj_name: str) -> str:
    blob = storage_client.bucket(bucket_name).blob(obj_name)
    return blob.download_as_text(encoding="utf-8")


def load_id_map_cached() -> Dict[str, Any]:
    global _id_map_cache, _id_map_cache_ts
    if not storage_client:
        raise HTTPException(status_code=500, detail="Storage client not initialized. Did startup run?")

    now = time.time()
    if _id_map_cache and (now - _id_map_cache_ts) < ID_MAP_TTL_SECONDS:
        return _id_map_cache

    blob = storage_client.bucket(BUCKET).blob(ID_MAP_OBJECT)
    if not blob.exists():
        raise HTTPException(status_code=500, detail=f"id_map not found: gs://{BUCKET}/{ID_MAP_OBJECT}")

    text = blob.download_as_text(encoding="utf-8")
    _id_map_cache = json.loads(text)
    _id_map_cache_ts = now
    return _id_map_cache


def parse_gcs_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Not a GCS URI: {uri}")
    parts = uri.replace("gs://", "").split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Bad GCS URI: {uri}")
    return parts[0], parts[1]


def pdf_name_from_gcs_uri(uri: str) -> str:
    if not uri:
        return ""
    return unquote(uri.split("/")[-1])


# ------------------------------------------------------------
# UPDATED: Text embedding using text-embedding-004
# ------------------------------------------------------------
def get_text_embedding(text: str) -> List[float]:
    """
    Generate text embedding using text-embedding-004.
    
    This matches the embedding model used in vector.py.
    """
    if not authed_session:
        raise HTTPException(status_code=500, detail="Authorized session not initialized. Did startup run?")

    url = (
        f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_NUMBER}"
        f"/locations/{LOCATION}/publishers/google/models/{EMBED_MODEL}:predict"
    )
    
    # UPDATED: New payload format for text-embedding-004
    payload = {
        "instances": [{"content": text}],
        "parameters": {"outputDimensionality": EMBED_DIMENSION}
    }

    resp = authed_session.post(url, json=payload, timeout=120)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Embedding failed: {resp.text}")

    # UPDATED: New response format for text-embedding-004
    pred0 = resp.json().get("predictions", [{}])[0]
    vec = pred0.get("embeddings", {}).get("values")
    
    if not vec:
        raise HTTPException(
            status_code=502, 
            detail=f"Embedding response missing embeddings.values. Keys: {list(pred0.keys())}"
        )
    
    if len(vec) != EMBED_DIMENSION:
        raise HTTPException(
            status_code=502,
            detail=f"Embedding dim mismatch: got {len(vec)}, expected {EMBED_DIMENSION}"
        )
    
    return vec


def find_neighbors(vec: List[float], top_k: int) -> List[Dict[str, Any]]:
    if not match_client:
        raise HTTPException(status_code=500, detail="MatchService client not initialized. Did startup run?")

    request = aiplatform_v1.FindNeighborsRequest(
        index_endpoint=INDEX_ENDPOINT,
        deployed_index_id=DEPLOYED_INDEX_ID,
        queries=[
            aiplatform_v1.FindNeighborsRequest.Query(
                datapoint=aiplatform_v1.IndexDatapoint(feature_vector=vec),
                neighbor_count=top_k,
            )
        ],
        return_full_datapoint=True,
    )
    resp = match_client.find_neighbors(request=request)

    neighbors: List[Dict[str, Any]] = []
    if resp.nearest_neighbors:
        for nn in resp.nearest_neighbors:
            for nb in nn.neighbors:
                neighbors.append(
                    {
                        "id": nb.datapoint.datapoint_id,
                        "score": float(nb.distance) if nb.distance is not None else None,
                    }
                )
    return neighbors


def build_sources(neighbors: List[Dict[str, Any]], id_map: Dict[str, Any], max_sources: int) -> List[QASource]:
    """
    Build sources from retrieved neighbors.
    
    UPDATED: Now handles chunked embeddings by grouping chunks from same page.
    """
    out: List[QASource] = []
    seen_pages = set()  # Track which pages we've already included
    
    for nb in neighbors:
        dp_id = nb["id"]
        meta = id_map.get(dp_id)
        if not meta:
            continue
        
        # Extract page identifier (works for both chunked and non-chunked)
        doc_id = str(meta.get("doc_id") or dp_id)
        page_number = int(meta.get("page_number", 0))
        page_key = f"{doc_id}_p{page_number}"
        
        # If we've already included this page, skip
        # (This prevents duplicate pages when multiple chunks from same page are retrieved)
        if page_key in seen_pages:
            continue
        
        seen_pages.add(page_key)
        
        # UPDATED: Extract new metadata fields
        out.append(
            QASource(
                id=dp_id,
                doc_id=doc_id,
                page_number=page_number,
                image_uri=meta.get("image_uri") or "",
                text_uri=meta.get("text_uri") or "",
                source_pdf=meta.get("source_pdf"),
                score=nb.get("score"),
                # NEW: Content metadata
                content_type=meta.get("content_type"),
                has_images=meta.get("has_images"),
                chunk_index=meta.get("chunk_index"),
                total_chunks=meta.get("total_chunks"),
            )
        )
        
        if len(out) >= max_sources:
            break
    
    return out


def load_text_excerpt(source: QASource, max_chars: int) -> str:
    """
    Load text excerpt from GCS.
    
    UPDATED: Now loads much larger excerpts (up to 100k chars).
    For chunked sources, loads the full page text (which may contain image descriptions).
    """
    if not source.text_uri:
        return ""
    try:
        bucket, obj = parse_gcs_uri(source.text_uri)
        raw = _gcs_read_text(bucket, obj).strip()
        
        # UPDATED: Return full text up to max_chars (much larger now)
        if len(raw) > max_chars:
            return raw[:max_chars] + "..."
        return raw
    except Exception:
        return ""


# ------------------------------------------------------------
# Language detect
# ------------------------------------------------------------
def detect_language(question: str) -> str:
    q = (question or "").strip()
    if not q:
        return "en"

    now = time.time()
    cached = _lang_cache.get(q)
    if cached and (now - cached[1]) < LANG_CACHE_TTL_SECONDS:
        return cached[0]

    lower = f" {q.lower()} "
    fr_hits = sum(1 for w in [" le ", " la ", " les ", " des ", " une ", " un ", " et ", " ou ", " pas ", " pour "] if w in lower)
    de_hits = sum(1 for w in [" der ", " die ", " das ", " und ", " nicht ", " ein ", " eine ", " für ", " mit "] if w in lower)

    lang = "fr" if fr_hits > de_hits and fr_hits >= 2 else "de" if de_hits > fr_hits and de_hits >= 2 else "en"
    _lang_cache[q] = (lang, now)
    return lang


# ------------------------------------------------------------
# Citation tags
# ------------------------------------------------------------
def _pdf_filename_from_source_pdf(source_pdf: Optional[str], fallback: str = "") -> str:
    if not source_pdf:
        return fallback
    return unquote(source_pdf.split("/")[-1]) or fallback


def _citation_tag_for_source(s: QASource) -> str:
    pdf_filename = _pdf_filename_from_source_pdf(s.source_pdf, fallback=str(s.doc_id))
    return f"[{pdf_filename}, p.{s.page_number}]"


# ------------------------------------------------------------
# Sources block
# ------------------------------------------------------------
def build_sources_block(sources: List[QASource]) -> str:
    """
    Build sources block for Gemini.
    
    UPDATED: Now includes content type info (text vs image pages).
    """
    blocks: List[str] = []
    for i, s in enumerate(sources, start=1):
        content_note = ""
        if s.content_type == "image":
            content_note = " (contains image description)"
        elif s.has_images:
            content_note = " (contains images)"
        
        blocks.append(
            f"[Source {i}]{content_note}\n"
            f"CITATION_TAG: {_citation_tag_for_source(s)}\n"
            f"EXCERPT:\n{s.text_excerpt or ''}\n"
        )
    return "INDEXED_SOURCES:\n" + "\n".join(blocks)


# ------------------------------------------------------------
# Allowlist helpers
# ------------------------------------------------------------
def _domain_is_allowed(url: str) -> bool:
    try:
        host = urlparse(url).netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return any(host == d or host.endswith("." + d) for d in APPROVED_WEB_DOMAINS)
    except Exception:
        return False


# ------------------------------------------------------------
# Enhanced Prompt with Comparison Table Template
# ------------------------------------------------------------
def get_system_instruction(target_lang: str, mode: QAMode) -> str:
    allowlist_str = ", ".join(APPROVED_WEB_DOMAINS)

    # Language-specific table headers
    if target_lang == "fr":
        table_template = """
TABLEAU DE COMPARAISON (si pertinent) :
Utilisez ce format markdown pour comparer des pathologies/traitements :

| Aspect | Condition A | Condition B |
|--------|-------------|-------------|
| Épidémiologie | ... | ... |
| Étiologie | ... | ... |
| Physiopathologie | ... | ... |
| Symptômes/Clinique | ... | ... |
| Diagnostics | **Gold standard** | **Gold standard** |
| Thérapie | ... | ... |
| Complications | ... | ... |
"""
    elif target_lang == "de":
        table_template = """
VERGLEICHSTABELLE (falls relevant):
Verwenden Sie dieses Markdown-Format zum Vergleichen von Pathologien/Therapien:

| Aspekt | Zustand A | Zustand B |
|--------|-----------|-----------|
| Epidemiologie | ... | ... |
| Ätiologie | ... | ... |
| Pathophysiologie | ... | ... |
| Symptome/Klinik | ... | ... |
| Diagnostik | **Goldstandard** | **Goldstandard** |
| Therapie | ... | ... |
| Komplikationen | ... | ... |
"""
    else:  # English
        table_template = """
COMPARISON TABLE (if relevant):
Use this markdown format to compare pathologies/treatments:

| Aspect | Condition A | Condition B |
|--------|-------------|-------------|
| Epidemiology | ... | ... |
| Etiology | ... | ... |
| Pathophysiology | ... | ... |
| Symptoms/Clinic | ... | ... |
| Diagnostics | **Gold standard** | **Gold standard** |
| Therapy | ... | ... |
| Complications | ... | ... |
"""

    base = (
        "Role: You are a Senior Medical Tutor and Clinical Examiner. You are rigorous, evidence-based, and concise.\n"
        f"Output Language: STRICTLY {target_lang.upper()} (match the user's input language).\n\n"
        "OUTPUT FORMAT (MANDATORY):\n"
        "- Write the answer BODY first.\n"
        "- The BODY must contain ZERO citations (no [Source N], no [PDF, p.X], no URLs).\n"
        "- Use clinical abbreviations where appropriate.\n"
        "- Bold key findings, gold standards, and critical values.\n"
        f"{table_template}\n"
        "- At the very end, output exactly a final section titled:\n"
        "  SOURCES:\n"
        "- In SOURCES, each line must be either a lecture source line or a web source line:\n"
        "  Lecture line format:\n"
        "    [Source N] \"<short verbatim quote>\" <CITATION_TAG>\n"
        "  Web line format:\n"
        "    [Web] \"<short verbatim quote>\" <URL>\n"
        "- Do not output any other sections after SOURCES.\n\n"
        "EVIDENCE RULE (STRICT):\n"
        "- Every non-trivial factual claim must be supported by evidence.\n"
        "- Do NOT invent numbers, labels, or definitions that are not present in evidence.\n\n"
        "IMAGE EVIDENCE:\n"
        "- Some sources contain image descriptions generated by AI vision analysis.\n"
        "- These descriptions are marked as '(contains image description)' in the source header.\n"
        "- Use these visual descriptions as evidence when relevant.\n\n"
        "CROSS-LINGUAL CITATION:\n"
        "- Your lecture sources may be in any language (EN/FR/DE).\n"
        "- If a source is in a different language than your response, cite it anyway.\n"
        "- Example: If answering in French but the source is German, cite:\n"
        "    [Source 1] \"Original German text\" [Document.pdf, p.5]\n"
        "  Then explain in French.\n\n"
        "WEB ALLOWLIST (HARD RULE):\n"
        "- If using web sources, you MUST ONLY use URLs from these allowed domains:\n"
        f"  {allowlist_str}\n"
        "- Ignore all other domains and keep searching until you find allowed domains.\n"
        "- If you cannot find allowed-domain evidence, say: 'insufficient evidence from allowed sources'.\n\n"
        "SOURCES SECTION RULES:\n"
        "- SOURCES must list ONLY evidence you actually used.\n"
        "- Each SOURCES line must include a short verbatim quote from the evidence.\n"
        "- For lecture sources: copy the CITATION_TAG exactly as provided.\n"
        "- Do NOT repeat the same [Source N] multiple times.\n"
    )

    if mode == QAMode.LECTURES:
        return base + (
            "\nMODE: LECTURES ONLY.\n"
            "- Use ONLY INDEXED_SOURCES.\n"
            "- Do NOT use web.\n"
            "- If INDEXED_SOURCES are insufficient, clearly state this.\n"
        )

    return base + (
        "\nMODE: WEB ONLY.\n"
        "- Use web search.\n"
        "- Cite at least 1 [Web] source from an allowed domain.\n"
        "- Prefer authoritative medical sources (UpToDate, NEJM, etc.).\n"
    )


# ------------------------------------------------------------
# Gemini response extraction
# ------------------------------------------------------------
def _extract_text_from_generate_content(resp_json: Dict[str, Any]) -> str:
    candidates = resp_json.get("candidates") or []
    if not candidates:
        raise KeyError("No candidates")

    for cand in candidates:
        content = cand.get("content") or {}
        parts = content.get("parts") or []
        chunks: List[str] = []
        for p in parts:
            t = p.get("text")
            if isinstance(t, str) and t.strip():
                chunks.append(t.strip())
        if chunks:
            return "\n".join(chunks).strip()

    raise KeyError("No text parts in candidates")


def call_gemini_orchestrator(
    *,
    question: str,
    target_lang: str,
    mode: QAMode,
    sources_for_model: List[QASource],
    image_uris: List[str],
    use_web: bool,
) -> str:
    url = (
        f"https://aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{GEN_LOCATION}"
        f"/publishers/google/models/{GEN_MODEL}:generateContent"
    )

    tools = [{"google_search": {}}] if use_web else []
    system_text = get_system_instruction(target_lang=target_lang, mode=mode)

    parts: List[Dict[str, Any]] = []
    
    # UPDATED: Image URIs are now for visualization only
    # The actual image descriptions are in the text excerpts
    for uri in image_uris:
        if uri:  # Only add if URI exists
            parts.append({"fileData": {"mimeType": "image/png", "fileUri": uri}})

    if sources_for_model:
        sources_text = build_sources_block(sources_for_model)
        user_content = f"QUESTION: {question}\n\n{sources_text}"
    else:
        user_content = f"QUESTION: {question}\n\nINDEXED_SOURCES: (none provided)"
    parts.append({"text": user_content})

    payload = {
        "systemInstruction": {"parts": [{"text": system_text}]},
        "contents": [{"role": "user", "parts": parts}],
        "tools": tools,
        "generationConfig": {
            "temperature": GEN_TEMPERATURE,
            "maxOutputTokens": GEN_MAX_OUTPUT_TOKENS,
            "topP": GEN_TOP_P,
        },
    }

    attempts = 1 + max(0, GEMINI_MAX_PARSE_RETRIES)
    last_err: Optional[Exception] = None

    for attempt in range(attempts):
        resp = authed_session.post(url, json=payload, timeout=180)
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Gemini API Error: {resp.text}")

        resp_json = resp.json()
        try:
            return _extract_text_from_generate_content(resp_json)
        except Exception as e:
            last_err = e
            if attempt < attempts - 1:
                time.sleep(GEMINI_RETRY_BACKOFF_S * (attempt + 1) + random.random() * GEMINI_RETRY_JITTER_S)
                continue
            raise HTTPException(
                status_code=502,
                detail=(
                    "Unexpected response format from Gemini. "
                    f"parse_error={str(last_err)} keys={list(resp_json.keys())} "
                    f"candidate_count={len(resp_json.get('candidates') or [])}"
                ),
            )


# ------------------------------------------------------------
# Post-processing
# ------------------------------------------------------------
def _split_sources_section(answer: str) -> Tuple[str, str]:
    m = re.search(r"(^|\n)\s*SOURCES\s*:\s*", answer, flags=re.IGNORECASE)
    if not m:
        return answer.strip(), ""
    body = answer[:m.start()].strip()
    sources = answer[m.start():].strip()
    return body, sources


def _strip_inline_citations(body: str) -> str:
    body = re.sub(r"\[\s*Source\s+\d+(?:[^\]]*)\]", "", body, flags=re.IGNORECASE)
    body = re.sub(r"\[[^\[\]]+?\.pdf\s*,\s*p\.\s*\d+\s*\]", "", body, flags=re.IGNORECASE)
    body = re.sub(r"https?://\S+", "", body, flags=re.IGNORECASE)
    body = re.sub(r"[ \t]{2,}", " ", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


def _extract_used_lecture_source_indices(sources_section: str, max_index: int) -> List[int]:
    used: List[int] = []
    for m in re.finditer(r"\[\s*Source\s+(\d+)\s*\]", sources_section or "", flags=re.IGNORECASE):
        try:
            n = int(m.group(1))
            if 1 <= n <= max_index and n not in used:
                used.append(n)
        except Exception:
            pass
    return used


def _extract_web_lines(sources_section: str) -> List[WebSource]:
    out: List[WebSource] = []
    if not sources_section:
        return out

    for m in re.finditer(r'\[\s*Web\s*\]\s*"([^"]+)"\s*(https?://\S+)', sources_section, flags=re.IGNORECASE):
        quote = m.group(1).strip()
        url = m.group(2).strip().rstrip(')"\'.')
        if url and quote:
            out.append(WebSource(url=url, quote=quote))
    
    seen = set()
    deduped: List[WebSource] = []
    for ws in out:
        if ws.url not in seen:
            seen.add(ws.url)
            deduped.append(ws)
    return deduped


def _filter_web_sources_allowlist(web_sources: List[WebSource]) -> List[WebSource]:
    return [ws for ws in web_sources if _domain_is_allowed(ws.url)]


def _build_fallback_sources_section_from_excerpts(sources_for_model: List[QASource]) -> str:
    if not sources_for_model:
        return "SOURCES:\n[Web] \"(missing evidence)\" https://who.int"

    chosen = list(enumerate(sources_for_model[: min(3, len(sources_for_model))], start=1))
    lines = ["SOURCES:"]
    for i, s in chosen:
        quote = (s.text_excerpt or "").replace("\n", " ").strip() or "N/A"
        if len(quote) > 220:
            quote = quote[:220].rstrip() + "…"
        lines.append(f"[Source {i}] \"{quote}\" {_citation_tag_for_source(s)}")
    return "\n".join(lines)


def _postprocess_answer(
    answer: str,
    *,
    mode: QAMode,
    sources_for_model: List[QASource],
    web_required: bool,
) -> Tuple[str, List[int], List[WebSource]]:
    body, sources_section = _split_sources_section(answer)
    body = _strip_inline_citations(body)

    if not sources_section:
        sources_section = _build_fallback_sources_section_from_excerpts(sources_for_model)

    used_lecture_indices = _extract_used_lecture_source_indices(sources_section, max_index=len(sources_for_model))

    web_sources = _extract_web_lines(sources_section)
    web_sources_allowed = _filter_web_sources_allowlist(web_sources)

    if mode == QAMode.WEB:
        lines = ["SOURCES:"]
        for ws in web_sources_allowed:
            lines.append(f'[Web] "{ws.quote}" {ws.url}')
        sources_section = "\n".join(lines)

    if web_required and not web_sources_allowed:
        sources_section = "SOURCES:\n[Web] \"(insufficient evidence from allowed sources)\" https://who.int"

    final = f"{body}\n\n{sources_section.strip()}".strip()
    return final, used_lecture_indices, web_sources_allowed


# ------------------------------------------------------------
# Citations builder
# ------------------------------------------------------------
def build_citations(used_indices: List[int], sources_for_model: List[QASource], web_sources: List[WebSource]) -> List[Citation]:
    out: List[Citation] = []

    for i in used_indices:
        if 1 <= i <= len(sources_for_model):
            s = sources_for_model[i - 1]
            out.append(
                Citation(
                    pdf_name=_pdf_filename_from_source_pdf(s.source_pdf, fallback=pdf_name_from_gcs_uri(s.text_uri)),
                    page_number=s.page_number,
                    source_pdf=s.source_pdf,
                    image_uri=s.image_uri,
                    text_uri=s.text_uri,
                    doc_id=s.doc_id,
                    kind="lecture",
                    chunk_index=s.chunk_index,
                    total_chunks=s.total_chunks,
                )
            )

    for ws in web_sources:
        out.append(
            Citation(
                pdf_name=ws.url,
                page_number=0,
                url=ws.url,
                kind="web",
            )
        )

    return out


# ------------------------------------------------------------
# Retrieval Configuration Resolution
# ------------------------------------------------------------
def resolve_retrieval_config(req: QARequest) -> Dict[str, Any]:
    config = REASONING_DEPTH_MAP[req.reasoning_depth].copy()
    
    if req.top_k is not None:
        config["top_k"] = req.top_k
    if req.expand_neighbors is not None:
        config["expand_neighbors"] = req.expand_neighbors
    if req.max_sources is not None:
        config["max_sources"] = req.max_sources
    
    config["reasoning_depth"] = req.reasoning_depth
    return config


# ------------------------------------------------------------
# Endpoint
# ------------------------------------------------------------
@router.on_event("startup")
def on_startup():
    init_clients()


@router.get("/qa_healthz")
def qa_healthz():
    return {
        "ok": True, 
        "module": "qa",
        "reasoning_depths_available": list(REASONING_DEPTH_MAP.keys()),
        "embed_model": EMBED_MODEL,
        "embed_dimension": EMBED_DIMENSION,
    }


@router.post("/qa", response_model=QAResponse)
def qa_endpoint(req: QARequest) -> QAResponse:
    init_clients()

    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    retrieval_config = resolve_retrieval_config(req)
    top_k = retrieval_config["top_k"]
    expand_neighbors = retrieval_config["expand_neighbors"]
    max_sources = retrieval_config["max_sources"]

    try:
        target_lang = detect_language(question)
    except Exception:
        target_lang = "en"

    retrieved_sources: List[QASource] = []
    image_uris: List[str] = []

    if req.mode == QAMode.LECTURES:
        try:
            vec = get_text_embedding(question)
            neighbors = find_neighbors(vec, top_k=top_k)
            id_map = load_id_map_cached()
            retrieved_sources = build_sources(neighbors, id_map, max_sources=max_sources)
            
            for s in retrieved_sources:
                # Load full text (includes image descriptions for image pages)
                s.text_excerpt = load_text_excerpt(s, MAX_TEXT_CHARS_PER_SOURCE)
                # Add image URIs for visualization (optional)
                if s.image_uri:
                    image_uris.append(s.image_uri)
                    
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

    use_web = req.mode == QAMode.WEB

    sources_for_model = retrieved_sources if req.mode == QAMode.LECTURES else []
    images_for_model = image_uris if req.mode == QAMode.LECTURES else []

    if req.mode == QAMode.LECTURES and not sources_for_model:
        msg = "Insufficient evidence in the provided documents."
        return QAResponse(
            language_hint=target_lang, 
            answer=msg, 
            citations=[], 
            sources=[], 
            used_sources=[], 
            web_sources=[],
            retrieval_config=retrieval_config
        )

    raw_answer = call_gemini_orchestrator(
        question=question,
        target_lang=target_lang,
        mode=req.mode,
        sources_for_model=sources_for_model,
        image_uris=images_for_model,
        use_web=use_web,
    )

    final_answer, used_lecture_indices, web_sources_allowed = _postprocess_answer(
        raw_answer,
        mode=req.mode,
        sources_for_model=sources_for_model,
        web_required=use_web,
    )

    used_sources = [sources_for_model[i - 1] for i in used_lecture_indices if 1 <= i <= len(sources_for_model)]
    citations = build_citations(used_lecture_indices, sources_for_model, web_sources_allowed)

    return QAResponse(
        language_hint=target_lang,
        answer=final_answer,
        citations=citations,
        sources=sources_for_model,
        used_sources=used_sources,
        web_sources=web_sources_allowed,
        retrieval_config=retrieval_config,
    )