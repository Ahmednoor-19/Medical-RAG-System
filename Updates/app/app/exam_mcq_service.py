# exam_mcq_service.py
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
ID_MAP_OBJECT = os.getenv("ID_MAP_OBJECT", "vector_data/pilot/mapping/id_map.json")
ID_MAP_TTL_SECONDS = int(os.getenv("ID_MAP_TTL_SECONDS", "300"))

EMBED_DIMENSION = int(os.getenv("EMBED_DIMENSION", "1408"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "multimodalembedding@001")

# Use Gemini 3 Pro preview for "full potential" (client feedback)
GEN_MODEL = os.getenv("GEN_MODEL", "gemini-3-pro-preview")

MAX_SOURCES_TO_SEND = int(os.getenv("MAX_SOURCES_TO_SEND", "10"))
MAX_TEXT_CHARS_PER_SOURCE = int(os.getenv("MAX_TEXT_CHARS_PER_SOURCE", "2500"))

# Robust parsing / retry
GEMINI_MAX_PARSE_RETRIES = int(os.getenv("GEMINI_MAX_PARSE_RETRIES", "2"))
GEMINI_RETRY_BACKOFF_S = float(os.getenv("GEMINI_RETRY_BACKOFF_S", "0.6"))
GEMINI_RETRY_JITTER_S = float(os.getenv("GEMINI_RETRY_JITTER_S", "0.25"))

# Generation defaults (client said: "Gemini 3 pro isn’t working with a lower temperature then 1")
GEN_TEMPERATURE = float(os.getenv("GEN_TEMPERATURE", "1.0"))
GEN_TOP_P = float(os.getenv("GEN_TOP_P", "0.95"))
GEN_MAX_OUTPUT_TOKENS = int(os.getenv("GEN_MAX_OUTPUT_TOKENS", "8192"))

# ------------------------------------------------------------
# Reasoning Depth presets (shared concept with /qa)
# ------------------------------------------------------------
REASONING_DEPTH_MAP = {
    "fast": {"top_k": 20, "expand_neighbors": 0, "max_sources": 5},
    "balanced": {"top_k": 40, "expand_neighbors": 1, "max_sources": 10},
    "deep": {"top_k": 50, "expand_neighbors": 2, "max_sources": 12},
}

# ------------------------------------------------------------
# Web allowlist (prompt + server-side filtering)
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

# ------------------------------------------------------------
# Router
# ------------------------------------------------------------
router = APIRouter(tags=["exam-mcq"])


class QAMode(str, Enum):
    LECTURES = "lectures"
    WEB = "web"


class ReasoningDepth(str, Enum):
    FAST = "fast"
    BALANCED = "balanced"
    DEEP = "deep"


class QASource(BaseModel):
    id: str
    doc_id: str
    page_number: int
    image_uri: str
    text_uri: str
    source_pdf: Optional[str] = None
    text_excerpt: Optional[str] = None
    score: Optional[float] = None


class MCQOptionIn(BaseModel):
    label: str = Field(..., min_length=1, max_length=5)
    text: str = Field(..., min_length=1)


class WebSource(BaseModel):
    url: str
    quote: str


class Citation(BaseModel):
    # Lecture-compatible fields (older clients can still show these)
    pdf_name: str
    page_number: int
    source_pdf: Optional[str] = None
    image_uri: Optional[str] = None
    text_uri: Optional[str] = None
    doc_id: Optional[str] = None

    # Web-compatible fields
    url: Optional[str] = None
    kind: Optional[str] = None  # "lecture" | "web"


class ExamMCQRequest(BaseModel):
    stem: str = Field(..., min_length=1)
    options: List[MCQOptionIn] = Field(..., min_length=2)

    mode: QAMode = QAMode.LECTURES
    reasoning_depth: ReasoningDepth = ReasoningDepth.BALANCED

    # Legacy overrides (optional)
    top_k: Optional[int] = Field(None, ge=1, le=50)
    expand_neighbors: Optional[int] = Field(None, ge=0, le=5)
    max_sources: Optional[int] = Field(None, ge=1, le=12)


class ExamMCQResponse(BaseModel):
    language: str
    answer: str

    citations: List[Citation] = []
    sources: List[QASource] = []          # sources sent to model (lectures mode)
    used_sources: List[QASource] = []     # sources actually used in SOURCES
    web_sources: List[WebSource] = []     # web sources actually used (web mode)

    retrieval_config: Optional[Dict[str, Any]] = None


# ------------------------------------------------------------
# Globals (clients + cache)
# ------------------------------------------------------------
creds = None
authed_session: Optional[AuthorizedSession] = None
storage_client: Optional[storage.Client] = None
match_client: Optional[aiplatform_v1.MatchServiceClient] = None

_id_map_cache: Dict[str, Any] = {}
_id_map_cache_ts: float = 0.0
_lang_cache: Dict[str, Tuple[str, float]] = {}
LANG_CACHE_TTL_SECONDS = int(os.getenv("LANG_CACHE_TTL_SECONDS", "3600"))


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


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def parse_gcs_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Not a GCS URI: {uri}")
    parts = uri.replace("gs://", "").split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Bad GCS URI: {uri}")
    return parts[0], parts[1]


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


def load_text_excerpt(text_uri: str, max_chars: int) -> str:
    if not text_uri:
        return ""
    try:
        b, o = parse_gcs_uri(text_uri)
        raw = _gcs_read_text(b, o).strip()
        raw = " ".join(raw.split())
        return (raw[:max_chars] + "...") if len(raw) > max_chars else raw
    except Exception:
        return ""


def detect_language_heuristic(text: str) -> str:
    q = (text or "").strip()
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


def _pdf_filename_from_source_pdf(source_pdf: Optional[str], fallback: str = "") -> str:
    if not source_pdf:
        return fallback
    return unquote(source_pdf.split("/")[-1]) or fallback


def _citation_tag_for_source(s: QASource) -> str:
    pdf_filename = _pdf_filename_from_source_pdf(s.source_pdf, fallback=str(s.doc_id))
    return f"[{pdf_filename}, p.{s.page_number}]"


# ------------------------------------------------------------
# Retrieval
# ------------------------------------------------------------
def get_text_embedding(text: str) -> List[float]:
    if not authed_session:
        raise HTTPException(status_code=500, detail="Authorized session not initialized. Did startup run?")

    url = (
        f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_NUMBER}"
        f"/locations/{LOCATION}/publishers/google/models/{EMBED_MODEL}:predict"
    )
    payload = {"instances": [{"text": text}], "parameters": {"dimension": EMBED_DIMENSION}}
    resp = authed_session.post(url, json=payload, timeout=120)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Embedding failed: {resp.text}")

    pred0 = resp.json().get("predictions", [{}])[0]
    vec = pred0.get("textEmbedding")
    if not vec:
        raise HTTPException(status_code=502, detail=f"Embedding response missing textEmbedding. Keys: {list(pred0.keys())}")
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
    out: List[QASource] = []
    for nb in neighbors:
        dp_id = nb["id"]
        meta = id_map.get(dp_id)
        if not meta:
            continue
        out.append(
            QASource(
                id=dp_id,
                doc_id=str(meta.get("doc_id") or meta.get("pdf_name") or dp_id),
                page_number=int(meta.get("page_number", 0)),
                image_uri=meta.get("image_uri") or "",
                text_uri=meta.get("text_uri") or "",
                source_pdf=meta.get("source_pdf"),
                score=nb.get("score"),
            )
        )
        if len(out) >= max_sources:
            break
    return out


def expand_with_neighbors(sources: List[QASource], expand_n: int, id_map: Dict[str, Any]) -> List[QASource]:
    if expand_n <= 0 or not sources:
        return sources

    expanded: Dict[str, QASource] = {s.id: s for s in sources}

    for s in sources:
        doc_id = s.doc_id
        page = s.page_number
        if not doc_id or page is None:
            continue

        for delta in range(-expand_n, expand_n + 1):
            if delta == 0:
                continue
            p2 = int(page) + delta
            if p2 <= 0:
                continue

            neighbor_id = f"{doc_id}_p{p2:04d}"
            if neighbor_id in expanded:
                continue

            meta = id_map.get(neighbor_id)
            if not meta:
                continue

            expanded[neighbor_id] = QASource(
                id=neighbor_id,
                doc_id=str(meta.get("doc_id") or meta.get("pdf_name") or neighbor_id),
                page_number=int(meta.get("page_number", p2)),
                image_uri=meta.get("image_uri") or "",
                text_uri=meta.get("text_uri") or "",
                source_pdf=meta.get("source_pdf"),
                score=s.score,
            )

    # keep original order, then append others
    ordered: List[QASource] = []
    seen = set()
    for s in sources:
        if s.id not in seen:
            ordered.append(s)
            seen.add(s.id)
    for k, v in expanded.items():
        if k not in seen:
            ordered.append(v)
            seen.add(k)

    return ordered


def resolve_retrieval_config(req: ExamMCQRequest) -> Dict[str, Any]:
    preset = REASONING_DEPTH_MAP[req.reasoning_depth].copy()
    if req.top_k is not None:
        preset["top_k"] = req.top_k
    if req.expand_neighbors is not None:
        preset["expand_neighbors"] = req.expand_neighbors
    if req.max_sources is not None:
        preset["max_sources"] = req.max_sources
    preset["reasoning_depth"] = req.reasoning_depth
    return preset


# ------------------------------------------------------------
# Prompt: "Tutor reasoning" + strict SOURCES section (like /qa)
# ------------------------------------------------------------
def build_sources_block(sources: List[QASource]) -> str:
    blocks: List[str] = []
    for i, s in enumerate(sources, start=1):
        blocks.append(
            f"[Source {i}]\n"
            f"CITATION_TAG: {_citation_tag_for_source(s)}\n"
            f"EXCERPT:\n{s.text_excerpt or ''}\n"
            f"IMAGE_URI: {s.image_uri}\n"
            f"TEXT_URI: {s.text_uri}\n"
        )
    return "INDEXED_SOURCES:\n" + "\n".join(blocks)


def get_system_instruction(target_lang: str, mode: QAMode) -> str:
    allowlist_str = ", ".join(APPROVED_WEB_DOMAINS)

    base = (
        "Role: You are a Senior Medical Tutor and Clinical Examiner.\n"
        "Goal: Provide well-balanced explanations with clinical reasoning (not a rigid fact-checker).\n"
        f"Output Language: STRICTLY {target_lang.upper()}.\n\n"
        "OUTPUT FORMAT (MANDATORY):\n"
        "- Write the answer BODY first.\n"
        "- In the BODY: do NOT include any citations, tags, or URLs.\n"
        "- At the very end, output exactly a final section titled:\n"
        "  SOURCES:\n"
        "- In SOURCES, each line must be either:\n"
        "  Lecture line: [Source N] \"<short verbatim quote>\" <CITATION_TAG>\n"
        "  Web line:    [Web] \"<short verbatim quote>\" <URL>\n"
        "- Do not output any sections after SOURCES.\n\n"
        "HOW TO ANSWER (EXAM STYLE):\n"
        "- Identify the best answer choice(s) and explain WHY.\n"
        "- Briefly explain why the main distractors are wrong or less likely.\n"
        "- If the question is ambiguous, say what extra info would disambiguate.\n"
        "- Use clinically standard abbreviations when appropriate.\n\n"
        "IMAGE EVIDENCE:\n"
        "- Some sources include images (tables/figures) as uploaded image parts.\n"
        "- If relevant, extract evidence from the image content.\n\n"
        "WEB ALLOWLIST (HARD RULE):\n"
        "- If using web sources, you MUST ONLY cite URLs from these allowed domains:\n"
        f"  {allowlist_str}\n"
        "- Ignore all other domains.\n"
        "- If you cannot find allowed-domain evidence, say: 'insufficient evidence from allowed sources'.\n\n"
        "SOURCES RULES:\n"
        "- List ONLY evidence you actually used.\n"
        "- Each SOURCES line must include a short verbatim quote from the evidence.\n"
        "- For lecture sources: copy the CITATION_TAG exactly as provided.\n"
        "- Do NOT repeat the same [Source N] more than once.\n"
    )

    if mode == QAMode.LECTURES:
        return base + (
            "\nMODE: LECTURES ONLY.\n"
            "- Use ONLY INDEXED_SOURCES.\n"
            "- Do NOT use web.\n"
            "- If INDEXED_SOURCES are insufficient, say so explicitly.\n"
        )

    return base + (
        "\nMODE: WEB ONLY.\n"
        "- Use web search.\n"
        "- Cite at least 1 [Web] source from an allowed domain.\n"
    )


# ------------------------------------------------------------
# Web allowlist helpers (server-side enforcement)
# ------------------------------------------------------------
def _domain_is_allowed(url: str) -> bool:
    try:
        host = urlparse(url).netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return any(host == d or host.endswith("." + d) for d in APPROVED_WEB_DOMAINS)
    except Exception:
        return False


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

    # dedupe by url
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
        return "SOURCES:\n[Web] \"(insufficient evidence from allowed sources)\" https://who.int"

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
        # rewrite SOURCES to allowed only
        lines = ["SOURCES:"]
        for ws in web_sources_allowed:
            lines.append(f'[Web] "{ws.quote}" {ws.url}')
        sources_section = "\n".join(lines)

    if web_required and not web_sources_allowed:
        sources_section = "SOURCES:\n[Web] \"(insufficient evidence from allowed sources)\" https://who.int"

    final = f"{body}\n\n{sources_section.strip()}".strip()
    return final, used_lecture_indices, web_sources_allowed


def build_citations(
    used_indices: List[int],
    sources_for_model: List[QASource],
    web_sources: List[WebSource],
) -> List[Citation]:
    out: List[Citation] = []

    for i in used_indices:
        if 1 <= i <= len(sources_for_model):
            s = sources_for_model[i - 1]
            out.append(
                Citation(
                    pdf_name=_pdf_filename_from_source_pdf(s.source_pdf, fallback=str(s.doc_id)),
                    page_number=s.page_number,
                    source_pdf=s.source_pdf,
                    image_uri=s.image_uri,
                    text_uri=s.text_uri,
                    doc_id=s.doc_id,
                    kind="lecture",
                )
            )

    for ws in web_sources:
        out.append(Citation(pdf_name=ws.url, page_number=0, url=ws.url, kind="web"))

    return out


# ------------------------------------------------------------
# Gemini call (supports images + web tool)
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


def call_gemini_exam(
    *,
    stem: str,
    options: List[MCQOptionIn],
    target_lang: str,
    mode: QAMode,
    sources_for_model: List[QASource],
    image_uris: List[str],
    use_web: bool,
) -> str:
    if not authed_session:
        raise HTTPException(status_code=500, detail="Authorized session not initialized.")

    url = (
        f"https://aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{GEN_LOCATION}"
        f"/publishers/google/models/{GEN_MODEL}:generateContent"
    )

    tools = [{"google_search": {}}] if use_web else []
    system_text = get_system_instruction(target_lang=target_lang, mode=mode)

    # Compose MCQ text
    mcq_lines = ["QUESTION:", stem.strip(), "", "OPTIONS:"]
    for o in options:
        mcq_lines.append(f"{o.label}) {o.text}")
    mcq_text = "\n".join(mcq_lines)

    parts: List[Dict[str, Any]] = []
    for uri in image_uris:
        if uri:
            parts.append({"fileData": {"mimeType": "image/png", "fileUri": uri}})

    if sources_for_model:
        sources_text = build_sources_block(sources_for_model)
        user_text = f"{mcq_text}\n\n{sources_text}"
    else:
        user_text = f"{mcq_text}\n\nINDEXED_SOURCES: (none provided)"
    parts.append({"text": user_text})

    payload = {
        "systemInstruction": {"parts": [{"text": system_text}]},
        "contents": [{"role": "user", "parts": parts}],
        "tools": tools,
        "generationConfig": {
            "temperature": GEN_TEMPERATURE,
            "topP": GEN_TOP_P,
            "maxOutputTokens": GEN_MAX_OUTPUT_TOKENS,
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
# Endpoints
# ------------------------------------------------------------
@router.on_event("startup")
def on_startup():
    init_clients()


@router.get("/exam_mcq_healthz")
def exam_mcq_healthz():
    return {
        "ok": True,
        "module": "exam-mcq",
        "modes": ["lectures", "web"],
        "reasoning_depths_available": list(REASONING_DEPTH_MAP.keys()),
    }


@router.post("/exam_mcq", response_model=ExamMCQResponse)
def exam_mcq(req: ExamMCQRequest) -> ExamMCQResponse:
    init_clients()

    stem = (req.stem or "").strip()
    if not stem:
        raise HTTPException(status_code=400, detail="stem must not be empty")

    if not req.options or len(req.options) < 2:
        raise HTTPException(status_code=400, detail="options must contain at least 2 choices")

    retrieval_config = resolve_retrieval_config(req)
    top_k = retrieval_config["top_k"]
    expand_neighbors_n = retrieval_config["expand_neighbors"]
    max_sources = retrieval_config["max_sources"]

    # language (heuristic)
    target_lang = detect_language_heuristic(stem)

    # Retrieval only in LECTURES mode
    retrieved_sources: List[QASource] = []
    image_uris: List[str] = []

    if req.mode == QAMode.LECTURES:
        try:
            opt_text = " ".join([f"{o.label}) {o.text}" for o in req.options])
            retrieval_query = f"STEM: {stem}\nOPTIONS: {opt_text}"

            id_map = load_id_map_cached()
            vec = get_text_embedding(retrieval_query)
            neighbors = find_neighbors(vec, top_k=top_k)
            retrieved_sources = build_sources(neighbors, id_map, max_sources=max_sources)

            # neighbor expansion (pages around hits)
            retrieved_sources = expand_with_neighbors(retrieved_sources, expand_neighbors_n, id_map)
            retrieved_sources = retrieved_sources[:max_sources]

            for s in retrieved_sources:
                s.text_excerpt = load_text_excerpt(s.text_uri, MAX_TEXT_CHARS_PER_SOURCE)
                if s.image_uri:
                    image_uris.append(s.image_uri)

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

        if not retrieved_sources:
            msg = "Insufficient evidence in the provided documents."
            return ExamMCQResponse(
                language=target_lang,
                answer=msg,
                citations=[],
                sources=[],
                used_sources=[],
                web_sources=[],
                retrieval_config=retrieval_config,
            )

    # Web tool usage
    use_web = req.mode == QAMode.WEB

    sources_for_model = retrieved_sources if req.mode == QAMode.LECTURES else []
    images_for_model = image_uris if req.mode == QAMode.LECTURES else []

    raw_answer = call_gemini_exam(
        stem=stem,
        options=req.options,
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

    return ExamMCQResponse(
        language=target_lang,
        answer=final_answer,
        citations=citations,
        sources=sources_for_model,
        used_sources=used_sources,
        web_sources=web_sources_allowed,
        retrieval_config=retrieval_config,
    )
