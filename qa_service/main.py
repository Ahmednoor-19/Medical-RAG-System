import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import google.auth
from google.auth.transport.requests import AuthorizedSession
from google.cloud import storage
from google.cloud import aiplatform_v1
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ------------------------------------------------------------
# ENV / CONFIG
# ------------------------------------------------------------
PROJECT_ID = os.getenv("PROJECT_ID", "slidesenhanced")
PROJECT_NUMBER = os.getenv("PROJECT_NUMBER", "69617700548") 
LOCATION = os.getenv("LOCATION", "europe-west6")
GEN_LOCATION = os.getenv("GEN_LOCATION", "global")

# Vector Search deployment
VECTOR_API_ENDPOINT = os.getenv("VECTOR_API_ENDPOINT", "94016133.europe-west6-69617700548.vdb.vertexai.goog")
INDEX_ENDPOINT = os.getenv("INDEX_ENDPOINT", "projects/69617700548/locations/europe-west6/indexEndpoints/1406301760204570624")
DEPLOYED_INDEX_ID = os.getenv("DEPLOYED_INDEX_ID", "embedding_v1_1769680840402")

# Bucket + mapping
BUCKET = os.getenv("BUCKET", "slidesenhanced-medslides")
ID_MAP_OBJECT = os.getenv("ID_MAP_OBJECT", "vector_data/pilot/mapping/id_map.json")

# Caching
ID_MAP_TTL_SECONDS = int(os.getenv("ID_MAP_TTL_SECONDS", "300"))

# Embedding model
EMBED_DIMENSION = int(os.getenv("EMBED_DIMENSION", "1408"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "multimodalembedding@001")  # Vertex model id

GEN_MODEL = os.getenv("GEN_MODEL", "gemini-2.0-flash")

# Context limits
MAX_SOURCES_TO_SEND = int(os.getenv("MAX_SOURCES_TO_SEND", "6"))  # max retrieved pages to feed the model
MAX_TEXT_CHARS_PER_SOURCE = int(os.getenv("MAX_TEXT_CHARS_PER_SOURCE", "2500"))

# Strictness knobs
REQUIRE_CITATIONS = os.getenv("REQUIRE_CITATIONS", "true").lower() == "true"

LANG_DETECT_MODEL = os.getenv("LANG_DETECT_MODEL", GEN_MODEL)
LANG_CACHE_TTL_SECONDS = int(os.getenv("LANG_CACHE_TTL_SECONDS", "3600"))

# ------------------------------------------------------------
# FastAPI
# ------------------------------------------------------------
app = FastAPI(title="Medical RAG QA Service", version="1.0")


class QARequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(8, ge=1, le=50)
    expand_neighbors: int = Field(1, ge=0, le=5)
    max_sources: int = Field(MAX_SOURCES_TO_SEND, ge=1, le=12)


class Citation(BaseModel):
    doc_id: str
    page_number: int
    source_pdf: Optional[str] = None
    image_uri: Optional[str] = None
    text_uri: Optional[str] = None


class QASource(BaseModel):
    id: str
    doc_id: str
    page_number: int
    image_uri: str
    text_uri: str
    source_pdf: Optional[str] = None
    text_excerpt: Optional[str] = None
    score: Optional[float] = None  # from Vector Search


class QAResponse(BaseModel):
    language_hint: Optional[str] = None
    answer: str
    citations: List[Citation]
    sources: List[QASource]


# ------------------------------------------------------------
# Globals (clients + cache)
# ------------------------------------------------------------
creds = None
authed_session: Optional[AuthorizedSession] = None
storage_client: Optional[storage.Client] = None
match_client: Optional[aiplatform_v1.MatchServiceClient] = None

_id_map_cache: Dict[str, Any] = {}
_id_map_cache_ts: float = 0.0
_lang_cache: Dict[str, Tuple[str, float]] = {}  # question -> (lang, ts)

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


def _gcs_read_text(bucket_name: str, obj_name: str) -> str:
    blob = storage_client.bucket(bucket_name).blob(obj_name)
    return blob.download_as_text(encoding="utf-8")


def load_id_map_cached() -> Dict[str, Any]:
    """Load id_map.json from GCS with TTL cache."""
    global _id_map_cache, _id_map_cache_ts

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
    """gs://bucket/path -> (bucket, path)"""
    if not uri.startswith("gs://"):
        raise ValueError(f"Not a GCS URI: {uri}")
    parts = uri.replace("gs://", "").split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Bad GCS URI: {uri}")
    return parts[0], parts[1]


def get_text_embedding(text: str) -> List[float]:
    """Vertex AI multimodalembedding@001 textEmbedding."""
    url = (
        f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_NUMBER}"
        f"/locations/{LOCATION}/publishers/google/models/{EMBED_MODEL}:predict"
    )
    payload = {"instances": [{"text": text}], "parameters": {"dimension": EMBED_DIMENSION}}

    resp = authed_session.post(url, json=payload, timeout=120)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Embedding failed: {resp.text}")

    pred0 = resp.json()["predictions"][0]
    vec = pred0.get("textEmbedding")
    if not vec:
        raise HTTPException(status_code=502, detail=f"Embedding response missing textEmbedding. Keys: {list(pred0.keys())}")

    if len(vec) != EMBED_DIMENSION:
        raise HTTPException(status_code=502, detail=f"Embedding dim mismatch: got {len(vec)}, expected {EMBED_DIMENSION}")
    return vec


def vector_find_neighbors(vec: List[float], top_k: int) -> aiplatform_v1.FindNeighborsResponse:
    datapoint = aiplatform_v1.IndexDatapoint(feature_vector=vec)
    query = aiplatform_v1.FindNeighborsRequest.Query(datapoint=datapoint, neighbor_count=top_k)

    request = aiplatform_v1.FindNeighborsRequest(
        index_endpoint=INDEX_ENDPOINT,
        deployed_index_id=DEPLOYED_INDEX_ID,
        queries=[query],
        return_full_datapoint=False,
    )
    return match_client.find_neighbors(request)


def expand_with_neighbors(results: List[Dict[str, Any]], expand_n: int, id_map: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Add page±expand_n neighbors for each hit (same doc_id)."""
    if expand_n <= 0:
        return results

    expanded: Dict[str, Dict[str, Any]] = {r["id"]: r for r in results}

    for r in results:
        doc_id = r.get("doc_id")
        page = r.get("page_number")
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
            expanded[neighbor_id] = {
                "id": neighbor_id,
                "score": r.get("score"),
                **meta,
            }

    # Keep original order then append added neighbors
    ordered = []
    seen = set()
    for r in results:
        if r["id"] not in seen:
            ordered.append(r)
            seen.add(r["id"])
    for k, v in expanded.items():
        if k not in seen:
            ordered.append(v)
            seen.add(k)
    return ordered


def load_page_text_excerpt(text_uri: str, max_chars: int) -> str:
    """Download extracted page text from GCS and truncate."""
    if not text_uri:
        return ""
    try:
        b, o = parse_gcs_uri(text_uri)
        txt = _gcs_read_text(b, o)
        txt = " ".join(txt.split())
        return txt[:max_chars]
    except Exception:
        return ""

def detect_language_llm(question: str) -> str:
    """
    Detect language using Gemini ONLY.
    Returns: "en" | "fr" | "de"
    Cached to avoid paying repeatedly for the same question.
    """
    global _lang_cache
    q = question.strip()
    now = time.time()

    # cache
    hit = _lang_cache.get(q)
    if hit and (now - hit[1]) < LANG_CACHE_TTL_SECONDS:
        return hit[0]

    url = (
        f"https://aiplatform.googleapis.com/v1/projects/{PROJECT_ID}"
        f"/locations/{GEN_LOCATION}/publishers/google/models/{LANG_DETECT_MODEL}:generateContent"
    )

    # Very strict constrained output instruction
    payload = {
        "systemInstruction": {
            "parts": [
                {"text": (
                    "You detect language. Output ONLY one of: en, fr, de. "
                    "No punctuation, no extra words."
                )}
            ]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": q}]
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 5
        }
    }

    resp = authed_session.post(url, json=payload, timeout=60)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Language detection failed: {resp.text}")

    data = resp.json()
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip().lower()
    except Exception:
        raise HTTPException(status_code=502, detail=f"Unexpected language detection response: {json.dumps(data)[:1200]}")

    # Normalize
    if text not in ("en", "fr", "de"):
        # fallback: default English if model returns weird stuff
        text = "en"

    _lang_cache[q] = (text, now)
    return text

def build_sources_block(sources: List[QASource]) -> str:
    """
    Build a text-only block listing sources with exact citation tags.
    """
    lines = []
    lines.append("SOURCES:")
    for s in sources:
        tag = f"[{s.doc_id}:p{s.page_number:04d}]"
        excerpt = (s.text_excerpt or "").strip() or "(No extracted text available.)"
        lines.append(f"\n{tag}")
        if s.source_pdf:
            lines.append(f"pdf: {s.source_pdf}")
        lines.append(f"text: {excerpt}")
    return "\n".join(lines)

def call_gemini_multimodal(question: str, target_lang: str, sources: List[QASource], image_uris: List[str]) -> str:
    """
    Gemini grounded answer: question + sources text + sources images.
    Uses systemInstruction for strict behavior + citation compliance.
    """

    url = (
        f"https://aiplatform.googleapis.com/v1/projects/{PROJECT_ID}"
        f"/locations/{GEN_LOCATION}/publishers/google/models/{GEN_MODEL}:generateContent"
    )

    system_text = f"""
You are a medical study assistant. You MUST follow these rules:

1) Grounding:
- Use ONLY the provided SOURCES (lecture pages) to answer.
- If the sources do not contain enough info, say so clearly.

2) Citations (MANDATORY):
- Every paragraph MUST contain at least one citation tag.
- Citation format: [doc_id:p####]
- Use ONLY citation tags that appear in SOURCES.
- Do NOT invent citations.

3) Language:
- Answer strictly in this language: {target_lang}
- Do not switch languages.

4) If answer not in sources:
- Output a short refusal in {target_lang} explaining the info is not in the provided pages.
- Still include citations pointing to the sources you checked.
""".strip()

    sources_block = build_sources_block(sources)

    # user parts: images first then text (matches Vertex examples)
    parts: List[Dict[str, Any]] = []

    for uri in image_uris:
        if uri:
            parts.append({"fileData": {"mimeType": "image/png", "fileUri": uri}})

    # Put question + sources in user content
    user_text = f"QUESTION:\n{question.strip()}\n\n{sources_block}"
    parts.append({"text": user_text})

    payload = {
        "systemInstruction": {"parts": [{"text": system_text}]},
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 1024,
        },
    }

    resp = authed_session.post(url, json=payload, timeout=180)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Gemini generateContent failed: {resp.text}")

    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        raise HTTPException(status_code=502, detail=f"Unexpected Gemini response shape: {json.dumps(data)[:1500]}")


def extract_citations_from_answer(answer: str, sources: List[QASource]) -> List[Citation]:
    """
    Extract citation tags like [docid:p0030] from model output.
    Return unique citations.
    """
    # Build allowed set from sources
    allowed = {(s.doc_id, s.page_number) for s in sources}
    found = set()

    pattern = re.compile(r"\[([A-Za-z0-9_\-]+):p(\d{4})\]")
    for m in pattern.finditer(answer):
        doc_id = m.group(1)
        page = int(m.group(2))
        if (doc_id, page) in allowed:
            found.add((doc_id, page))

    # Map back to full citation details
    by_key = {(s.doc_id, s.page_number): s for s in sources}
    out = []
    for (doc_id, page) in sorted(found, key=lambda x: (x[0], x[1])):
        s = by_key[(doc_id, page)]
        out.append(
            Citation(
                doc_id=doc_id,
                page_number=page,
                source_pdf=s.source_pdf,
                image_uri=s.image_uri,
                text_uri=s.text_uri,
            )
        )
    return out


def guess_language_hint(question: str) -> str:
    """
    very lightweight heuristic. The model is instructed to respond
    in the same language anyway; this is just a hint in response.
    """
    q = question.lower()
    # crude hints
    if any(w in q for w in [" der ", " die ", " das ", " und ", " nicht ", " warum "]):
        return "de"
    if any(w in q for w in [" le ", " la ", " les ", " pourquoi ", " comment ", " est-ce "]):
        return "fr"
    return "en"


@app.on_event("startup")
def startup():
    global creds, authed_session, storage_client, match_client
    _require_env()

    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    authed_session = AuthorizedSession(creds)

    storage_client = storage.Client()

    match_client = aiplatform_v1.MatchServiceClient(
        client_options={"api_endpoint": VECTOR_API_ENDPOINT}
    )


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/qa", response_model=QAResponse)
def qa(req: QARequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    id_map = load_id_map_cached()

    # 1) Embed query (text-only)
    qvec = get_text_embedding(question)

    # 2) Vector Search retrieve
    resp = vector_find_neighbors(qvec, req.top_k)

    if not resp.nearest_neighbors or not resp.nearest_neighbors[0].neighbors:
        # No retrieval => cannot answer grounded
        msg = (
            "I can’t find relevant content in the provided lecture PDFs to answer this question."
            if guess_language_hint(question) == "en"
            else "Je ne trouve pas de contenu pertinent dans les cours fournis pour répondre à cette question."
        )
        return QAResponse(language_hint=guess_language_hint(question), answer=msg, citations=[], sources=[])

    # 3) Convert neighbors to meta using id_map
    raw_hits: List[Dict[str, Any]] = []
    for n in resp.nearest_neighbors[0].neighbors:
        dp_id = n.datapoint.datapoint_id
        meta = id_map.get(dp_id, {})
        raw_hits.append(
            {
                "id": dp_id,
                "score": float(n.distance),
                **meta,
            }
        )

    # 4) Expand neighbors pages
    expanded_hits = expand_with_neighbors(raw_hits, req.expand_neighbors, id_map)

    # 5) Build sources list
    max_sources = min(req.max_sources, MAX_SOURCES_TO_SEND)
    expanded_hits = expanded_hits[:max_sources]

    sources: List[QASource] = []
    image_uris: List[str] = []

    for h in expanded_hits:
        # Required keys from id_map
        doc_id = h.get("doc_id")
        page_number = h.get("page_number")
        image_uri = h.get("image_uri")
        text_uri = h.get("text_uri")

        if not (doc_id and isinstance(page_number, int) and image_uri and text_uri):
            # skip malformed entries
            continue

        excerpt = load_page_text_excerpt(text_uri, MAX_TEXT_CHARS_PER_SOURCE)

        sources.append(
            QASource(
                id=h["id"],
                doc_id=doc_id,
                page_number=page_number,
                image_uri=image_uri,
                text_uri=text_uri,
                source_pdf=h.get("source_pdf"),
                text_excerpt=excerpt,
                score=h.get("score"),
            )
        )
        image_uris.append(image_uri)

    if not sources:
        raise HTTPException(status_code=500, detail="Retrieved neighbors but could not build valid sources from id_map")

    # 6) Build strict grounding prompt
    target_lang = detect_language_llm(question)  # "en" | "fr" | "de"

    # 7) Call Gemini with text prompt + images
    answer = call_gemini_multimodal(
        question=question,
        target_lang=target_lang,
        sources=sources,
        image_uris=image_uris
    )

    # 8) Extract citations actually used
    citations = extract_citations_from_answer(answer, sources)

    # 9) Enforce citations if required
    if REQUIRE_CITATIONS and not citations:
        # Soft enforcement: append a note and include sources list
        answer += (
            "\n\nNote: I could not detect citation tags in the model output. "
            "Please ensure citations are included in the format [doc_id:p####]."
        )

    return QAResponse(
        language_hint=target_lang,
        answer=answer,
        citations=citations,
        sources=sources,
    )

