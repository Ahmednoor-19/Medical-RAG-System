# retriever_service.py
"""
FastAPI retriever service (updated for pilot-new + text-embedding-004 + chunked datapoint ids)

Changes vs old version:
- Uses text-embedding-004 (outputDimensionality) instead of multimodalembedding@001
- EMBED_DIMENSION default 768
- ID_MAP_OBJECT default pilot-new mapping path
- Expand-neighbors supports BOTH:
    - non-chunked ids: docid_p0003
    - chunked ids:     docid_p0003_c00, docid_p0003_c01, ...
  by using a cached page->datapoint_ids index derived from id_map.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import google.auth
from google.auth.transport.requests import AuthorizedSession
from google.cloud import storage
from google.cloud import aiplatform_v1
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field


# -----------------------------
# Config
# -----------------------------
PROJECT_NUMBER = os.getenv("PROJECT_NUMBER", "")
LOCATION = os.getenv("LOCATION", "europe-west6")

# UPDATED: match your new vector pipeline
EMBED_DIMENSION = int(os.getenv("EMBED_DIMENSION", "768"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-004")

# Vector Search deployment
VECTOR_API_ENDPOINT = os.getenv("VECTOR_API_ENDPOINT", "")
INDEX_ENDPOINT = os.getenv("INDEX_ENDPOINT", "")
DEPLOYED_INDEX_ID = os.getenv("DEPLOYED_INDEX_ID", "")

# Mapping file in GCS (UPDATED)
BUCKET = os.getenv("BUCKET", "slidesenhanced-medslides")
ID_MAP_OBJECT = os.getenv("ID_MAP_OBJECT", "vector_data/pilot-new/mapping/id_map.json")

# Cache controls
ID_MAP_TTL_SECONDS = int(os.getenv("ID_MAP_TTL_SECONDS", "300"))

# NOTE: This is NOT your "human-friendly citation" layer.
# Retriever is debug/inspection. It returns raw meta from id_map.


# -----------------------------
# Router
# -----------------------------
router = APIRouter(tags=["retriever"])


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(10, ge=1, le=200)
    expand_neighbors: int = Field(0, ge=0, le=5)
    # Optional: if True, avoid returning multiple chunks from the same page
    dedupe_by_page: bool = Field(True)


class PageHit(BaseModel):
    id: str
    distance: float
    doc_id: Optional[str] = None
    page_number: Optional[int] = None

    # Old fields
    image_uri: Optional[str] = None
    text_uri: Optional[str] = None
    source_pdf: Optional[str] = None

    # New metadata fields (if present)
    content_type: Optional[str] = None
    has_images: Optional[bool] = None
    chunk_index: Optional[int] = None
    total_chunks: Optional[int] = None
    chunk_start_char: Optional[int] = None
    chunk_end_char: Optional[int] = None
    text_chars: Optional[int] = None


class RetrieveResponse(BaseModel):
    query: str
    top_k: int
    expand_neighbors: int
    dedupe_by_page: bool
    results: List[PageHit]


# -----------------------------
# Global clients + caches
# -----------------------------
creds = None
authed_session: Optional[AuthorizedSession] = None
storage_client: Optional[storage.Client] = None
match_client: Optional[aiplatform_v1.MatchServiceClient] = None

_id_map_cache: Dict[str, Any] = {}
_id_map_cache_ts: float = 0.0

# NEW: page->datapoint_ids cache to support chunked ids expansion
# key: (doc_id, page_number) -> [datapoint_id1, datapoint_id2, ...]
_page_index_cache: Dict[Tuple[str, int], List[str]] = {}
_page_index_cache_ts: float = 0.0


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
    """
    Called once from app/main.py startup (or lazily at first request).
    """
    global creds, authed_session, storage_client, match_client
    if authed_session and storage_client and match_client:
        return

    _require_env()

    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    authed_session = AuthorizedSession(creds)
    storage_client = storage.Client()
    match_client = aiplatform_v1.MatchServiceClient(client_options={"api_endpoint": VECTOR_API_ENDPOINT})


def load_id_map_cached() -> Dict[str, Any]:
    """Load id_map.json from GCS with TTL cache."""
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


def _build_page_index_cached(id_map: Dict[str, Any]) -> Dict[Tuple[str, int], List[str]]:
    """
    Build (doc_id, page_number) -> [datapoint_ids...] index from id_map.
    Cached with same TTL behavior to avoid re-scanning id_map frequently.
    """
    global _page_index_cache, _page_index_cache_ts

    now = time.time()
    if _page_index_cache and (now - _page_index_cache_ts) < ID_MAP_TTL_SECONDS:
        return _page_index_cache

    page_index: Dict[Tuple[str, int], List[str]] = {}

    for dp_id, meta in (id_map or {}).items():
        try:
            doc_id = meta.get("doc_id")
            page_number = meta.get("page_number")
            if not doc_id or page_number is None:
                continue
            page_number = int(page_number)
            key = (str(doc_id), page_number)
            page_index.setdefault(key, []).append(dp_id)
        except Exception:
            continue

    # For stable ordering (nice for debugging): sort chunked ids
    for key in list(page_index.keys()):
        page_index[key].sort()

    _page_index_cache = page_index
    _page_index_cache_ts = now
    return _page_index_cache


def get_text_embedding(text: str) -> List[float]:
    """
    Vertex AI text-embedding-004:predict
    Payload:
      instances: [{"content": "..."}]
      parameters: {"outputDimensionality": 768}
    Response:
      predictions[0].embeddings.values
    """
    if not authed_session:
        raise HTTPException(status_code=500, detail="Authorized session not initialized. Did startup run?")

    url = (
        f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_NUMBER}"
        f"/locations/{LOCATION}/publishers/google/models/{EMBED_MODEL}:predict"
    )
    payload = {
        "instances": [{"content": text}],
        "parameters": {"outputDimensionality": EMBED_DIMENSION},
    }

    resp = authed_session.post(url, json=payload, timeout=120)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Embedding failed: {resp.text}")

    pred0 = (resp.json().get("predictions") or [{}])[0]
    vec = (pred0.get("embeddings") or {}).get("values")

    if not vec:
        raise HTTPException(
            status_code=502,
            detail=f"Embedding response missing embeddings.values. Keys: {list(pred0.keys())}",
        )

    if len(vec) != EMBED_DIMENSION:
        raise HTTPException(
            status_code=502,
            detail=f"Embedding dim mismatch: got {len(vec)}, expected {EMBED_DIMENSION}",
        )
    return vec


def find_neighbors(vec: List[float], top_k: int) -> aiplatform_v1.FindNeighborsResponse:
    if not match_client:
        raise HTTPException(status_code=500, detail="Match client not initialized. Did startup run?")

    datapoint = aiplatform_v1.IndexDatapoint(feature_vector=vec)
    query = aiplatform_v1.FindNeighborsRequest.Query(datapoint=datapoint, neighbor_count=top_k)

    request = aiplatform_v1.FindNeighborsRequest(
        index_endpoint=INDEX_ENDPOINT,
        deployed_index_id=DEPLOYED_INDEX_ID,
        queries=[query],
        return_full_datapoint=False,
    )
    return match_client.find_neighbors(request)


def _dedupe_by_page(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Keep only the first hit per (doc_id, page_number).
    Helpful when Vector Search returns multiple chunks from same page.
    """
    out: List[Dict[str, Any]] = []
    seen: set[Tuple[str, int]] = set()

    for r in results:
        doc_id = r.get("doc_id")
        page_number = r.get("page_number")
        if not doc_id or page_number is None:
            out.append(r)
            continue
        key = (str(doc_id), int(page_number))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def expand_with_neighbors(
    results: List[Dict[str, Any]],
    expand_n: int,
    id_map: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Given top hits, include page neighbors within same doc: page±expand_n.
    UPDATED: supports chunked ids by expanding to all datapoints for that page.
    """
    if expand_n <= 0:
        return results

    page_index = _build_page_index_cached(id_map)

    expanded: Dict[str, Dict[str, Any]] = {r["id"]: r for r in results}

    for r in results:
        doc_id = r.get("doc_id")
        page = r.get("page_number")
        if not doc_id or page is None:
            continue

        doc_id = str(doc_id)
        page = int(page)

        for delta in range(-expand_n, expand_n + 1):
            if delta == 0:
                continue
            p2 = page + delta
            if p2 <= 0:
                continue

            # Expand to *all* datapoints for that neighbor page (chunked + non-chunked)
            dp_ids = page_index.get((doc_id, p2), [])
            for neighbor_dp_id in dp_ids:
                if neighbor_dp_id in expanded:
                    continue
                meta = id_map.get(neighbor_dp_id)
                if not meta:
                    continue
                expanded[neighbor_dp_id] = {
                    "id": neighbor_dp_id,
                    "distance": r.get("distance", 0.0),
                    **meta,
                }

    # preserve original ordering first, then appended neighbors (stable)
    ordered: List[Dict[str, Any]] = []
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


@router.get("/retriever_healthz")
def retriever_healthz():
    return {
        "ok": True,
        "module": "retriever",
        "embed_model": EMBED_MODEL,
        "embed_dimension": EMBED_DIMENSION,
        "id_map_object": ID_MAP_OBJECT,
    }


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest):
    init_clients()

    q = (req.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="query must not be empty")

    id_map = load_id_map_cached()

    qvec = get_text_embedding(q)
    resp = find_neighbors(qvec, req.top_k)

    if not resp.nearest_neighbors or not resp.nearest_neighbors[0].neighbors:
        return RetrieveResponse(
            query=q,
            top_k=req.top_k,
            expand_neighbors=req.expand_neighbors,
            dedupe_by_page=req.dedupe_by_page,
            results=[],
        )

    raw_hits: List[Dict[str, Any]] = []
    for n in resp.nearest_neighbors[0].neighbors:
        dp_id = n.datapoint.datapoint_id
        dist = float(n.distance) if n.distance is not None else 0.0
        meta = id_map.get(dp_id, {})
        raw_hits.append({"id": dp_id, "distance": dist, **meta})

    # Optional: de-dupe chunk duplicates (recommended for debugging UX)
    hits = _dedupe_by_page(raw_hits) if req.dedupe_by_page else raw_hits

    # Expand page neighbors, chunk-aware
    hits = expand_with_neighbors(hits, req.expand_neighbors, id_map)

    results = [PageHit(**h) for h in hits]
    return RetrieveResponse(
        query=q,
        top_k=req.top_k,
        expand_neighbors=req.expand_neighbors,
        dedupe_by_page=req.dedupe_by_page,
        results=results,
    )
