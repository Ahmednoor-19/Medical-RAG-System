import json
import os
import time
from typing import Any, Dict, List, Optional

import google.auth
from google.auth.transport.requests import AuthorizedSession
from google.cloud import storage
from google.cloud import aiplatform_v1
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# -----------------------------
# Config
# -----------------------------
PROJECT_NUMBER = os.getenv("PROJECT_NUMBER", "")
LOCATION = os.getenv("LOCATION", "europe-west6")
EMBED_DIMENSION = int(os.getenv("EMBED_DIMENSION", "1408"))

# Vector Search deployment
VECTOR_API_ENDPOINT = os.getenv("VECTOR_API_ENDPOINT", "")  
INDEX_ENDPOINT = os.getenv("INDEX_ENDPOINT", "")            
DEPLOYED_INDEX_ID = os.getenv("DEPLOYED_INDEX_ID", "")

# Mapping file in GCS
BUCKET = os.getenv("BUCKET", "slidesenhanced-medslides")
ID_MAP_OBJECT = os.getenv("ID_MAP_OBJECT", "vector_data/pilot/id_map.json")

# Cache controls
ID_MAP_TTL_SECONDS = int(os.getenv("ID_MAP_TTL_SECONDS", "300")) 


# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(title="Medical RAG Retriever", version="1.0")


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(10, ge=1, le=50)
    expand_neighbors: int = Field(0, ge=0, le=5)


class PageHit(BaseModel):
    id: str
    distance: float
    doc_id: Optional[str] = None
    page_number: Optional[int] = None
    image_uri: Optional[str] = None
    text_uri: Optional[str] = None
    source_pdf: Optional[str] = None


class RetrieveResponse(BaseModel):
    query: str
    top_k: int
    expand_neighbors: int
    results: List[PageHit]


# -----------------------------
# Global clients
# -----------------------------
creds = None
authed_session: Optional[AuthorizedSession] = None
storage_client: Optional[storage.Client] = None
match_client: Optional[aiplatform_v1.MatchServiceClient] = None

_id_map_cache: Dict[str, Any] = {}
_id_map_cache_ts: float = 0.0


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


def load_id_map_cached() -> Dict[str, Any]:
    """Load id_map.json from GCS with TTL cache."""
    global _id_map_cache, _id_map_cache_ts

    now = time.time()
    if _id_map_cache and (now - _id_map_cache_ts) < ID_MAP_TTL_SECONDS:
        return _id_map_cache

    blob = storage_client.bucket(BUCKET).blob(ID_MAP_OBJECT)
    text = blob.download_as_text(encoding="utf-8")
    _id_map_cache = json.loads(text)
    _id_map_cache_ts = now
    return _id_map_cache


def get_mm_text_embedding(text: str) -> List[float]:
    """Vertex AI multimodalembedding@001:predict (text only)."""
    url = (
        f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_NUMBER}"
        f"/locations/{LOCATION}/publishers/google/models/multimodalembedding@001:predict"
    )
    payload = {"instances": [{"text": text}], "parameters": {"dimension": EMBED_DIMENSION}}

    resp = authed_session.post(url, json=payload, timeout=120)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Embedding failed: {resp.text}")

    pred0 = resp.json()["predictions"][0]
    if "textEmbedding" not in pred0:
        raise HTTPException(status_code=502, detail=f"Embedding response missing textEmbedding. Keys: {list(pred0.keys())}")
    vec = pred0["textEmbedding"]

    if len(vec) != EMBED_DIMENSION:
        raise HTTPException(status_code=502, detail=f"Embedding dim mismatch: got {len(vec)}, expected {EMBED_DIMENSION}")
    return vec


def find_neighbors(vec: List[float], top_k: int) -> aiplatform_v1.FindNeighborsResponse:
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
    """
    Given top hits, include page neighbors within same doc: page±expand_n.
    Keeps uniqueness by datapoint ID.
    """
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
                "distance": r.get("distance", 0.0),
                **meta,
            }

    # preserve original ordering first, then appended neighbors
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


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest):
    q = req.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="query must not be empty")

    id_map = load_id_map_cached()

    qvec = get_mm_text_embedding(q)
    resp = find_neighbors(qvec, req.top_k)

    if not resp.nearest_neighbors or not resp.nearest_neighbors[0].neighbors:
        return RetrieveResponse(query=q, top_k=req.top_k, expand_neighbors=req.expand_neighbors, results=[])

    raw_hits = []
    for n in resp.nearest_neighbors[0].neighbors:
        dp_id = n.datapoint.datapoint_id
        dist = float(n.distance)
        meta = id_map.get(dp_id, {})
        raw_hits.append({"id": dp_id, "distance": dist, **meta})

    # Optional neighbor-page expansion
    hits = expand_with_neighbors(raw_hits, req.expand_neighbors, id_map)

    # Convert to typed output
    results = [PageHit(**h) for h in hits]
    return RetrieveResponse(query=q, top_k=req.top_k, expand_neighbors=req.expand_neighbors, results=results)
