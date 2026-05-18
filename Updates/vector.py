import json
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple

from google.cloud import storage
import google.auth
from google.auth.transport.requests import AuthorizedSession

# ------------------------
# CONFIG
# ------------------------
PROJECT_ID = os.getenv("PROJECT_ID", "slidesenhanced")
LOCATION = os.getenv("LOCATION", "europe-west6")

BUCKET = os.getenv("BUCKET", "slidesenhanced-medslides")
META_PREFIX = os.getenv("META_PREFIX", "processed-new/metadata/")
CSV_PREFIX = os.getenv("CSV_PREFIX", "processed-new/csv/")

# We'll store vector files here:
OUT_PREFIX = os.getenv("OUT_PREFIX", "vector_data/pilot-new/")
EMBEDDINGS_PREFIX = os.getenv("EMBEDDINGS_PREFIX", f"{OUT_PREFIX}embeddings/")
MAPPING_PREFIX = os.getenv("MAPPING_PREFIX", f"{OUT_PREFIX}mapping/")

# Only index PDFs that came from pilot/
PILOT_PREFIX = os.getenv("PILOT_PREFIX", "pilot-new/")

# Use text-embedding-004 for text-only embeddings
# Supported dimensions: 128, 256, 512, 768, 1024
EMBED_DIMENSION = int(os.getenv("EMBED_DIMENSION", "768"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-004")

# Chunking configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "15000"))  # ~3750 words per chunk (safe for embedding API)
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "500"))  # Character overlap between chunks
MIN_CHUNK_SIZE = int(os.getenv("MIN_CHUNK_SIZE", "1000"))  # Min chars to create a chunk

# Whether to chunk or not
ENABLE_CHUNKING = os.getenv("ENABLE_CHUNKING", "true").lower() == "true"

# ------------------------
# Helpers
# ------------------------
def gcs_uri(bucket: str, obj: str) -> str:
    return f"gs://{bucket}/{obj}"


def list_metadata_jsons(client: storage.Client) -> List[str]:
    bucket = client.bucket(BUCKET)
    blobs = bucket.list_blobs(prefix=META_PREFIX)
    return [b.name for b in blobs if b.name.endswith(".json")]


def load_json_from_gcs(client: storage.Client, object_name: str) -> Dict[str, Any]:
    blob = client.bucket(BUCKET).blob(object_name)
    return json.loads(blob.download_as_bytes().decode("utf-8"))


def load_full_text_from_gcs_uri(client: storage.Client, uri: str) -> str:
    """
    Load FULL extracted page text from gs://...
    
    This now includes:
    - For text pages: Pure PyMuPDF extracted text
    - For image pages: Gemini image description + PyMuPDF text
    
    Returns:
        Full text content (no truncation here - we handle in chunking)
    """
    if not uri or not uri.startswith("gs://"):
        return ""
    
    parts = uri.replace("gs://", "").split("/", 1)
    if len(parts) != 2:
        return ""
    
    bucket_name, obj_name = parts
    if not bucket_name or not obj_name:
        return ""

    blob = client.bucket(bucket_name).blob(obj_name)
    try:
        txt = blob.download_as_text(encoding="utf-8")
    except Exception:
        return ""

    # Normalize whitespace for better embeddings
    txt = " ".join(txt.split())
    
    return txt


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Tuple[str, int, int]]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Full text to chunk
        chunk_size: Target size of each chunk in characters
        overlap: Overlap between chunks in characters
    
    Returns:
        List of (chunk_text, start_char, end_char) tuples
    """
    if len(text) <= chunk_size:
        # Text fits in one chunk
        return [(text, 0, len(text))]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # If this isn't the last chunk, try to break at a sentence boundary
        if end < len(text):
            # Look for sentence endings in the last 20% of the chunk
            search_start = end - int(chunk_size * 0.2)
            search_text = text[search_start:end]
            
            # Find last sentence boundary (., !, ?, or newline)
            last_period = max(
                search_text.rfind('. '),
                search_text.rfind('! '),
                search_text.rfind('? '),
                search_text.rfind('\n')
            )
            
            if last_period != -1:
                # Adjust end to sentence boundary
                end = search_start + last_period + 1
        
        chunk_text = text[start:end].strip()
        
        if len(chunk_text) >= MIN_CHUNK_SIZE:
            chunks.append((chunk_text, start, end))
        
        # Move start forward (with overlap)
        start = end - overlap
        
        # Avoid infinite loop
        if start <= chunks[-1][1] if chunks else 0:
            start = end
    
    return chunks


def get_text_embedding_predict(
    session: AuthorizedSession, 
    project_id: str, 
    location: str, 
    text: str, 
    dimension: int
) -> List[float]:
    """
    Generate text embedding using text-embedding-004.
    
    This is optimized for text-to-text semantic matching.
    Much better than multimodal embeddings for text queries.
    
    Args:
        session: Authenticated session
        project_id: GCP project ID
        location: Region (e.g., 'europe-west6')
        text: Text to embed (max ~20k tokens = ~80k chars)
        dimension: Embedding dimension (128, 256, 512, 768, or 1024)
    
    Returns:
        Embedding vector as list of floats
    """
    # Ensure we don't exceed API limits
    if len(text) > 80000:
        text = text[:80000]
    
    url = (
        f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}"
        f"/locations/{location}/publishers/google/models/{EMBED_MODEL}:predict"
    )
    
    payload = {
        "instances": [{"content": text}],
        "parameters": {"outputDimensionality": dimension}
    }
    
    resp = session.post(url, json=payload, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"Text embedding failed ({resp.status_code}): {resp.text}")
    
    result = resp.json()
    embeddings = result["predictions"][0]["embeddings"]["values"]
    
    if len(embeddings) != dimension:
        raise RuntimeError(f"Expected {dimension} dimensions, got {len(embeddings)}")
    
    return embeddings


def process_page_with_chunking(
    session: AuthorizedSession,
    page_data: Dict[str, Any],
    doc_id: str,
    project_id: str,
    location: str,
    dimension: int,
    storage_client: storage.Client
) -> List[Dict[str, Any]]:
    """
    Process a single page, optionally creating multiple chunks.
    
    Returns:
        List of embedding records (one per chunk, or one for whole page)
    """
    page_number = page_data["page_number"]
    text_uri = page_data["text_uri"]
    
    # Load full text
    page_full_text = load_full_text_from_gcs_uri(storage_client, text_uri)
    
    if not page_full_text.strip():
        return []  # Skip empty pages
    
    records = []
    
    if ENABLE_CHUNKING and len(page_full_text) > CHUNK_SIZE:
        # Create multiple chunks for this page
        chunks = chunk_text(page_full_text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        
        for chunk_idx, (chunk_text, start_char, end_char) in enumerate(chunks):
            # Generate embedding for this chunk
            emb = get_text_embedding_predict(session, project_id, location, chunk_text, dimension)
            
            # Unique chunk ID
            datapoint_id = f"{doc_id}_p{page_number:04d}_c{chunk_idx:02d}"
            
            record = {
                "id": datapoint_id,
                "embedding": emb,
                "restricts": [
                    {"namespace": "doc_id", "allow": [doc_id]},
                    {"namespace": "content_type", "allow": [page_data.get("content_type", "text")]},
                ],
            }
            records.append(record)
            
            # Metadata for this chunk
            yield {
                "datapoint_id": datapoint_id,
                "record": record,
                "metadata": {
                    "doc_id": doc_id,
                    "page_number": page_number,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks),
                    "chunk_start_char": start_char,
                    "chunk_end_char": end_char,
                    "content_type": page_data.get("content_type", "text"),
                    "has_images": page_data.get("has_images", False),
                    "image_uri": page_data.get("image_uri"),
                    "text_uri": text_uri,
                    "text_chars": len(chunk_text),
                    "source_pdf": page_data.get("source_pdf"),
                }
            }
    else:
        # Single embedding for entire page (small content)
        emb = get_text_embedding_predict(session, project_id, location, page_full_text, dimension)
        
        datapoint_id = f"{doc_id}_p{page_number:04d}"
        
        record = {
            "id": datapoint_id,
            "embedding": emb,
            "restricts": [
                {"namespace": "doc_id", "allow": [doc_id]},
                {"namespace": "content_type", "allow": [page_data.get("content_type", "text")]},
            ],
        }
        
        yield {
            "datapoint_id": datapoint_id,
            "record": record,
            "metadata": {
                "doc_id": doc_id,
                "page_number": page_number,
                "chunk_index": 0,
                "total_chunks": 1,
                "content_type": page_data.get("content_type", "text"),
                "has_images": page_data.get("has_images", False),
                "image_uri": page_data.get("image_uri"),
                "text_uri": text_uri,
                "text_chars": len(page_full_text),
                "source_pdf": page_data.get("source_pdf"),
            }
        }


def main():
    print("=" * 60)
    print("VECTOR EMBEDDING GENERATION (WITH CHUNKING)")
    print("=" * 60)
    print(f"Project: {PROJECT_ID}")
    print(f"Location: {LOCATION}")
    print(f"Model: {EMBED_MODEL}")
    print(f"Dimension: {EMBED_DIMENSION}")
    print(f"Chunking: {'ENABLED' if ENABLE_CHUNKING else 'DISABLED'}")
    if ENABLE_CHUNKING:
        print(f"  - Chunk size: {CHUNK_SIZE} chars (~{CHUNK_SIZE // 4} words)")
        print(f"  - Overlap: {CHUNK_OVERLAP} chars")
        print(f"  - Min chunk: {MIN_CHUNK_SIZE} chars")
    print("=" * 60)
    
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    session = AuthorizedSession(creds)
    storage_client = storage.Client()

    meta_files = list_metadata_jsons(storage_client)
    if not meta_files:
        raise RuntimeError(f"No metadata JSON found under gs://{BUCKET}/{META_PREFIX}")

    print(f"\nFound {len(meta_files)} metadata files")

    out_dir = Path("/tmp/vector_out")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "pages.json"
    idmap_path = out_dir / "id_map.json"

    id_map = {}
    total_pages = 0
    total_chunks = 0
    image_pages = 0
    text_pages = 0

    with json_path.open("w", encoding="utf-8") as f:
        for meta_idx, meta_obj in enumerate(meta_files):
            meta = load_json_from_gcs(storage_client, meta_obj)

            # ------------------------
            # Normalize source (supports both old and new metadata formats)
            # ------------------------
            source_field = meta.get("source", "")

            if isinstance(source_field, str):
                # New format: "gs://bucket/pilot-new/file.pdf"
                source_pdf = source_field
                if source_field.startswith("gs://"):
                    source_obj = source_field.replace("gs://", "").split("/", 1)[1]
                else:
                    source_obj = source_field

            elif isinstance(source_field, dict):
                # Old format
                source_pdf = source_field.get("gcs_uri")
                source_obj = source_field.get("object", "")

            else:
                source_pdf = None
                source_obj = ""

            # Only index pilot PDFs
            if not source_obj.startswith(PILOT_PREFIX):
                continue

            doc_id = meta["doc_id"]
            pages = meta.get("pages", [])
            
            print(f"\n[{meta_idx + 1}/{len(meta_files)}] Processing: {source_obj}")
            print(f"  Doc ID: {doc_id}")
            print(f"  Pages: {len(pages)}")

            for page_data in pages:
                page_number = page_data["page_number"]
                content_type = page_data.get("content_type", "text")
                
                # Add source_pdf to page_data for metadata
                page_data["source_pdf"] = source_pdf
                
                try:
                    # Process page (may create multiple chunks)
                    chunks_created = 0
                    for chunk_data in process_page_with_chunking(
                        session=session,
                        page_data=page_data,
                        doc_id=doc_id,
                        project_id=PROJECT_ID,
                        location=LOCATION,
                        dimension=EMBED_DIMENSION,
                        storage_client=storage_client
                    ):
                        # Write embedding record
                        f.write(json.dumps(chunk_data["record"]) + "\n")
                        
                        # Store metadata
                        id_map[chunk_data["datapoint_id"]] = chunk_data["metadata"]
                        
                        chunks_created += 1
                        total_chunks += 1
                    
                    if chunks_created > 0:
                        total_pages += 1
                        if content_type == "image":
                            image_pages += 1
                        else:
                            text_pages += 1
                        
                        if chunks_created > 1:
                            print(f"    Page {page_number}: {chunks_created} chunks ({content_type})")
                    
                except Exception as e:
                    print(f"  ⚠️  Page {page_number}: Error - {e}")
                    continue
                
                if total_chunks % 50 == 0 and total_chunks > 0:
                    print(f"  ✓ Generated {total_chunks} embeddings...")

    # Save ID map locally
    idmap_path.write_text(json.dumps(id_map, ensure_ascii=False, indent=2), encoding="utf-8")
    
    print("\n" + "=" * 60)
    print("EMBEDDING GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total pages processed: {total_pages}")
    print(f"  - Image pages: {image_pages}")
    print(f"  - Text pages: {text_pages}")
    print(f"Total chunks/embeddings: {total_chunks}")
    print(f"Average chunks per page: {total_chunks / total_pages:.1f}" if total_pages > 0 else "")
    print(f"Local JSON: {json_path}")
    print(f"Local ID map: {idmap_path}")
    print("=" * 60)

    # Upload both files to GCS for Vector Search + app usage
    bucket = storage_client.bucket(BUCKET)
    
    print("\nUploading to GCS...")
    bucket.blob(f"{EMBEDDINGS_PREFIX}pages.json").upload_from_filename(
        str(json_path),
        content_type="application/json"
    )
    print(f"✓ {gcs_uri(BUCKET, f'{EMBEDDINGS_PREFIX}pages.json')}")
    
    bucket.blob(f"{MAPPING_PREFIX}id_map.json").upload_from_filename(
        str(idmap_path),
        content_type="application/json"
    )
    print(f"✓ {gcs_uri(BUCKET, f'{MAPPING_PREFIX}id_map.json')}")
    
    print("\n" + "=" * 60)
    print("SUCCESS: Vector embeddings ready for Vector Search index")
    print("=" * 60)


if __name__ == "__main__":
    main()