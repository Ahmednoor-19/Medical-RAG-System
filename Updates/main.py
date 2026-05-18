import base64
import hashlib
import json
import os
import io
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List, Tuple

import fitz  # PyMuPDF
from PIL import Image
from fastapi import FastAPI, HTTPException, Request
from google.cloud import storage
import google.auth
from google.auth.transport.requests import AuthorizedSession

app = FastAPI()
gcs = storage.Client()

INPUT_PREFIX = os.getenv("INPUT_PREFIX", "pilot-new/")
OUT_PREFIX = os.getenv("OUT_PREFIX", "processed-new/")
OUT_IMAGES_PREFIX = os.getenv("OUT_IMAGES_PREFIX", f"{OUT_PREFIX}page_images/")
OUT_TEXT_PREFIX = os.getenv("OUT_TEXT_PREFIX", f"{OUT_PREFIX}page_text/")
OUT_META_PREFIX = os.getenv("OUT_META_PREFIX", f"{OUT_PREFIX}metadata/")
OUT_CSV_PREFIX = os.getenv("OUT_CSV_PREFIX", f"{OUT_PREFIX}csv/")
DEFAULT_BUCKET = os.getenv("BUCKET_NAME", "slidesenhanced-medslides")
MAX_PAGES = int(os.getenv("MAX_PAGES", "2000"))

PROJECT_ID = os.getenv("PROJECT_ID", "slidesenhanced")
LOCATION = os.getenv("LOCATION", "europe-west6")
GEN_LOCATION = os.getenv("GEN_LOCATION", "global")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-pro-preview")

# Image detection thresholds
IMAGE_SIZE_THRESHOLD = int(os.getenv("IMAGE_SIZE_THRESHOLD", "10000"))  # Min pixels (100x100)
IMAGE_COUNT_THRESHOLD = int(os.getenv("IMAGE_COUNT_THRESHOLD", "1"))  # Min images per page

# Global auth session for Gemini
_auth_session: Optional[AuthorizedSession] = None


def _get_auth_session() -> AuthorizedSession:
    """Get or create authenticated session for Gemini API"""
    global _auth_session
    if _auth_session is None:
        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        _auth_session = AuthorizedSession(creds)
    return _auth_session


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_filename(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".", "/") else "_" for ch in name)


def _doc_id(bucket: str, object_name: str) -> str:
    h = hashlib.sha1(f"{bucket}/{object_name}".encode("utf-8")).hexdigest()
    return h


def _download_gcs_to_tmp(bucket: str, object_name: str) -> str:
    if not object_name.lower().endswith(".pdf"):
        raise ValueError("Not a PDF")
    b = gcs.bucket(bucket)
    blob = b.blob(object_name)
    if not blob.exists():
        raise FileNotFoundError(f"GCS object not found: gs://{bucket}/{object_name}")

    local_path = f"/tmp/{hashlib.md5(object_name.encode()).hexdigest()}.pdf"
    blob.download_to_filename(local_path)
    return local_path


def _upload_bytes(bucket: str, object_name: str, data: bytes, content_type: str) -> str:
    b = gcs.bucket(bucket)
    blob = b.blob(object_name)
    blob.upload_from_string(data, content_type=content_type)
    return f"gs://{bucket}/{object_name}"


def _upload_text(bucket: str, object_name: str, text: str) -> str:
    return _upload_bytes(bucket, object_name, text.encode("utf-8"), "text/plain; charset=utf-8")


def _upload_json(bucket: str, object_name: str, obj: Dict[str, Any]) -> str:
    return _upload_bytes(bucket, object_name, json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8"), "application/json")


def detect_images_on_page(page: fitz.Page) -> Tuple[bool, int, List[Dict[str, Any]]]:
    """
    Detect if a page contains significant images.
    
    Returns:
        (has_images, image_count, image_list)
        - has_images: True if page has images above threshold
        - image_count: Number of images found
        - image_list: List of dicts with image metadata
    """
    image_list = page.get_images(full=True)
    
    if not image_list:
        return False, 0, []
    
    significant_images = []
    
    for img_index, img_info in enumerate(image_list):
        xref = img_info[0]
        
        try:
            # Get image metadata
            base_image = page.parent.extract_image(xref)
            width = base_image.get("width", 0)
            height = base_image.get("height", 0)
            size = width * height
            
            # Filter out small images (icons, logos, bullets)
            if size >= IMAGE_SIZE_THRESHOLD:
                significant_images.append({
                    "xref": xref,
                    "width": width,
                    "height": height,
                    "size": size,
                    "ext": base_image.get("ext", "png"),
                })
        except Exception:
            # Skip problematic images
            continue
    
    has_images = len(significant_images) >= IMAGE_COUNT_THRESHOLD
    return has_images, len(significant_images), significant_images


def extract_page_image_as_png(page: fitz.Page, zoom: float = 2.0) -> bytes:
    """
    Render page as PNG image.
    
    Args:
        page: PyMuPDF page object
        zoom: Scale factor (2.0 = 2x resolution)
    
    Returns:
        PNG bytes
    """
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("png")


def describe_image_with_gemini(image_bytes: bytes, page_text: str = "") -> str:
    """
    Send image to Gemini for description.
    
    Args:
        image_bytes: PNG image bytes
        page_text: Optional OCR text found on the page (for context)
    
    Returns:
        Gemini's description of the image
    """
    session = _get_auth_session()
    
    url = (
        f"https://aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{GEN_LOCATION}"
        f"/publishers/google/models/{GEMINI_MODEL}:generateContent"
    )
    
    # Convert image to base64
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")
    
    # Build prompt
    if page_text.strip():
        prompt = f"""Describe this medical lecture slide/image in detail. Focus on:
- Diagrams, charts, tables, or anatomical illustrations
- Key medical concepts shown visually
- Any text visible in the image that's part of diagrams/labels

Additional text context found on this page:
{page_text[:500]}

Provide a comprehensive description suitable for medical students."""
    else:
        prompt = """Describe this medical lecture slide/image in detail. Focus on:
- Diagrams, charts, tables, or anatomical illustrations  
- Key medical concepts shown visually
- Any text visible in the image that's part of diagrams/labels
- Visual content that cannot be captured by text extraction alone

Provide a comprehensive description suitable for medical students."""
    
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": img_b64
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 2048,
        }
    }
    
    try:
        resp = session.post(url, json=payload, timeout=120)
        if resp.status_code != 200:
            return f"[ERROR: Gemini API failed: {resp.status_code}]"
        
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return "[ERROR: No response from Gemini]"
        
        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        if not parts:
            return "[ERROR: Empty response from Gemini]"
        
        description = parts[0].get("text", "").strip()
        return description if description else "[ERROR: Empty description]"
        
    except Exception as e:
        return f"[ERROR: {str(e)}]"


def process_page_with_detection(
    page: fitz.Page, 
    page_no: int, 
    bucket: str, 
    doc_id: str,
    images_base: str,
    text_base: str
) -> Dict[str, Any]:
    """
    Process a single page with image detection logic.
    
    Returns page metadata with content information.
    """
    # Step 1: Detect images on page
    has_images, img_count, img_metadata = detect_images_on_page(page)
    
    # Step 2: Extract text (always - needed for context)
    try:
        page_text = page.get_text("text") or ""
    except Exception as e:
        page_text = ""
    
    page_data = {
        "page_number": page_no,
        "has_images": has_images,
        "image_count": img_count,
        "text_chars": len(page_text),
        "content_type": None,  # Will be set below
        "image_uri": None,
        "text_uri": None,
        "description": None,
    }
    
    # Step 3: Branch based on image detection
    if has_images:
        # IMAGE PATH: Render page as image + get Gemini description
        page_data["content_type"] = "image"
        
        # Render and save page image
        img_bytes = extract_page_image_as_png(page, zoom=2.0)
        img_obj = f"{images_base}page_{page_no:04d}.png"
        img_uri = _upload_bytes(bucket, img_obj, img_bytes, "image/png")
        page_data["image_uri"] = img_uri
        
        # Get Gemini description
        description = describe_image_with_gemini(img_bytes, page_text)
        page_data["description"] = description
        
        # Save combined content (description + any text)
        combined_content = f"[IMAGE DESCRIPTION]\n{description}\n\n[PAGE TEXT]\n{page_text}"
        text_obj = f"{text_base}page_{page_no:04d}.txt"
        text_uri = _upload_text(bucket, text_obj, combined_content)
        page_data["text_uri"] = text_uri
        
    else:
        # TEXT PATH: Just extract and save text
        page_data["content_type"] = "text"
        
        # Save extracted text
        text_obj = f"{text_base}page_{page_no:04d}.txt"
        text_uri = _upload_text(bucket, text_obj, page_text)
        page_data["text_uri"] = text_uri
        page_data["description"] = None
    
    return page_data


def create_csv_record(doc_id: str, page_data: Dict[str, Any], source_info: Dict[str, Any]) -> Dict[str, str]:
    """
    Create a CSV-compatible record for the page.
    
    Returns a flat dictionary suitable for CSV export.
    """
    return {
        "doc_id": doc_id,
        "source_pdf": source_info.get("object", ""),
        "page_number": str(page_data["page_number"]),
        "content_type": page_data["content_type"],
        "has_images": str(page_data["has_images"]),
        "image_count": str(page_data["image_count"]),
        "text_chars": str(page_data["text_chars"]),
        "image_uri": page_data.get("image_uri", ""),
        "text_uri": page_data.get("text_uri", ""),
        "description_preview": (page_data.get("description", "") or "")[:200],  # First 200 chars
    }


def save_csv_index(bucket: str, csv_obj: str, csv_records: List[Dict[str, str]]):
    """
    Save CSV index file to GCS.
    """
    if not csv_records:
        return
    
    # Create CSV content
    import csv
    import io
    
    output = io.StringIO()
    
    # Get headers from first record
    headers = list(csv_records[0].keys())
    writer = csv.DictWriter(output, fieldnames=headers)
    
    writer.writeheader()
    writer.writerows(csv_records)
    
    csv_content = output.getvalue()
    output.close()
    
    # Upload to GCS
    _upload_bytes(bucket, csv_obj, csv_content.encode("utf-8"), "text/csv")


def process_pdf(bucket: str, object_name: str) -> Dict[str, Any]:
    if not bucket:
        raise ValueError("Missing bucket")
    if not object_name:
        raise ValueError("Missing object name")

    if not object_name.lower().endswith(".pdf"):
        # Ignore non-PDF objects cleanly
        return {"skipped": True, "reason": "not_pdf", "bucket": bucket, "object": object_name}

    local_pdf = _download_gcs_to_tmp(bucket, object_name)
    doc_id = _doc_id(bucket, object_name)
    safe_obj = _safe_filename(object_name)

    # Output base paths
    images_base = f"{OUT_IMAGES_PREFIX}{doc_id}/"
    text_base = f"{OUT_TEXT_PREFIX}{doc_id}/"
    meta_path = f"{OUT_META_PREFIX}{doc_id}.json"
    csv_path = f"{OUT_CSV_PREFIX}{doc_id}.csv"

    meta: Dict[str, Any] = {
        "doc_id": doc_id,
        "source": {
            "gcs_uri": f"gs://{bucket}/{object_name}",
            "bucket": bucket,
            "object": object_name,
            "object_safe": safe_obj,
        },
        "created_at_utc": _utc_now_iso(),
        "outputs": {
            "images_prefix": f"gs://{bucket}/{images_base}",
            "text_prefix": f"gs://{bucket}/{text_base}",
            "metadata_uri": f"gs://{bucket}/{meta_path}",
            "csv_uri": f"gs://{bucket}/{csv_path}",
        },
        "pages": [],
        "warnings": [],
        "stats": {
            "total_pages": 0,
            "image_pages": 0,
            "text_pages": 0,
        }
    }

    pdf = fitz.open(local_pdf)
    num_pages = pdf.page_count
    meta["num_pages"] = num_pages
    meta["stats"]["total_pages"] = num_pages

    if num_pages > MAX_PAGES:
        raise RuntimeError(f"PDF too large: {num_pages} pages (MAX_PAGES={MAX_PAGES})")

    csv_records = []

    # Process each page
    for i in range(num_pages):
        page = pdf.load_page(i)
        page_no = i + 1
        
        print(f"Processing page {page_no}/{num_pages}...")
        
        try:
            page_data = process_page_with_detection(
                page=page,
                page_no=page_no,
                bucket=bucket,
                doc_id=doc_id,
                images_base=images_base,
                text_base=text_base
            )
            
            meta["pages"].append(page_data)
            
            # Update stats
            if page_data["content_type"] == "image":
                meta["stats"]["image_pages"] += 1
            else:
                meta["stats"]["text_pages"] += 1
            
            # Create CSV record
            csv_record = create_csv_record(doc_id, page_data, meta["source"])
            csv_records.append(csv_record)
            
        except Exception as e:
            meta["warnings"].append({
                "page": page_no,
                "type": "processing_error",
                "detail": str(e)
            })
            print(f"Error processing page {page_no}: {e}")

    pdf.close()

    # Save CSV index
    try:
        save_csv_index(bucket, csv_path, csv_records)
        print(f"CSV index saved: gs://{bucket}/{csv_path}")
    except Exception as e:
        meta["warnings"].append({
            "type": "csv_save_error",
            "detail": str(e)
        })

    # Save metadata JSON
    meta_uri = _upload_json(bucket, meta_path, meta)
    meta["outputs"]["metadata_uri"] = meta_uri
    
    print(f"Processing complete: {meta['stats']}")
    return meta


def _parse_eventarc_gcs_event(body: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    Eventarc Cloud Storage "Object finalized" events typically include:
      body["bucket"], body["name"]  (object name)
    Sometimes wrapped in CloudEvent envelope. We handle common variants.
    """
    # Variant A: direct GCS payload
    if isinstance(body.get("bucket"), str) and isinstance(body.get("name"), str):
        return {"bucket": body["bucket"], "name": body["name"]}

    # Variant B: cloudevent with "data"
    data = body.get("data")
    if isinstance(data, dict) and isinstance(data.get("bucket"), str) and isinstance(data.get("name"), str):
        return {"bucket": data["bucket"], "name": data["name"]}

    return None


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/events/gcs")
async def handle_gcs_event(request: Request):
    """
    Eventarc calls this endpoint on object finalize.
    We only process objects under PILOT_PREFIX (default: 'pilot/').
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    parsed = _parse_eventarc_gcs_event(body)
    if not parsed:
        raise HTTPException(status_code=400, detail="Unsupported event payload (missing bucket/name)")

    bucket = parsed["bucket"]
    name = parsed["name"]

    if not name.startswith(INPUT_PREFIX):
        return {
            "ok": True,
            "ignored": True,
            "reason": f"Object not under INPUT_PREFIX='{INPUT_PREFIX}'",
            "bucket": bucket,
            "name": name,
        }

    try:
        result = process_pdf(bucket, name)
        return {
            "ok": True,
            "result_summary": {
                "doc_id": result.get("doc_id"),
                "num_pages": result.get("num_pages"),
                "stats": result.get("stats"),
                "skipped": result.get("skipped", False),
            },
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")


@app.post("/process")
async def manual_process(request: Request):
    """
    Manual endpoint for pilot/backfill.
    Send JSON:
      {"bucket": "slidesenhanced-medslides", "name": "pilot/your.pdf"}
    If bucket omitted, uses BUCKET_NAME.
    """
    body = await request.json()
    bucket = body.get("bucket") or DEFAULT_BUCKET
    name = body.get("name")
    if not bucket or not name:
        raise HTTPException(status_code=400, detail="Provide 'name' and either 'bucket' or BUCKET_NAME env var")

    try:
        result = process_pdf(bucket, name)
        return {
            "ok": True,
            "doc_id": result.get("doc_id"),
            "num_pages": result.get("num_pages"),
            "stats": result.get("stats"),
            "metadata_uri": result.get("outputs", {}).get("metadata_uri"),
            "csv_uri": result.get("outputs", {}).get("csv_uri"),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")


@app.post("/process_batch")
async def manual_process_batch(request: Request):
    """
    Optional helper for pilot: process a list of PDFs.
    JSON:
      {"bucket":"...", "names":["pilot/a.pdf","pilot/b.pdf"]}
    """
    body = await request.json()
    bucket = body.get("bucket") or DEFAULT_BUCKET
    names = body.get("names")
    if not bucket or not isinstance(names, list) or not names:
        raise HTTPException(status_code=400, detail="Provide 'names' list and either 'bucket' or BUCKET_NAME env var")

    results = []
    for name in names:
        try:
            r = process_pdf(bucket, str(name))
            results.append({
                "name": name,
                "ok": True,
                "doc_id": r.get("doc_id"),
                "num_pages": r.get("num_pages"),
                "stats": r.get("stats"),
                "metadata_uri": r.get("outputs", {}).get("metadata_uri"),
                "csv_uri": r.get("outputs", {}).get("csv_uri"),
            })
        except Exception as e:
            results.append({"name": name, "ok": False, "error": str(e)})

    return {"ok": True, "results": results}