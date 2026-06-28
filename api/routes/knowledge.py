from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from utils.knowledge_agent import kb
import uuid
import structlog

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/knowledge", tags=["Knowledge"])

@router.post("/upload")
async def upload_doc(user_id: str = Form(...), file: UploadFile = File(...)):
    content = (await file.read()).decode("utf-8", errors="replace")
    doc_id = str(uuid.uuid4())[:8]
    kb.add_doc(user_id, doc_id, content, {"file_name": file.filename})
    log.info("doc_uploaded", user_id=user_id, file=file.filename)
    return {"doc_id": doc_id, "file_name": file.filename, "status": "ok"}

@router.post("/upload-text")
async def upload_text(user_id: str = Form(...), content: str = Form(...), name: str = Form("manual")):
    doc_id = str(uuid.uuid4())[:8]
    kb.add_doc(user_id, doc_id, content, {"file_name": name})
    return {"doc_id": doc_id, "file_name": name, "status": "ok"}

@router.get("/search")
async def search_kb(user_id: str, query: str, top_k: int = 3):
    results = kb.search(user_id, query, top_k)
    return {"user_id": user_id, "results": results, "total": len(results)}
