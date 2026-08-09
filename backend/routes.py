import json

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import StreamingResponse

from agent.config import MAX_RETRIES, QUALITY_THRESHOLD
from agent.retrieval import add_documents, build_retriever
from backend import placeholder_agent, runtime_config, session_store
from backend.llm_client import MissingAPIKeyError, get_available_models
from backend.schemas import (
    ApiKeyUpdate,
    ConfigResponse,
    ModelsResponse,
    RunRequest,
    StatusResponse,
    UploadResponse,
)

router = APIRouter()


@router.get("/config", response_model=ConfigResponse)
def get_config():
    return ConfigResponse(
        max_retries=MAX_RETRIES,
        quality_threshold=QUALITY_THRESHOLD,
        has_key=runtime_config.has_key(),
        masked_key=runtime_config.masked_key(),
    )


@router.post("/config", response_model=StatusResponse)
def set_config(body: ApiKeyUpdate, request: Request):
    runtime_config.set_override_key(body.api_key)
    # Refresh the cached model list now that the key changed — otherwise
    # a user who starts the server without a key, then pastes one into
    # the sidebar, would still see the fallback model list until restart.
    try:
        request.app.state.available_models = get_available_models()
    except MissingAPIKeyError:
        pass
    return StatusResponse(status="ok")


@router.get("/models", response_model=ModelsResponse)
def get_models(request: Request):
    return ModelsResponse(models=request.app.state.available_models)


@router.post("/upload", response_model=UploadResponse)
async def upload(session_id: str, files: list[UploadFile] = File(...)):
    file_bytes = [(f.filename, await f.read()) for f in files]

    existing = session_store.get(session_id)
    if existing is None:
        retriever = build_retriever(file_bytes)
    else:
        retriever = add_documents(existing, file_bytes)
    session_store.set(session_id, retriever)

    return UploadResponse(status="ok", files_indexed=len(file_bytes))


@router.post("/run")
async def run(body: RunRequest):
    async def event_stream():
        async for event in placeholder_agent.run(
            body.session_id, body.goal, body.model, body.temperature
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
