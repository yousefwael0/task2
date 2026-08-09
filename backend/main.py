from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.llm_client import FALLBACK_MODELS, MissingAPIKeyError, get_available_models
from backend.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fetch once at startup, not per-request — see llm_client.py. A missing
    # key at boot isn't fatal: the sidebar can set one later via POST
    # /config, which refreshes this same cache (see routes.set_config).
    try:
        app.state.available_models = get_available_models()
    except MissingAPIKeyError:
        app.state.available_models = FALLBACK_MODELS
    yield


app = FastAPI(title="Research Agent Backend", lifespan=lifespan)

# Local dev only — the frontend runs on a different port (Streamlit's
# default 8501) than this backend (8000), so the browser needs CORS
# to allow it. Tighten allow_origins before this goes anywhere real.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok"}
