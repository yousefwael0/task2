from pydantic import BaseModel


class ConfigResponse(BaseModel):
    max_retries: int
    quality_threshold: float
    has_key: bool
    masked_key: str


class ApiKeyUpdate(BaseModel):
    api_key: str


class StatusResponse(BaseModel):
    status: str


class ModelsResponse(BaseModel):
    models: list[str]


class UploadResponse(BaseModel):
    status: str
    files_indexed: int


class RunRequest(BaseModel):
    session_id: str
    goal: str
    model: str
    temperature: float
