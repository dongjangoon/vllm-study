from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

from .models import DebateRequest, DebateResponse
from .debate import run_debate, langfuse
from .config import LANGFUSE_HOST


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    langfuse.flush()


app = FastAPI(
    title="Multi-Agent Debate Service",
    description="두 LLM 에이전트가 주어진 주제에 대해 찬반 토론을 수행합니다. Langfuse로 트레이싱됩니다.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/debate", response_model=DebateResponse)
async def debate_endpoint(request: DebateRequest):
    try:
        turns, trace_id = run_debate(
            topic=request.topic,
            rounds=request.rounds,
            language=request.language,
        )
        return DebateResponse(
            topic=request.topic,
            rounds=request.rounds,
            turns=turns,
            trace_id=trace_id,
            langfuse_url=f"{LANGFUSE_HOST}/trace/{trace_id}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {
        "service": "multi-agent-debate",
        "endpoints": {
            "POST /debate": "토론 시작 (topic, rounds, language)",
            "GET /health": "헬스 체크",
        },
    }
