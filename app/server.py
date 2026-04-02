from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .env import SupportTriageEnv
from .models import EnvStateModel, ResetRequest, StepResultModel, SupportAction


app = FastAPI(title="Support Triage OpenEnv", version="0.1.0")
env = SupportTriageEnv()


@app.get("/")
async def root() -> dict:
    return {"status": "ok", "service": "support-triage-openenv"}


@app.get("/ping")
async def ping() -> dict:
    return {"status": "ok"}


@app.post("/reset", response_model=StepResultModel)
async def reset(payload: ResetRequest | None = None) -> StepResultModel:
    try:
        observation = env.reset(task_id=payload.task_id if payload else None)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return StepResultModel(
        observation=observation,
        reward=0.0,
        done=False,
        info=env._finalize_step(0.0, ["Environment reset."]).info,
    )


@app.post("/step", response_model=StepResultModel)
async def step(action: SupportAction) -> StepResultModel:
    try:
        return env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state", response_model=EnvStateModel)
async def state() -> EnvStateModel:
    try:
        return env.state()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
