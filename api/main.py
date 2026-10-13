"""FastAPI app (spec 13).

POST /score      Green Home Score with component contributions (fast, no ranges)
POST /plan       full results JSON: feasible list, customised plan, savings, carbon, value
POST /whatif     scenario (structured or a chat message) -> rerun + diff (+ explanation)
POST /explain    LLM explanation of results, number-verified, template fallback
GET  /assumptions inputs, sources, versions and the not-a-guarantee notice

    uvicorn api.main:app --reload
"""
from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from engine import memory
from engine.inputs import CustomerInput, CustomiseOptions
from engine.pipeline import ENGINE_VERSION, analyse, assumptions_summary
from engine.whatif import run_whatif
from llm.explainer import explain
from llm.parser import parse
from llm.providers import status as llm_status
from llm.schema import Scenario

app = FastAPI(title="Green Home Engine", version=ENGINE_VERSION,
              description="Physics-based home simulator, ML surrogates, retrofit planner and verified LLM explanations.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PlanRequest(BaseModel):
    input: CustomerInput
    options: Optional[CustomiseOptions] = None
    with_ranges: bool = True
    use_memory: bool = False


class WhatIfRequest(BaseModel):
    input: CustomerInput
    options: Optional[CustomiseOptions] = None
    scenario: Optional[Scenario] = None
    message: Optional[str] = None
    explain: bool = True


class ExplainRequest(BaseModel):
    input: Optional[CustomerInput] = None
    results: Optional[dict] = None
    options: Optional[CustomiseOptions] = None
    question: str = ""


class MemoryRequest(BaseModel):
    completed: list[str] = []
    declined: list[str] = []
    budget: Optional[float] = None
    disruption_tolerance: Optional[str] = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "engine_version": ENGINE_VERSION, "llm": llm_status()}


@app.post("/score")
def score(inp: CustomerInput) -> dict:
    res = analyse(inp, with_ranges=False)
    if res["status"] != "ok":
        return res
    return {
        "status": "ok",
        "score": res["score"]["current"],
        "confidence": res["confidence"],
        "efficiency_flag": res["efficiency_flag"],
        "rating": res["rating"],
        "disclaimer": res["disclaimer"],
    }


@app.post("/plan")
def plan(req: PlanRequest) -> dict:
    return analyse(req.input, req.options, with_ranges=req.with_ranges, use_memory=req.use_memory)


@app.post("/whatif")
def whatif(req: WhatIfRequest) -> dict:
    if req.scenario is None and not req.message:
        raise HTTPException(422, "give a scenario or a message")
    parsed_by = "structured"
    sc = req.scenario
    if sc is None:
        sc, parsed_by = parse(req.message)
    out = run_whatif(req.input, sc, req.options)
    out["parsed_by"] = parsed_by
    if req.explain:
        ex = explain(out["results"], req.message or "", requested=out["requested_infeasible"], diff=out["diff"], question_type=sc.question_type)
        out["explanation"] = {k: ex[k] for k in ("text", "source", "verified", "attempts")}
    return out


@app.post("/explain")
def explain_endpoint(req: ExplainRequest) -> dict:
    results = req.results
    if results is None:
        if req.input is None:
            raise HTTPException(422, "give input or results")
        results = analyse(req.input, req.options)
    ex = explain(results, req.question)
    return {k: ex[k] for k in ("text", "source", "verified", "attempts", "errors")}


@app.get("/assumptions")
def assumptions() -> dict:
    return assumptions_summary()


@app.get("/memory/{customer_id}")
def get_memory(customer_id: str) -> dict:
    return memory.load(customer_id)


@app.post("/memory/{customer_id}")
def set_memory(customer_id: str, req: MemoryRequest) -> dict:
    return memory.record(customer_id, req.completed, req.declined, req.budget, req.disruption_tolerance)
