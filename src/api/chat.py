from __future__ import annotations

import os, json, uuid
from datetime import datetime
from typing import Optional, Any

from fastapi import APIRouter
from pydantic import BaseModel, Field
import time
from agents.langgraph_app import build_graph

router = APIRouter()
graph = build_graph()

RUNS_DIR = os.getenv("RUNS_DIR", "runs")
os.makedirs(RUNS_DIR, exist_ok=True)

class ChatMetrics(BaseModel):
    total_latency: float
    spark_time: float      # Time spent in the tool
    inference_time: float  # Time the LLM was thinking
    tokens_per_sec: float  # Throughput

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    session_id: str
    assistant: dict[str, Any]   # your crew_output (structured)
    metrics: dict[str, float]
    meta: dict[str, Any]        # attempts/errors/etc.


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    start_time = time.perf_counter()
    session_id = req.session_id or str(uuid.uuid4())
    
    # Configure the thread for memory
    config = {"configurable": {"thread_id": session_id}}

    state = {
        "user_query": req.message,
    }

    # Pass the config to the graph#added for memory
    out = graph.invoke(state, config=config)
    end_time = time.perf_counter()
    total_latency = round(time.perf_counter() - start_time, 2)

    # Persist transcript + output (simple long-term storage)
    rec = {
        "ts": datetime.utcnow().isoformat(),
        "session_id": session_id,
        "user": req.message,
        "result": out.get("crew_output"),
        "crew_attempt": out.get("crew_attempt"),
        "crew_error": out.get("crew_error"),
    }
    with open(os.path.join(RUNS_DIR, f"{session_id}.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")

    return {
        "session_id": session_id,
        "assistant": out.get("crew_output"),
        "metrics": {
            "total_latency": total_latency,
            # Use .get() with a default of 0.0 to prevent crashes
            "spark_time": out.get("spark_latency", 0.0), 
            "inference_time": out.get("llm_latency", 0.0),
        },
        "meta": {
            "crew_attempt": out.get("crew_attempt"),
            "crew_error": out.get("crew_error"),
        },
    }
