from __future__ import annotations

from typing import TypedDict, Optional, Any
from pydantic import ValidationError

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from agents.crew_team import build_sales_expert_crew
from schemas import CrewEnvelope

from langgraph.checkpoint.memory import MemorySaver
MAX_CREW_ATTEMPTS = 3
import time

class OrchestratorState(TypedDict, total=False):
    user_query: str

    # Crew execution + reflection
    crew_attempt: int
    crew_error: Optional[str]
    crew_reflection_hint: Optional[str]

    # Successful structured output
    crew_output: Optional[dict[str, Any]]
    llm_latency: float
    spark_latency: float # We'll simulate this or pull from tool logs
    total_latency: float


def run_crew_team(state: OrchestratorState) -> Command:
    attempt = int(state.get("crew_attempt", 0)) + 1
    user_query = state["user_query"]
    reflection = state.get("crew_reflection_hint")

    # Reflection/Self-correction pattern: feed the prior failure back in
    # (simple but effective; you can expand into a richer “critic” later)
    inputs = {
        "user_query": (
            user_query
            if not reflection
            else f"{user_query}\n\nPrevious attempt failed. Fix issues:\n{reflection}"
        )
    }
    start_llm = time.perf_counter()

    try:
        crew = build_sales_expert_crew()
        result = crew.kickoff(inputs=inputs)
        llm_dur = round(time.perf_counter() - start_llm, 2)

        # CrewAI returns structured output in different places depending on versions;
        # safest: try pydantic first, then json_dict, then raw parse.
        envelope: CrewEnvelope | None = None

        if getattr(result, "pydantic", None):
            envelope = result.pydantic
        elif getattr(result, "json_dict", None):
            envelope = CrewEnvelope.model_validate(result.json_dict)
        else:
            # If all else fails, this will throw ValidationError (caught below)
            envelope = CrewEnvelope.model_validate_json(result.raw)

        # Extra “incomplete JSON” checks beyond schema (optional)
        if not envelope.finding.recommendations or envelope.finding.confidence < 0.2:
            raise ValueError("Output incomplete/low-confidence: add recommendations and justify confidence >= 0.2")

        return Command(
            update={
                "crew_attempt": attempt,
                "crew_error": None,
                "crew_reflection_hint": None,
                "crew_output": envelope.model_dump(),
                "llm_latency": llm_dur,
                "spark_latency": 0.45,
            },
            goto="done",
        )

    except (ValidationError, ValueError) as e:
        hint = (
            "Return ONLY valid JSON matching the schema. "
            "Ensure required fields are present, non-empty lists where required, "
            "confidence is 0..1, and no extra commentary/markdown.\n"
            f"Validation error: {e}"
        )
        if attempt < MAX_CREW_ATTEMPTS:
            return Command(
                update={
                    "crew_attempt": attempt,
                    "crew_error": str(e),
                    "crew_reflection_hint": hint,
                },
                goto="run_crew_team",  # loop back (self-correction) <citation src="8"></citation>
            )

        return Command(
            update={
                "crew_attempt": attempt,
                "crew_error": str(e),
                "crew_reflection_hint": hint,
            },
            goto="fallback",
        )

    except Exception as e:
        # Unknown failure: still loop a couple times with a simpler hint.
        hint = f"Crew execution failed with exception: {type(e).__name__}: {e}"
        if attempt < MAX_CREW_ATTEMPTS:
            return Command(
                update={
                    "crew_attempt": attempt,
                    "crew_error": str(e),
                    "crew_reflection_hint": hint,
                },
                goto="run_crew_team",
            )
        return Command(update={"crew_attempt": attempt, "crew_error": str(e)}, goto="fallback")


def fallback(state: OrchestratorState) -> dict:
    return {
        "crew_output": {
            "finding": {
                "title": "Fallback: Unable to produce a validated finding",
                "exec_summary": "Crew failed after retries; inspect crew_error for details.",
                "kpis_used": [],
                "risks": [state.get("crew_error") or "unknown error"],
                "recommendations": ["Check Spark API healthz", "Verify dataset columns", "Reduce schema strictness temporarily"],
                "confidence": 0.1,
            }
        }
    }


def done(state: OrchestratorState) -> dict:
    # In a full system: route to a “tool execution” node or “report rendering” node.
    return {}


def build_graph():
    memory = MemorySaver()  # optional: persist state across runs for better long-term learning
    g = StateGraph(OrchestratorState)
    g.add_node("run_crew_team", run_crew_team)
    g.add_node("fallback", fallback)
    g.add_node("done", done)

    g.add_edge(START, "run_crew_team")
    g.add_edge("done", END)
    g.add_edge("fallback", END)

    return g.compile(checkpointer=memory)
