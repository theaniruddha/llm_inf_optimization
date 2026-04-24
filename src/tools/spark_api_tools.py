from __future__ import annotations

import os
import httpx
from crewai.tools import tool

# BASE_URL = os.getenv("SPARK_API_BASE_URL", "http://127.0.0.1:8000")

# @tool("kpi_team_win_rate_and_cycle")
# def kpi_team_win_rate_and_cycle() -> dict:
#     """Fetch win-rates aggregated by Spark. Returns JSON."""
#     r = httpx.get(f"{BASE_URL}/kpi/win-rates/by-team", timeout=60)
#     r.raise_for_status()
#     return r.json()

class SparkSalesTools:
    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.getenv("SPARK_API_BASE_URL", "http://127.0.0.1:8000")

    # 1. Define the logic WITHOUT the @tool decorator here
    def _run_kpi_logic(self, query: str = "") -> dict:
        """The actual code that calls your FastAPI server."""
        r = httpx.get(f"{self.base_url}/kpi/win-rates/by-team", timeout=60)
        r.raise_for_status()
        return r.json()

# 2. Create the instance
spark_instance = SparkSalesTools()

# 3. Create the tool by wrapping the instance method
# This tells the decorator: "The AI only needs to provide 'query', I've already handled 'self'."
@tool("kpi_team_win_rate_and_cycle")
def kpi_team_win_rate_and_cycle(query: str = ""):
    """Fetch win-rates aggregated by Spark. Returns JSON."""
    return spark_instance._run_kpi_logic(query)