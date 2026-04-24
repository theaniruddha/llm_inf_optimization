from __future__ import annotations

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pyspark.sql import SparkSession

from spark.jobs import SparkEnv, kpi_team_win_rate_and_cycle
from api.chat import router as chat_router

class WinRatesResponse(BaseModel):
    rows: list[dict]


def create_app() -> FastAPI:
    app = FastAPI(title="Spark KPI Tools", version="0.1.0")
    app.include_router(chat_router, prefix="/api/v1", tags=["Chat"])

    spark = (
        SparkSession.builder
        .appName("spark-kpi-tools")
        .master(os.getenv("SPARK_MASTER", "local[*]"))
        .getOrCreate()
    )
    from spark.jobs import SparkEnv, load_crm_tables
    env = SparkEnv(spark=spark, crm_path=os.environ["CRM_DATA_PATH"])
    print("🔥 Warming up Spark Cache...")
    env.tables = load_crm_tables(env)

    @app.get("/healthz")
    def healthz():
        return {"ok": True}
    @app.get("/kpi/win-rates/by-team", response_model=WinRatesResponse)
    # def win_rates_by_team():
    #     try:
    #         return {"rows": kpi_team_win_rate_and_cycle(env)}
    #     except Exception as e:
    #         raise HTTPException(status_code=500, detail=str(e))
    def win_rates_by_team():
        try:
            # Now 'env.tables' is guaranteed to exist
            return {"rows": kpi_team_win_rate_and_cycle(env)}
        except Exception as e:
            # This helps you see the REAL error in your terminal
            import traceback
            print(traceback.format_exc()) 
            raise HTTPException(status_code=500, detail=str(e))


    return app


app = create_app()
