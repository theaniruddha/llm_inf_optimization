from __future__ import annotations

from dataclasses import dataclass, field
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F


@dataclass
class SparkEnv:
    spark: SparkSession
    crm_path: str
    cached_data: dict[str, DataFrame] = None #
    tables: dict[str, DataFrame] = field(default_factory=dict)

def get_data(env: SparkEnv):
    # Minimal Change: Only load from disk if we haven't already
    if env.cached_data is None:
        print("Reading from disk for the first (and last) time...")
        tables = load_crm_tables(env) 
        
        # We tell Spark to keep these in RAM
        for name in tables:
            tables[name] = tables[name].cache()
            tables[name].count() # This "Action" forces the memory to fill up
            
        env.cached_data = tables
    
    return env.cached_data


def load_crm_tables(env: SparkEnv) -> dict[str, DataFrame]:
    s = env.spark
    base = env.crm_path.rstrip("/")

    accounts = s.read.option("header", True).csv(f"{base}/accounts.csv")
    products = s.read.option("header", True).csv(f"{base}/products.csv")
    teams = s.read.option("header", True).csv(f"{base}/sales_teams.csv")
    pipeline = s.read.option("header", True).csv(f"{base}/sales_pipeline.csv")

    # Normalize types used in KPIs
    pipeline = (
        pipeline
        .withColumn("engage_date", F.to_date("engage_date"))
        .withColumn("close_date", F.to_date("close_date"))
        .withColumn("close_value", F.col("close_value").cast("double"))
    )

    return {
        "accounts": accounts,
        "products": products,
        "sales_teams": teams,
        "sales_pipeline": pipeline,
    }


def kpi_team_win_rate_and_cycle(env: SparkEnv) -> list[dict]:
    #t = load_crm_tables(env)
    t = get_data(env)  # Minimal Change: Use cached data if available
    p = t["sales_pipeline"]
    teams = t["sales_teams"]

    df = (
        p.join(teams, on="sales_agent", how="left")
        .withColumn("is_closed", F.col("deal_stage").isin("Won", "Lost"))
        .withColumn("is_won", (F.col("deal_stage") == F.lit("Won")).cast("int"))
        .withColumn("is_closed_i", F.col("is_closed").cast("int"))
        .withColumn("cycle_days", F.datediff(F.col("close_date"), F.col("engage_date")))
    )

    agg = (
        df.groupBy("manager", "regional_office")
        .agg(
            F.sum("is_closed_i").alias("closed_deals"),
            F.sum("is_won").alias("won_deals"),
            F.sum(F.when(F.col("deal_stage") == "Won", F.col("close_value")).otherwise(F.lit(0.0))).alias("won_revenue"),
            F.avg(F.when(F.col("deal_stage").isin("Won", "Lost"), F.col("cycle_days"))).alias("avg_cycle_days_closed"),
        )
        .withColumn(
            "win_rate",
            F.when(F.col("closed_deals") > 0, F.col("won_deals") / F.col("closed_deals")).otherwise(F.lit(None))
        )
        .orderBy(F.desc("win_rate"))
    )

    return [r.asDict(recursive=True) for r in agg.collect()]
