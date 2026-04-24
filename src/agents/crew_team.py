from __future__ import annotations

from crewai import Agent, Crew, Process, Task, LLM
from schemas import CrewEnvelope
from tools.spark_api_tools import kpi_team_win_rate_and_cycle


def build_sales_expert_crew() -> Crew:
    llm = LLM(
        model="ollama/nemotron-3-nano:30b",  # set to your exact Ollama model name
        base_url="http://127.0.0.1:11434",
        temperature=0.2,
    )
    # spark_tools = SparkSalesTools()

    researcher = Agent(
        role="Researcher",
        goal="Pull KPI evidence via tools; do not invent numbers.",
        backstory="Meticulous analyst. Only uses tool outputs; never fabricates fields or metrics.",
        llm=llm,
        tools=[kpi_team_win_rate_and_cycle],
        verbose=True,
    )

    analyst = Agent(
        role="Financial Analyst",
        goal="Produce decision-ready output strictly matching the Pydantic schema.",
        backstory="Staff-level finance partner. Outputs concise, structured JSON only.",
        llm=llm,
        verbose=True,
    )

    t1 = Task(
        description=(
        "Use KPI tools to gather evidence for: {user_query}.\n"
        "Use and cite these fields when relevant: win_rate, closed_deals, won_deals, won_revenue, avg_cycle_days_closed.\n"
        "Do not invent any missing fields."
        ),
        expected_output="Evidence summary grounded in tool JSON.",
        agent=researcher,
    )

    t2 = Task(
        description="Return ONLY JSON matching the schema for: {user_query}. No markdown.",
        expected_output="A JSON object matching the schema exactly.",
        agent=analyst,
        output_pydantic=CrewEnvelope,  # structured output <citation src="8"></citation>
    )

    return Crew(
        agents=[researcher, analyst],
        tasks=[t1, t2],
        process=Process.sequential,  # sequential crew <citation src="12"></citation>
        verbose=True,
        memory=False,
    )
