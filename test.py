#%%
print("Hello world")

#%%
from agents.langgraph_app import build_graph
app = build_graph()

state = {"user_query": "Which regions/managers have lowest win rate and what should we do first?"}
out = app.invoke(state)

out["crew_attempt"], out.get("crew_error")
out["crew_output"]
# %%
