import streamlit as st
import requests
import uuid

def show_performance_dashboard(metrics):
    cols = st.columns(3)
    with cols[0]:
        st.metric("Total Latency", f"{metrics['total_latency']}s", delta_color="inverse")
    with cols[1]:
        # Helpful to show Spark speed to prove your cache is working!
        st.metric("Spark Job", f"{metrics['spark_time']}s")
    with cols[2]:
        st.metric("LLM Speed", f"{metrics['inference_time']}s")

st.set_page_config(page_title="Spark-RAG Analyst", page_icon="⚡")
st.title("⚡ Sales Intelligence Portal")

# Initialize a persistent session ID for the multi-turn memory
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask about win rates or managers..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("H100 is analyzing..."):
            payload = {
                "message": prompt,
                "session_id": st.session_state.session_id
            }
            # Point this to your FastAPI backend
            response = requests.post("http://localhost:8000/api/v1/chat", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                if 'metrics' in data:
                    show_performance_dashboard(data['metrics'])
                else:
                    st.warning("Metrics not found in API response. Is the backend updated?")
                # Extract the summary from your structured Pydantic output
                answer = data["assistant"]["finding"]["exec_summary"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                st.error("Backend connection failed.")