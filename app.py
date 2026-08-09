"""
Streamlit dashboard for the research agent.

Talks to the backend only through frontend/http_client.py — every
function there is a placeholder today (see its docstring) and will be
swapped for real httpx calls once backend/ exists. Nothing below this
line should need to change when that swap happens.
"""

import streamlit as st

from frontend.http_client import (
    get_available_models,
    get_api_key_status,
    set_api_key,
    get_config,
    upload_files,
    run_agent_stream,
    new_session_id,
)

st.set_page_config(page_title="Research Agent", layout="wide")


# ---------- session state init (must happen before anything reads these) ----------
if "session_id" not in st.session_state:
    st.session_state.session_id = new_session_id()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "tokens" not in st.session_state:
    st.session_state.tokens = 0
if "agent_state" not in st.session_state:
    st.session_state.agent_state = {
        "tasks": [],
        "findings": [],
        "critique": "",
        "quality_score": None,
        "retry_count": 0,
        "last_decision": None,
    }
if "last_report" not in st.session_state:
    st.session_state.last_report = None
if "uploaded_file_names" not in st.session_state:
    st.session_state.uploaded_file_names = []


def _on_api_key_change():
    new_key = st.session_state.api_key_input
    set_api_key(new_key)


def report_to_markdown(report: dict) -> str:
    lines = [f"# {report['title']}", "", report["summary"], ""]
    for section in report["sections"]:
        lines.append(f"## {section['heading']}")
        lines.append(section["content"])
        lines.append("")
    if report.get("sources"):
        lines.append("## Sources")
        for src in report["sources"]:
            lines.append(f"- {src}")
    return "\n".join(lines)


# ---------- sidebar: key + model/temperature controls + token counter ----------
with st.sidebar:
    st.header("⚙️ Controls")

    key_status = get_api_key_status()
    st.text_input(
        "Groq API key",
        value=key_status["masked"] if key_status["has_key"] else "",
        type="password",
        key="api_key_input",
        on_change=_on_api_key_change,
        help="Auto-loaded from the backend's .env. Override here if needed.",
    )
    st.caption("🔑 Key loaded" if key_status["has_key"] else "⚪ No key set")

    model = st.selectbox("Model", get_available_models())
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)

    tokens_placeholder = st.empty()
    tokens_placeholder.metric("Tokens used (session)", st.session_state.tokens)

    config = get_config()
    st.caption(
        f"Quality threshold: {config['quality_threshold']} · "
        f"Max retries: {config['max_retries']}"
    )

    st.divider()
    uploaded = st.file_uploader(
        "Upload research documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    if uploaded and st.button("Index documents"):
        result = upload_files(st.session_state.session_id, uploaded)
        st.session_state.uploaded_file_names.extend(f.name for f in uploaded)
        st.success(f"Indexed {result['files_indexed']} file(s)")

    if st.session_state.uploaded_file_names:
        st.caption("Indexed: " + ", ".join(st.session_state.uploaded_file_names))

    st.divider()
    if st.button("Reset chat"):
        st.session_state.messages = []
        st.session_state.tokens = 0
        st.session_state.agent_state = {
            "tasks": [],
            "findings": [],
            "critique": "",
            "quality_score": None,
            "retry_count": 0,
            "last_decision": None,
        }
        st.session_state.last_report = None
        tokens_placeholder.metric("Tokens used (session)", 0)


# ---------- main area ----------
st.title("Research agent")

chat_col, state_col = st.columns([2, 1])

with chat_col:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    goal = st.chat_input("Enter a research objective...")

    if goal:
        st.session_state.messages.append({"role": "user", "content": goal})
        with st.chat_message("user"):
            st.markdown(goal)

        with st.chat_message("assistant"):
            run_tokens = 0
            with st.status("Running agent...", expanded=True) as status:
                for event in run_agent_stream(
                    st.session_state.session_id, goal, model, temperature
                ):
                    node = event["node"]
                    output = event["output"]
                    run_tokens += output.get("tokens_used", 0)

                    # live reasoning stream — one line per node as it fires
                    if node == "planner":
                        status.write(
                            f"**Planner** produced {len(output['tasks'])} tasks"
                        )
                        st.session_state.agent_state["tasks"] = output["tasks"]
                    elif node == "researcher":
                        status.write(
                            f"**Researcher** found {len(output['findings'])} findings"
                        )
                        st.session_state.agent_state["findings"] = output["findings"]
                    elif node == "critic":
                        status.write(f"**Critic** scored {output['quality_score']:.2f}")
                        st.session_state.agent_state["critique"] = output["critique"]
                        st.session_state.agent_state["quality_score"] = output[
                            "quality_score"
                        ]
                    elif node == "decision":
                        decision_label = (
                            "retry -> Planner"
                            if output["next"] == "planner"
                            else "approved -> Reporting"
                        )
                        status.write(f"**Decision**: {decision_label}")
                        st.session_state.agent_state["retry_count"] = output[
                            "retry_count"
                        ]
                        st.session_state.agent_state["last_decision"] = decision_label
                    elif node == "reporting":
                        status.write("**Reporting** assembled the final report")
                        st.session_state.last_report = output["report"]

                status.update(label="Run complete", state="complete", expanded=False)

            st.session_state.tokens += run_tokens
            tokens_placeholder.metric("Tokens used (session)", st.session_state.tokens)

            summary = "Report ready — see the export button in the sidebar panel below."
            st.markdown(summary)
            st.session_state.messages.append({"role": "assistant", "content": summary})

with state_col:
    st.subheader("State")
    state = st.session_state.agent_state

    st.markdown("**Tasks**")
    if state["tasks"]:
        for t in state["tasks"]:
            st.markdown(f"- {t}")
    else:
        st.caption("No run yet")

    if state["quality_score"] is not None:
        threshold = get_config()["quality_threshold"]
        passed = state["quality_score"] >= threshold
        st.metric(
            "Quality score",
            f"{state['quality_score']:.2f}",
            delta="above threshold" if passed else "below threshold",
            delta_color="normal" if passed else "inverse",
        )

    st.metric("Retry count", state["retry_count"])

    if state["last_decision"]:
        st.markdown(f"**Last decision:** {state['last_decision']}")

    if state["critique"]:
        with st.expander("Latest critique"):
            st.markdown(state["critique"])

    st.divider()
    if st.session_state.last_report:
        report_md = report_to_markdown(st.session_state.last_report)
        st.download_button(
            "Download report (.md)",
            data=report_md,
            file_name="research_report.md",
            mime="text/markdown",
        )
