"""
Streamlit UI for the LangGraph Agent Pipeline
Week 8 — Single-Agent Systems, Tools & Evaluation
"""

import sys
import os
import time
import json
import random
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LangGraph Agent Pipeline",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 40%, #16213e 100%);
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e1b4b 0%, #1e3a5f 100%);
    border-right: 1px solid rgba(99,102,241,0.3);
}
[data-testid="stSidebar"] * { color: #c7d2fe !important; }

/* Header */
.hero {
    background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(168,85,247,0.15) 100%);
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
    text-align: center;
}
.hero h1 { font-size: 2.2rem; font-weight: 700;
    background: linear-gradient(135deg, #818cf8, #c084fc, #38bdf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 8px; }
.hero p { color: #94a3b8; font-size: 1rem; margin: 0; }

/* Cards */
.card {
    background: rgba(30,27,75,0.6);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
    backdrop-filter: blur(10px);
}
.card-title {
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: #818cf8; margin-bottom: 12px;
}

/* Route badges */
.badge {
    display: inline-block; padding: 3px 12px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.05em;
}
.badge-calc   { background: rgba(52,211,153,0.2); color: #34d399; border: 1px solid rgba(52,211,153,0.4); }
.badge-kw     { background: rgba(251,191,36,0.2);  color: #fbbf24; border: 1px solid rgba(251,191,36,0.4); }
.badge-gen    { background: rgba(96,165,250,0.2);  color: #60a5fa; border: 1px solid rgba(96,165,250,0.4); }
.badge-err    { background: rgba(248,113,113,0.2); color: #f87171; border: 1px solid rgba(248,113,113,0.4); }

/* Chat messages */
.msg-user {
    background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(168,85,247,0.2));
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 12px 12px 4px 12px;
    padding: 12px 16px; margin: 8px 0; margin-left: 60px;
    color: #e2e8f0;
}
.msg-agent {
    background: rgba(30,41,59,0.8);
    border: 1px solid rgba(51,65,85,0.6);
    border-radius: 12px 12px 12px 4px;
    padding: 12px 16px; margin: 8px 0; margin-right: 60px;
    color: #e2e8f0;
}
.msg-label { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.08em;
             text-transform: uppercase; margin-bottom: 4px; }
.msg-user .msg-label  { color: #818cf8; }
.msg-agent .msg-label { color: #34d399; }

/* Trajectory step */
.traj-step {
    display: flex; align-items: center; gap: 10px;
    padding: 6px 12px; margin: 4px 0;
    border-radius: 8px; font-size: 0.82rem; font-family: 'JetBrains Mono', monospace;
}
.traj-ok   { background: rgba(52,211,153,0.1); border-left: 3px solid #34d399; color: #a7f3d0; }
.traj-fail { background: rgba(248,113,113,0.1); border-left: 3px solid #f87171; color: #fca5a5; }

/* Metric boxes */
.metric-box {
    background: rgba(30,27,75,0.7);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 12px; padding: 18px; text-align: center;
}
.metric-value {
    font-size: 2rem; font-weight: 700;
    background: linear-gradient(135deg, #818cf8, #c084fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.metric-label { font-size: 0.75rem; color: #64748b; text-transform: uppercase;
                letter-spacing: 0.08em; margin-top: 4px; }

/* Input area */
.stTextInput input {
    background: rgba(30,41,59,0.8) !important;
    border: 1px solid rgba(99,102,241,0.4) !important;
    border-radius: 8px !important; color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextInput input:focus {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 3px rgba(129,140,248,0.15) !important;
}

/* Buttons */
.stButton button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.stButton button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.4) !important;
}

/* Graph topology display */
.graph-box {
    background: rgba(0,0,0,0.3);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 10px; padding: 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem; color: #94a3b8;
    white-space: pre;
}

/* Dividers */
hr { border-color: rgba(99,102,241,0.2) !important; }

/* Section headers */
h2 { color: #c7d2fe !important; font-weight: 600 !important; }
h3 { color: #a5b4fc !important; font-weight: 500 !important; }

/* Expander */
[data-testid="stExpander"] {
    background: rgba(30,27,75,0.4) !important;
    border: 1px solid rgba(99,102,241,0.25) !important;
    border-radius: 10px !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: rgba(30,27,75,0.3); }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.5); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── Import agent pipeline ─────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

@st.cache_resource(show_spinner="Compiling LangGraph...")
def load_agent():
    import importlib
    mod = importlib.import_module("agent_pipeline")
    return mod

try:
    pipe = load_agent()
    AGENT_OK = True
except Exception as e:
    AGENT_OK = False
    AGENT_ERR = str(e)


# ── Session state init ────────────────────────────────────────────────────────
if "history"       not in st.session_state: st.session_state.history       = []
if "agent"         not in st.session_state: st.session_state.agent         = None
if "tasks_run"     not in st.session_state: st.session_state.tasks_run     = 0
if "tasks_ok"      not in st.session_state: st.session_state.tasks_ok      = 0
if "total_cost"    not in st.session_state: st.session_state.total_cost    = 0.0
if "par_results"   not in st.session_state: st.session_state.par_results   = []


def get_agent():
    if st.session_state.agent is None and AGENT_OK:
        random.seed(None)
        st.session_state.agent = pipe.SingleAgent()
    return st.session_state.agent


def badge_html(route: str) -> str:
    cls = {"calculator": "badge-calc", "keyword_extractor": "badge-kw",
           "general": "badge-gen", "error": "badge-err"}.get(route, "badge-gen")
    label = {"calculator": "🧮 Calculator", "keyword_extractor": "🔑 Keywords",
             "general": "💬 General", "error": "⚠️ Error"}.get(route, route)
    return f'<span class="badge {cls}">{label}</span>'


def traj_html(traj_steps: list) -> str:
    rows = []
    for s in traj_steps:
        cls   = "traj-ok" if s["success"] else "traj-fail"
        icon  = "✓" if s["success"] else "✗"
        rows.append(
            f'<div class="traj-step {cls}">'
            f'<span>{icon}</span>'
            f'<span style="flex:1">{s["node"]}</span>'
            f'<span style="color:#64748b">{s["duration_ms"]:.1f} ms</span>'
            f'</div>'
        )
    return "\n".join(rows)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 LangGraph Pipeline")
    st.markdown("**Week 8 — Agent Systems**")
    st.markdown("---")

    st.markdown("### Graph Topology")
    st.markdown("""
<div class="graph-box">START
  │
  ▼
router ──────────────────┐
  │ add_conditional_edges │
  ├──► calculator         │
  ├──► keyword_extractor  │
  ├──► general            │
  └──► error_handler──►END│
            │              │
            ▼              │
        retry_gate ◄───────┘
            │  Q4 CYCLE
            ├──► router (retry)
            └──► END
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Quiz Concepts")
    concepts = [
        ("Q1", "Stateful Graph",     "AgentState TypedDict"),
        ("Q2", "Nodes & Edges",      "add_node / add_edge"),
        ("Q3", "Conditional Route",  "add_conditional_edges"),
        ("Q4", "Retry Loops",        "retry_gate cycle"),
        ("Q5", "Multi-Role Agent",   "Analyser/Executor/Eval"),
        ("Q6", "JSON Schema Tools",  "TOOL_SCHEMAS"),
        ("Q7", "Seq vs Parallel",    "ThreadPoolExecutor"),
        ("Q8", "Error Handling",     "try/except + fallback"),
        ("Q9", "Trajectory Eval",    "_record() logging"),
        ("Q10","Completion & Cost",  "SingleAgent.report()"),
    ]
    for q, title, impl in concepts:
        st.markdown(
            f'<div style="display:flex;gap:8px;align-items:center;margin:4px 0;">'
            f'<span style="background:rgba(99,102,241,0.25);color:#818cf8;'
            f'padding:1px 7px;border-radius:4px;font-size:0.7rem;font-weight:700">{q}</span>'
            f'<span style="font-size:0.82rem;color:#c7d2fe">{title}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    # Live metrics
    st.markdown("### Performance (Q10)")
    m1, m2 = st.columns(2)
    tasks  = st.session_state.tasks_run
    ok     = st.session_state.tasks_ok
    rate   = round(ok / tasks * 100, 1) if tasks else 0
    m1.metric("Tasks", tasks)
    m2.metric("Success", f"{rate}%")
    st.metric("Simulated Cost", f"${st.session_state.total_cost:.4f}")

    st.markdown("---")
    if st.button("🗑️ Clear History"):
        st.session_state.history     = []
        st.session_state.agent       = None
        st.session_state.tasks_run   = 0
        st.session_state.tasks_ok    = 0
        st.session_state.total_cost  = 0.0
        st.session_state.par_results = []
        st.rerun()


# ── Main content ──────────────────────────────────────────────────────────────
if not AGENT_OK:
    st.error(f"Failed to load agent_pipeline.py: {AGENT_ERR}")
    st.stop()

# Hero header
st.markdown("""
<div class="hero">
  <h1>🤖 LangGraph Agent Pipeline</h1>
  <p>Single-Agent System with Conditional Routing, Retry Loops & Trajectory Evaluation</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_batch, tab_parallel, tab_traj = st.tabs([
    "💬 Interactive Chat",
    "🔁 Sequential Batch",
    "⚡ Parallel Batch",
    "📊 Trajectory Viewer",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Interactive Chat
# ════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown("### Chat with the Agent")
    st.markdown(
        '<p style="color:#64748b;font-size:0.85rem;">Try: '
        '<code>calculate 99*3+7</code> &nbsp;·&nbsp; '
        '<code>keywords: machine learning deep neural networks</code> &nbsp;·&nbsp; '
        '<code>Hello, what can you do?</code></p>',
        unsafe_allow_html=True,
    )

    # Chat history display
    chat_container = st.container()
    with chat_container:
        for item in st.session_state.history:
            # User message
            st.markdown(
                f'<div class="msg-user">'
                f'<div class="msg-label">You</div>'
                f'{item["query"]}'
                f'</div>',
                unsafe_allow_html=True,
            )
            # Agent response
            out    = item["result"]["output"]
            route  = item["route"]
            retries= item["result"]["retries"]
            result_val = out.get("result", "")
            if isinstance(result_val, list):
                result_str = ", ".join(result_val)
            else:
                result_str = str(result_val)

            retry_note = (
                f' <span style="color:#fbbf24;font-size:0.75rem">({retries} retry)</span>'
                if retries > 0 else ""
            )
            st.markdown(
                f'<div class="msg-agent">'
                f'<div class="msg-label">Agent &nbsp; {badge_html(route)}{retry_note}</div>'
                f'<div style="margin-top:6px">{result_str}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Input row
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        query = st.text_input(
            "Query", placeholder="Type a query and press Send…",
            label_visibility="collapsed", key="chat_input",
        )
    with col_btn:
        send = st.button("Send ➤", use_container_width=True)

    if send and query.strip():
        agent = get_agent()
        with st.spinner("Agent thinking…"):
            result = agent.run(query)

        # Update metrics
        route = result["output"].get("type", "general")
        if route == "error":
            route = "error"
        elif route == "calculation":
            route = "calculator"
        elif route == "keywords":
            route = "keyword_extractor"
        else:
            route = "general"

        st.session_state.history.append({
            "query":  query,
            "result": result,
            "route":  route,
        })
        st.session_state.tasks_run  += 1
        if result["output"].get("type") != "error":
            st.session_state.tasks_ok   += 1
        st.session_state.total_cost += result["trajectory"]["api_calls_simulated"] * 0.001
        st.rerun()

    elif send and not query.strip():
        st.warning("Please enter a query.")

    # Quick example buttons
    st.markdown("#### Quick Examples")
    ex_cols = st.columns(3)
    examples = [
        ("🧮 Calculate",  "calculate (15 ** 2) - 100"),
        ("🔑 Keywords",   "keywords: LangGraph stateful agents tools routing"),
        ("💬 General",    "What is a directed graph in agent systems?"),
    ]
    for col, (label, ex_query) in zip(ex_cols, examples):
        with col:
            if st.button(label, use_container_width=True, key=f"ex_{label}"):
                agent = get_agent()
                with st.spinner("Running…"):
                    result = agent.run(ex_query)
                out_type = result["output"].get("type", "general")
                route_map = {"calculation": "calculator", "keywords": "keyword_extractor",
                             "general": "general", "error": "error"}
                route = route_map.get(out_type, "general")
                st.session_state.history.append({
                    "query": ex_query, "result": result, "route": route,
                })
                st.session_state.tasks_run  += 1
                if out_type != "error":
                    st.session_state.tasks_ok += 1
                st.session_state.total_cost += result["trajectory"]["api_calls_simulated"] * 0.001
                st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Sequential Batch (Q7)
# ════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown("### Sequential Tool Calls  *(Q7)*")
    st.markdown(
        '<p style="color:#64748b;font-size:0.85rem;">'
        'All 7 test queries run one-after-another. '
        'Each waits for the previous to complete.</p>',
        unsafe_allow_html=True,
    )

    DEFAULT_QUERIES = [
        "calculate 3 * (4 + 2)",
        "calculate 100 / 4 + 25",
        "extract keywords from: Machine learning is a subset of artificial intelligence",
        "Hello! What can you do?",
        "",
        "calculate 10 ^ 3 - 500",
        "keywords: Python data science neural networks deep learning",
    ]

    run_batch = st.button("▶ Run Sequential Batch", use_container_width=True)

    if run_batch:
        agent = get_agent()
        results = []
        prog = st.progress(0, text="Running sequential batch…")
        for i, q in enumerate(DEFAULT_QUERIES):
            with st.spinner(f"Task {i+1}/{len(DEFAULT_QUERIES)}: {q[:40] or '(empty)'}"):
                r = agent.run(q)
                results.append((q, r))
                st.session_state.tasks_run  += 1
                if r["output"].get("type") != "error":
                    st.session_state.tasks_ok += 1
                st.session_state.total_cost += r["trajectory"]["api_calls_simulated"] * 0.001
            prog.progress((i + 1) / len(DEFAULT_QUERIES),
                          text=f"Completed {i+1}/{len(DEFAULT_QUERIES)}")

        prog.empty()
        st.success(f"✅ Sequential batch complete — {len(results)} tasks")

        for i, (q, r) in enumerate(results, 1):
            out      = r["output"]
            out_type = out.get("type", "?")
            result_v = out.get("result", "")
            if isinstance(result_v, list): result_v = ", ".join(result_v)
            route_key = {"calculation": "calculator", "keywords": "keyword_extractor",
                         "general": "general", "error": "error"}.get(out_type, "general")
            retries = r["retries"]

            with st.expander(
                f"Task {i}: {q[:55] or '(empty query)'} — {out_type}", expanded=False
            ):
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.markdown(f"**Route:** {badge_html(route_key)}", unsafe_allow_html=True)
                    st.markdown(f"**Result:** `{str(result_v)[:120]}`")
                with c2:
                    st.metric("Retries", retries)
                    st.metric("Nodes", r["trajectory"]["total_nodes_visited"])
                # Mini trajectory
                st.markdown("**Trajectory:**")
                st.markdown(traj_html(r["trajectory"].get("steps", [])), unsafe_allow_html=True)

    else:
        empty_label = '<em style="color:#f87171">empty — error path</em>'
        rows = "".join(
            f'<div style="padding:4px 0;font-family:JetBrains Mono,monospace;font-size:0.82rem;color:#94a3b8">'
            f'<span style="color:#6366f1">#{i+1}</span> &nbsp; {q if q else empty_label}'
            f'</div>'
            for i, q in enumerate(DEFAULT_QUERIES)
        )
        st.markdown(
            '<div class="card"><div class="card-title">Test Queries</div>' + rows + "</div>",
            unsafe_allow_html=True,
        )


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Parallel Batch (Q7)
# ════════════════════════════════════════════════════════════════════════════
with tab_parallel:
    st.markdown("### Parallel Tool Calls  *(Q7)*")
    st.markdown(
        '<p style="color:#64748b;font-size:0.85rem;">'
        '3 independent queries execute simultaneously in a ThreadPoolExecutor. '
        'Results arrive out-of-order (by completion time).</p>',
        unsafe_allow_html=True,
    )

    par_qs = [
        "calculate 55 + 45",
        "extract keywords from: The quick brown fox jumps over the lazy dog",
        "What is the capital of France?",
    ]

    run_par = st.button("⚡ Run Parallel Batch", use_container_width=True)

    if run_par:
        with st.spinner("Executing 3 queries in parallel…"):
            t0  = time.time()
            res = pipe.run_parallel_tools(par_qs)
            elapsed = (time.time() - t0) * 1000
        st.session_state.par_results = list(zip(par_qs, res))

        st.success(f"✅ All 3 parallel tasks completed in **{elapsed:.1f} ms**")
        st.session_state.tasks_run   += 3
        st.session_state.tasks_ok    += sum(1 for _, r in zip(par_qs, res) if r.get("type") != "error")
        st.session_state.total_cost  += len(par_qs) * 0.002

    if st.session_state.par_results:
        cols = st.columns(3)
        route_map = {"calculation": "calculator", "keywords": "keyword_extractor",
                     "general": "general", "error": "error"}
        for col, (q, r) in zip(cols, st.session_state.par_results):
            with col:
                out_type = r.get("type", "general")
                rv       = r.get("result", "")
                if isinstance(rv, list): rv = "\n• ".join([""] + rv)
                route_key = route_map.get(out_type, "general")
                st.markdown(
                    f'<div class="card">'
                    f'<div class="card-title">Query</div>'
                    f'<div style="font-size:0.85rem;color:#cbd5e1;margin-bottom:12px">{q[:60]}</div>'
                    f'<div class="card-title">Route</div>'
                    f'{badge_html(route_key)}'
                    f'<div class="card-title" style="margin-top:12px">Result</div>'
                    f'<div style="font-family:JetBrains Mono,monospace;font-size:0.82rem;'
                    f'color:#a5f3fc;word-break:break-word">{str(rv)[:150]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.info("Click **⚡ Run Parallel Batch** to execute all 3 queries simultaneously.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Trajectory Viewer (Q9)
# ════════════════════════════════════════════════════════════════════════════
with tab_traj:
    st.markdown("### Trajectory Evaluation  *(Q9)*")
    st.markdown(
        '<p style="color:#64748b;font-size:0.85rem;">'
        'Inspect the full node-by-node execution path of any query. '
        'Trajectory evaluation reveals intermediate decisions beyond the final output.</p>',
        unsafe_allow_html=True,
    )

    traj_query = st.text_input(
        "Query to trace", value="calculate 10 ^ 3 - 500",
        placeholder="Enter any query…", key="traj_input",
    )
    trace_btn = st.button("🔍 Trace Query", use_container_width=True)

    if trace_btn and traj_query.strip():
        agent = get_agent()
        with st.spinner("Tracing…"):
            result = agent.run(traj_query)

        out      = result["output"]
        traj     = result["trajectory"]
        out_type = out.get("type", "?")
        route_k  = {"calculation": "calculator", "keywords": "keyword_extractor",
                    "general": "general", "error": "error"}.get(out_type, "general")

        st.markdown("---")

        # Summary metrics
        mc = st.columns(5)
        mc[0].markdown(f'<div class="metric-box"><div class="metric-value">{traj["total_nodes_visited"]}</div><div class="metric-label">Nodes Visited</div></div>', unsafe_allow_html=True)
        mc[1].markdown(f'<div class="metric-box"><div class="metric-value">{traj["successful_nodes"]}</div><div class="metric-label">Successful</div></div>', unsafe_allow_html=True)
        mc[2].markdown(f'<div class="metric-box"><div class="metric-value">{traj["failed_nodes"]}</div><div class="metric-label">Failed</div></div>', unsafe_allow_html=True)
        mc[3].markdown(f'<div class="metric-box"><div class="metric-value">{traj["total_time_ms"]:.0f}ms</div><div class="metric-label">Total Time</div></div>', unsafe_allow_html=True)
        mc[4].markdown(f'<div class="metric-box"><div class="metric-value">{result["retries"]}</div><div class="metric-label">Retries</div></div>', unsafe_allow_html=True)

        st.markdown("---")

        col_l, col_r = st.columns([1, 1])
        with col_l:
            st.markdown("#### Route Taken")
            st.markdown(f"{badge_html(route_k)}", unsafe_allow_html=True)

            st.markdown("#### Final Output")
            rv = out.get("result", "")
            if isinstance(rv, list):
                for kw in rv:
                    st.markdown(f'<span style="background:rgba(99,102,241,0.2);color:#c7d2fe;padding:2px 10px;border-radius:12px;margin:3px;display:inline-block;font-size:0.82rem">{kw}</span>', unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div style="background:rgba(0,0,0,0.3);border:1px solid rgba(99,102,241,0.3);'
                    f'border-radius:8px;padding:12px;font-family:JetBrains Mono,monospace;'
                    f'font-size:0.9rem;color:#a5f3fc">{rv}</div>',
                    unsafe_allow_html=True,
                )

        with col_r:
            st.markdown("#### Node Execution Sequence")
            st.markdown(
                '<p style="color:#64748b;font-size:0.78rem">Each step shows node name, success, and time</p>',
                unsafe_allow_html=True,
            )
            # We build trajectory steps from the state trajectory list
            state_traj = result.get("trajectory", {})
            # Use the raw trajectory from SingleAgent run
            # Re-run to get step details
            init_state = {
                "query": traj_query, "route": "unknown", "tool_input": {},
                "tool_output": {}, "error": None, "retries": 0, "trajectory": [],
            }
            final = pipe.COMPILED_GRAPH.invoke(init_state)
            steps = final.get("trajectory", [])
            if steps:
                st.markdown(traj_html(steps), unsafe_allow_html=True)
            else:
                st.info("Node detail not available — see summary metrics above.")

        # Raw JSON view
        with st.expander("🔎 Raw trajectory JSON"):
            st.json(result)

    elif trace_btn:
        st.warning("Please enter a query to trace.")
    else:
        # placeholder
        st.markdown("""
<div class="card">
  <div class="card-title">How Trajectory Evaluation Works (Q9)</div>
  <div style="color:#94a3b8;font-size:0.85rem;line-height:1.7">
    Every node visited during a query is recorded as a <code>TrajectoryStep</code> with:<br>
    &nbsp;&nbsp;• <strong>Node name</strong> — which component ran<br>
    &nbsp;&nbsp;• <strong>Success / failure</strong> — did it complete without error<br>
    &nbsp;&nbsp;• <strong>Duration</strong> — how long it took<br>
    &nbsp;&nbsp;• <strong>Input / Output</strong> — data flowing through the node<br><br>
    Retry cycles (Q4) are visible as repeated node sequences in the trajectory.
  </div>
</div>
""", unsafe_allow_html=True)
