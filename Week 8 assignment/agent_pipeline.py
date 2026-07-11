"""
=============================================================================
  Single-Agent Pipeline System  --  Built with LangGraph
  Week 8 - Agent Systems, Tools & Evaluation
=============================================================================
  Concepts demonstrated:
    Q1  - Stateful directed graph   -> LangGraph StateGraph + TypedDict state
    Q2  - Nodes & Edges             -> graph.add_node() / add_edge()
    Q3  - Conditional routing       -> graph.add_conditional_edges()
    Q4  - Retry loops (cycles)      -> cycle edge back to router + retry counter
    Q5  - Single-agent multi-role   -> SingleAgent wrapping the compiled graph
    Q6  - JSON-schema tools         -> TOOL_SCHEMAS + validate_json_schema
    Q7  - Sequential vs parallel    -> sequential_tool_calls / run_parallel_tools
    Q8  - Error handling            -> try/except in nodes + error_handler node
    Q9  - Trajectory evaluation     -> TrajectoryRecorder logging each node step
    Q10 - Task completion & cost    -> SingleAgent.report()
=============================================================================
"""

import json
import math
import re
import time
import random
import logging
import concurrent.futures
from typing import Any, Dict, List, Optional, Literal
from typing_extensions import TypedDict

# ── LangGraph imports ────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, END, START

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("LangGraphPipeline")


# ===========================================================================
# Q6 -- JSON-Schema Tool Definitions
# ===========================================================================
TOOL_SCHEMAS: Dict[str, Dict] = {
    "calculator": {
        "name": "calculator",
        "description": "Evaluates a mathematical expression.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "e.g. 3*(4+2)"}
            },
            "required": ["expression"],
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "type":   {"type": "string", "enum": ["calculation"]},
                "result": {"type": ["number", "string"]},
            },
            "required": ["type", "result"],
        },
    },
    "keyword_extractor": {
        "name": "keyword_extractor",
        "description": "Extracts significant keywords from text.",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "type":   {"type": "string", "enum": ["keywords"]},
                "result": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["type", "result"],
        },
    },
    "general_responder": {
        "name": "general_responder",
        "description": "Handles conversational or unclassified queries.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "type":   {"type": "string", "enum": ["general"]},
                "result": {"type": "string"},
            },
            "required": ["type", "result"],
        },
    },
}


def validate_json_schema(data: Dict, schema: Dict):
    """Lightweight JSON-schema validator (Q6)."""
    mapping = {
        "string": str, "number": (int, float), "integer": int,
        "boolean": bool, "array": list, "object": dict, "null": type(None),
    }
    for f in schema.get("required", []):
        if f not in data:
            return False, f"Missing required field: {f!r}"
    for f, fschema in schema.get("properties", {}).items():
        if f not in data:
            continue
        ftype = fschema.get("type")
        py_types = (
            tuple(mapping.get(t, object) for t in ftype) if isinstance(ftype, list)
            else (mapping.get(ftype, object),) if ftype else ()
        )
        if py_types and not isinstance(data[f], py_types):
            return False, f"Field {f!r}: expected {ftype}, got {type(data[f]).__name__}"
    return True, "OK"


# ===========================================================================
# Q1 -- Stateful Directed Graph State  (LangGraph TypedDict)
# ===========================================================================
class AgentState(TypedDict):
    """
    Q1 -- The shared state that LangGraph passes between every node.
    This IS the stateful directed graph's memory.
    Every node receives this dict and returns an updated copy.
    """
    query:         str
    route:         str                    # which tool node to call
    tool_input:    Dict[str, Any]
    tool_output:   Dict[str, Any]
    error:         Optional[str]
    retries:       int
    trajectory:    List[Dict[str, Any]]   # Q9 -- step log


# ===========================================================================
# Q9 -- Trajectory helpers
# ===========================================================================
def _record(state: AgentState, node: str, success: bool,
            output: Any = None, error: str = None, duration_ms: float = 0.0) -> List:
    """Appends a trajectory step to the state list (Q9)."""
    step = {
        "node":        node,
        "output":      output,
        "success":     success,
        "error":       error,
        "duration_ms": round(duration_ms, 2),
    }
    status = "OK  " if success else "FAIL"
    logger.info(f"    [{status}]  Node={node:<28}  ({duration_ms:.1f} ms)")
    return state["trajectory"] + [step]


# ===========================================================================
# Q2 -- LangGraph NODES
# ===========================================================================

# ── Node: Router (Q3 -- conditional routing) ─────────────────────────────────
def router_node(state: AgentState) -> AgentState:
    """
    Q3 -- Rule-based conditional routing node.
    Classifies the query and sets state['route'] so that
    add_conditional_edges() can direct to the correct tool node.
    """
    t0    = time.time()
    query = state["query"].lower().strip()

    if "calculate" in query or re.search(r"[\d].*[+\-*/^]|[+\-*/^].*[\d]", query):
        route = "calculator"
        m = re.search(r"calculate\s+(.+)", query)
        tool_input = {"expression": m.group(1) if m else query}

    elif "keywords" in query or "extract" in query or "key words" in query:
        route      = "keyword_extractor"
        tool_input = {"text": state["query"]}

    elif not state["query"].strip():
        route      = "error"
        tool_input = {}

    else:
        route      = "general"
        tool_input = {"query": state["query"]}

    logger.info(f"  [->] Router: route={route!r}  query={state['query'][:55]!r}")
    ms = (time.time() - t0) * 1000
    traj = _record(state, "RouterNode", True, {"route": route}, duration_ms=ms)

    return {
        **state,
        "route":      route,
        "tool_input": tool_input,
        "error":      "Empty query received - nothing to process." if route == "error" else state.get("error"),
        "trajectory": traj,
    }


# ── Q3 -- Routing function used by add_conditional_edges ─────────────────────
def routing_decision(state: AgentState) -> Literal["calculator", "keyword_extractor", "general", "error_handler"]:
    """
    Q3 -- This function is passed to add_conditional_edges().
    It reads state['route'] and returns the name of the next node.
    LangGraph uses the return value to pick the edge to follow.
    """
    route = state.get("route", "general")
    if route == "error":
        return "error_handler"
    return route   # "calculator" | "keyword_extractor" | "general"


# ── Node: Calculator ──────────────────────────────────────────────────────────
def calculator_node(state: AgentState) -> AgentState:
    """Q2 node -- Q6 schema-validated, Q8 try/except error handling."""
    t0 = time.time()

    ok, msg = validate_json_schema(state["tool_input"], TOOL_SCHEMAS["calculator"]["input_schema"])
    if not ok:
        ms = (time.time() - t0) * 1000
        traj = _record(state, "CalculatorNode", False, error=msg, duration_ms=ms)
        return {**state, "error": f"Schema error: {msg}", "trajectory": traj}

    try:                                     # Q8 -- error handling
        expr   = re.sub(r"[^0-9+\-*/().\s^%]", "", state["tool_input"]["expression"])
        expr   = expr.replace("^", "**")
        result = eval(expr, {"__builtins__": {}}, {"math": math})  # noqa: S307
        output = {"type": "calculation", "result": round(float(result), 6)}
        err    = None
    except Exception as exc:
        output = {"type": "calculation", "result": f"Error: {exc}"}
        err    = str(exc)

    validate_json_schema(output, TOOL_SCHEMAS["calculator"]["output_schema"])
    ms   = (time.time() - t0) * 1000
    traj = _record(state, "CalculatorNode", err is None, output, err, ms)
    return {**state, "tool_output": output, "error": err, "trajectory": traj}


# ── Node: Keyword Extractor ───────────────────────────────────────────────────
def keyword_extractor_node(state: AgentState) -> AgentState:
    """Q2 node -- Q6 schema-validated keyword extraction."""
    t0 = time.time()

    ok, msg = validate_json_schema(state["tool_input"], TOOL_SCHEMAS["keyword_extractor"]["input_schema"])
    if not ok:
        ms = (time.time() - t0) * 1000
        traj = _record(state, "KeywordExtractorNode", False, error=msg, duration_ms=ms)
        return {**state, "error": f"Schema error: {msg}", "trajectory": traj}

    stopwords = {
        "the","a","an","is","are","was","were","in","on","at","to","of",
        "and","or","but","for","with","this","that","it","be","by","from",
        "as","not","no","do","did","does","have","has","had",
        "i","we","you","he","she","they","my","your","his","her","its",
    }
    words    = re.findall(r"[a-zA-Z]{3,}", state["tool_input"]["text"].lower())
    keywords = list(dict.fromkeys(w for w in words if w not in stopwords))[:10]
    output   = {"type": "keywords", "result": keywords}
    validate_json_schema(output, TOOL_SCHEMAS["keyword_extractor"]["output_schema"])

    ms   = (time.time() - t0) * 1000
    traj = _record(state, "KeywordExtractorNode", True, output, duration_ms=ms)
    return {**state, "tool_output": output, "error": None, "trajectory": traj}


# ── Node: General Responder ───────────────────────────────────────────────────
def general_responder_node(state: AgentState) -> AgentState:
    """Q2 node -- Fallback for unclassified queries."""
    t0    = time.time()
    query = state["tool_input"].get("query", state["query"])
    output = {
        "type":   "general",
        "result": (f"Received: \"{query}\". "
                   "Say 'calculate' for math or 'keywords' for keyword extraction."),
    }
    validate_json_schema(output, TOOL_SCHEMAS["general_responder"]["output_schema"])
    ms   = (time.time() - t0) * 1000
    traj = _record(state, "GeneralResponderNode", True, output, duration_ms=ms)
    return {**state, "tool_output": output, "error": None, "trajectory": traj}


# ── Node: Error Handler (Q8) ──────────────────────────────────────────────────
def error_handler_node(state: AgentState) -> AgentState:
    """
    Q8 -- Converts any pipeline error into a structured JSON output.
    The agent never crashes; it always returns a valid response.
    """
    t0     = time.time()
    output = {"type": "error", "result": state.get("error") or "Unknown error"}
    ms     = (time.time() - t0) * 1000
    traj   = _record(state, "ErrorHandlerNode", False, output,
                     state.get("error"), ms)
    return {**state, "tool_output": output, "trajectory": traj}


# ── Node: Retry Gate (Q4 -- cycle) ───────────────────────────────────────────
MAX_RETRIES = 3

def retry_gate_node(state: AgentState) -> AgentState:
    """
    Q4 -- Retry loop node. If the previous tool node returned an error
    AND we have retries left, this node resets the error and sends
    execution back to the router (creating a CYCLE in the graph).
    A 20% transient failure is simulated on the first attempt.
    """
    t0 = time.time()

    # Simulate a 20% transient failure on the very first attempt
    if state["retries"] == 0 and random.random() < 0.20:
        logger.warning(f"  [~] Transient failure on attempt 1/{MAX_RETRIES} -- retrying...")
        ms = (time.time() - t0) * 1000
        traj = _record(state, "RetryGateNode", False,
                       {"action": "retry", "retries": state['retries'] + 1},
                       "Simulated transient failure", ms)
        return {
            **state,
            "error":   "Simulated transient failure -- will retry",
            "retries": state["retries"] + 1,
            "trajectory": traj,
        }

    if state.get("error") and state["retries"] < MAX_RETRIES:
        logger.warning(
            f"  [!] Retry {state['retries'] + 1}/{MAX_RETRIES}: {state['error']}"
        )
        ms = (time.time() - t0) * 1000
        traj = _record(state, "RetryGateNode", False,
                       {"action": "retry", "retries": state["retries"] + 1},
                       state["error"], ms)
        time.sleep(0.05 * (state["retries"] + 1))    # exponential back-off
        return {
            **state,
            "error":      None,          # clear error so router tries again
            "retries":    state["retries"] + 1,
            "trajectory": traj,
        }

    # No error or retries exhausted -- pass through
    ms   = (time.time() - t0) * 1000
    traj = _record(state, "RetryGateNode", True, {"action": "pass_through"}, duration_ms=ms)
    return {**state, "trajectory": traj}


# ── Q4 -- Retry routing function for add_conditional_edges ───────────────────
def retry_routing(state: AgentState) -> Literal["router", "end"]:
    """
    Q4 -- Called after retry_gate_node.
    If there is still an error AND retries remain, loop BACK to router.
    Otherwise, proceed to END.
    """
    if state.get("error") and state["retries"] < MAX_RETRIES:
        return "router"    # CYCLE back -- creates the loop in the graph
    return "end"


# ===========================================================================
# Q1 + Q2 + Q3 + Q4 -- Build the LangGraph StateGraph
# ===========================================================================
def build_graph() -> Any:
    """
    Constructs and compiles the LangGraph StateGraph.

    Graph topology:
                         ┌─────────────┐
        START ──────────►│  router     │
                         └──────┬──────┘
               Q3               │  conditional_edges
         ┌─────────────┬─────────┼──────────┬─────────────┐
         v             v         v          v             v
     calculator  keyword_extractor  general  error_handler
         └─────────────┴─────────┬──────────┘
                                 v
                          retry_gate  ──── Q4 cycle ──► router
                                 │
                               END
    """
    # Q1 -- StateGraph uses AgentState as the shared state schema
    graph = StateGraph(AgentState)

    # Q2 -- Add nodes
    graph.add_node("router",             router_node)
    graph.add_node("calculator",         calculator_node)
    graph.add_node("keyword_extractor",  keyword_extractor_node)
    graph.add_node("general",            general_responder_node)
    graph.add_node("error_handler",      error_handler_node)
    graph.add_node("retry_gate",         retry_gate_node)

    # Q2 -- Entry edge: START -> router
    graph.add_edge(START, "router")

    # Q3 -- Conditional edges from router based on routing_decision()
    graph.add_conditional_edges(
        "router",
        routing_decision,
        {
            "calculator":        "calculator",
            "keyword_extractor": "keyword_extractor",
            "general":           "general",
            "error_handler":     "error_handler",
        },
    )

    # Q2 -- All tool nodes feed into retry_gate
    graph.add_edge("calculator",        "retry_gate")
    graph.add_edge("keyword_extractor", "retry_gate")
    graph.add_edge("general",           "retry_gate")
    graph.add_edge("error_handler",     END)

    # Q4 -- Conditional cycle: retry_gate loops back to router OR ends
    graph.add_conditional_edges(
        "retry_gate",
        retry_routing,
        {
            "router": "router",   # CYCLE -- the retry loop
            "end":    END,
        },
    )

    return graph.compile()


# Cache the compiled graph (compile once, reuse)
COMPILED_GRAPH = build_graph()
logger.info("LangGraph compiled successfully.")


# ===========================================================================
# Q9 -- Trajectory summary helper
# ===========================================================================
def trajectory_summary(state: AgentState) -> Dict:
    steps      = state.get("trajectory", [])
    total      = len(steps)
    successful = sum(1 for s in steps if s["success"])
    total_ms   = sum(s["duration_ms"] for s in steps)
    return {
        "total_nodes_visited": total,
        "successful_nodes":    successful,
        "failed_nodes":        total - successful,
        "total_time_ms":       round(total_ms, 2),
        "api_calls_simulated": total,
        "completion_rate_pct": round(successful / total * 100, 1) if total else 0,
    }


# ===========================================================================
# Q5 -- SingleAgent  (one agent, three internal roles)
# ===========================================================================
class SingleAgent:
    """
    Q5 -- One agent simulating three internal roles:
      Role 1 . Analyser  : routes the query (RouterNode in the graph)
      Role 2 . Executor  : runs the correct tool node with retry protection
      Role 3 . Evaluator : logs trajectory and tracks performance metrics
    """
    def __init__(self):
        self.task_count    = 0
        self.success_count = 0
        self.total_cost    = 0.0

    def run(self, query: str) -> Dict:
        self.task_count += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"  Task #{self.task_count}: {query!r}")
        logger.info("="*60)

        # Build initial state (Q1)
        initial_state: AgentState = {
            "query":       query,
            "route":       "unknown",
            "tool_input":  {},
            "tool_output": {},
            "error":       None,
            "retries":     0,
            "trajectory":  [],
        }

        # Run the LangGraph (Roles 1 + 2: Analyser + Executor)
        final_state = COMPILED_GRAPH.invoke(initial_state)

        # Role 3 -- Evaluator (Q9 + Q10)
        traj = trajectory_summary(final_state)
        self.total_cost += traj["api_calls_simulated"] * 0.001
        if final_state.get("tool_output", {}).get("type") not in ("error", None):
            self.success_count += 1

        result = {
            "query":      query,
            "output":     final_state.get("tool_output", {}),
            "retries":    final_state.get("retries", 0),
            "trajectory": traj,
        }
        logger.info(f"  Output  : {json.dumps(result['output'], ensure_ascii=False)}")
        logger.info(f"  Retries : {result['retries']}  |  Nodes: {traj['total_nodes_visited']}")
        return result

    def report(self):
        """Q10 -- Task completion rate and cost metrics."""
        rate = self.success_count / max(self.task_count, 1) * 100
        logger.info(f"\n{'='*60}")
        logger.info("  AGENT PERFORMANCE REPORT  (Q10)")
        logger.info("="*60)
        logger.info(f"  Tasks attempted      : {self.task_count}")
        logger.info(f"  Tasks succeeded      : {self.success_count}")
        logger.info(f"  Task completion rate : {rate:.1f}%")
        logger.info(f"  Total simulated cost : ${self.total_cost:.4f}")
        logger.info(f"  Cost per task        : ${self.total_cost/max(self.task_count,1):.4f}")
        logger.info("="*60)


# ===========================================================================
# Q7 -- Sequential tool calls
# ===========================================================================
def sequential_tool_calls(queries: List[str], agent: SingleAgent) -> List[Dict]:
    """Q7 -- Execute queries one-at-a-time (preferred when tasks are dependent)."""
    return [agent.run(q) for q in queries]


# ===========================================================================
# Q7 -- Parallel tool calls
# ===========================================================================
def run_parallel_tools(queries: List[str]) -> List[Dict]:
    """
    Q7 -- Run multiple independent queries in parallel using ThreadPoolExecutor.
    Each query gets its own graph invocation (LangGraph is thread-safe).
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"  [||] Parallel execution: {len(queries)} queries")
    logger.info("="*60)

    def _process(query: str) -> Dict:
        initial: AgentState = {
            "query": query, "route": "unknown", "tool_input": {},
            "tool_output": {}, "error": None, "retries": 0, "trajectory": [],
        }
        final = COMPILED_GRAPH.invoke(initial)
        return final.get("tool_output", {})

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(queries)) as pool:
        futures = {pool.submit(_process, q): q for q in queries}
        for fut in concurrent.futures.as_completed(futures):
            try:    results.append(fut.result())
            except Exception as exc:
                results.append({"type": "error", "result": str(exc)})
    return results


# ===========================================================================
# Interactive REPL  (while True loop -- Q4 cycle)
# ===========================================================================
def interactive_loop(agent: SingleAgent):
    """
    Q4 -- A while-True cycle accepting live queries.
    Q8 -- Empty / bad input is handled safely.
    """
    print("\n" + "="*60)
    print("  Interactive Agent Shell  (type 'exit' to quit)")
    print("="*60)
    while True:
        try:
            query = input("\n  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  [Bye!]")
            break
        if query.lower() in ("exit", "quit", "q"):
            print("  [Bye!]")
            break
        if not query:
            print("  [!] Please enter a query.")
            continue
        result = agent.run(query)
        out    = result["output"]
        print(f"  Agent [{out.get('type','?')}]: {out.get('result','(no result)')}")


# ===========================================================================
# Main -- Automated Validation Suite
# ===========================================================================
def main():
    random.seed(42)
    agent = SingleAgent()

    # --- 1. Sequential batch (Q7: sequential calls) -----------------------
    print("\n" + "#"*60)
    print("  SEQUENTIAL TOOL CALLS")
    print("#"*60)
    seq_queries = [
        "calculate 3 * (4 + 2)",
        "calculate 100 / 4 + 25",
        "extract keywords from: Machine learning is a subset of artificial intelligence",
        "Hello! What can you do?",
        "",                                               # error path
        "calculate 10 ^ 3 - 500",
        "keywords: Python data science neural networks deep learning",
    ]
    sequential_tool_calls(seq_queries, agent)
    agent.report()

    # --- 2. Parallel batch (Q7: parallel calls) ---------------------------
    print("\n" + "#"*60)
    print("  PARALLEL TOOL CALLS")
    print("#"*60)
    par_queries = [
        "calculate 55 + 45",
        "extract keywords from: The quick brown fox jumps over the lazy dog",
        "What is the capital of France?",
    ]
    t0      = time.time()
    par_res = run_parallel_tools(par_queries)
    elapsed = (time.time() - t0) * 1000
    logger.info(f"\n  Parallel results ({elapsed:.1f} ms total):")
    for i, r in enumerate(par_res, 1):
        logger.info(f"    [{i}] {json.dumps(r, ensure_ascii=False)}")

    # --- 3. Interactive mode ----------------------------------------------
    interactive_loop(agent)


if __name__ == "__main__":
    main()
