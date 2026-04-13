# How LangGraph Orchestrates Agent Workflows

> **⚠️ Best-effort draft:** this report was saved after reaching the maximum number of revise cycles and may still contain unresolved gaps noted by the Critic.

# How LangGraph Orchestrates Agent Workflows

## Executive summary

LangGraph is a graph-based orchestration framework for building long‑running, stateful LLM and multi‑agent workflows. Instead of a single opaque “agent loop,” you explicitly define a directed graph of nodes (functions) that share a typed state. LangGraph then:

- executes nodes step‑by‑step,
- merges their partial updates into a shared state via reducers,
- follows static or conditional edges (including cycles) to decide what runs next,
- persists checkpoints after each step for durability and replay,
- and supports human‑in‑the‑loop via interrupts and resume commands.

Agents, tools, and LLMs live *inside* nodes; LangGraph provides the scheduling, control flow, and state management around them.

Below is a concise, technical walkthrough organized around six key aspects of how LangGraph orchestrates agent workflows.

---

## 1. Core concepts: graphs, nodes, edges, agents, tools, state

### Graphs

- A **graph** is the top‑level workflow: a directed graph over a shared application state.
- In Python you typically use `StateGraph(StateType)` to define it.
- A compiled graph represents an executable state machine that advances via node execution and edge traversal until an end condition is reached.

### Nodes

- A **node** is a Python callable that takes the current state and returns a partial state update:

  ```python
  def some_node(state: State) -> dict:
      # read from state
      ...
      # return only fields you want to change
      return {"messages": [new_message], "flag": True}
  ```

- Semantically, a node can be:
  - an LLM “agent” (LLM + prompt + tools),
  - a tool executor,
  - a router/decision function,
  - or arbitrary business logic.

### Edges

- **Static edges**: fixed transitions.

  ```python
  graph.add_edge("planner", "worker")  # always planner → worker
  ```

- **Conditional edges**: branching based on state.

  ```python
  def route(state: State) -> str:
      if state.get("done"):
          return "END"
      if state.get("needs_more_work"):
          return "WORKER"
      return "CRITIC"

  graph.add_conditional_edges(
      "router",
      route,  # function returning a key
      {
          "WORKER": "worker",
          "CRITIC": "critic",
          "END": END,
      },
  )
  ```

- Cycles in the graph (e.g., `worker → router → worker`) create loops.

### Agents

- LangGraph does **not** impose a special “Agent” class.
- An **agent** is usually implemented as one or more nodes that:
  - wrap an LLM (often a LangChain `ChatModel` or Runnable),
  - optionally bind tools,
  - read some subset of the shared state (e.g., `messages`, `plan`),
  - and write back outputs (new messages, updated plan, control flags).
- Multi‑agent systems are graphs with multiple such nodes (planner, worker, critic, supervisor, etc.) routed via edges.

### Tools

- Tools are typically LangChain tools (`@tool` / `BaseTool`) or other callables.
- Within a node you usually:
  - bind tools to an LLM (`llm.bind_tools([...])`) so the model can emit structured tool calls, or
  - call tools explicitly from Python code when appropriate.
- Tool calls and results are recorded in the shared state, usually on a `messages` channel or specific fields like `documents`, `analysis`, etc.

### State, channels, and reducers

- The **state** is a single, shared, typed mapping that flows through the graph. In Python you commonly use a `TypedDict` or Pydantic model:

  ```python
  from typing import Annotated, TypedDict
  from langgraph.graph.message import add_messages

  class GraphState(TypedDict):
      messages: Annotated[list, add_messages]
      plan: str
      done: bool
  ```

- Each field is effectively a **channel**. LangGraph requires a **reducer** function for each field that defines how updates are merged:
  - `messages` often uses `add_messages` (append / concatenate lists).
  - Scalar fields (`plan`, `done`) can use last‑write‑wins or a custom reducer.
- When a node returns a partial update, LangGraph merges it into the existing state using these reducers. This enables:
  - multiple nodes contributing to the same channel (e.g., many messages),
  - consistent shared memory across agents,
  - and clean persistence/checkpointing of the whole state.

---

## 2. How workflows are defined and executed

### Defining a workflow

1. **Define the state schema**:

   ```python
   from typing import Annotated, TypedDict
   from langgraph.graph import StateGraph, END
   from langgraph.graph.message import add_messages

   class GraphState(TypedDict):
       messages: Annotated[list, add_messages]
       plan: str
       done: bool
   ```

2. **Define nodes** (functions that operate on state):

   ```python
   def planner_node(state: GraphState) -> dict:
       user_msg = state["messages"][0]["content"]
       plan = f"Plan for: {user_msg}"
       return {"plan": plan}
   ```

3. **Create a `StateGraph` and register nodes & edges**:

   ```python
   graph = StateGraph(GraphState)
   graph.add_node("planner", planner_node)
   graph.set_entry_point("planner")
   graph.add_edge("planner", END)
   ```

4. **Compile** into an executable app:

   ```python
   app = graph.compile()
   ```

### Executing: `invoke` and `stream`

- **Synchronous run to completion**:

  ```python
  result_state = app.invoke({
      "messages": [{"role": "user", "content": "Help me learn Python."}],
      "plan": "",
      "done": False,
  })
  ```

- **Streaming intermediate steps** (node‑level events or partial state):

  ```python
  for event in app.stream({"messages": [...], "plan": "", "done": False}):
      # event typically includes which node ran and what changed
      print(event)
  ```

### Conceptual execution loop

Internally, LangGraph orchestrates like this:

1. Initialize state from your input.
2. Set the **current node** to the entry point.
3. Repeat until termination:
   - Run the current node with the current state.
   - Get a partial state update from the node.
   - Merge the update into the global state using reducers.
   - **Checkpoint** the new state.
   - Consult the graph’s edges (static or conditional) to pick the next node.
4. Stop when:
   - the next node is `END`,
   - an **interrupt** is raised (for human‑in‑the‑loop), or
   - an unrecoverable error occurs.

This explicit step‑wise model is what makes orchestration transparent and debuggable.

---

## 3. Control flow: branching, loops, conditions, retries, human‑in‑the‑loop

### Branching and routing

- Branching is done through **conditional edges** driven by small routing functions:

  ```python
  def router(state: GraphState) -> str:
      if state.get("done"):
          return "END"
      if not state.get("plan"):
          return "PLAN"
      return "EXECUTE"

  graph.add_node("router", router)
  graph.add_conditional_edges(
      "router",
      router,
      {
          "PLAN": "planner",
          "EXECUTE": "worker",
          "END": END,
      },
  )
  graph.set_entry_point("router")
  ```

- The router inspects state (e.g., flags, last message, task queue) to decide which node to run next.

### Loops

- Loops are just cycles in the graph:

  ```python
  graph.add_edge("worker", "router")  # worker → router → worker → ...
  ```

- Typical patterns:
  - planner ⇄ worker until all subtasks are done,
  - writer ⇄ critic until quality threshold met,
  - supervisor loops over workers until `done=True`.

### Retries and error handling

- LangGraph’s orchestration unit is a **step** (one node execution). After each step, state is checkpointed.
- Error handling is usually implemented in **user code**:
  - wrap LLM/tool calls in retry logic (e.g., using LangChain’s retry utilities),
  - catch exceptions inside a node and:
    - set error fields in state, and/or
    - route to an "error_handler" node via conditional edges.
- The framework’s main contribution is:
  - isolating failures to a single step,
  - ensuring successful steps are durable (you don’t lose progress),
  - making it easy to resume from a checkpoint after fixing a bug.

### Human‑in‑the‑loop: interrupts and Commands

LangGraph has first‑class support for pausing and resuming workflows when human input is required.

**Interrupts**

- Inside a node, you can call LangGraph’s interrupt API (e.g., `interrupt()` helper in Python/JS) when you need human input instead of proceeding automatically.
- When a node triggers an interrupt:
  - execution **pauses** immediately,
  - the current state (and position in the graph) is **checkpointed**,
  - `invoke` / `stream` returns control to your application with information about the interrupt.

**Pause → Human → Resume with `Command`**

- Your application (backend / UI) receives the interrupt and:
  - shows the current state or a question to a human,
  - collects the human’s response.
- To resume, you call `app.invoke` (or `app.stream`) again with:
  - the same **thread/run identifiers** (to pick the correct checkpoint), and
  - a **Command** object containing the **resume value** (the human’s answer or decision).
- LangGraph loads the last checkpoint, merges the resume data into state, and continues executing from the interrupted point.

**Commands and control messages from LLMs**

- On top of the explicit `Command` objects you pass from your app, you can model *logical commands* inside LLM outputs, such as:
  - `{ "action": "REQUEST_HUMAN", "draft": "..." }`,
  - `{ "action": "ESCALATE" }`.
- A router or control node reads these structured messages from state and then either:
  - calls `interrupt()` (to involve a human), or
  - changes routing (e.g., escalates to a different agent node).

This separation—LLMs producing structured intentions, nodes interpreting them, and LangGraph providing the pause/resume mechanism—is a core piece of human‑in‑the‑loop orchestration.

---

## 4. State and memory management across steps and agents

### State evolution and shared memory

- The state is a **single source of truth** for a workflow run.
- At `t=0` you provide an initial state (often just the user’s message and some defaults).
- Each node sees the **current** state, computes updates, and returns **only the deltas** it wants to apply.
- LangGraph merges these deltas using reducers to form the next state.
- Over many steps this produces rich shared memory that can include:
  - conversation history across all agents (`messages`),
  - plans and task lists (`plan`, `tasks`),
  - intermediate tool results (`retrieved_docs`, `code_outputs`),
  - control flags (`done`, `needs_human`, `error_type`).

### Channels and reducers

- Each state field behaves like a **channel** with a specific merge policy:
  - list‑like channels (`messages`, `tasks`) usually append new items,
  - scalar channels (`plan`, `done`) are overwritten or custom‑reduced.
- This allows multiple nodes—even running in parallel in more advanced setups—to contribute to the same logical stream of data.

### Threads, runs, and checkpoints

- LangGraph organizes long‑running workflows into **threads** (or sessions):
  - each thread corresponds to one logical conversation or job,
  - each thread accumulates a sequence of **checkpoints** (one per step).
- A **checkpoint** is a snapshot of the full graph state plus execution position at a point in time, identified by a monotonically increasing ID.
- Checkpoints are stored using a pluggable backend (SQLite, Postgres, or custom implementations that conform to the checkpoint saver interface).

This enables:

- **Durable execution**: long workflows can span many steps and survive restarts; after each step the state is safely persisted.
- **Resume**: if a process crashes or you intentionally stop (e.g., after an interrupt), you can resume from the last checkpoint by calling `invoke` with the same thread/run IDs.
- **Time‑travel & branching**:
  - you can inspect older checkpoints in a thread to debug or audit,
  - you can “fork” from a previous checkpoint to explore alternative behaviors or fix earlier logic without restarting from scratch.

In multi‑agent systems this means all agents see a consistent, evolving state, and their interactions are fully reproducible.

---

## 5. LLM and tool integration

LangGraph is intentionally low‑level and integrates tightly with LangChain’s LLM and tool primitives.

### Typical pattern: LLM agent node with tools

1. **Define tools** using LangChain’s `@tool` decorator:

   ```python
   from langchain_core.tools import tool

   @tool
   def search_docs(query: str) -> str:
       """Search internal documentation."""
       # Implementation elided
       return f"Results for {query}"
   ```

2. **Define state** with a `messages` channel:

   ```python
   from typing import Annotated, TypedDict
   from langgraph.graph.message import add_messages

   class GraphState(TypedDict):
       messages: Annotated[list, add_messages]
   ```

3. **Bind tools to a chat model and define a node**:

   ```python
   from langchain_openai import ChatOpenAI

   llm = ChatOpenAI(model="gpt-4o-mini")
   tools = [search_docs]
   agent_llm = llm.bind_tools(tools)

   def worker_agent(state: GraphState) -> dict:
       # Use full message history as context
       response = agent_llm.invoke(state["messages"])

       # In a fuller pattern, you'd also:
       # - detect tool calls in `response`
       # - execute them (possibly in another node)
       # - append tool results back to messages

       return {"messages": [response]}
   ```

4. **Wire into a graph**:

   ```python
   from langgraph.graph import StateGraph, END

   graph = StateGraph(GraphState)
   graph.add_node("worker", worker_agent)
   graph.set_entry_point("worker")
   graph.add_edge("worker", END)
   app = graph.compile()

   final_state = app.invoke({
       "messages": [{"role": "user", "content": "What is 2+2?"}]
   })
   ```

### Orchestrating tool use across nodes

A common pattern is to separate concerns across nodes:

- **LLM agent node**:
  - reads `messages`,
  - emits either:
    - natural language replies, or
    - structured tool call messages (e.g., `{"tool": "search_docs", "tool_input": {...}}`).

- **Tool executor node**:
  - scans `messages` for pending tool calls,
  - executes the corresponding Python functions,
  - appends tool results back into `messages`.

- **Follow‑up LLM node**:
  - reads both previous conversation and tool results,
  - generates a final answer or the next action.

LangGraph orchestrates this by:

- defining nodes for each phase (agent → tools → agent),
- routing between them via edges (static or conditional depending on whether tools were requested),
- merging all intermediate outputs into the shared `messages` channel.

Because state is persisted at each step, you get a fully traceable multi‑step tool‑use workflow.

---

## 6. End‑to‑end multi‑agent example: planner → worker → critic

Below is a compact multi‑agent workflow that shows how LangGraph orchestrates agents via shared state and routing.

### Goal

Handle a user request using:

- **planner** – breaks the request into tasks.
- **worker** – executes each task (simulated with an LLM call).
- **critic** – reviews progress and either continues or finalizes.
- **router** – decides which node runs next.

### Code (conceptual)

```python
from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

# ---------- State ----------

class GraphState(TypedDict):
    messages: Annotated[list, add_messages]    # conversation + agent messages
    plan: List[str]                            # list of tasks
    current_task_index: int                    # index in plan
    final_answer: str                          # final output
    done: bool                                 # termination flag

llm = ChatOpenAI(model="gpt-4o-mini")

# ---------- Nodes ----------

def planner(state: GraphState) -> dict:
    """Decompose user request into a small plan."""
    user_msg = state["messages"][0]["content"]
    prompt = f"Break this request into 2-3 concrete tasks:\n\n{user_msg}"
    resp = llm.invoke([{"role": "user", "content": prompt}])

    tasks = [line.strip("- ").strip()
             for line in resp.content.split("\n") if line.strip()]

    return {
        "plan": tasks,
        "current_task_index": 0,
        "messages": [{"role": "assistant", "content": f"Plan: {tasks}"}],
    }

def worker(state: GraphState) -> dict:
    """Execute the current task (mocked)."""
    idx = state["current_task_index"]
    task = state["plan"][idx]
    prompt = f"Execute this task and summarize:\n\nTask {idx+1}: {task}"
    resp = llm.invoke([{"role": "user", "content": prompt}])

    worker_msg = {
        "role": "assistant",
        "name": "worker",
        "content": f"Task {idx+1} result: {resp.content}",
    }
    return {"messages": [worker_msg]}

def critic(state: GraphState) -> dict:
    """Decide whether to continue or finalize."""
    plan = state["plan"]
    idx = state["current_task_index"]

    prompt = (
        "You are a critic. Given the plan and progress, decide whether we "
        "should continue with more tasks or can finalize. Respond with the "
        "word CONTINUE or FINAL only.\n\n"
        f"Plan: {plan}\nCurrent task index: {idx}"
    )
    resp = llm.invoke(state["messages"] + [{"role": "user", "content": prompt}])

    decision = "FINAL" if "FINAL" in resp.content.upper() else "CONTINUE"

    if decision == "FINAL" or idx >= len(plan) - 1:
        final_answer = f"Completed {idx+1} tasks. Here is your result."
        return {
            "final_answer": final_answer,
            "done": True,
            "messages": [
                {"role": "assistant", "name": "critic",
                 "content": f"Decision: FINAL. {final_answer}"}
            ],
        }
    else:
        return {
            "current_task_index": idx + 1,
            "messages": [
                {"role": "assistant", "name": "critic",
                 "content": "Decision: CONTINUE with next task."}
            ],
        }

def router(state: GraphState) -> str:
    """Route between planner, worker, critic, and END."""
    if not state.get("plan"):
        return "PLAN"
    if state.get("done"):
        return "END"

    # If last message was from worker, go to critic, else go to worker
    last_msg = state["messages"][-1]
    if last_msg.get("name") == "worker":
        return "CRITIC"
    else:
        return "WORKER"

# ---------- Build graph ----------

graph = StateGraph(GraphState)

graph.add_node("planner", planner)
graph.add_node("worker", worker)
graph.add_node("critic", critic)
graph.add_node("router", router)

graph.set_entry_point("planner")

# planner → router
graph.add_edge("planner", "router")

# router decides next node
graph.add_conditional_edges(
    "router",
    router,
    {
        "PLAN": "planner",
        "WORKER": "worker",
        "CRITIC": "critic",
        "END": END,
    },
)

# worker and critic both go back to router
graph.add_edge("worker", "router")
graph.add_edge("critic", "router")

app = graph.compile()

# ---------- Run ----------

initial_state = {
    "messages": [{"role": "user", "content": "Design a simple 1-week Python study plan."}],
    "plan": [],
    "current_task_index": 0,
    "final_answer": "",
    "done": False,
}

final_state = app.invoke(initial_state)
print(final_state["final_answer"])
```

### Orchestration walkthrough

1. **Entry:** `planner` runs first, reads the user’s request from `messages`, and writes a `plan`, sets `current_task_index = 0`, and appends a plan message.
2. **Planner → Router:** via a static edge; `router` inspects state. Since `plan` exists and `done` is False, it returns `"WORKER"`.
3. **Router → Worker:** `worker` executes the current task via LLM, appends a worker message. State is checkpointed. Static edge `worker → router` runs.
4. **Router → Critic:** seeing the last message came from `worker`, `router` routes to `critic`.
5. **Critic:** examines `plan`, `current_task_index`, and the conversation; either:
   - sets `done=True` and `final_answer` (then router will route to `END`), or
   - increments `current_task_index` and appends a “CONTINUE” message.
6. **Loop:** `critic → router`; router decides to either:
   - send flow back to `worker` for the next task, or
   - end the workflow if `done=True`.

Throughout this process, LangGraph:

- maintains a single shared state object across all agents,
- persists a checkpoint after each node execution,
- drives routing through explicit edges and the router function,
- and makes it straightforward to insert human approval (via interrupts) at, for example, the critic node.

That is how LangGraph orchestrates complex, stateful, multi‑agent workflows in a transparent, controllable way.

---

## Sources

- LangGraph overview & concepts – LangChain Docs (Python):
  - https://docs.langchain.com/oss/python/langgraph/overview
- Interrupts & human‑in‑the‑loop – LangGraph Docs:
  - Python: https://docs.langchain.com/oss/python/langgraph/interrupts
  - JS: https://docs.langchain.com/oss/javascript/langgraph/interrupts
- Durable execution & checkpoints – LangGraph Docs and reference:
  - JS durable execution overview: https://docs.langchain.com/oss/javascript/langgraph/durable-execution
  - Python checkpoints reference: https://reference.langchain.com/python/langgraph/checkpoints
- Checkpoint backends – GitHub:
  - https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint/README.md
- Functional API / orchestration patterns – LangGraph Docs:
  - https://docs.langchain.com/oss/javascript/langgraph/functional-api
- Multi‑agent patterns – external but aligned tutorials:
  - https://www.getzep.com/ai-agents/langchain-agents-langgraph/
