#TODO: medieum: deleate the task to one or more agents
#TODO: loop is not allowed at the moment (consider in the high level)
#TODO: have not explicitly give number constraint yet (consider in the high)
#TODO: take context length into account? should genrate smaller sub-task and deligate to more than 1 search agent considering the context length
#TODO: parse should do as much as work as possible, as we can design
# e.g., we want the input to be handled by the parser
#TODO: update example, to accomodate the decompose tasks, cont... (allow more complex cases)


base = """
Please solve the given question by creating one or more agents and connecting them into a valid computational graph that collaboratively produces the final answer. To create an agent, you must define that agent by outputting <agent> with agent_id (a unique id for the agent, must be unique and contain only alphanumeric or underscore characters (e.g., A1, Refine_1, WS_Japan)), agent_name (exactly one of: CoTAgent, SCAgent, DebateAgent, ReflexionAgent and WebSearchAgent), agent_description, required_arguments (must include at least one <agent_input> tag. DebateAgents must define <debate_roles> with two or more roles. If  <agent_input> left empty (""), the parser will automatically replace it with the original question.). 

After defining all agents, you must build a valid graph by specifying edges that describe the data flow between agents. Output exactly one <edge> block, Each <from> <to> pair connects the output of one agent to the input of another: 

<edge>
<from>source_agent_id</from>
<to>target_agent_id</to>
</edge>

You can output multiple <from> and </to> inside the <edge>. Each <from> and <to> value must exactly match an existing <agent_id>.

To be valid, your graph must satisfy all of the following constraints:

1. Node consistency: Every <from> and <to> must reference a valid <agent_id> that appears in an <agent> block.
2. Directionality: Edges are directed: data flows from <from> → <to>.
3. Connectivity: Every agent must be connected directly or indirectly to the main flow. Isolated agents are not allowed.
4. Start node(s): At least one agent must have no incoming edge. These are “entry points” (e.g., WebSearch or initial reasoning).
5. Sink node: There must be exactly one agent with no outgoing edge — this is the FINAL agent that produces the answer.
6. No undefined edges: It is invalid to reference an agent in <from> or <to> that was not declared.
7. No loops or cycles: No self-loop: <from>X</from><to>X</to> is not allowed. No cycles: The graph must be acyclic; a topological order must exist.
8. Parallelism allowed: Multiple agents may have the same <from> or <to> (fan-out/fan-in).
9. Unambiguous sink: The parser will reject graphs with multiple sinks (add a final “collector” agent if needed).
10. Order-independent: The XML order of edges does not need to follow execution order; topological sorting is handled automatically.
11. Sink answer completeness: The unique sink agent's output must directly answer the original question in a user-ready form. It must not be an intermediate artifact (e.g., notes, critique, raw table) unless the question explicitly asks for that artifact. If <agent_input> is empty for the sink, it inherits the original question and must return the final answer. If <agent_input> is non-empty, the runner still prepends the original question as context; the sink must still produce the final, user-facing answer to that question.
12. Edge-Data Flow Consistency (BIDIRECTIONAL): Edges represent execution order. ${} represents data flow. As a result, you must ensure they are consistent with each other.
    a) If an <agent_input> references ${X}, there MUST be an edge <from>X</from><to>THIS_AGENT</to>
    b) If there is an edge <from>X</from><to>Y</to>, then Y's <agent_input> MUST reference ${X}
    In other words: edges exist if and only if there is actual data passing from one agent to another. Do not create edges solely for execution ordering without data flow.

Thinking Section (Required):

Before defining agents and edges, you must include a <thinking> section.
It should naturally describe why multiple agents are needed, why each type was chosen, and why the graph has that structure (parallel, sequential or hybrid).
It must justify both planning and design rationale.

Example structure:

<thinking>
  Explain why a single agent is insufficient.
  Describe each agent's role and how they connect.
  Justify the flow pattern (parallel, sequential, hybrid).
  End by clearly stating which agent produces the final output.
</thinking>


Single-agent example:

If you decide to solve via single agent, you will output the following. In this case, since the agent_input is the same as the original task, you must set the agent_input as empty (""), and the parser will replace it with the original question.

Multi-Agent Example: 

If you decide to solve via multiple agents, you must first decompose the original question into smaller, well-defined sub-tasks, each representing a single sub-goal. Then, create one agent per sub-task and connect them into a coherent, acyclic computational graph.

In this case, the agent_input is not empty and serves as the specific sub-task for the agent to solve, while the parser automatically prepends the original question as context before the provided agent_input content.

When decomposing:

1. Keep sub-tasks minimal and focused. Each agent should handle one atomic objective (e.g., one query, one reasoning step, or one verification task).
2. Use multiple WebSearchAgents for different pieces of factual evidence, rather than a single broad search.
3. Connect agents logically so that information flows toward a single final agent (the sink) that directly answers the original question.

Final Checklist:
1. Unique IDs: Every <agent_id> is unique and uses only letters, digits, or underscores (A1, WS_Grid, Final_1, etc.). Bellow is WRONG as there are two FINAL agents:
<agent>
  <agent_id>FINAL</agent_id>
  <agent_name>CoTAgent</agent_name>
  <required_arguments><agent_input></agent_input></required_arguments>
</agent>

<agent>
  <agent_id>FINAL</agent_id>
  <agent_name>SCAgent</agent_name>
  <required_arguments><agent_input></agent_input></required_arguments>
</agent>
2. Declared First	All <agent> blocks appear before any <edge> definitions.
3. Connected Graph	No isolated agents — each must connect directly or indirectly to the main flow. Bellow is WRONG as B is isolated agent: 
<agent>
  <agent_id>A</agent_id>
  <agent_name>CoTAgent</agent_name>
  <required_arguments><agent_input></agent_input></required_arguments>
</agent>

<agent>
  <agent_id>B</agent_id>
  <agent_name>CoTAgent</agent_name>
  <required_arguments><agent_input></agent_input></required_arguments>
</agent>

<edge>
  <from>A</from><to>FINAL</to>
</edge>
4. At Least One Start Node:	At least one agent has no incoming edge (e.g., WebSearch or initial reasoning).
5. Exactly One Sink Node	Exactly one agent has no outgoing edge — this is the final output (sink). Bellow is WRONG as there are two sink nodes (B and C):
<edge>
  <from>A</from><to>B</to>
  <from>A</from><to>C</to>
</edge>
6. No Dangling Edges:	Every <from> and <to> exactly matches an existing <agent_id>. Bellow is WRONG as B is not an existing <agent_id>: 
<agent>
  <agent_id>A</agent_id>
  <agent_name>CoTAgent</agent_name>
  <agent_description>...</agent_description>
  <required_arguments><agent_input></agent_input></required_arguments>
</agent>

<edge>
  <from>A</from><to>B</to>
</edge>
7. No Duplicates in Edges: Do not define the same <from> <to> link twice. Bellow is WRONG as it defines the same <from> <to> link twice:
<edge>
  <from>A</from><to>B</to>
  <from>A</from><to>B</to>
</edge>
8. Parallel Allowed	Fan-out and fan-in are fine (e.g., multiple <from>s into one <to>).
9. Edge-Data Flow Consistency: Every edge must correspond to actual data flow. If <from>X</from><to>Y</to> exists, then Y's <agent_input> MUST include ${X}. Conversely, if Y uses ${X}, you MUST have an edge <from>X</from><to>Y</to>. Below is WRONG because edge C→D exists but D doesn't use ${C}:
<agent>
  <agent_id>C</agent_id>
  <agent_name>CoTAgent</agent_name>
  <required_arguments><agent_input>Analyze data.</agent_input></required_arguments>
</agent>
<agent>
  <agent_id>D</agent_id>
  <agent_name>CoTAgent</agent_name>
  <required_arguments><agent_input>Make final decision.</agent_input></required_arguments>
</agent>
<edge>
  <from>C</from><to>D</to>
</edge>
The correct version should have D use ${C}: <agent_input>Make final decision based on: ${C}</agent_input>
10. Consistent Agent Types:	Each agent uses only one of the allowed names: CoTAgent, SCAgent, DebateAgent, ReflexionAgent, or WebSearchAgent.
11. No Self-Loops: Never link an agent to itself. No Cycles: The graph must be a DAG (topological order exists). Bellow is WRONG as there is a cycle (A -> B -> C -> A) and cycle (REFLECT -> REFLECT):
<edge>
  <from>A</from><to>B</to>
  <from>B</from><to>C</to>
  <from>C</from><to>A</to>
  <from>REFLECT</from><to>REFLECT</to>
</edge>


Below is the question to solve:\n\n[QUESTION]
"""

