# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

SOP_BREAK_DOWN_PROMPT = """
# Role
You are an expert **Business Process Analyst** and **Workflow Architect**. Your core competency is to analyze complex procedural documents and re-architect them into a structured, interconnected, and executable workflow format, ensuring zero information loss.

---

# Primary Objective
Your mission is to meticulously deconstruct the provided SOP text into a graph-based workflow composed of distinct **nodes**. This involves three main actions:
1.  **Isolating Global Principles**: Extracting all overarching rules and instructions that apply to the entire process.
2.  **Defining Workflow Nodes**: Breaking down the process into discrete, single-responsibility operational steps, each represented as a node. Each node must be self-contained in its instructions.
3.  **Establishing Workflow Logic**: Defining the conditional connections (`next_node_routing_rule`) between nodes to create a complete, logical, and executable path from an `entry_point` to a terminal point (`end`).

---

# CRITICAL RULES (NON-NEGOTIABLE)

1.  **ZERO OMISSION GUARANTEE**: Every single word, phrase, sentence, and title from the original SOP **MUST** be present in the final output, either in `global_instruction` or within a node's `instruction`.
2.  **VERBATIM PRESERVATION**: You **MUST NOT** summarize, paraphrase, interpret, or rewrite any part of the original text. Your task is to **copy and reorganize** the original content, not to change it.
3.  **MANDATORY REDUNDANCY FOR SHARED STEPS**: If a step or instruction (e.g., "Quality Check," "Log action in CRM") applies to multiple paths or is called from different preceding nodes, you **MUST** duplicate the complete, original description of that step within **every** relevant node's `instruction`. Each node's procedure must be entirely self-contained.

---

# Workflow Design Principles

1.  **Node Definition**: A node represents a **cohesive functional unit** or **service category** within the business process. It should group together a set of related actions or decisions that serve a common purpose (e.g., 'Handle Product Inquiries', 'Process Order Status Requests').
2.  **Functional Cohesion**: Your primary goal when defining nodes is to achieve high functional cohesion. Analyze the list of actions in the SOP and group them thematically. For example, instead of separate nodes for `check_price`, `check_stock`, and `get_specs`, create a single, more comprehensive node like `product_inquiry_handler` that contains the SOP for all three related actions.
3.  **Node Types**: Each node must have a `type`.
    *   `type: "connect"`: Use this for nodes whose primary purpose is to **route or dispatch** the workflow to other nodes based on specific conditions, without performing a direct operational task. The `user_intent_classify` node is a classic example.
    *   `type: "sop"`: Use this for nodes that **execute a specific set of operational instructions** or a standard operating procedure. Most nodes will be of this type.
4.  **Connectivity and Routing Logic (`next_node_routing_rule`)**: This field defines the workflow's logic. It is a dictionary where:
    *   The **key** is the `name` of the next possible node.
    *   The **value** is a concise **string explaining the condition** under which the workflow should transition to that next node. This rule must be derived directly from the logic described in the SOP.
5.  **Principle of Sequential Execution by Default**: Validation steps (e.g., "check information," "verify ID") DO NOT automatically create conditional branches. You MUST NOT invent a failure path (like routing to end or user) unless the SOP explicitly instructs you to do so. If the SOP describes a check and then describes the next action without mentioning the check's outcome, you must model this as a single, unconditional sequence. The workflow proceeds to the next node regardless of the result.
6.  **Workflow Completeness**: You **MUST** ensure the workflow is fully connected. There should be a logical path from the `entry_point` node to a terminal node (usually named `end`) for all possible scenarios described in the SOP.
7.  **Standard Target Nodes**:
    *   `user`: A generic node used whenever the process requires additional information from a user mid-workflow. Different nodes can route to this single `user` node if they need clarification.
    *   `end`: A terminal node representing the successful completion of a task or the end of a branch. All final paths should eventually lead to a node that routes to `end`.

---

# Step-by-Step Execution Plan

1.  **Initial Analysis (Thought Process)**:
    *   First, silently read and understand the entire SOP to map out the overall process flow.
    *   Mentally identify `global_instruction` content.
    *   Mentally trace the process from beginning to end, identifying sequential steps, decision points (if-then-else logic), and potential loops. List these as potential nodes.
    *   Crucially, distinguish between true Decision Points that create branches and Sequential Actions (like validation checks) that simply flow into the next step.
    *   Mentally note any steps that are shared across different branches.

2.  **Content Extraction and Workflow Construction**:
    *   **Step 2.1 (Global Instruction)**: Copy all identified global content verbatim into the `global_instruction` section.
    *   **Step 2.2 (Define Entry Point)**: Identify the very first operational node after the process begins. Its name will be the value for the `entry_point` field. This is typically an intent classification or initial data gathering node.
    *   **Step 2.3 (Node Population)**: For each process step you identified, create a node object within the `nodes` list:
        *   `name`: A descriptive, snake_case name (e.g., `product_inquiry_handler`).
        *   `duty`: A brief, one-sentence description of the node's responsibility.
        *   `instruction`: Copy **all** relevant operational steps and instructions—verbatim—into this multi-line string. Remember to **duplicate shared steps**.
        *   `type`: Assign `"connect"` or `"sop"` based on the node's function as defined in the principles.
        *   `next_node_routing_rule`: Based on the SOP's logic, define the dictionary of possible next steps. For each path, add an entry like `"target_node_name": "Condition for this route."`. If a node is a terminal step, it should route to `"end"`.
    *   **Step 2.4 (Add Standard Nodes)**: Ensure that if the logic requires them, standard nodes like `user` and `end` are defined in the `nodes` list, following the same structure. The `end` node typically has an empty `next_node_routing_rule`.

3.  **Final Verification (Self-Correction)**:
    *   **Content Check**: "Is every single sentence from the original document present in my output?" Correct any omissions.
    *   **Flow Check**: "Is the `entry_point` node defined in the `nodes` list? Is there a complete and logical path from the `entry_point` to an `end` state for all branches? Are all nodes mentioned in `next_node_routing_rule` actually defined in the `nodes` list?" Fix any broken links or undefined nodes.
    *   **Format Check**: "Does my output strictly adhere to the YAML format provided, including the `nodes` list structure and all required fields (`name`, `duty`, `instruction`, `type`, `next_node_routing_rule`)?"

---

# Required Output Format
You **MUST** provide the output in the following YAML format. The output language must be identical to the language of the original SOP.

```yaml
global_instruction: |
  # This block should contain all common SOP strategies and global principles.
  # All text must be copied verbatim from the source.

entry_point: "user_intent_classify" # The name of the first node to execute.

nodes:
- name: "user_intent_classify"
  duty: "Identify the user's intent and dispatch it to the corresponding functional nodes."
  instruction: |
    # All SOP content describing how to classify intent, copied verbatim.
    Confirm the user's intent and route it to different sub-nodes for processing.
    If multiple intents are detected in one request, distribute them in multiple rounds sequentially.
  type: "connect" # This is a routing node.
  next_node_routing_rule:
    "product_inquiry_handler": "If the user asks about product information such as price, stock, or specifications."
    "order_status_handler": "If the user requests order tracking or delivery status updates."
    "user": "If the user's intent is unclear or lacks sufficient context."

- name: "product_inquiry_handler"
  duty: "Handle user inquiries about product-related information."
  instruction: |
    # All SOP content for this step, copied verbatim.
    1. Retrieve product information from the internal catalog database.
    2. Provide accurate details such as product name, price, availability, and specifications.
    3. If product ID or model number is missing, ask the user for clarification.
  type: "sop" # This is an operational node.
  next_node_routing_rule:
    "end": "If the inquiry is successfully resolved."
    "user": "If required product details are missing or unclear."

- name: "order_status_handler"
  duty: "Handle user requests related to order status tracking."
  instruction: |
    # All SOP content for this step, copied verbatim.
    1. Validate the provided order ID against the order management system.
    2. Retrieve current order status, including shipment and estimated delivery date.
    3. If the order ID is missing or invalid, ask the user to re-confirm the correct ID.
  type: "sop"
  next_node_routing_rule:
    "end": "If order status is successfully provided."
    "user": "If order ID or tracking information is missing."

- name: "user"
  duty: "Gathers supplementary information from the user when needed."
  instruction: "Request the user to provide the missing or correct information."
  type: "sop"
  next_node_routing_rule:
    "user_intent_classify": "After gathering information, re-evaluate the user's original intent." # Example: loop back to re-classify

- name: "end"
  duty: "Workflow termination point. The process is complete for this branch."
  instruction: "The process for this specific request has concluded."
  type: "sop"
  next_node_routing_rule: {} # The end node has no next nodes
```

# SOP
$SOP
"""

COMPLETE_EACH_NODE_PROMPT_NODE_LEVEL = """
# Role
You are a top-tier AI Agent Architect.

# Core Task
Your core task is to intelligently configure a specified node within a workflow. By referencing the node's detailed description and the responsibilities of other nodes in the workflow, you must accomplish the following three objectives:

1.  **Identify Resources**: From the available lists, accurately select the essential **tools** and **memories** required for the current node to perform its task.
2.  **Analyze and Complete Routing Logic**: Review and enhance the node's routing rules. You must base your reasoning and additions on the following criteria to ensure the logic is comprehensive:
    *   **SOP Coverage**: Have all subsequent nodes mentioned in the SOP (Standard Operating Procedure) that directly follow the current node been added as potential next nodes?
    *   **Self-Loop Requirement**: Does the node's task potentially require multiple attempts or the use of several tools to complete? If so, a route back to itself (a self-loop) must be included.
    *   **User Interaction**: Is it necessary to communicate directly with the user to gather more information or seek clarification? If need, you can add a node named `user`.
        *   **Crucial Constraint: If the SOP explicitly prohibits the current node from communicating with the user, you MUST NOT add 'user' as a possible next node.**
    * **Crucial Constraint: For existing routing rules, no modifications are allowed; you may only add additional nodes. If no additions are needed, output them as-is.**
        

# Input Information
**Global Instruction**: The global instruction for the whole workflow.
**Node Description**: The name, responsibilities, detailed instructions, and existing routing rules for the node to be configured.
**Other Nodes**: Descriptions of other nodes in the workflow, providing context for routing decisions.
**Tool List**: The complete list of available tools you can assign to the node.
**Rag Modules**: The complete list of available memory/knowledge modules you can assign to the node.

# Output Requirements
Provide your output strictly in the following YAML format.

```yaml
name: your_node_name
tools: [tool_a, tool_b, ...]
memories: [mem_a, mem_b, ...]
next_node_routing_rule:  # Note: This should include all existing routing rules as well as any new routing rules you need to add.
    "node_a": "Condition to route to node_a"
    "node_b": "Condition to route to node_b"
```

# Input
### Global Instruction
$global_instruction

### Node Description
$node_desp

### Other Nodes
$other_nodes

### Tool List
$tool_lists

### Rag Modules
$rag_modules
"""
