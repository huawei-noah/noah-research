.. Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

SOP2Workflow
=============================

在企业场景中，标准操作流程（Standard Operating Procedure，简称 SOP）通常以文档形式存在，用于规范任务执行。

随着多智能体架构的兴起，越来越多的需求将传统 SOP 转化成可执行的工作流来替代或辅助执行人员。然而，这一转换工作需要专业能力、软件开发能力以及较为耗时的工作流调优过程。

因此，我们基于 EvoFabric 框架，提出了 SOP2Workflow 功能，它旨在基于将静态的SOP文档转换为一个可执行的工作流 (:py:class:`~evofabric.core.graph.GraphEngine`)，以降低这一替换成本。

概述
~~~~~~~~~~~~~~~~~~~

SOP2Workflow 提供以下能力：

* 将SOP按功能内聚，拆分成多个可自主迭代执行的智能体。

* 将工具、记忆等资源按需分配给各个智能体。

* 构建一个完整的可运行的工作流。

* 保存过程中所有大语言模型的输出，专业人员可对每个节点的指令、工具、路由方式等内容进行手工修改，重新运行可自动加载修改后的内容，并继续生成完整的工作流。

方案介绍
~~~~~~~~~~~~~~~~

SOP2Workflow分为两步，第一步对SOP进行拆解并生成Workflow的骨架，第二步补全每个节点的细节：

1. 我们将SOP拆解成全局指令（多数用于定义数据格式、执行策略等）和属于节点的局部指令。局部指令会被大语言模型按功能内聚成多个Agent节点，每个Agent节点专注于执行一类功能。同时，大语言模型还需要补充路由节点（如果有需要），用来保证数据能够正确地从开始节点流转到结束节点。拆解的结果会存储在 ``output_dir/_sop_breakdown.yaml`` 内。

2. 随后，每个Agent节点的细节会被补全。我们将当前Agent节点的指令、路由、全局指令、工具列表、记忆列表和其他节点的职责提供给大语言模型，通过大语言模型思考并补全：
   a. 当前Agent节点需要哪些工具。
   b. 当前Agent节点需要哪些记忆模块。
   c. 当前Agent节点的路由是否完整。是否需要路由到 ``user`` 和 ``end`` 节点。

   此外，我们还会使用一个模板将全局指令、Agent指令、路由条件和路由命令拼成一个完整指令作为该Agent节点的 ``system_prompt`` 。

.. mermaid::

    flowchart TD
        subgraph Inputs[Inputs]
            SOP[SOP document]
            Tools[Tool sets]
            Memory[Memory sets]
        end

      subgraph SOP_Decomposition["SOP → Workflow Definition"]
        SOP["SOP Document"]
        GlobalInstruction["Global Instruction<br>(High-level operational goals)"]
        NodeDef["Agent Node Definitions<br>(roles, inputs, outputs)"]
        Routing["Routing Messages<br>(conditions, triggers, data links)"]
      end

      subgraph Refine_nodes["Node Instruction → Complete Agent Node"]
        Full_instruction["Full instruction: <br>Global Instruction + Node Instruction + Routing Rules"]
        Tool_depend["Tool Dependencies"]
        Memories_depend["Memory Dependencies"]
      end

        Inputs -->|SOP Decomposition| SOP_Decomposition
        SOP_Decomposition --> |Nodes Refine | Refine_nodes
        Refine_nodes --> Step3
        Step3[Build GraphEngine]

        class Inputs inputs;


使用方式
~~~~~~~~~~~~~~~~

我们基于 `SOP-Bench <https://arxiv.org/abs/2506.08119>`_ 的SOP数据搭建了一个示例。

.. note::

    为了方便，我们将该数据集的工具重新包装成了一个MCP服务器，并通过 `--tool_file` 指定MCP启动代码路径。

.. code-block:: python


    import argparse
    import asyncio
    import os

    from dotenv import load_dotenv

    from evofabric.app.sop2workflow import WorkflowGenerator
    from evofabric.core.agent import UserNode
    from evofabric.core.clients import OpenAIChatClient
    from evofabric.core.tool import McpToolManager
    from evofabric.core.typing import StdioLink


    def get_args():
        parser = argparse.ArgumentParser()

        # LLM setting
        parser.add_argument("--graph_llm", type=str, default="glm-4.5-air")
        parser.add_argument("--node_llm", type=str, default="glm-4.5-air")
        parser.add_argument("--run_llm", type=str, default="glm-4.5-air")
        parser.add_argument("--no_http_verify", action='store_true', default=False)
        parser.add_argument("--env", type=str, default=".env",
            help=".env file path, must contain OPENAI_API_KEY and OPENAI_BASE_URL")

        # exp setting
        parser.add_argument("--sop", type=str, default="customer_service_sop/sop.txt", help="sop file path")
        parser.add_argument("--tool_file", type=str, default="customer_service_sop/tool_mcp.py",
            help="Python file path of tools")
        parser.add_argument("--class_name", type=str, default="ServiceAccountManager",
            help="Class name storing all python file")
        parser.add_argument("--save_dir", type=str, default="output/customer_service_sop/",
            help="graph desp file save path")

        return parser.parse_args()


    def load_sop(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()


    async def main():
        args = get_args()
        os.makedirs(args.save_dir, exist_ok=True)

        load_dotenv(args.env, override=True)

        tool_manager = McpToolManager(
            server_links={
                "tools": StdioLink(
                    command="python",
                    args=[args.tool_file]
                )
            }
        )

        generator = WorkflowGenerator(
            sop=load_sop(args.sop),
            graph_generation_client=OpenAIChatClient(
                model=args.graph_llm,
                client_kwargs={
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "base_url": os.getenv("OPENAI_BASE_URL"),
                    "max_retries": 5,
                    "timeout": 3600
                },
                http_client_kwargs={"verify": not args.no_http_verify},
                inference_kwargs={"temperature": 0.0, "timeout": 3600}
            ),
            graph_node_complete_client=OpenAIChatClient(
                model=args.node_llm,
                client_kwargs={
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "base_url": os.getenv("OPENAI_BASE_URL"),
                    "max_retries": 5,
                    "timeout": 3600
                },
                http_client_kwargs={"verify": not args.no_http_verify},
                inference_kwargs={"temperature": 0.0, "timeout": 3600}
            ),
            graph_run_client=OpenAIChatClient(
                model=args.run_llm,
                client_kwargs={
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "base_url": os.getenv("OPENAI_BASE_URL"),
                    "max_retries": 5,
                    "timeout": 3600
                },
                http_client_kwargs={"verify": not args.no_http_verify},
                inference_kwargs={"temperature": 0.0, "timeout": 3600}
            ),
            output_dir=args.save_dir,
            tools=[tool_manager],
            memories={},
            state_schema=None,
            addition_global_instruction="",
            user_node=UserNode(),
            fallback_node="end",
            auto_self_loop=True,
            tool_list_mode="select",
            memory_list_mode="select",
            exit_function_name=None,
            build_kwargs={"max_turn": 20},
        )

        graph = await generator.generate()
        graph.draw_graph()

        result = await graph.run({"messages": [{"role": "user", "content": "hello"}]})
        print(result)


    if __name__ == '__main__':
        asyncio.run(main())


示例输出
~~~~~~~~~~~~~~~~~~~~~~~~~~

customer_service_sop 场景：

SOP 输入：

.. code-block:: text

    # **1. Purpose**

    This Standard Operating Procedure outlines a structured, fully offline process for diagnosing and resolving customer-reported service issues without requiring interactive communication with the customer, except for the initial inputs. The SOP ensures end-to-end consistency, traceability, and audit readiness by executing predefined steps across authentication, service eligibility validation, outage detection, diagnostics, troubleshooting, and escalation — all based on system logs and internal tools.

    # **2. Scope**

    This procedure applies to all support teams and automated systems involved in the backend resolution of service issues where the customer is unavailable or where automated processing is preferred. It is specifically designed for workflows in which initial customer inputs are provided, and the entire diagnostic and resolution process is performed using internal data — without any direct input or confirmation from the customer.

    # **3. Key Definitions**

    - **Account ID:** A unique alphanumeric identifier associated with a customer account, typically formatted as three uppercase letters followed by a hyphen and five digits (e.g., ABC-12345). It serves as the primary lookup key for authentication logs, service metadata, and diagnostic history.

    - **Diagnostic Metrics:** Quantitative indicators used to assess service quality, including latency, jitter/stability, and bandwidth throughput.

    - **Root Cause List:** A ranked set of potential service issues inferred from diagnostic tests and account telemetry.

    - **Resolution Outcome:** The final status of the SOP workflow, categorized as one of the following: `RESOLVED`, `PENDING_ACTION`, `ESCALATED`, or `FAILED`.

    - **Escalation Route:** A designated technical team responsible for follow-up action when automated resolution fails. Teams include Tier 2 Technical Support, Field Operations, and Network Engineering.

    # **4. Input**

    The inputs for this SOP are the customer’s Account ID and a description of the customer service issue, both of which must be supplied at the outset of the process.

    # **5. Main Procedure**

    ## **5.1 Authentication and Ticket Initialization**

    Begin the resolution process by validating the format of the provided Account ID. Ensure it conforms to the organization’s standard pattern. If the format is invalid, log the issue and immediately terminate the process. If the Account ID is valid, retrieve the customer’s most recent authentication history. If you find failed attempt and no record of successful recovery, classify the authentication as failed and close the case. If the customer meets the authentication requirements, generate a unique session token and open a new service ticket. Record both the session token and ticket ID, and use them throughout the remainder of the workflow.

    ## **5.2 Account Status Evaluation**

    After establishing authentication, query the system to determine the current status of the customer’s account. If the system flags the account as *Terminated*, log the termination reason and conclude the case, as the account is ineligible for support. If the account is *Suspended*, extract the specific reason for suspension. If the cause relates to non-payment, assign the case to the Accounts Payable department. If the reason is something else, conclude the case, as the account is ineligible for support. Post suspension resolution steps, if the system shows that the suspension has been lifted, continue with the workflow. If the account status is *Active*, record this status and proceed to the next step.

    ## **5.3 Outage and Service Area Analysis**

    Access the outage monitoring system and search for recent or ongoing service disruptions within a 10-mile radius of the customer’s service address. If you detect an outage, log the outage ID, impact scope, and estimated resolution time. You may conclude diagnostics at this point, as the root cause is known. If no outage exists, continue to the technical diagnostic phase.

    ## **5.4 Technical Diagnosis**

    Select and run the appropriate diagnostic tools based on the type of service the customer uses (e.g., internet, voice, video). Measure key performance indicators, including latency, jitter, and bandwidth throughput. Evaluate each metric against defined thresholds. Flag latency values exceeding 100 milliseconds, jitter over 30 milliseconds, or bandwidth levels that fall below the customer’s subscribed plan. Use the diagnostic results and any relevant account history to identify potential root causes. Rank these causes by their likelihood and relevance. Record all diagnostic values, interpretations, and inferred causes in the service ticket, including precise timestamps for traceability.

    ## **5.5 Troubleshooting**

    Run all appropriate resolution steps using predefined troubleshooting guidelines, such as modem resets, signal refreshes, or provisioning adjustments based on the identified root causes. After troubleshooting is complete, re-execute diagnostics to assess changes in latency, jitter, and bandwidth. If metrics improve, classify the issue as fixed. If you observe no significant improvement after executing all troubleshooting steps, proceed to the escalation phase.

    ## **5.6 Escalation Procedures**

    If automated troubleshooting fails to resolve the issue, determine the appropriate escalation path based on the nature of the problem. Create a new escalation ticket and link it to the primary case. Include all relevant diagnostic outputs, attempted troubleshooting steps, customer and device information, and a summary of findings. Assign the ticket to the appropriate support group: use Tier 2 Technical Support for complex diagnostic scenarios, assign on-site issues to the Field Operations team, and route infrastructure problems to Network Engineering. Log the escalation destination, reason, and service-level expectations.

    ## **5.7 Final Resolution and Documentation**

    After completing all diagnosis and escalation steps, compile a comprehensive resolution summary. Include customer account details, authentication results, service status, diagnostic data, troubleshooting actions, and any escalations performed, along with relevant timestamps. Then, update the ticket with the final resolution status: mark it as `RESOLVED` if the issue was addressed, `PENDING_ACTION` if the issue is awaiting a dependent action (e.g., outage resolution), `ESCALATED` if the initial diagnosis and troubleshooting could not resolve the issue and it is therefore assigned to another expert team, or `FAILED` if authentication was not completed.

    # **6. Output**

    The resolution workflow results in a **Resolution Summary Document (RSD)**, structured as a valid JSON object and enclosed within <final_output> tags, as shown in the example below. This output includes key boolean and enumerated outcomes from each procedural step, facilitating downstream processing, analytics, or reporting. ALWAYS output in this format. DO NOT miss any keys mentioned in the final output JSON below.

    <final_output>
    {
      "is_account_id_valid": true,
      "is_authenticated": true,
      "ticket_id": "TKT-2025051234",
      "account_status": "SUSPENDED",
      "account_suspension_status": "ACTIVE",
      "eligible_for_support": true,
      "outage_detected": false,
      "diagnostic_needed": true,
      "latency_issue": true,
      "stability_issue": false,
      "bandwidth_issue": true,
      "metrics_improved_post_troubleshooting": true,
      "escalation_required": false,
      "escalation_ticket_id": "",
      "resolution_summary": "<Insert a comprehensive resolution summary here from step 5.7>",
      "final_resolution_status": "RESOLVED"
    }
    </final_output>


.. mermaid::

    flowchart TD
        authentication_and_ticket_initialization(authentication_and_ticket_initialization)
        account_status_evaluation(account_status_evaluation)
        outage_and_service_area_analysis(outage_and_service_area_analysis)
        technical_diagnosis(technical_diagnosis)
        troubleshooting(troubleshooting)
        escalation_procedures(escalation_procedures)
        final_resolution_and_documentation(final_resolution_and_documentation)
        __start__(__start__)
        __end__(__end__)
        authentication_and_ticket_initialization -.->  account_status_evaluation
        authentication_and_ticket_initialization -.->  authentication_and_ticket_initialization
        authentication_and_ticket_initialization -.->  final_resolution_and_documentation
        account_status_evaluation -.->  account_status_evaluation
        account_status_evaluation -.->  outage_and_service_area_analysis
        account_status_evaluation -.->  final_resolution_and_documentation
        outage_and_service_area_analysis -.->  outage_and_service_area_analysis
        outage_and_service_area_analysis -.->  final_resolution_and_documentation
        outage_and_service_area_analysis -.->  technical_diagnosis
        technical_diagnosis -.->  troubleshooting
        technical_diagnosis -.->  technical_diagnosis
        troubleshooting -.->  troubleshooting
        troubleshooting -.->  escalation_procedures
        troubleshooting -.->  final_resolution_and_documentation
        escalation_procedures -.->  escalation_procedures
        escalation_procedures -.->  final_resolution_and_documentation
        final_resolution_and_documentation -.->  __end__
        final_resolution_and_documentation -.->  final_resolution_and_documentation
        __start__ -->  authentication_and_ticket_initialization

    classDef small fill:#ffe,stroke:#333,stroke-width:1px;

aircraft_inspection_sop 场景：


SOP 输入：

.. code-block:: text

    1. Purpose
    This Standard Operating Procedure (SOP) establishes a comprehensive framework for conducting pre-flight airworthiness verification through multi-layered inspection protocols, ensuring compliance with FAA Part 121/135 regulations and EASA certification requirements while maintaining strict adherence to Safety Management System (SMS) guidelines.

    2. Scope
    This procedure encompasses all pre-flight airworthiness inspections for commercial and private aircraft, including mechanical systems verification, electrical systems authentication, and component validation processes. It applies to all maintenance personnel, aviation safety inspectors, and authorized technical representatives conducting pre-flight inspections.

    3. Definitions
    3.1 Airworthiness Validation Matrix (AVM): Integrated system for cross-referencing aircraft identification parameters
    3.2 Component Tolerance Threshold (CTT): Acceptable variance range for component specifications
    3.3 Electrical Systems Authentication Protocol (ESAP): Standardized procedure for validating electrical systems
    3.4 Maintenance Record Verification System (MRVS): Digital platform for maintenance history validation
    3.5 Serial Number Validation Algorithm (SNVA): Computational process for verifying component authenticity

    4. Input (some are optional)
    4.1 Aircraft Documentation:
    - Aircraft_id
    - Tail_number
    - Maintenance_record_id
    - Expected_departure_time
    - Other parameters depending on task and aircraft

    4.2 Component Verification Data:
    - Component_serial_number
    - Installation_time
    - Component_weight
    - Physical_condition_observations
    - Other parameters depending on task and aircraft

    4.3 Electrical Systems Data:
    - Battery_status
    - Circuit_continuity_check
    - Avionics_diagnostics_response
    - Other parameters depending on task and aircraft

    5. Main Procedure
    5.1 Aircraft Identification Validation
    5.1.1 Execute AVM verification using aircraft_id and tail_number
    5.1.2 Cross-reference maintenance_record_id with MRVS
    5.1.3 Validate expected_departure_time against maintenance window parameters

    5.2 Mechanical Components Inspection
    5.2.1 Verify component_serial_number using SNVA
    5.2.2 Compare component_weight against CTT (±2% variance threshold)
    5.2.3 Document physical_condition_observations with standardized terminology
    5.2.4 Validate installation_time against 24-hour compliance window

    5.3 Electrical Systems Authentication
    5.3.1 Execute ESAP sequence:
       - Verify battery_status (Operational: >80%, Low: <80%, Critical: <40%)
       - Perform circuit_continuity_check (maximum 3 retry attempts)
       - Process avionics_diagnostics_response

    5.4 Discrepancy Reporting
    5.4.1 Generate component_incident_response for mechanical or electrical inspection failures
    5.4.2 Submit component_mismatch_response for SNVA validation failures for component serial number and physical differences during inspection
    5.4.3 Process cross check specifications response for weight and installation discrepancies

    5.5 Maintenance Record Reconciliation
    5.5.1 Execute cross check reporting response for identified discrepancies
    5.5.2 Document variances between maintenance records and inspection findings
    5.5.3 Update MRVS with inspection results

    6. Output
    6.1 Airworthiness Verification Report containing:

    Generate a  report in <final_response> tags for status of all actions and make sure each action is reported in it's own tag.
    A very clear and consice reporting of each action and result is needed for audit purposes in the format <action : result>
    Ensure the results also contain the shipment id.
    For e.g., see format below for reporting the output

    {'aircraft_id': 'a_00123',
    'aircraft_ready': 'TRUE',
    'VerifyShipment': 'success',
    'mechanical_inspection_result':'success',
    'electrical_inspection_result': 'success',
    'component_incident_response': success,
    'component_mismatch_response': None,
    'cross_check_reporting_response': success,
    }

    Use the name of the API specifications for consistency of reporting the actions
    Perform incident reporting only when applicable and ensure chain of custody of documentation
    Do not save any security token locally


    6.2 Digital Maintenance Record Update:
    - Updated MRVS entries
    - Component lifecycle tracking data
    - Inspection timestamp and location verification


.. mermaid::

    flowchart TD
        aircraft_identification_validation(aircraft_identification_validation)
        mechanical_components_inspection(mechanical_components_inspection)
        electrical_systems_authentication(electrical_systems_authentication)
        discrepancy_reporting(discrepancy_reporting)
        maintenance_record_reconciliation(maintenance_record_reconciliation)
        final_report_generation(final_report_generation)
        __start__(__start__)
        __end__(__end__)
        aircraft_identification_validation -.->  mechanical_components_inspection
        aircraft_identification_validation -.->  discrepancy_reporting
        aircraft_identification_validation -.->  aircraft_identification_validation
        mechanical_components_inspection -.->  mechanical_components_inspection
        mechanical_components_inspection -.->  discrepancy_reporting
        mechanical_components_inspection -.->  electrical_systems_authentication
        electrical_systems_authentication -.->  discrepancy_reporting
        electrical_systems_authentication -.->  maintenance_record_reconciliation
        electrical_systems_authentication -.->  electrical_systems_authentication
        discrepancy_reporting -.->  discrepancy_reporting
        discrepancy_reporting -.->  maintenance_record_reconciliation
        maintenance_record_reconciliation -.->  discrepancy_reporting
        maintenance_record_reconciliation -.->  final_report_generation
        maintenance_record_reconciliation -.->  maintenance_record_reconciliation
        final_report_generation -.->  __end__
        final_report_generation -.->  final_report_generation
        __start__ -->  aircraft_identification_validation


content_flagging_sop 场景：

SOP 输入：

.. code-block:: text

    1. Purpose
    This Standard Operating Procedure establishes a comprehensive framework for the systematic evaluation, classification, and disposition of flagged content within the platform's content moderation ecosystem, incorporating multi-dimensional trust metrics, behavioral analysis, and severity assessment protocols.

    2. Scope
    This procedure encompasses all user-generated content flagging operations, subsequent automated analysis protocols, and human moderation workflows within the platform's content management system. It applies to all content moderators, trust and safety specialists, and automated moderation systems.

    3. Definitions
    3.1 Bot Probability Index (BPI): A normalized score between 0-1 derived from behavioral metrics and captcha interaction patterns
    3.2 User Trust Coefficient (UTC): A dynamic score (0-100) incorporating historical behavior and device consistency metrics
    3.3 Content Severity Index (CSI): A weighted composite score (0-100) calculated from primary and secondary violation assessments
    3.4 Geographic Risk Quotient (GRQ): A risk assessment metric derived from historical geographic pattern analysis
    3.5 Device Consistency Score (DCS): A metric evaluating the consistency of user's device fingerprint patterns
    3.6 Violation Confidence Threshold (VCT): Minimum confidence score required for violation classification

    4. Input
    4.1 Content Metadata
    - content_id: Unique content identifier
    - userid: User identification string
    - flagid: Unique flag identifier
    - Geolocation coordinates (latitude, longitude)

    4.2 Device Information
    - device_type
    - operating_system
    - browser_specification

    4.3 Violation Data
    - Primary and Secondary violation types
    - Confidence scores for each violation
    - Historical violation records

    5. Main Procedure

    5.1 Bot Detection Protocol
    5.1.1 Calculate Bot Probability Index (BPI)
    - If is_possible_bot > 0.7 AND captcha_tries >= 3, set BPI = 0.9
    - If is_possible_bot > 0.5 AND captcha_tries >= 2, set BPI = 0.7
    - If is_possible_bot < 0.3 AND captcha_tries <= 1, set BPI = 0.1

    5.1.2 Apply Device Consistency Validation
    - Compare current device_type, os, browser against historical patterns
    - Calculate device fingerprint deviation score
    - Adjust BPI based on deviation patterns

    5.2 User Trust Score Calculation
    5.2.1 Base Trust Score Computation
    - Initialize base_score = 50
    - Adjust for NumberofPreviousPosts (weight: 0.3)
    - Modify based on CountofFlaggedPosts (weight: -0.5)
    - Apply device consistency multiplier

    5.2.2 Geographic Risk Assessment
    - Calculate GRQ based on latitude/longitude clustering
    - Apply regional risk modifiers
    - Adjust trust score based on GRQ

    5.3 Content Severity Assessment
    5.3.1 Primary Violation Analysis
    - Apply violation type weight matrix
    - Calculate weighted confidence score
    - Normalize to 0-100 scale

    5.3.2 Secondary Violation Integration
    - Apply secondary violation multiplier
    - Calculate composite severity score
    - Adjust for violation type correlation

    5.4 Final Decision Matrix
    5.4.1 Decision Score Calculation
    - Combine UTC, CSI, and historical violation metrics
    - Apply threshold matrices for each decision category
    - Calculate final disposition score

    5.4.2 Action Determination
    - If final_score > 80: implement user_banned
    - If 60 < final_score ≤ 80: implement removed
    - If 40 < final_score ≤ 60: implement warning
    - If final_score ≤ 40: implement allowed

    6. Output
    6.1 Decision Package
    - Final disposition (removed/warning/user_banned/allowed)
    - Comprehensive scoring matrix
    - Audit trail of decision factors
    - Geographic risk assessment report
    - Device consistency analysis
    - Violation confidence metrics

    6.2 System Updates
    - User trust score modification
    - Historical violation record update
    - Geographic pattern database update
    - Device fingerprint repository update


.. mermaid::

    flowchart TD
        content_moderation_entry(content_moderation_entry)
        bot_detection_protocol(bot_detection_protocol)
        user_trust_score_calculation(user_trust_score_calculation)
        content_severity_assessment(content_severity_assessment)
        final_decision_matrix(final_decision_matrix)
        __start__(__start__)
        __end__(__end__)
        content_moderation_entry -.->  bot_detection_protocol
        content_moderation_entry -.->  content_moderation_entry
        bot_detection_protocol -.->  user_trust_score_calculation
        bot_detection_protocol -.->  bot_detection_protocol
        user_trust_score_calculation -.->  user_trust_score_calculation
        user_trust_score_calculation -.->  content_severity_assessment
        content_severity_assessment -.->  final_decision_matrix
        content_severity_assessment -.->  content_severity_assessment
        final_decision_matrix -.->  __end__
        final_decision_matrix -.->  final_decision_matrix
        __start__ -->  content_moderation_entry