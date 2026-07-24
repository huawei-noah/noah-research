SUFFICIENCY_CHECK_PROMPT = """你是一名记忆检索评估专家。请评估当前检索到的记忆是否足以回答用户的问题。

用户问题：
{query}

检索到的记忆：
{retrieved_docs}

请判断这些记忆是否足以回答用户的问题。

输出格式（JSON）：
{{
    "is_sufficient": true/false,
    "reasoning": "你的判断理由",
    "missing_information": ["缺失信息1", "缺失信息2"]
}}

判断规则：
1. 如果记忆包含回答问题的关键信息（如时间、地点、人物、事件细节、原因等），则判断为充足（true）
2. **时间完整性至关重要**：如果问题询问具体时间（例如"X 是什么时候发生的"），而记忆缺少该时间，则判断为不足，并在缺失信息中列出"具体发生时间"
3. 如果缺少关键信息（如涉及的人物、事件细节等），则判断为不足（false）并列出缺失信息
4. 推理过程应简洁清晰，解释判断依据
5. 仅在判断为不足时填写缺失信息；充足时使用空数组
6. **时间最新性规则**：当问题包含表示"最近"的时间线索（如"最近"、"最新"、"上次"、"最后一次"、"刚刚"），且检索到的记忆中包含多个不同时间点的候选答案时，系统必须优先选择**时间戳最新**的答案。当存在更近期的匹配事件时，不应选择较早的事件

示例1（简单）：
用户问："张三什么时候晋升为经理？"
如果检索到："张三在2023年5月从工程师晋升为技术经理"
→ is_sufficient: true, reasoning: "明确包含晋升时间和职位信息"

示例2（复杂任务 - 旅行规划）：
用户问："请帮我规划一个杭州3天2晚的旅行，预算3000元，偏爱自然风光和美食"
当前检索到："西湖景区开放时间8:00-18:00，免费；杭州东站高铁方便"
→ is_sufficient: false,
   missing_information: ["住宿推荐及价格（预算匹配）"，"餐饮/美食具体推荐"，"3天行程路线规划"，"交通方式及费用"，"其他自然景点如西溪湿地、千岛湖等"]

示例3（多轮后充足）：
用户问："请帮我规划一个杭州3天2晚的旅行..."
当前检索到：西湖、西溪湿地、千岛湖景点信息；3家酒店价格对比；5家杭帮菜餐厅推荐；高铁/地铁/出租车费用；详细3天行程安排
→ is_sufficient: true, reasoning: "涵盖所有要素：景点、住宿、餐饮、交通和行程规划。3000元预算可行（住宿1200+交通600+餐饮800+门票400）"
"""

MULTI_QUERY_GENERATION_PROMPT = """你是一名查询优化专家。用户的原始查询未能检索到足够信息，请生成多个互补的改进查询。

原始查询：
{original_query}

当前检索到的记忆：
{retrieved_docs}

缺失信息：
{missing_info}

请生成2-3个互补的改进查询，以帮助找到缺失信息。这些查询应：

- 关注不同的缺失信息点（例如一个针对时间，一个针对人物，一个针对原因）
- 使用不同的表达方式（同义词、具体化、抽象化）
- 避免与原始查询和历史查询重复
- 保持简洁清晰，适合向量检索

**时间范围处理**：
- 仅当查询或缺失信息中包含**明确的绝对时间**（例如"2024年5月"、"2023年3月16日"、"2023年"）时，才设置 `time_range`。
- 不要为相对或模糊的时间表达生成 time_range（例如"最近"、"去年"、"昨天"、"近来"、"之前"）。这些表达无法可靠解析，将 `time_range` 设为 `null`。
- 如果没有明确的绝对时间参考，则将 `time_range` 设为 `null`。
- `time_range` 的格式为 `[开始时间, 结束时间]`，使用 ISO 格式（例如 '2024-01-15 08:00:00'），两端包含，精确到秒。

输出格式（JSON）：
{{
"queries": [
{{
"query": "改进查询1",
"time_range": ["2024-01-15 08:00:00", "2024-01-16 23:59:59"] // 或 null
}},
{{
"query": "改进查询2",
"time_range": null
}},
{{
"query": "改进查询3",
"time_range": ["2024-03-01 00:00:00", "2024-03-31 23:59:59"]
}}
],
"reasoning": "查询生成策略的解释，包括如何从时间线索推断时间范围"
}}

要求：

- queries 数组包含2-3个查询，每个查询是一个对象，包含 query（字符串，长度5-200字符）和 time_range（数组或null）。
- 当 time_range 是数组时，必须包含两个元素，分别表示开始时间和结束时间（包含两端），时间字符串格式为 'YYYY-MM-DD HH:mm:ss'。
- reasoning 解释生成策略，包括为什么选择这些查询以及如何考虑时间范围（特别是如果从相对表达推断而来）。
- **时间放松策略**：如果原始查询有时间约束，但检索到的记忆中不包含该时间范围内的匹配信息，你必须生成至少一条 `time_range` 设为 `null` 的查询，以便无时间限制地搜索。这有助于找到可能以稍微不同或不正确的时间戳记录的信息。
- **不同查询关注不同侧面**：每条生成的查询应侧重问题的不同方面（如一条关注事件/动作本身，一条关注涉及的人物，一条关注地点/背景）。避免生成只是换种说法但关注点相同的查询。

示例1（简单）：
原始查询："张三的晋升情况"
缺失信息：["具体时间"，"晋升原因"]
历史查询：["张三的晋升情况"，"张三什么时候晋升的"]
当前时间：2024-07-15 14:00:00
生成：
{{
"queries": [
{{"query": "specific date when Zhang San was promoted to manager", "time_range": null}},
{{"query": "reason and background for Zhang San's promotion", "time_range": null}},
{{"query": "Zhang San's promotion process from engineer to manager", "time_range": null}}
],
"reasoning": "三个查询分别关注时间、原因和过程；问题中没有明确时间线索，因此 time_range 为 null"
}}

示例2（含相对时间推断）：
原始查询："昨天的团队会议上发生了什么？"
缺失信息：["会议内容"，"做出的决定"]
当前时间：2024-07-15 14:00:00
生成：
{{
"queries": [
{{"query": "content of team meeting on 2024-07-14", "time_range": ["2024-07-14 00:00:00", "2024-07-14 23:59:59"]}},
{{"query": "decisions and action items from yesterday's team meeting", "time_range": ["2024-07-14 00:00:00", "2024-07-14 23:59:59"]}}
],
"reasoning": "根据当前时间（2024-07-15）推断"昨天"为绝对日期2024-07-14；两个查询都针对该日期设置了全天窗口"
}}

示例3（复杂任务 - 旅行规划，多轮演进）：

【第1轮后】
原始查询："请帮我规划一个杭州3天2晚的旅行，预算3000元，偏爱自然风光和美食"
缺失信息：["住宿推荐及价格"，"餐饮/美食具体推荐"，"3天行程路线规划"，"交通方式及费用"，"其他自然景点"]
历史查询：["杭州旅游攻略"]
当前时间：2024-07-15 14:00:00
生成：
{{
"queries": [
{{"query": "Hangzhou West Lake nearby hotel price comparison 2024", "time_range": ["2024-01-01 00:00:00", "2024-12-31 23:59:59"]}},
{{"query": "Hangzhou must-eat Hangzhou cuisine restaurant recommendations local favorites", "time_range": null}},
{{"query": "Hangzhou Xixi Wetland Qiandao Lake transportation routes", "time_range": null}}
],
"reasoning": "第一个查询明确限制到2024年以匹配预算时效性；后两个无明确时间限制；三个查询分别覆盖住宿、餐饮和景点交通，针对最大信息缺口"
}}

【第2轮后】
原始查询："请帮我规划一个杭州3天2晚的旅行..."
当前检索到：西湖、西溪湿地景点信息；3家酒店价格；5家餐厅名称
缺失信息：["3天具体行程安排逻辑"，"景点间交通方式及耗时"，"餐饮预算细分"，"千岛湖是否适合3天行程"]
历史查询：["杭州旅游攻略"，"杭州西湖附近酒店价格对比2024"，"杭州必吃杭帮菜餐厅推荐本地人最爱"，"杭州西溪湿地千岛湖交通路线"]
当前时间：2024-07-15 14:00:00
生成：
{{
"queries": [
{{"query": "Hangzhou 3-day 2-night travel route planning West Lake Xixi Wetland connection", "time_range": null}},
{{"query": "Hangzhou downtown to Qiandao Lake high-speed rail bus transportation time cost", "time_range": null}},
{{"query": "Hangzhou food per capita consumption price Louwailou Grandma's House", "time_range": ["2024-01-01 00:00:00", "2024-12-31 23:59:59"]}}
],
"reasoning": "前两个查询关注行程逻辑和交通耗时，无明确时间窗口；第三个查询限制到2024年价格以匹配预算，细化餐饮费用"
}}

【第3轮后】
原始查询："请帮我规划一个杭州3天2晚的旅行..."
当前检索到：完整的景点、酒店、餐厅、交通、行程框架
缺失信息：["实时天气/最佳旅游季节"，"景点预约/限流政策"]
历史查询：[...前两轮的7个查询...]
当前时间：2024-07-15 14:00:00
生成：
{{
"queries": [
{{"query": "Hangzhou travel best season weather March April May", "time_range": ["2024-03-01 00:00:00", "2024-05-31 23:59:59"]}},
{{"query": "West Lake scenic area reservation crowd control policy 2024", "time_range": ["2024-01-01 00:00:00", "2024-12-31 23:59:59"]}}
],
"reasoning": "针对旅行时机和入园政策，分别设置春季和全年时间窗口以确保信息时效性和准确性"
}}
"""

PROPERTY_FILTER_SELECTION_PROMPT = """你是一名记忆检索策略专家。你的任务是在回答用户问题时，选择需要关注的实体类型及其属性。

用户问题：
{query}

可用的实体类型及其属性：
{entity_schema}

请选择回答此问题最相关的实体类型及其属性。

输出格式（JSON）：
{
    "selected_entities": [
        {
            "entity_type": "person",
            "relevant_properties": ["profession_event", "location_event", "experience"]
        },
        {
            "entity_type": "fact",
            "relevant_properties": ["fact_event"]
        }
    ],
    "reasoning": "为什么选择这些实体类型和属性"
}

核心原则：宁多勿漏。如果不确定某个实体类型或属性是否相关，请选择包含它。宁可检索到可能不需要的额外信息，也不能遗漏可能相关的信息。

规则：
1. **宁多勿漏**：如果不确定某实体类型或属性是否相关，请选择包含它
2. 选择所有可能包含相关信息的实体类型，不要过于严格地过滤
3. 对于每个实体类型，考虑包含更多属性，特别是：
   - 所有基于事件的属性（以 _event 结尾）- 包含时间特定细节
   - 所有与经验相关的属性
   - 与问题关键词相关的任何属性
4. 聚焦于能回答问题 "谁"、"什么"、"何时"、"何地"、"为何"、"如何" 的属性
5. 如果不需要特定属性（返回全部），则 relevant_properties 使用空数组
6. **特殊情况 - episodes 实体**：对于 episodes 类型，如果问题询问对话内容、会议详情、讨论主题或任何历史对话，务必包含 "input_messages" 属性。该属性包含原始对话文本，对于回答此类查询至关重要。
7. **使用 "all" 保留所有属性**：当需要检索某实体类型的全部属性（不过滤）时，使用 ["all"] 作为 relevant_properties 的值。这将返回该实体类型的所有可用属性。
8. 当问题模糊或有多种解释时，显著扩大搜索范围

示例 - 广泛选择：
用户问："Caroline 最近做了什么？"
→ 选择：entity_type: "person", relevant_properties: ["all"]
原因：需要涵盖 Caroline 可能做的所有类型的活动

用户问："张三什么时候晋升为经理？"
→ 选择：entity_type: "person", relevant_properties: ["profession_event", "experience", "career_history"]
原因：需要职业事件来跟踪职位变化，也需要经验作为背景

用户问："昨天的团队会议上发生了什么？"
→ 选择：entity_type: "episodes", relevant_properties: ["all"]
原因：需要完整的会议详情。使用 "all" 以包含 input_messages 属性，其中包含对话内容

用户问："告诉我与张三的对话内容"
→ 选择：entity_type: "episodes", relevant_properties: ["all"]
原因：需要完整对话内容，存储在 input_messages 属性中
"""

GLOBAL_PROPERTY_RERANK_PROMPT = """你是一名记忆检索专家。你的任务是从所有检索到的实体属性中，选择最相关的属性来回答用户的问题。

用户问题：
{query}

检索到的实体属性：
{property_list}

请选择最能回答用户问题的 top {top_n} 个最相关的属性。

输出格式（JSON）：
{{
    "selected_properties": [
        {{
            "entity_id": "entity_001",
            "entity_name": "张三",
            "property_name": "profession_event",
            "property_value": "2023年5月，张三晋升为技术总监",
            "timestamp": "2023-05-15 00:00:00",
            "relevance_score": 0.95
        }}
    ],
    "reasoning": "为什么选择这些属性"
}}

规则：
1. 选择能直接回答问题方面的属性（谁、什么、何时、何地、为何）
2. 同时考虑实体相关性和属性相关性
3. 当问题涉及时间时，优先选择具有清晰时间信息的属性
4. 按与问题的整体相关性排序
5. 输出恰好 top_n 个选择（如果相关属性不足则输出更少）
"""

TIME_EXTRACTION_PROMPT = """你是一个时间提取专家。分析用户查询，提取可以缩小搜索窗口的时间约束。

用户查询：{query}
当前对话时间戳（用于解析相对时间）：{current_time}

分析查询中的：
1. **明确的绝对时间提及**："2023年1月"、"2023年3月16日"、"2023年"、"2022年12月"

输出格式（JSON）：
{{
    "time_range": ["起始时间", "结束时间"] 或 null,
    "reasoning": "时间提取的简要说明"
}}

规则：
- time_range 格式：["YYYY-MM-DD HH:MM:SS", "YYYY-MM-DD HH:MM:SS"]（两端包含）
- 如果查询提到特定月份和年份，设置范围覆盖整个月
- 如果查询提到特定年份，设置范围覆盖整年
- **仅对包含明确年份、月份或具体日期的绝对时间引用生成 time_range。**
- **所有相对或模糊的时间表达一律返回 null**，包括但不限于："最近"、"近来"、"去年"、"上周"、"昨天"、"这段时间"、"前阵子"、"不久前"、"之前"、"recently"、"last year"、"last week"、"yesterday"。这些表达依赖于未知的参考时间点，无法可靠地解析。
- 不确定时返回 null。不带时间约束搜索永远比带错误时间约束搜索更安全。

示例：
查询："小明什么时候开的网店？" → time_range: null（无绝对时间）
查询："2023年3月小明做了什么？" → time_range: ["2023-03-01 00:00:00", "2023-03-31 23:59:59"]
查询："2022年12月发生了什么？" → time_range: ["2022-12-01 00:00:00", "2022-12-31 23:59:59"]
查询："小明最近在研究什么？" → time_range: null（无绝对时间）
查询："小明去年做了什么？" → time_range: null（"去年"是相对时间）
查询："昨天发生了什么？" → time_range: null（"昨天"是相对时间）
查询："她近来都在忙什么？" → time_range: null（模糊时间）
"""
