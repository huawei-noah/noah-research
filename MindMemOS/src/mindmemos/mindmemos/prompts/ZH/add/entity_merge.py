DUPLICATE_NAME_RESOLUTION_PROMPT = """
你是一名记忆实体冲突处理专家。新提取的实体与数据库中已有实体**同名**。你需要判断：是**改名**创建新实体，还是**合并**到已有实体中。

## 冲突信息

### 新实体（刚从对话中提取）
- 名称：{new_entity_name}
- 类型：{new_entity_type}
- 描述：{new_entity_description}

### 已有实体（数据库中已存在）
- 名称：{existing_entity_name}
- 类型：{existing_entity_type}
- 描述：{existing_entity_description}

## 判定规则

### ⚠️ 关键规则：Episodes 实体 —— 只能改名
**如果实体类型是 "episodes"，必须选择 "rename"。Episodes 实体代表独立的对话片段，绝对不能合并。**
- 每个 episode 都是独立的对话记录——即使话题相似，也是不同的事件
- 改名时应突出新 episode 的独特侧重点（如添加日期、具体子话题、区分性细节）

### 对于非 Episodes 实体：

**选择 "merge"（合并）当（默认选择——除非有明确证据表明不同）：**
1. 同名 + 同类型（几乎肯定是同一现实世界实体）
2. 同名 + 描述兼容或描述同一实体的不同方面
3. 新信息是对已有实体的状态更新、新事件或新侧面
4. 一个叫"小明"的人既是银行职员又喜欢跳舞——人有多个方面，应合并到同一实体

**选择 "rename"（改名）仅当：**
1. 同名但有**明确、具体的证据**表明是不同实体（如"北京的张伟"和"上海的张伟"且有不兼容的背景信息）
2. 实体类型根本不同（如同名的人物和组织）
3. 核心描述**直接矛盾**，无法调和（不是不同话题——而是不同身份）

**不确定时，优先选择 "merge"——把信息整合到一个实体比分散到多个重复实体更好。**

## 输出格式
输出 JSON 对象：

**改名**时：
```json
{{
    "action": "rename",
    "new_name": "更具体的名称，以区分已有实体",
    "reason": "简要说明"
}}
```

**合并**时：
```json
{{
    "action": "merge",
    "reason": "简要说明为什么这是同一实体"
}}
```

只输出 JSON，不要包含额外文字。
"""

DES_UPDATE_PROMPT = """
你是记忆总结专家。
将新信息融合到现有描述中，不要从头重写。

规则：
1. 保留当前描述中的所有关键事实（身份、爱好、宠物、关系、习惯、偏好）。
2. 补充最新属性中尚未覆盖的新事实。
3. 如果某项事实已变化（如职位更新），用新值替换旧值。
4. 最多10句话。优先级：身份 > 关系 > 经常性活动 > 近期事件。

输出格式：
<description>融合后的描述。</description>

实体：{entity_name}（类型：{entity_type}）
当前描述：{current_description}
需要融合的新属性：
{latest_properties}

融合后的描述：
"""

SINGLE_ENTITY_MERGE_PROMPT = """
你是一名记忆融合专家。请判断以下新提取的实体应该 CREATE（创建新实体）还是 UPDATE（更新现有实体）。

## 新实体
- 名称：{entity_name}
- 类型：{entity_type}
- 描述：{entity_description}

## 现有候选实体（向量检索结果）
{existing_entities}

## 判定标准

### 使用 UPDATE 当（优先选择——除非明确不同）：
1. 实体名称匹配或与某个候选相似（描述相同的人、物或事）
2. 实体可能与某个候选是同一现实世界实体
3. 同名带有附加上下文（如新信息中"小明"在跳舞，和已有的做银行工作的"小明"是同一人——人有多个方面）
4. target_entity 名称必须是上面候选列表中的名称

### ⚠️ 关键规则：基础名称匹配
当新实体和某个候选实体的**基础名称**（括号前的部分）相同时，它们几乎肯定是同一实体——即使括号内的修饰语不同。
- "Toby(German Shepherd)" 和 "Toby(golden retriever)" → **UPDATE**（同一只狗 Toby——品种描述不一致是数据问题，不代表是两只不同的狗）
- "Fox Hollow(远足小径)" 和 "Fox Hollow(自然保护区)" → **UPDATE**（同一个地方，不同描述）
- "Ferrari(跑车)" 和 "Ferrari(488 GTB)" → **UPDATE**（同一辆车，不同细节层级）

**原因：** 括号内的修饰语如"(golden retriever)"或"(German Shepherd)"是描述性标注，不是身份定义特征。在同一对话者之间的对话中，相同命名的实体就是同一个现实世界事物。修饰语冲突说明描述不够精确，而非指向不同实体。

**同一对话者上下文增强合并信心：** 如果新实体和候选实体都出现在涉及相同对话者的对话中（如都来自 Andrew-Audrey 的对话），这是它们指向同一现实世界实体的强证据。人们通常不会有两个同名的宠物/物品/地点。

### 使用 CREATE 仅当：
1. 候选列表中没有任何可能匹配的实体
2. 有**明确、具体的证据**表明这是不同实体（如"纽约的Jon Smith"和"东京的Jon Lee"——不同全名的明显不同的人）
3. 实体类型根本不兼容（如同名的人物和组织）
4. **仅括号内修饰语不同不构成 CREATE 的充分理由** ——需要根本不同的身份

## 输出格式（JSON 对象，不是数组）

CREATE 时：
```json
{{
    "action": "create",
    "relation_candidates": [
        {{"target_entity": "现有实体名称", "relation": "关系描述"}}
    ]
}}
```

UPDATE 时：
```json
{{
    "action": "update",
    "target_entity": "要更新的现有实体名称（必须在候选列表中）"
}}
```

## 规则
1. 只输出一个决策
2. **不确定时优先 UPDATE**——同名的人几乎总是同一个人，除非有明确证据表明不同。人有多个方面（职业、爱好、关系），这些都应该在同一个实体上。
3. 只在有**具体证据**表明是真正不同的实体时才使用 CREATE（不同全名、不同地点、不同身份）
4. **基础名称相同 = 同一实体**：如果新实体去掉括号后的名称与候选去掉括号后的名称匹配，且 entity_type 相同，必须 UPDATE。括号内修饰语不同（品种、型号、子标题）永远不构成 CREATE 的理由。
5. relation_candidates 只包含有明确关系的实体
6. 如果没有明确关系，relation_candidates 可以为空 []
7. UPDATE 时 target_entity 必须精确匹配候选列表中的名称

只输出 JSON，不要包含额外文字。
"""

MERGE_DECISION_PROMPT = """
你是一名记忆融合专家。请分析新提取信息与现有实体库之间的关系。

## 任务
1. 对每个新提取的实体，决定是 CREATE（创建新实体）还是 UPDATE（更新现有实体）。
2. 判断CREATE实体与现有实体之间是否存在逻辑关系，需要建立边连接

## 判定标准

### 使用 CREATE 当：
1. 实体在现有库中不存在（全新实体）
2. 实体名称相同但代表不同事物（例如两个不同的人都叫"张三"），同名且具体名称的两个实体通常表示相同的事物，除非有明确信息表明不同。
3. 核心属性与现有实体冲突（如同名但类型不同、核心信息矛盾）

### 使用 UPDATE 当：
1. 实体与现有某个实体**明确匹配**（描述相同的人、物或事）
2. **只有当被更新的目标实体明确出现在下方"现有实体库"列表中时，才使用 UPDATE**

## 输出格式
对每个新提取的实体，输出以下格式之一：

### CREATE 格式：
```json
{
    "action": "create",
    "entity_name": "新提取的实体名称",
    "entity_type": "实体类型",
    "relation_candidates": [
        {
            "target_entity": "现有实体名称（必须在现有实体库中）",
            "relation": "关系描述"
        }
    ]
}
```

### UPDATE 格式：
```json
{
    "action": "update",
    "target_entity": "要更新的现有实体名称（必须在现有实体库中）",
    "new_entity_name": "触发更新的新实体名称",
    "new_entity_info": "新增信息的简要摘要"
}
```

## 重要规则（必须遵守）
1. 每个新提取必须有且仅有一个对应的操作（CREATE 或 UPDATE）
2. 不能跳过任何新提取 - 必须全部映射
3. **只有当 target_entity 明确出现在"现有实体库"中时，才使用 UPDATE**
4. 如果不确定目标实体是否存在，优先使用 CREATE（安全优先）
5. relation_candidates 只包含确实存在明确关系描述的现有实体
6. 如果没有明确关系，relation_candidates 可以为空 []


## 时间处理
时间会自动处理，只需关注实体匹配判断。

现有实体库
{existing_entities}

新提取信息列表
{new_extractions}

只输出 JSON 数组，不要包含额外文字。

示例：
现有实体：王伟（ID: axbecew，type: person），陈明（ID: mendnetn，type: person）
新提取信息：李华（陈明的同事），王伟（2024年5月晋升为技术总监）
输出：
```json
[
    {
        "action": "create",
        "entity_name": "李华",
        "entity_type": "person",
        "relation_candidates": [
            {"target_entity": "陈明", "relation": "同事"}
        ]
    },
    {
        "action": "update",
        "target_entity": "王伟",
        "new_entity_name": "王伟",
        "new_entity_info": "2024年5月晋升为技术总监"
    }
]
```

错误示例（禁止这样做）：
```json
[
    {
        "action": "update",
        "target_entity": "张三",  // 错误：张三不在现有实体库中
        "new_entity_name": "张三",
        "new_entity_info": "一些信息"
    }
]
```
"""
