"""Handle episode objectify."""

EPISODE_OBJECTIFY_PROMPT = """
You are an episodic memory generation expert. Please convert the following conversation content into a structured, objective third-person narrative.

Conversation timestamp: {conversation_timestamp}
Conversation content:
{conversation_text}

Speaker note: Lines may be formatted as `speaker=Name: ...` for named-speaker dialogue. Treat `Name` as the real speaker of that line; first-person statements in that line belong to `Name`, not automatically to the user. Use explicit speaker names in the narrative whenever available.

Please generate a comprehensive, factual record of the conversation. Write it as a chronological account focusing on observable actions, direct statements, and specific details that aid keyword search. Preserve ALL information from the original — do not summarize or compress.

**STRICTLY FORBIDDEN — NO INTERPRETATION OR INFERENCE**:
- Do NOT add interpretive commentary like "this suggests...", "this implies...", "this demonstrates...", "this conveys...", "indicating that..."
- Do NOT analyze speakers' intentions, motivations, willingness, or emotional states beyond what they explicitly said
- Do NOT speculate about potential outcomes, future possibilities, or underlying meanings
- Do NOT add phrases like "showing his willingness to...", "expressing her desire to...", "implying a potential..."
- ONLY record what was explicitly said and done — if someone said "I'd love to go bowling with you", write exactly that, do NOT add "this statement indicates his interest in shared activities"
- The output should read like a court transcript or news report: pure facts, zero editorializing

IMPORTANT TIME HANDLING:
- The conversation timestamp includes the day of the week (e.g., "2023-07-15 13:51:00 (Saturday)"). Use this to calculate relative dates precisely.
- When the conversation mentions relative times (e.g., "yesterday", "last week", "last Friday"), preserve both the original relative expression AND calculate the absolute date
- Format time references as: "relative time (absolute date)" - e.g., "last Friday (July 14, 2023)"
- All absolute time calculations should be based on the provided conversation timestamp
- **CRITICAL — "last [weekday]" calculation rule**:
  - "last Friday" means the MOST RECENT Friday BEFORE the conversation date, NOT the Friday of the previous calendar week
  - Example: If conversation is on Saturday July 15, "last Friday" = July 14 (yesterday), NOT July 7
  - Example: If conversation is on Wednesday Sept 13, "last Monday" = Sept 11 (2 days ago), NOT Sept 4
  - Example: If conversation is on Thursday Feb 9, "last Wednesday" = Feb 8 (yesterday), NOT Feb 1
  - Step-by-step: (1) Note the conversation day of week, (2) Count backwards to find the nearest target weekday, (3) That is the answer
- **"last weekend" rule**: means the most recent Saturday-Sunday before the conversation date
  - Example: If conversation is on Wednesday May 24, "last weekend" = May 20-21, NOT May 13-14

Requirements:
1. Convert the dialogue format into a third-person narrative description.
2. Maintain chronological order and causal relationships.
3. Use specific names consistently rather than pronouns to avoid ambiguity in retrieval.
4. Include the conversation timestamp at the beginning to establish when this episode occurred.
5. **DO NOT COMPRESS** — LENGTH PRESERVATION IS CRITICAL:
   - The output MUST be at least as long as the original conversation, or longer
   - Do NOT summarize, condense, or shorten any factual content
   - Convert dialogue format to narrative format, but preserve ALL information density
   - Every statement, opinion, reaction, and detail from the original conversation must appear in the output
   - If someone expressed an emotion about something specific, include both the emotion and the specific thing
   - For redundant questions and filler tone: if they contain useful information, preserve it by merging into other sentences; only discard if truly zero-information
   - When in doubt, include more rather than less
6. CONTENT FILTERING — OMIT ONLY the following (nothing else):
   - Pure greetings with zero content ("Hi!", "Hey there!")
   - Pure farewells with zero content ("Bye!", "See you!")
   - Single-word acknowledgments ("OK", "Sure", "Got it")
   Do NOT omit: emotional reactions with context, questions that provide topic framing, expressions of interest or surprise about specific topics, social exchanges that reveal preferences or personality.
7. CRITICAL DETAIL PRESERVATION:
   - Person Names: Always include full names (e.g., "Caroline met with her colleague Amy" not "Caroline met with a colleague")
   - Special Nouns & Entities: Preserve all proper nouns, brand names, place names, organization names exactly as mentioned
   - Item Names: Include specific product names, book titles, movie names, restaurant names, game names
   - Quantities & Numbers: Record exact numbers, amounts, prices, percentages, dates, times, counts
   - Specific Activities: Use precise activity descriptions (e.g., "practiced hot yoga" not just "exercised")
   - Time Points: Include all specific times mentioned (e.g., "at 3:30 PM", "every Tuesday", "twice a week")
   - Named Objects & Personal Items: Preserve exact names of pets, toys, keepsakes (e.g., "stuffed animal dog named Tilly", NOT "a comfort item")
   - Descriptive Adjectives: Preserve exact adjectives (e.g., "red and purple lighting", NOT "adjustable lighting")
   - Nicknames: Preserve nicknames people use for each other (e.g., "Nate calls Joanna 'Jo'")
   - Suggestions & Recommendations: Include the exact item/title recommended
   - Pet & Animal Details: Include pet names, species, breeds
   - Locations: Include specific addresses, venue names, geographic details
8. FREQUENCY INFORMATION:
   - Record recurring activities and their frequency (e.g., "goes to yoga class every Tuesday and Thursday")
   - Include habitual actions (e.g., "usually has coffee at 8 AM before work")
9. IMAGE CAPTION PRESERVATION:
   - When a message contains image content (indicated by [Shared image: ...] or [Image context: ...]), preserve the COMPLETE original image caption exactly as provided
   - Format: Include the original caption in brackets: [Original caption: ...]
10. ALIAS PRESERVATION:
    - When different names refer to the same entity, preserve all variants using parentheses
    - For named items whose category is not obvious, annotate with type: "Labubu(a PopMart designer toy)", "Toby(golden retriever puppy)", "Catan(a strategy board game)"

When in doubt between brevity and preserving a specific detail, ALWAYS keep the detail. The output should be AT LEAST as long as the original conversation.

**OBJECTIVE DESCRIPTION:**
"""

EPISODE_OBJECTIFY_PROMPT_ZH = """
您是情节记忆提取专家。请将以下原始对话转换为结构化的、客观的第三人称叙述。

对话时间戳: {conversation_timestamp}
对话内容:
{conversation_text}

说话人说明：对话行可能使用 `speaker=Name: ...` 表示具名说话人。请将 `Name` 视为该行真实说话人；该行中的第一人称陈述属于 `Name`，不要自动归因到用户。叙述中如存在显式说话人姓名，应使用该姓名。

请生成完整、事实性的对话记录，保留所有有效信息的同时去除冗余的提问和语气。

重要的时间处理规则：
- 对话时间戳包含星期几信息（如 "2023-07-15 13:51:00 (Saturday)"），请利用此信息精确计算相对日期
- 对于相对时间提及（昨天、上周等），同时保留原始相对表达和计算出的绝对日期
- 格式："相对时间（绝对日期）" - 如："上周五（2023年7月14日）"
- 所有绝对时间计算应基于提供的对话时间戳
- **关键 — "last [星期X]" 计算规则**：
  - "last Friday" 指的是对话日期之前最近的那个周五，而不是上一个日历周的周五
  - 示例：如果对话在周六 July 15，"last Friday" = July 14（昨天），不是 July 7
  - 示例：如果对话在周三 Sept 13，"last Monday" = Sept 11（2天前），不是 Sept 4
  - 示例：如果对话在周四 Feb 9，"last Wednesday" = Feb 8（昨天），不是 Feb 1
  - 计算步骤：(1) 确认对话是星期几，(2) 向前数找到最近的目标星期，(3) 即为答案
- **"last weekend" 规则**：指对话日期之前最近的周六-周日
  - 示例：如果对话在周三 May 24，"last weekend" = May 20-21，不是 May 13-14

要求：
1. 将对话格式转换为第三人称叙述描述。
2. 保持时间顺序和因果关系。
3. 一致使用具体姓名而非代词，避免检索歧义。
4. 在开头包含对话时间戳以确立此情节发生的时间。
5. 简洁性与去冗余：
   - 去除冗余表达和啰嗦描述
   - 避免以不同方式重复相同信息
   - 消除不必要的填充词和对话标记
   - 保持句子直接和事实性
   - 目标长度与原对话相似或更短
6. 内容过滤 — 省略以下内容：
   - 纯粹的问候和告别（"嗨！"、"再见！"、"早上好"、"回头见"）
   - 纯粹的确认（"好的"、"可以"、"收到"、"谢谢"、"没问题"）
   - 没有具体内容的纯情感感叹（"哇！"、"太好了！"）
   - 在同一episode中已被回答的问题（只保留带有上下文的回答）
   - 纯粹的社交寒暄（"你好吗？"、"祝你愉快！"）
   只包含含有可检索事实、计划、带有具体细节的观点或建议的内容。
7. 关键细节保护：
   - 人名：若信息可知，则始终包含全名（如："Caroline与她的同事Amy见面"而非"Caroline与一位同事见面"）
   - 特殊实体：准确保留所有专有名词、品牌名、地名、组织名等
   - 具体物品：包含确切的产品名、书名、电影名、餐厅名、游戏名
   - 数量和数字：记录精确的数字、金额、价格、百分比、日期、时间、计数
   - 活动：使用精确的活动描述（如："练习热瑜伽"而非"锻炼"）
   - 时间点：包含所有提及的具体时间（如："下午3:30"、"每周二"、"每周两次"）
   - 命名物品与个人物品：保留宠物、玩具、纪念品的确切名称（如："名叫Tilly的毛绒狗玩偶"，而非"一个舒适物品"）
   - 描述性形容词：保留确切形容词（如："红色和紫色灯光"，而非"可调灯光"）
   - 昵称：保留人们互相使用的昵称（如："Nate叫Joanna 'Jo'"）
   - 建议与推荐：包含具体推荐的名称/标题
   - 宠物与动物细节：包含宠物名字、物种、品种
   - 位置：包含具体地址、场所名称、地理细节
8. 频率信息：
   - 记录重复性活动及其频率（如："每周二和周四上瑜伽课"）
   - 包含习惯性行为（如："通常早上8点上班前喝咖啡"）
9. 图片描述保留：
   - 当消息中包含图片内容（以 [Shared image: ...] 或 [Image context: ...] 标识）时，必须完整保留原始图片描述
   - 格式：用方括号附上原始caption：[图片原始caption: ...]
10. 别称保留：
    - 当不同名称指代同一事物时，用括号保留所有名称变体
    - 对于仅凭名字无法明确类别的命名物品，标注类型："拉布布（PopMart设计师玩偶）"、"Toby（金毛寻回犬幼犬）"、"卡坦岛（策略桌游）"

在简洁和保留具体细节之间抉择时，始终选择保留细节。

**客观化描述:**
"""
