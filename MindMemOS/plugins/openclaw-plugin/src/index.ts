import { createHash } from "node:crypto";
import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import { buildJsonPluginConfigSchema, definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import { spawnFileJson, spawnFileOk } from "./mindmemos-cli.js";

type PluginConfig = {
  enabled: boolean;
  cli: string;
  topK: number;
  addMode: "sync" | "async";
  userId?: string;
  appId: string;
  sessionId?: string;
  minQueryLength: number;
  maxConversationMessages: number;
};

type MemoryHit = {
  id?: string;
  memory?: string;
  last_update_at?: string | null;
  event_time?: string | null;
  source_timestamp?: string | null;
};

type MemorySearchResult = {
  request_id?: string | null;
  memories?: MemoryHit[];
};

type MemoryMessage = {
  role: "user" | "assistant" | "system" | "tool";
  content: string;
  timestamp: number;
};

type SkillContext = {
  name: string;
  content_hash: string;
  base_version_id: string;
  version_label?: string;
  usage?: "injected" | "modified";
};

const MEMORY_CONTEXT_OPEN = "<relevant-memories>";
const MEMORY_CONTEXT_CLOSE = "</relevant-memories>";

const DEFAULT_CONFIG: PluginConfig = {
  enabled: true,
  cli: "mindmemos",
  topK: 5,
  addMode: "async",
  appId: "openclaw",
  minQueryLength: 2,
  maxConversationMessages: 80,
};

const CONFIG_SCHEMA = {
  type: "object",
  additionalProperties: false,
  properties: {
    enabled: {
      type: "boolean",
    },
    cli: {
      type: "string",
    },
    topK: {
      type: "integer",
      minimum: 1,
    },
    addMode: {
      type: "string",
      enum: ["sync", "async"],
    },
    userId: {
      type: "string",
    },
    appId: {
      type: "string",
    },
    sessionId: {
      type: "string",
    },
    minQueryLength: {
      type: "integer",
      minimum: 1,
    },
    maxConversationMessages: {
      type: "integer",
      minimum: 1,
    },
  },
} as const;

const entry: ReturnType<typeof definePluginEntry> = definePluginEntry({
  id: "mindmemos-memory",
  name: "MindMemOS Memory",
  description:
    "Searches MindMemOS memories before each prompt and stores completed OpenClaw conversations through the mindmemos CLI.",
  configSchema: buildJsonPluginConfigSchema(CONFIG_SCHEMA),
  register,
});

export default entry;

function register(api: OpenClawPluginApi): void {
  const config = loadConfig(api.pluginConfig ?? {});
  if (!config.enabled) {
    api.logger.info("[mindmemos] plugin disabled by config");
    return;
  }

  api.on("before_prompt_build", async (event: unknown, ctx: unknown) => {
    const query = extractQuery(event);
    const sessionId = resolveSessionId(config, event, ctx);
    if (!query || query.trim().length < config.minQueryLength) {
      return {};
    }

    try {
      const result = await searchMemories(config, query, sessionId);
      const context = formatMemoryContext(result.memories ?? [], config.userId, isSubagentSession(sessionId));
      if (!context) {
        return {};
      }
      api.logger.info(`[mindmemos] recall hit ${result.memories?.length ?? 0} memories, injected ${context.length} chars`);
      return { prependContext: context };
    } catch (error) {
      api.logger.warn(`[mindmemos] memory search failed: ${errorMessage(error)}`);
      return {};
    }
  });

  // Stateless capture: on each completed turn we store only the latest turn
  // (the last user message and the assistant replies that follow it). Earlier
  // turns were already stored by their own agent_end, so nothing is re-sent and
  // there is no client-side buffer to keep in sync. Deduplication of repeated
  // facts is handled server-side by the mindmemos add pipeline.
  api.on("agent_end", async (event: unknown, ctx: unknown) => {
    if (!isSuccessfulRun(event)) {
      return;
    }
    const messages = pickLastTurnMessages(event, config.maxConversationMessages);
    if (messages.length === 0) {
      return;
    }
    const sessionId = resolveSessionId(config, event, ctx, messages);

    try {
      await addConversation(config, messages, sessionId);
      api.logger.info(`[mindmemos] stored ${messages.length} message(s) from last turn (session_id=${sessionId})`);
    } catch (error) {
      api.logger.warn(`[mindmemos] memory add failed: ${errorMessage(error)}`);
    }
  });

  api.logger.info("[mindmemos] plugin loaded");
}

function loadConfig(raw: unknown): PluginConfig {
  const obj = isRecord(raw) ? raw : {};
  const addMode = obj.addMode === "sync" ? "sync" : "async";
  return {
    enabled: obj.enabled !== false,
    cli: typeof obj.cli === "string" && obj.cli.trim() ? obj.cli.trim() : DEFAULT_CONFIG.cli,
    topK: positiveInteger(obj.topK, DEFAULT_CONFIG.topK),
    addMode,
    appId: typeof obj.appId === "string" && obj.appId.trim() ? obj.appId.trim() : DEFAULT_CONFIG.appId,
    sessionId: typeof obj.sessionId === "string" && obj.sessionId.trim() ? obj.sessionId.trim() : undefined,
    userId: typeof obj.userId === "string" && obj.userId.trim() ? obj.userId.trim() : undefined,
    minQueryLength: positiveInteger(obj.minQueryLength, DEFAULT_CONFIG.minQueryLength),
    maxConversationMessages: positiveInteger(
      obj.maxConversationMessages,
      DEFAULT_CONFIG.maxConversationMessages,
    ),
  };
}

async function searchMemories(config: PluginConfig, query: string, sessionId: string): Promise<MemorySearchResult> {
  const args = ["memory", "search", query, "--top-k", String(config.topK), "--json"];
  args.push("--app-id", config.appId, "--session-id", sessionId);
  if (config.userId) {
    args.push("--user-id", config.userId);
  }
  return spawnFileJson<MemorySearchResult>(config.cli, args);
}

async function addConversation(config: PluginConfig, messages: MemoryMessage[], sessionId: string): Promise<void> {
  const args = ["memory", "add", "--messages-json-file", "-", "--json"];
  args.push("--app-id", config.appId, "--session-id", sessionId);
  if (config.addMode === "async") {
    args.push("--async");
  }
  if (config.userId) {
    args.push("--user-id", config.userId);
  }
  const skillContext = detectSkillContext(messages);
  if (skillContext.length > 0) {
    args.push("--skill-context-json", JSON.stringify(skillContext));
  }
  args.push("--metadata-json", JSON.stringify({ source: "openclaw-plugin" }));
  await spawnFileOk(config.cli, args, `${JSON.stringify(messages)}\n`);
}

function detectSkillContext(messages: MemoryMessage[]): SkillContext[] {
  const candidates = new Map<string, { path: string; content: string; usage: "injected" | "modified" }>();
  for (let i = 0; i < messages.length; i += 1) {
    const message = messages[i];
    if (message.role !== "assistant") {
      continue;
    }
    const call = parseToolCall(message.content);
    if (!call) {
      continue;
    }
    const path = toolArgPath(call.args);
    if (!path || !isSkillMdPath(path)) {
      continue;
    }
    const key = skillDirKey(path);
    if (call.tool === "read") {
      const content = messages[i + 1]?.role === "tool" ? messages[i + 1].content : "";
      if (content) {
        candidates.set(key, { path, content, usage: strongestUsage(candidates.get(key)?.usage, "injected") });
      }
    } else if (call.tool === "write" || call.tool === "edit") {
      const content = call.tool === "edit" ? editContent(call.args) : toolArgText(call.args, "content");
      if (content) {
        candidates.set(key, { path, content, usage: "modified" });
      }
    }
  }
  return [...candidates.values()].map((candidate) => {
    const metadata = parseSkillMetadata(candidate.content);
    return {
      name: metadata.name || skillNameFromPath(candidate.path),
      content_hash: computeSkillContentHash(candidate.content),
      base_version_id: "",
      ...(metadata.version ? { version_label: metadata.version } : {}),
      usage: candidate.usage,
    };
  });
}

function parseToolCall(content: string): { tool: string; args: Record<string, unknown> & { path?: string } } | null {
  const match = content.match(/^\s*\[tool_call\]\s*([A-Za-z0-9_.-]+)\((.*)\)\s*$/s);
  if (!match) {
    return null;
  }
  let parsed: unknown;
  try {
    parsed = match[2].trim() ? JSON.parse(match[2]) : {};
  } catch {
    return null;
  }
  if (!isRecord(parsed)) {
    return null;
  }
  const path = typeof parsed.path === "string" ? parsed.path : "";
  return { tool: match[1].toLowerCase(), args: { ...parsed, path } };
}

function toolArgPath(args: Record<string, unknown>): string {
  for (const key of ["path", "file_path", "filepath"]) {
    const value = args[key];
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
  }
  return "";
}

function toolArgText(args: Record<string, unknown>, key: string): string {
  const value = args[key];
  return typeof value === "string" ? value : "";
}

function editContent(args: Record<string, unknown>): string {
  for (const key of ["content", "new_content", "replacement", "replace"]) {
    const value = toolArgText(args, key);
    if (value) {
      return value;
    }
  }
  return "";
}

function isSkillMdPath(path: unknown): path is string {
  return typeof path === "string" && /(^|[/\\])SKILL\.md$/.test(path);
}

function skillDirKey(path: string): string {
  return path.replace(/\\/g, "/").replace(/\/SKILL\.md$/, "");
}

function skillNameFromPath(path: string): string {
  const parts = skillDirKey(path).split("/").filter(Boolean);
  return parts[parts.length - 1] || "skill";
}

function strongestUsage(
  current: "injected" | "modified" | undefined,
  next: "injected" | "modified",
): "injected" | "modified" {
  return current === "modified" || next === "modified" ? "modified" : "injected";
}

function parseSkillMetadata(content: string): { name?: string; version?: string } {
  return {
    name: simpleFrontmatterField(content, "name"),
    version: simpleFrontmatterField(content, "version"),
  };
}

function simpleFrontmatterField(content: string, field: string): string | undefined {
  const match = content.match(new RegExp(`^\\s*${field}\\s*:\\s*["']?([^"'\\n#]+)`, "m"));
  return match?.[1]?.trim() || undefined;
}

function computeSkillContentHash(content: string): string {
  const normalized = content.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  const canonical = JSON.stringify([{ content: normalized, path: "SKILL.md" }]);
  return createHash("sha256").update(canonical).digest("hex");
}

function formatMemoryContext(memories: MemoryHit[], userId: string | undefined, isSubagent: boolean): string {
  const lines = memories
    .map((hit, index) => {
      const text = typeof hit.memory === "string" ? hit.memory.trim() : "";
      if (!text) {
        return null;
      }
      const label = hit.id ? `${index + 1}. [${hit.id}]` : `${index + 1}.`;
      const when = hit.last_update_at ?? hit.event_time ?? hit.source_timestamp;
      return when ? `${label} ${text} (${when})` : `${label} ${text}`;
    })
    .filter((line): line is string => line !== null);

  if (lines.length === 0) {
    return "";
  }
  const preamble = buildPreamble(userId, isSubagent);
  return [MEMORY_CONTEXT_OPEN, preamble, ...lines, MEMORY_CONTEXT_CLOSE].join("\n");
}

/**
 * One-line instruction placed before the recalled memories. For a subagent the
 * memories are someone else's; we frame them as background only and tell the
 * model not to take on that user's identity.
 */
function buildPreamble(userId: string | undefined, isSubagent: boolean): string {
  if (isSubagent) {
    const who = userId ? `user "${userId}"` : "the user";
    return `The following stored memories for ${who} are provided as background context only. You are a subagent — use them for context but do not assume you are this user.`;
  }
  return userId ? `Relevant memories for ${userId}:` : "Relevant memories:";
}

/** Subagent sessions carry a `:subagent:` segment in their session key. */
function isSubagentSession(sessionId: string): boolean {
  return /:subagent:/i.test(sessionId);
}

/**
 * The recall query for the current turn. In `before_prompt_build` the live user
 * input lives on `event.prompt`; `event.messages` only holds prior session
 * history (empty on the first turn, stale afterwards). Prefer `prompt` and fall
 * back to scanning messages only when it is absent.
 */
function extractQuery(event: unknown): string {
  const prompt = getPath(event, ["prompt"]);
  if (typeof prompt === "string" && prompt.trim()) {
    return prompt.trim();
  }
  return extractLatestUserText(event);
}

function extractLatestUserText(event: unknown): string {
  const messages = getMessagesArray(event);
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const message = normalizeMessage(messages[i]);
    if (message?.role === "user" && !isInternalMemoryMessage(message)) {
      return message.content;
    }
  }

  const directMessages = [
    getPath(event, ["message"]),
    getPath(event, ["latestMessage"]),
    getPath(event, ["turn", "message"]),
    getPath(event, ["request", "message"]),
  ];
  for (const candidate of directMessages) {
    const message = normalizeMessage(candidate);
    if (message?.role === "user" && !isInternalMemoryMessage(message)) {
      return message.content;
    }
  }

  return "";
}

/** Whether the finished run succeeded; missing flag is treated as success. */
function isSuccessfulRun(event: unknown): boolean {
  return getPath(event, ["success"]) !== false;
}

/**
 * Select only the latest turn: the last user message and every assistant/tool
 * message that follows it. Returns [] when there is no user message. `maxMessages`
 * is a safety cap (a single turn is normally well under it).
 */
function pickLastTurnMessages(event: unknown, maxMessages: number): MemoryMessage[] {
  const all = getMessagesArray(event)
    .map((message) => normalizeMessage(message))
    .filter((message): message is MemoryMessage => message !== null && !isInternalMemoryMessage(message));

  let lastUserIndex = -1;
  for (let i = all.length - 1; i >= 0; i -= 1) {
    if (all[i].role === "user") {
      lastUserIndex = i;
      break;
    }
  }
  if (lastUserIndex === -1) {
    return [];
  }

  const turn = all.slice(lastUserIndex);
  if (turn.length <= maxMessages) {
    return turn;
  }
  return turn.slice(turn.length - maxMessages);
}

function resolveSessionId(
  config: PluginConfig,
  event: unknown,
  ctx: unknown,
  messages: MemoryMessage[] = [],
): string {
  if (config.sessionId) {
    return sanitizeSessionId(config.sessionId);
  }

  // The agent_end event payload does not carry a session id; the canonical
  // session key lives on the hook context (`ctx.sessionKey`). Prefer it so all
  // turns of one conversation share a session_id on the mindmemos side.
  const candidates = [
    getPath(ctx, ["sessionKey"]),
    getPath(ctx, ["sessionId"]),
    getPath(ctx, ["session_id"]),
    getPath(ctx, ["session", "id"]),
    getPath(event, ["sessionId"]),
    getPath(event, ["session_id"]),
    getPath(event, ["session", "id"]),
    getPath(event, ["session", "sessionId"]),
    getPath(event, ["session", "session_id"]),
    getPath(event, ["conversationId"]),
    getPath(event, ["conversation_id"]),
    getPath(event, ["conversation", "id"]),
    getPath(event, ["conversation", "conversationId"]),
    getPath(event, ["threadId"]),
    getPath(event, ["thread_id"]),
    getPath(event, ["thread", "id"]),
    getPath(event, ["request", "sessionId"]),
    getPath(event, ["request", "session_id"]),
  ];
  for (const candidate of candidates) {
    if (typeof candidate === "string" && candidate.trim()) {
      return sanitizeSessionId(candidate.trim());
    }
    if (typeof candidate === "number" && Number.isFinite(candidate)) {
      return String(candidate);
    }
  }

  // Fallback: derive a stable, NUL-free id from the first message. The raw
  // signature contains NUL separators, so hash it — passing NUL bytes as a
  // CLI argument makes child_process.spawn throw and the add silently fail.
  if (messages.length > 0) {
    const digest = createHash("sha1").update(messageSignature(messages[0])).digest("hex");
    return `openclaw:${digest}`;
  }
  return "openclaw:default";
}

/** Strip control characters (notably NUL) so the value is safe as a CLI argument. */
function sanitizeSessionId(value: string): string {
  // eslint-disable-next-line no-control-regex
  const cleaned = value.replace(/[\u0000-\u001f\u007f]/g, "");
  return cleaned || "openclaw:default";
}

function getMessagesArray(event: unknown): unknown[] {
  const candidates = [
    getPath(event, ["messages"]),
    getPath(event, ["conversation", "messages"]),
    getPath(event, ["session", "messages"]),
    getPath(event, ["finalMessages"]),
    getPath(event, ["transcript"]),
  ];
  for (const candidate of candidates) {
    if (Array.isArray(candidate)) {
      return candidate;
    }
  }
  return [];
}

function normalizeMessage(raw: unknown): MemoryMessage | null {
  if (!isRecord(raw)) {
    return null;
  }

  const role = normalizeRole(raw.role ?? raw.type ?? raw.sender);
  const content = textFromUnknown(raw.content ?? raw.text ?? raw.message);
  if (!role || !content) {
    return null;
  }

  return {
    role,
    content,
    timestamp: timestampMillis(raw.timestamp ?? raw.createdAt ?? raw.time),
  };
}

function messageSignature(message: MemoryMessage): string {
  return `${message.timestamp}\u0000${message.role}\u0000${message.content}`;
}

function isInternalMemoryMessage(message: MemoryMessage): boolean {
  const text = message.content.trim();
  if (!text) {
    return true;
  }
  if (text === "HEARTBEAT_OK" || text === "[OpenClaw heartbeat poll]") {
    return true;
  }
  return (
    text.includes("An async command completion event was triggered") ||
    text.includes("user delivery is disabled for this run") ||
    text.includes("reply HEARTBEAT_OK only") ||
    text.includes("Do not mention, summarize, or reuse command output")
  );
}

function normalizeRole(raw: unknown): MemoryMessage["role"] | null {
  if (typeof raw !== "string") {
    return null;
  }
  const role = raw.toLowerCase();
  if (role === "human") {
    return "user";
  }
  if (role === "ai" || role === "model") {
    return "assistant";
  }
  // OpenClaw tool result messages carry the role "toolResult"; map them to the
  // generic "tool" role so they are persisted as tool replies.
  if (role === "toolresult" || role === "tool_result") {
    return "tool";
  }
  if (role === "user" || role === "assistant" || role === "system" || role === "tool") {
    return role;
  }
  return null;
}

function textFromUnknown(value: unknown): string {
  if (typeof value === "string") {
    return value.trim();
  }
  if (Array.isArray(value)) {
    return value.map(textFromUnknown).filter(Boolean).join("\n").trim();
  }
  if (isRecord(value)) {
    const toolCall = textFromToolCall(value);
    if (toolCall) {
      return toolCall;
    }
    return textFromUnknown(value.text ?? value.content ?? value.value);
  }
  return "";
}

/**
 * Render an OpenClaw assistant tool-call content block as text. The block looks
 * like `{ type: "toolCall", name, arguments }` and carries no text/content
 * field, so without this it would serialize to an empty string and the call
 * would be lost. The tool call stays on the assistant message; we emit a single
 * line `[tool_call] name(<json args>)`.
 */
function textFromToolCall(value: Record<string, unknown>): string {
  const isToolCall = value.type === "toolCall" || (typeof value.name === "string" && "arguments" in value);
  if (!isToolCall || typeof value.name !== "string") {
    return "";
  }
  const name = value.name.trim();
  if (!name) {
    return "";
  }
  let args = "";
  if (value.arguments !== undefined && value.arguments !== null) {
    try {
      args = typeof value.arguments === "string" ? value.arguments : JSON.stringify(value.arguments);
    } catch {
      args = "";
    }
  }
  return `[tool_call] ${name}(${args})`;
}

function timestampMillis(value: unknown): number {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value > 10_000_000_000 ? Math.round(value) : Math.round(value * 1000);
  }
  if (typeof value === "string" && value.trim()) {
    const parsed = Date.parse(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return Date.now();
}

function getPath(value: unknown, path: string[]): unknown {
  let cursor = value;
  for (const key of path) {
    if (!isRecord(cursor)) {
      return undefined;
    }
    cursor = cursor[key];
  }
  return cursor;
}

function positiveInteger(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isInteger(value) && value > 0 ? value : fallback;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
