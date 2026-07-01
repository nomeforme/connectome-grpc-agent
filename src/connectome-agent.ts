/**
 * ConnectomeAgent — central class wrapping pi-agent-core's Agent inside the VEIL system.
 *
 * Implements connectome's AgentInterface behavioral methods for backward compatibility
 * with AgentComponent, while exposing pi-agent's full capabilities (tool loop, steering,
 * follow-up, streaming events).
 *
 * This is a composition of three pieces:
 *   - pi-agent Agent (LLM loop, tool execution, streaming)
 *   - VEILContextAdapter (VEIL state -> AgentMessage[], system prompt)
 *   - VEILToolBridge (VEIL facet actions -> AgentTool[])
 */

import { Agent } from '@mariozechner/pi-agent-core';
import { streamSimple } from '@mariozechner/pi-ai';
import type { Message as PiMessage, Context as PiContext } from '@mariozechner/pi-ai';
import type {
  AgentMessage,
  AgentEvent,
  AgentTool,
  AgentContext,
  ConnectomeAgentConfig,
  ConnectomeAgentPoolConfig,
  ConnectomeCycleResult,
  OutgoingVEILOperation,
  AgentCommand,
} from './types.js';
import { VEILContextAdapter } from './veil-context-adapter.js';
import { VEILToolBridge, toolHandlerToAgentTool } from './veil-tool-bridge.js';
import {
  loadSkillsFromPaths,
  formatSkillsForPrompt,
  getSkillContent,
  type Skill,
} from './skill-loader.js';
import { initRlmState, resetRlmStateForCycle } from './rlm/state.js';
import { createRlmQueryTool, createRlmCheckJobTool, createRlmCostTool } from './rlm/tools.js';
import { buildRlmSystemPromptFragment } from './rlm/system-prompt.js';
import type { RlmState } from './rlm/types.js';
import { PiAuthProvider } from './pi-auth-provider.js';
import { wrapAnthropicWithRefusalCapture } from './anthropic-refusal-capture.js';

/**
 * Behavioral state for the agent (sleeping, ignoring sources, etc.)
 * Mirrors connectome-ts AgentState but kept as a local concern.
 */
interface ConnectomeAgentBehaviorState {
  sleeping: boolean;
  ignoringSources: Set<string>;
  attentionThreshold: number;
}

/** Per-stream pi-agent pool entry. */
interface PiAgentEntry {
  agent: Agent;
  lastUsedAt: number;
}

/** Stream key used when no streamRef is provided (e.g. legacy runPiCycle calls). */
const DEFAULT_STREAM_KEY = '__default__';

const DEFAULT_POOL_CONFIG: Required<ConnectomeAgentPoolConfig> = {
  idleTtlMs: 10 * 60 * 1000, // 10 minutes
  maxStreams: 50,
  sweepIntervalMs: 60 * 1000, // 60s
};

export class ConnectomeAgent {
  /** Per-stream pi-agent instances. The pi-agent's `_state.isStreaming`,
   *  `_state.messages`, abortController, and listeners are per-stream so
   *  cross-stream cycles don't collide. */
  private piAgents: Map<string, PiAgentEntry> = new Map();
  /** Resolved pi-agent constructor opts — used by lazy spawn. */
  private piAgentInitialState: { model: any; thinkingLevel: any; systemPrompt: string };
  private piAgentStreamFn: ((model: any, context: any, options?: any) => any) | undefined;
  private piAgentGetApiKey: ((provider: string) => Promise<string | undefined> | string | undefined) | undefined;
  /** Pool tuning + lifecycle. */
  private poolConfig: Required<ConnectomeAgentPoolConfig>;
  private poolSweepInterval?: ReturnType<typeof setInterval>;
  /** Set when pool is being torn down — short-circuits new spawns. */
  private disposed = false;

  private contextAdapter: VEILContextAdapter;
  private toolBridge: VEILToolBridge;
  private config: ConnectomeAgentConfig;
  private agentState: ConnectomeAgentBehaviorState;
  private agentId: string;
  private loadedSkills: Skill[] = [];
  private skillPromptFragment: string = '';
  private convertedHandlerTools: AgentTool[] = [];
  private rlmState: RlmState | null = null;
  private rlmTools: AgentTool[] = [];
  private rlmPromptFragment: string = '';
  private _maxOutputTokens: number | undefined;

  constructor(config: ConnectomeAgentConfig) {
    this.config = config;
    this.agentId = this.createAgentId(config.name);
    this._maxOutputTokens = config.maxOutputTokens;

    // Initialize behavioral state
    this.agentState = {
      sleeping: false,
      ignoringSources: new Set(),
      attentionThreshold: 0.5,
    };

    // Build stream function — wrap to inject config overrides (always wrap so maxOutputTokens can be changed at runtime)
    const baseFn = config.streamFn ?? streamSimple;
    const needsCacheOverride = config.promptCaching === false;
    const regionOverride = config.awsRegion;
    const self = this;
    let streamFn: typeof baseFn | undefined = (model: any, context: any, options?: any) => {
      const overrides: Record<string, any> = {};
      if (needsCacheOverride) overrides.cacheRetention = 'none';
      if (typeof self._maxOutputTokens === 'number') overrides.maxTokens = self._maxOutputTokens;
      // Vercel AI Gateway: pi-ai's env-key map for "vercel-ai-gateway" reads
      // AI_GATEWAY_API_KEY. We want a clearer name (VERCEL_AI_GATEWAY_API_KEY)
      // so the credential is identifiable in env dumps. Inject it as
      // options.apiKey — pi-ai checks options.apiKey BEFORE falling back to
      // the env-key map.
      if (model?.provider === 'vercel-ai-gateway' && !options?.apiKey) {
        const key = process.env.VERCEL_AI_GATEWAY_API_KEY;
        if (key) overrides.apiKey = key;
      }
      // Local self-hosted OpenAI-compatible endpoints (llama-server / LM Studio /
      // vLLM over Tailscale) don't authenticate, but pi-ai's openai-completions
      // provider requires a non-empty key. Inject a throwaway one — override with
      // LOCAL_LLM_API_KEY for servers started with --api-key.
      if (model?.provider === 'local-llm' && !options?.apiKey) {
        overrides.apiKey = process.env.LOCAL_LLM_API_KEY || 'sk-local';
      }
      return baseFn(model, context, { ...options, ...overrides });
    };

    // Initialize pi auth provider (reads ~/.pi/agent/auth.json for OAuth tokens).
    // Falls through to ANTHROPIC_API_KEY env var if no auth.json exists.
    // Skip OAuth when useApiKey is set (for models not on Claude subscription).
    const skipOAuth = config.useApiKey || !!config.getApiKey;
    const authProvider = skipOAuth ? undefined : new PiAuthProvider();
    const resolvedGetApiKey = config.getApiKey ?? authProvider?.getApiKey;

    if (regionOverride) {
      console.log(`[ConnectomeAgent:${config.name}] AWS region override: ${regionOverride}`);
      process.env.AWS_REGION = regionOverride;
    }

    if (config.useApiKey) {
      console.log(`[ConnectomeAgent:${config.name}] Using API key auth (useApiKey=true)`);
    } else if (authProvider?.hasCredentials('anthropic')) {
      console.log(`[ConnectomeAgent:${config.name}] Using pi OAuth auth (Claude subscription)`);
    } else if (!config.getApiKey) {
      console.log(`[ConnectomeAgent:${config.name}] No pi auth.json found — using ANTHROPIC_API_KEY env var`);
    }

    // Capture pi-agent recipe (used by lazy per-stream spawn). We do NOT
    // construct a default pi-agent up front — the first runWithContext call
    // materialises one for its streamId.
    this.piAgentInitialState = {
      model: config.model,
      thinkingLevel: config.thinkingLevel ?? 'off',
      systemPrompt: config.systemPrompt ?? '',
    };
    // Outer wrap: enrich pi-ai's stripped "An unknown error occurred" (which
    // is what a mid-stream classifier trip surfaces as) with Anthropic's
    // real stop_details.category + explanation via a non-streaming replay.
    // No-op for non-anthropic providers.
    this.piAgentStreamFn = wrapAnthropicWithRefusalCapture(streamFn as any, resolvedGetApiKey) as any;
    this.piAgentGetApiKey = resolvedGetApiKey;

    // Pool config + idle sweep
    this.poolConfig = { ...DEFAULT_POOL_CONFIG, ...(config.agentPool ?? {}) };
    this.startPoolSweep();

    // Initialize the VEIL adapters
    this.contextAdapter = new VEILContextAdapter({
      agentId: this.agentId,
      agentName: config.name ?? this.agentId,
      systemPrompt: config.systemPrompt ?? '',
      contextTokenBudget: config.contextTokenBudget,
    });

    this.toolBridge = new VEILToolBridge();

    // Load skills from configured paths
    if (config.skillPaths && config.skillPaths.length > 0) {
      this.loadSkillsInternal(config.skillPaths);
    }

    // Convert ToolHandler[] to AgentTool[]
    if (config.toolHandlers && config.toolHandlers.length > 0) {
      this.convertedHandlerTools = config.toolHandlers.map(toolHandlerToAgentTool);
    }

    // Initialize RLM (recursive sub-agent) if configured.
    // RlmState remains shared across streams — concurrent RLM-using cycles
    // will share counters. This is a known limitation; per-stream RlmState
    // is a follow-up if it becomes a practical issue.
    if (config.rlm) {
      this.rlmState = initRlmState(config.rlm);

      // Wire native execution: child agents use the same stream function,
      // model, and tools as the parent (tools evaluated lazily each call).
      this.rlmState.streamFn = streamFn;
      this.rlmState.parentModel = config.model;
      this.rlmState.getApiKey = this.piAgentGetApiKey;
      this.rlmState.getParentTools = () => {
        const veilTools = this.toolBridge.getAllTools();
        const extraTools = this.config.extraTools ?? [];
        return [...veilTools, ...this.convertedHandlerTools, ...this.rlmTools, ...extraTools];
      };

      this.rlmTools = [
        createRlmQueryTool(config.rlm, this.rlmState),
        createRlmCheckJobTool(config.rlm, this.rlmState),
        createRlmCostTool(config.rlm, this.rlmState),
      ];
      this.rlmPromptFragment = buildRlmSystemPromptFragment(config.rlm, this.rlmState);
    }
  }

  // ---------------------------------------------------------------------------
  // Per-stream pi-agent pool
  // ---------------------------------------------------------------------------

  /**
   * Get or lazily create the pi-agent for a given stream. Each stream gets its
   * own pi-agent instance so concurrent activations on different streams don't
   * race on pi-agent's shared `_state.isStreaming` / messages / abort controller.
   *
   * Updates the entry's lastUsedAt for LRU tracking. Triggers an LRU eviction
   * if the cap is reached.
   */
  private getOrCreatePiAgent(streamId: string): Agent {
    const key = streamId || DEFAULT_STREAM_KEY;
    const existing = this.piAgents.get(key);
    if (existing) {
      existing.lastUsedAt = Date.now();
      return existing.agent;
    }

    // At cap — evict LRU non-busy entry to make room
    if (this.piAgents.size >= this.poolConfig.maxStreams) {
      this.evictLru();
    }

    const agent = new Agent({
      initialState: { ...this.piAgentInitialState },
      streamFn: this.piAgentStreamFn,
      getApiKey: this.piAgentGetApiKey,
    });
    this.piAgents.set(key, { agent, lastUsedAt: Date.now() });
    console.log(
      `[ConnectomeAgent:${this.name}] pi-agent spawn streamId=${key} (active=${this.piAgents.size})`,
    );
    return agent;
  }

  /** Returns the pi-agent for a stream if it exists — never lazily creates. */
  private peekPiAgent(streamId: string): Agent | undefined {
    return this.piAgents.get(streamId || DEFAULT_STREAM_KEY)?.agent;
  }

  /** True if the pi-agent for this stream is mid-cycle. */
  private isPiAgentBusy(entry: PiAgentEntry): boolean {
    return !!(entry.agent as any).state?.isStreaming;
  }

  /** Evict the least-recently-used non-busy entry. If all are busy, no-op
   *  (the new entry is still allowed — correctness wins over the cap). */
  private evictLru(): void {
    let oldestKey: string | undefined;
    let oldestTime = Infinity;
    for (const [key, entry] of this.piAgents) {
      if (this.isPiAgentBusy(entry)) continue;
      if (entry.lastUsedAt < oldestTime) {
        oldestTime = entry.lastUsedAt;
        oldestKey = key;
      }
    }
    if (oldestKey) {
      this.disposePiAgentEntry(oldestKey, 'lru-cap');
    } else {
      console.warn(
        `[ConnectomeAgent:${this.name}] pool over cap (${this.piAgents.size}) — all streams busy, allowing growth`,
      );
    }
  }

  /** Sweep idle entries past the TTL. */
  private sweepIdle(): void {
    if (this.disposed) return;
    const now = Date.now();
    const ttl = this.poolConfig.idleTtlMs;
    for (const [key, entry] of this.piAgents) {
      if (this.isPiAgentBusy(entry)) continue;
      if (now - entry.lastUsedAt > ttl) {
        this.disposePiAgentEntry(key, 'idle');
      }
    }
  }

  private startPoolSweep(): void {
    if (this.poolSweepInterval) clearInterval(this.poolSweepInterval);
    this.poolSweepInterval = setInterval(
      () => this.sweepIdle(),
      this.poolConfig.sweepIntervalMs,
    );
    // Don't keep the process alive for the sweep interval
    if (typeof (this.poolSweepInterval as any).unref === 'function') {
      (this.poolSweepInterval as any).unref();
    }
  }

  /** Dispose a single per-stream pi-agent (abort, drop). */
  private disposePiAgentEntry(streamKey: string, reason: string): void {
    const entry = this.piAgents.get(streamKey);
    if (!entry) return;
    try {
      entry.agent.abort();
      (entry.agent as any).reset?.();
    } catch { /* ignore */ }
    this.piAgents.delete(streamKey);
    console.log(
      `[ConnectomeAgent:${this.name}] pi-agent evict streamId=${streamKey} reason=${reason} (active=${this.piAgents.size})`,
    );
  }

  /** Public: dispose a specific stream's pi-agent. */
  disposeStream(streamId: string): void {
    this.disposePiAgentEntry(streamId || DEFAULT_STREAM_KEY, 'manual');
  }

  /** Reset a stream's pi-agent state — clears stuck "isStreaming" after a
   *  failed cycle so subsequent activations on the same stream don't see
   *  a wedged agent. Safe no-op if no pi-agent exists for the stream. */
  resetStream(streamId: string): void {
    const agent = this.peekPiAgent(streamId);
    if (!agent) return;
    try {
      agent.abort();
      (agent as any).reset?.();
    } catch { /* ignore */ }
  }

  /** Dispose all per-stream pi-agents and stop the sweep. Called on bot shutdown. */
  dispose(): void {
    this.disposed = true;
    if (this.poolSweepInterval) {
      clearInterval(this.poolSweepInterval);
      this.poolSweepInterval = undefined;
    }
    const keys = Array.from(this.piAgents.keys());
    for (const key of keys) this.disposePiAgentEntry(key, 'shutdown');
  }

  /** List currently held per-stream pi-agent keys (observability). */
  getActiveStreams(): string[] {
    return Array.from(this.piAgents.keys());
  }

  // ---------------------------------------------------------------------------
  // Identity
  // ---------------------------------------------------------------------------

  /** Agent ID (from config.name, slugified, or generated) */
  get id(): string {
    return this.agentId;
  }

  /** Agent display name */
  get name(): string {
    return this.config.name ?? this.agentId;
  }

  // ---------------------------------------------------------------------------
  // Runtime config
  // ---------------------------------------------------------------------------

  /** Get current max output tokens override (undefined = model default) */
  getMaxOutputTokens(): number | undefined {
    return this._maxOutputTokens;
  }

  /** Set max output tokens at runtime. Pass undefined to reset to model default. */
  setMaxOutputTokens(value: number | undefined): void {
    this._maxOutputTokens = value;
    console.log(`[ConnectomeAgent:${this.name}] maxOutputTokens set to ${value ?? 'model default'}`);
  }

  // ---------------------------------------------------------------------------
  // Primary API — runPiCycle
  // ---------------------------------------------------------------------------

  /**
   * Run a full pi-agent cycle from VEIL state.
   *
   * This is the PRIMARY method. It replaces both BasicAgent.runCycle and the
   * old ToolLoopAgent.runCycle with pi-agent's built-in tool loop.
   *
   * Flow:
   * 1. Convert VEIL state to AgentMessage[] via contextAdapter
   * 2. Build system prompt from VEIL state via contextAdapter
   * 3. Configure pi-agent (system prompt, messages, tools, model, thinking)
   * 4. Prompt pi-agent with the latest user message
   * 5. Wait for completion (pi-agent runs its full tool loop internally)
   * 6. Extract new messages by comparing before/after message counts
   * 7. Convert new assistant output to VEIL operations via contextAdapter
   * 8. Return ConnectomeCycleResult
   */
  async runPiCycle(
    veilState: any, // VEILStateManager or VEILState-like object
    streamRef?: { streamId: string; streamType?: string },
  ): Promise<ConnectomeCycleResult> {
    // Reset RLM per-activation counters (timeout + call count)
    if (this.rlmState) resetRlmStateForCycle(this.rlmState);

    // Resolve per-stream pi-agent (lazy spawn)
    const streamKey = streamRef?.streamId || DEFAULT_STREAM_KEY;
    const piAgent = this.getOrCreatePiAgent(streamKey);

    // 1. Convert VEIL state to messages
    const messages = this.contextAdapter.renderToMessages(veilState, streamRef);

    // 2. Build system prompt (base + ambient facets + skills + RLM)
    const systemPrompt =
      this.contextAdapter.getSystemPrompt(veilState, streamRef) + this.skillPromptFragment + this.rlmPromptFragment;

    // Pass composed system prompt to RLM so child agents inherit it
    if (this.rlmState) this.rlmState.parentSystemPrompt = systemPrompt;

    // 3. Configure pi-agent for this cycle
    piAgent.setSystemPrompt(systemPrompt);
    piAgent.setModel(this.config.model);

    if (this.config.thinkingLevel) {
      piAgent.setThinkingLevel(this.config.thinkingLevel);
    }

    // Combine VEIL-discovered tools, converted handler tools, RLM tools, and extra tools
    const veilTools = this.toolBridge.getAllTools();
    const extraTools = this.config.extraTools ?? [];
    piAgent.setTools([...veilTools, ...this.convertedHandlerTools, ...this.rlmTools, ...extraTools]);

    // Separate the latest user message from the conversation history.
    // pi-agent.prompt() expects the new input message(s) to be passed as
    // an argument, while prior context should already be in piAgent.state.messages.
    // We split messages into history (all but the last user message) and
    // the prompt (the trailing user message).
    const { history, userMessage } = this.splitMessages(messages);

    // Set the conversation history (everything before the latest user input)
    piAgent.replaceMessages(history);

    // Record message count before prompting so we can extract new output
    const messageCountBefore = piAgent.state.messages.length;

    // 4. Prompt the agent — this runs the full tool loop and resolves when done
    if (userMessage) {
      await piAgent.prompt(userMessage);
    } else {
      // No user message found — pass the full messages as history and
      // use continue() or prompt with an empty nudge
      piAgent.replaceMessages(messages);
      await piAgent.prompt('Continue.');
    }

    // 5. Wait for idle (should already be done since prompt() is async, but just in case)
    await piAgent.waitForIdle();

    // Check for errors caught internally by pi-agent
    if (piAgent.state.error) {
      throw new Error(piAgent.state.error);
    }

    // 6. Extract new messages
    const allMessages = piAgent.state.messages;
    const newMessages = allMessages.slice(messageCountBefore);

    // 7. Extract text content and token usage from new assistant messages
    const content = this.extractTextContent(newMessages);
    const tokensUsed = this.extractTokenUsage(newMessages);

    // 8. Convert new messages to VEIL operations
    const operations = this.contextAdapter.messagesToVEILOps(newMessages, streamRef);

    return {
      content,
      operations,
      messages: newMessages,
      tokensUsed,
    };
  }

  // ---------------------------------------------------------------------------
  // Secondary API — runWithContext (pre-rendered messages)
  // ---------------------------------------------------------------------------

  /**
   * Run a pi-agent cycle from pre-rendered context (messages + system prompt).
   *
   * This is the path used by the ConnectomeEffector. It accepts an AgentContext
   * (produced by either VEILContextAdapter or a gRPC context provider) and
   * drives the pi-agent directly without going through VEILContextAdapter.
   *
   * Use runPiCycle() when you have raw VEIL state (server-side).
   * Use runWithContext() when you have pre-rendered messages (client-side/gRPC).
   */
  async runWithContext(
    context: AgentContext,
    streamRef?: { streamId: string; streamType?: string },
    continuation?: boolean,
  ): Promise<ConnectomeCycleResult> {
    // Reset RLM per-activation counters (timeout + call count)
    if (this.rlmState) resetRlmStateForCycle(this.rlmState);

    const { messages, systemPrompt } = context;

    // Resolve per-stream pi-agent (lazy spawn). Streams without a streamRef
    // share DEFAULT_STREAM_KEY which keeps legacy single-stream callers safe.
    const streamKey = streamRef?.streamId || DEFAULT_STREAM_KEY;
    const piAgent = this.getOrCreatePiAgent(streamKey);

    // Configure pi-agent (append skill descriptions + RLM to system prompt)
    const composedPrompt = systemPrompt + this.skillPromptFragment + this.rlmPromptFragment;
    piAgent.setSystemPrompt(composedPrompt);
    piAgent.setModel(this.config.model);

    // Pass composed system prompt to RLM so child agents inherit it
    if (this.rlmState) this.rlmState.parentSystemPrompt = composedPrompt;

    if (this.config.thinkingLevel) {
      piAgent.setThinkingLevel(this.config.thinkingLevel);
    }

    const veilTools = this.toolBridge.getAllTools();
    const extraTools = this.config.extraTools ?? [];
    piAgent.setTools([...veilTools, ...this.convertedHandlerTools, ...this.rlmTools, ...extraTools]);

    if (continuation) {
      // Continuation mode: resume from the bot's last assistant turn.
      //
      // Two paths:
      // - Pseudo-prefill (4.5/4.6): CLI framing workaround via completeSimple()
      // - Direct API prefill (all others): call the Anthropic API directly with
      //   the last assistant text as a prefill turn. This bypasses pi-ai's
      //   completeSimple() which mangles the request through middleware that
      //   doesn't understand prefill (drops content blocks, enables thinking
      //   incorrectly, etc). The direct call is simple and correct.
      const modelId = this.config.model?.id ?? '';
      // Only 4.6 models need pseudo-prefill — 4.5 and earlier support true API prefill
      const needsPseudoPrefill = /claude-(opus|sonnet|haiku)-4[._-]6\b/i.test(modelId)
        || /claude-4[._-]6\b/i.test(modelId);

      console.log(`[ConnectomeAgent:${this.name}] Continuation mode (${needsPseudoPrefill ? 'pseudo-prefill' : 'direct API prefill'}) model=${modelId}`);
      return this.runContinuation(messages, streamRef, needsPseudoPrefill, context.rawMessages);
    }

    // Normal mode: split into history + latest user message
    const { history, userMessage } = this.splitMessages(messages);
    piAgent.replaceMessages(history);

    const messageCountBefore = piAgent.state.messages.length;

    if (userMessage) {
      await piAgent.prompt(userMessage);
    } else {
      piAgent.replaceMessages(messages);
      await piAgent.prompt('Continue.');
    }

    await piAgent.waitForIdle();

    // Check for errors caught internally by pi-agent
    if (piAgent.state.error) {
      throw new Error(piAgent.state.error);
    }

    // Extract new messages
    const allMessages = piAgent.state.messages;
    const newMessages = allMessages.slice(messageCountBefore);

    const content = this.extractTextContent(newMessages);
    const tokensUsed = this.extractTokenUsage(newMessages);

    // Convert to VEIL operations
    const operations = this.contextAdapter.messagesToVEILOps(newMessages, streamRef);

    return { content, operations, messages: newMessages, tokensUsed };
  }

  // ---------------------------------------------------------------------------
  // Continuation (true prefill + pseudo-prefill for 4.5/4.6)
  // ---------------------------------------------------------------------------

  /**
   * Run a continuation cycle using completeSimple() directly.
   * Bypasses pi-agent's tool loop — "m continue" is text-only continuation.
   *
   * @param pseudoPrefill - If true, use CLI framing for 4.5/4.6 models
   */
  private async runContinuation(
    messages: AgentMessage[],
    streamRef?: { streamId: string; streamType?: string },
    pseudoPrefill: boolean = false,
    rawMessages?: Array<{ role: string; content: string; metadata?: any }>,
  ): Promise<ConnectomeCycleResult> {
    const prefix = `[ConnectomeAgent:${this.name}]`;

    // Extract the bot's last assistant content and its index
    let lastAssistantText = '';
    let lastAssistantIdx = -1;
    for (let i = messages.length - 1; i >= 0; i--) {
      const msg = messages[i] as any;
      if (msg.role === 'assistant') {
        const textBlocks = Array.isArray(msg.content)
          ? msg.content.filter((b: any) => b.type === 'text').map((b: any) => b.text)
          : [String(msg.content ?? '')];
        lastAssistantText = textBlocks.join('\n').trim();
        if (lastAssistantText) {
          lastAssistantIdx = i;
          break;
        }
      }
    }

    if (!lastAssistantText) {
      console.warn(`${prefix} Continuation requested but no prior assistant content found, falling back to prompt`);
      const piAgent = this.getOrCreatePiAgent(streamRef?.streamId || DEFAULT_STREAM_KEY);
      piAgent.replaceMessages(messages);
      const countBefore = piAgent.state.messages.length;
      await piAgent.prompt('Continue.');
      await piAgent.waitForIdle();
      if (piAgent.state.error) throw new Error(piAgent.state.error);
      const newMsgs = piAgent.state.messages.slice(countBefore);
      return {
        content: this.extractTextContent(newMsgs),
        operations: this.contextAdapter.messagesToVEILOps(newMsgs, streamRef),
        messages: newMsgs,
        tokensUsed: this.extractTokenUsage(newMsgs),
      };
    }

    // Single path: conversation-log pseudo-prefill works for all models.
    // Full VEIL context is formatted as a participant-labeled log and wrapped
    // in the cat/cut <cmd> pattern. The model sees it as a "file" to continue.
    return this.runDirectPrefill(rawMessages ?? messages, lastAssistantText, streamRef);
  }

  // ---------------------------------------------------------------------------
  // Direct API calls — bypass pi-ai for clean prefill (both true + pseudo)
  // ---------------------------------------------------------------------------

  /**
   * Create an API client for direct prefill calls.
   * Routes to Anthropic (OAuth or API key) or Bedrock based on model provider.
   * Both return the same messages.create() interface.
   */
  private async createPrefillClient(): Promise<{ client: any; isOAuth: boolean; isBedrock: boolean }> {
    const provider = this.config.model?.provider ?? 'anthropic';

    if (provider === 'amazon-bedrock') {
      // @ts-ignore — bedrock SDK available at runtime
      const { AnthropicBedrock } = await import('@anthropic-ai/bedrock-sdk');
      const client = new AnthropicBedrock({
        awsRegion: this.config.awsRegion || process.env.AWS_REGION || 'us-east-1',
      });
      return { client, isOAuth: false, isBedrock: true };
    }

    // Anthropic direct (OAuth or API key)
    // @ts-ignore — anthropic SDK is available at runtime via transitive dep
    const Anthropic = (await import('@anthropic-ai/sdk')).default;

    const apiKey = this.piAgentGetApiKey
      ? await this.piAgentGetApiKey(this.config.model.provider)
      : process.env.ANTHROPIC_API_KEY;

    if (!apiKey) {
      throw new Error('No API key available for prefill');
    }

    const isOAuth = apiKey.includes('sk-ant-oat');

    const client = isOAuth
      ? new Anthropic({
          apiKey: null as any,
          authToken: apiKey,
          dangerouslyAllowBrowser: true,
          defaultHeaders: {
            'accept': 'application/json',
            'anthropic-dangerous-direct-browser-access': 'true',
            'anthropic-beta': 'oauth-2025-04-20',
            'user-agent': 'claude-cli/0.0.0 (external, cli)',
            'x-app': 'cli',
          },
        } as any)
      : new Anthropic({ apiKey });

    return { client, isOAuth, isBedrock: false };
  }

  /**
   * Continuation via CLI-framed pseudo-prefill (works for all models).
   * Builds a conversation log from all VEIL messages with participant names,
   * then wraps it in the cat/cut <cmd> pattern so the model continues from it.
   *
   * Structure:
   *   user:      <cmd>cut -c 1-{N} < untitled.txt</cmd>
   *   assistant: {full conversation log with participant names}
   *   user:      <cmd>cat untitled.txt</cmd>
   *
   * The model sees the entire conversation as a "file" and continues it.
   */
  private async runDirectPrefill(
    messages: AgentMessage[] | Array<{ role: string; content: string; metadata?: any }>,
    prefillText: string,
    streamRef?: { streamId: string; streamType?: string },
  ): Promise<ConnectomeCycleResult> {
    const prefix = `[ConnectomeAgent:${this.name}]`;
    const { client, isOAuth, isBedrock } = await this.createPrefillClient();

    // Build conversation log from last N VEIL messages with participant labels
    // Uses raw unmerged messages when available (each VEIL frame = one message)
    const N = 5;
    const recentMessages = messages.slice(-N);
    console.log(`${prefix} [PREFILL] total msgs=${messages.length}, sliced to last ${N}, got ${recentMessages.length}`);
    for (let i = 0; i < recentMessages.length; i++) {
      const m = recentMessages[i] as any;
      const text = typeof m.content === 'string' ? m.content : '(blocks)';
      console.log(`${prefix} [PREFILL]   [${i}] role=${m.role} len=${text.length} "${text.substring(0, 80)}..."`);
    }
    let conversationLog = '';
    let lastParticipant = '';

    for (const msg of recentMessages) {
      const m = msg as any;
      const role = m.role as string;
      if (role !== 'user' && role !== 'assistant') continue;

      // Raw messages have string content; merged AgentMessages have content blocks
      let text = '';
      if (typeof m.content === 'string') {
        text = m.content;
      } else if (Array.isArray(m.content)) {
        text = m.content
          .filter((b: any) => b.type === 'text')
          .map((b: any) => b.text)
          .join('\n');
      }
      if (!text.trim()) continue;

      // Filter out continuation trigger messages (signal doesn't delete them)
      if (role === 'user' && /^\s*(<\S+>\s+)?m\s+continue\s*$/i.test(text)) continue;

      const participant = role === 'assistant' ? this.name : 'User';

      // Merge consecutive assistant messages (chained continuations) into one entry.
      // User messages always get their own line — different speakers.
      if (participant === lastParticipant && participant === this.name) {
        // Append to previous assistant entry (continuation chain)
        conversationLog = conversationLog.trimEnd() + ' ' + text.trimStart() + '\n\n';
      } else {
        conversationLog += `${participant}: ${text}\n\n`;
      }
      lastParticipant = participant;
    }

    conversationLog = conversationLog.trimEnd();

    const charCount = conversationLog.length;

    // Tail-cut: model outputs only new content (no echo of the log)
    // Bedrock: native prefill (assistant-last). Anthropic: CLI-framed tail-cut.
    const cliDirective = 'The assistant is in CLI simulation mode, and responds to the user\'s CLI commands only with the output of the command.';

    let apiMessages: any[];
    let system: any;

    if (isBedrock) {
      // CLI-framed cut with implied large file
      apiMessages = [
        { role: 'user' as const, content: `<cmd>cut -c 1-${charCount} < untitled.txt</cmd>` },
        { role: 'assistant' as const, content: conversationLog },
        { role: 'user' as const, content: `<cmd>cut -c ${charCount + 1}-50000 < untitled.txt</cmd>` },
      ];
      system = cliDirective;
    } else {
      // CLI-framed tail-cut — model outputs only new content
      apiMessages = [
        { role: 'user' as const, content: [{ type: 'text' as const, text: `<cmd>cut -c 1-${charCount} < untitled.txt</cmd>` }] },
        { role: 'assistant' as const, content: [{ type: 'text' as const, text: conversationLog }] },
        { role: 'user' as const, content: [{ type: 'text' as const, text: `<cmd>cut -c ${charCount + 1}- < untitled.txt</cmd>` }] },
      ];
      system = isOAuth
        ? [
            { type: 'text' as const, text: 'You are Claude Code, Anthropic\'s official CLI for Claude.' },
            { type: 'text' as const, text: cliDirective },
          ]
        : cliDirective;
    }

    console.log(`${prefix} [PREFILL] model=${this.config.model?.id}, provider=${isBedrock ? 'bedrock' : 'anthropic'}, isOAuth=${isOAuth}, log=${charCount} chars, msgs=${messages.length}, recent=${recentMessages.length}`);
    console.log(`${prefix} [PREFILL] ---- CONVERSATION LOG ----`);
    console.log(conversationLog);
    console.log(`${prefix} [PREFILL] ---- END LOG ----`);

    try {
      const requestParams: any = {
        model: this.config.model?.id ?? 'claude-opus-4-6',
        max_tokens: this._maxOutputTokens || 4096,
        temperature: 1,
        messages: apiMessages,
        stop_sequences: ['\nUser:'],
      };
      if (system) requestParams.system = system;
      // Only send thinking param for models that support it (3.7+, 4.x)
      const modelId = this.config.model?.id ?? '';
      const supportsThinking = /claude-(3-7|4|opus-4|sonnet-4|haiku-4)/i.test(modelId);
      if (!isBedrock && supportsThinking) requestParams.thinking = { type: 'disabled' };

      const response = await client.messages.create(requestParams);

      // Tail-cut: response is only new content, no stripping needed
      const newContent = (response as any).content
        .filter((b: any) => b.type === 'text')
        .map((b: any) => b.text)
        .join('');

      const tokensUsed = ((response as any).usage?.input_tokens ?? 0) + ((response as any).usage?.output_tokens ?? 0);
      console.log(`${prefix} [PREFILL] Response: ${newContent.length} chars, stopReason=${(response as any).stop_reason}, tokens=${tokensUsed}`);
      console.log(`${prefix} [PREFILL] Preview: "${newContent.substring(0, 200)}..."`);

      const newMessages: AgentMessage[] = [{
        role: 'assistant',
        content: [{ type: 'text' as const, text: newContent }],
        stopReason: (response as any).stop_reason,
        usage: { input: (response as any).usage?.input_tokens ?? 0, output: (response as any).usage?.output_tokens ?? 0 },
      } as any];

      const operations = this.contextAdapter.messagesToVEILOps(newMessages, streamRef);
      return { content: newContent, operations, messages: newMessages, tokensUsed };

    } catch (err: any) {
      console.error(`${prefix} [PREFILL] API error: ${err.status} ${err.message}`);
      throw err;
    }
  }

  // ---------------------------------------------------------------------------
  // Agent control (steering, abort, subscribe)
  // ---------------------------------------------------------------------------

  /**
   * Steer the agent mid-run (inject a user message into the conversation).
   * The steering message is delivered after the current tool execution completes,
   * skipping any remaining tool calls in the current batch.
   *
   * When streamId is provided, only that stream's pi-agent is steered. When
   * omitted, every active in-flight pi-agent is steered (legacy behavior).
   */
  steer(message: string, streamId?: string): void {
    const payload = {
      role: 'user' as const,
      content: [{ type: 'text' as const, text: message }],
      timestamp: Date.now(),
    };
    if (streamId !== undefined) {
      const piAgent = this.peekPiAgent(streamId);
      if (piAgent) piAgent.steer(payload);
      return;
    }
    for (const entry of this.piAgents.values()) {
      try { entry.agent.steer(payload); } catch { /* ignore non-streaming */ }
    }
  }

  /**
   * Abort the current cycle. The pi-agent will stop streaming and tool execution.
   * Also resets the agent state to clear any stuck "processing" state.
   *
   * When streamId is provided, only that stream's pi-agent is aborted. When
   * omitted, every active in-flight pi-agent is aborted (legacy behavior —
   * matches how `!stop` halts everything the bot is doing).
   */
  abort(streamId?: string): void {
    if (streamId !== undefined) {
      const piAgent = this.peekPiAgent(streamId);
      if (!piAgent) return;
      try { piAgent.abort(); } catch { /* ignore */ }
      try { (piAgent as any).reset?.(); } catch { /* ignore if already idle */ }
      return;
    }
    for (const entry of this.piAgents.values()) {
      try { entry.agent.abort(); } catch { /* ignore */ }
      try { (entry.agent as any).reset?.(); } catch { /* ignore */ }
    }
  }

  /**
   * Subscribe to pi-agent events (for streaming UI updates, logging, etc.)
   * Returns an unsubscribe function.
   *
   * When streamId is provided, subscribes only to that stream's pi-agent. The
   * pi-agent is lazily created if it doesn't exist yet, so the effector can
   * subscribe BEFORE calling runWithContext for the stream and still receive
   * events from the cycle that runWithContext kicks off.
   *
   * When streamId is omitted, subscribes to events from EVERY currently held
   * pi-agent (broadcast). New pi-agents created later are NOT auto-subscribed.
   * Prefer the per-stream form for cycle-scoped event handling.
   */
  subscribe(fn: (e: AgentEvent) => void, streamId?: string): () => void {
    if (streamId !== undefined) {
      const piAgent = this.getOrCreatePiAgent(streamId);
      return piAgent.subscribe(fn);
    }
    const unsubs: Array<() => void> = [];
    for (const entry of this.piAgents.values()) {
      unsubs.push(entry.agent.subscribe(fn));
    }
    return () => {
      for (const u of unsubs) {
        try { u(); } catch { /* ignore */ }
      }
    };
  }

  // ---------------------------------------------------------------------------
  // AgentInterface compatibility (for AgentComponent)
  // ---------------------------------------------------------------------------

  /**
   * Check if this agent should respond to an activation.
   * Respects sleeping state, ignored sources, and agent targeting.
   */
  shouldActivate(activation: any, _state: any): boolean {
    // Don't activate when sleeping
    if (this.agentState.sleeping) {
      return false;
    }

    // Check ignored sources
    if (activation.source && this.agentState.ignoringSources.has(activation.source)) {
      return false;
    }

    // Check if activation targets a specific agent
    if (activation.targetAgent && activation.targetAgent !== this.config.name) {
      return false;
    }

    if (activation.targetAgentId && activation.targetAgentId !== this.agentId) {
      return false;
    }

    return true;
  }

  /**
   * Handle agent commands (sleep, wake, ignore, unignore, setThreshold).
   */
  handleCommand(command: AgentCommand): void {
    switch (command.type) {
      case 'sleep':
        this.agentState.sleeping = true;
        break;

      case 'wake':
        this.agentState.sleeping = false;
        break;

      case 'ignore':
        this.agentState.ignoringSources.add(command.source);
        break;

      case 'unignore':
        this.agentState.ignoringSources.delete(command.source);
        break;

      case 'setThreshold':
        this.agentState.attentionThreshold = command.threshold;
        break;
    }
  }

  /**
   * Get the current behavioral state (sleeping, ignoringSources, attentionThreshold).
   */
  getAgentState(): { sleeping: boolean; ignoringSources: Set<string>; attentionThreshold: number } {
    return {
      sleeping: this.agentState.sleeping,
      ignoringSources: new Set(this.agentState.ignoringSources),
      attentionThreshold: this.agentState.attentionThreshold,
    };
  }

  // ---------------------------------------------------------------------------
  // Skill management
  // ---------------------------------------------------------------------------

  /** Get all loaded skills. */
  getSkills(): readonly Skill[] {
    return this.loadedSkills;
  }

  /** Look up a loaded skill by name. */
  getSkill(name: string): Skill | undefined {
    return this.loadedSkills.find((s) => s.name === name);
  }

  /**
   * Read a skill's full content (markdown body wrapped in XML).
   * Used for /skill:name command expansion.
   */
  getSkillContent(name: string): string | null {
    const skill = this.getSkill(name);
    if (!skill) return null;
    return getSkillContent(skill);
  }

  /**
   * Reload skills from the given paths (or from the original config paths).
   * Replaces all previously loaded skills.
   */
  reloadSkills(paths?: string[]): void {
    const skillPaths = paths ?? this.config.skillPaths ?? [];
    this.loadSkillsInternal(skillPaths);
  }

  // ---------------------------------------------------------------------------
  // Accessors for sub-components
  // ---------------------------------------------------------------------------

  /**
   * Access the underlying VEILToolBridge (for registering additional tools,
   * discovering VEIL tools from facets, etc.)
   */
  getToolBridge(): VEILToolBridge {
    return this.toolBridge;
  }

  /**
   * Access the underlying VEILContextAdapter (for custom rendering, etc.)
   */
  getContextAdapter(): VEILContextAdapter {
    return this.contextAdapter;
  }

  /**
   * Access the per-stream pi-agent Agent instance (for advanced usage —
   * streaming subscriptions, direct message manipulation, etc.)
   *
   * When streamId is provided, returns/spawns that stream's pi-agent.
   * When omitted, returns the default-stream pi-agent (legacy).
   */
  getPiAgent(streamId?: string): Agent {
    return this.getOrCreatePiAgent(streamId || DEFAULT_STREAM_KEY);
  }

  /**
   * Access the RLM runtime state (depth, call count, cost, async jobs).
   * Returns null if RLM is not configured.
   */
  getRlmState(): RlmState | null {
    return this.rlmState;
  }

  // ---------------------------------------------------------------------------
  // Private helpers
  // ---------------------------------------------------------------------------

  /**
   * Load skills from paths and cache the formatted prompt fragment.
   */
  private loadSkillsInternal(paths: string[]): void {
    const result = loadSkillsFromPaths(paths);
    this.loadedSkills = result.skills;
    this.skillPromptFragment = formatSkillsForPrompt(result.skills);
  }

  /**
   * Split messages into conversation history and the latest user message.
   *
   * pi-agent.prompt() appends the provided message(s) to its internal
   * state.messages before running the loop. So we set history via
   * replaceMessages() and pass the latest user message to prompt().
   */
  private splitMessages(messages: AgentMessage[]): {
    history: AgentMessage[];
    userMessage: AgentMessage | null;
  } {
    if (messages.length === 0) {
      return { history: [], userMessage: null };
    }

    // Walk backwards to find the last user message
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === 'user') {
        return {
          history: messages.slice(0, i),
          userMessage: messages[i],
        };
      }
    }

    // No user message found — return all as history
    return { history: messages, userMessage: null };
  }

  /**
   * Extract text content from new assistant messages.
   * Concatenates all text blocks from assistant messages into a single string.
   */
  private extractTextContent(messages: AgentMessage[]): string {
    const textParts: string[] = [];

    for (const msg of messages) {
      if (msg.role !== 'assistant') continue;

      // Assistant message content is an array of TextContent | ThinkingContent | ToolCall
      if (Array.isArray(msg.content)) {
        for (const block of msg.content) {
          if (block.type === 'text' && typeof block.text === 'string') {
            textParts.push(block.text);
          }
        }
      }
    }

    return textParts.join('\n').trim();
  }

  /**
   * Extract total token usage from new assistant messages.
   * Sums usage.totalTokens (or input + output) across all assistant messages.
   */
  private extractTokenUsage(messages: AgentMessage[]): number {
    let total = 0;

    for (const msg of messages) {
      if (msg.role !== 'assistant') continue;

      const usage = (msg as any).usage;
      if (!usage) continue;

      if (typeof usage.totalTokens === 'number') {
        total += usage.totalTokens;
      } else if (typeof usage.input === 'number' && typeof usage.output === 'number') {
        total += usage.input + usage.output;
      }
    }

    return total;
  }

  /**
   * Create a stable agent ID from a name, or generate a random one.
   */
  private createAgentId(name?: string): string {
    if (name) {
      return name.toLowerCase().replace(/[^a-z0-9]+/g, '-');
    }
    return `agent-${Math.random().toString(36).substring(2, 11)}`;
  }
}
