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
import type {
  AgentMessage,
  AgentEvent,
  AgentTool,
  AgentContext,
  ConnectomeAgentConfig,
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

/**
 * Behavioral state for the agent (sleeping, ignoring sources, etc.)
 * Mirrors connectome-ts AgentState but kept as a local concern.
 */
interface ConnectomeAgentBehaviorState {
  sleeping: boolean;
  ignoringSources: Set<string>;
  attentionThreshold: number;
}

export class ConnectomeAgent {
  private piAgent: Agent;
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
    const self = this;
    let streamFn: typeof baseFn | undefined = (model: any, context: any, options?: any) => {
      const overrides: Record<string, any> = {};
      if (needsCacheOverride) overrides.cacheRetention = 'none';
      if (typeof self._maxOutputTokens === 'number') overrides.maxTokens = self._maxOutputTokens;
      return baseFn(model, context, { ...options, ...overrides });
    };

    // Initialize pi auth provider (reads ~/.pi/agent/auth.json for OAuth tokens).
    // Falls through to ANTHROPIC_API_KEY env var if no auth.json exists.
    // Skip OAuth when useApiKey is set (for models not on Claude subscription).
    const skipOAuth = config.useApiKey || !!config.getApiKey;
    const authProvider = skipOAuth ? undefined : new PiAuthProvider();
    const resolvedGetApiKey = config.getApiKey ?? authProvider?.getApiKey;

    if (config.useApiKey) {
      console.log(`[ConnectomeAgent:${config.name}] Using API key auth (useApiKey=true)`);
    } else if (authProvider?.hasCredentials('anthropic')) {
      console.log(`[ConnectomeAgent:${config.name}] Using pi OAuth auth (Claude subscription)`);
    } else if (!config.getApiKey) {
      console.log(`[ConnectomeAgent:${config.name}] No pi auth.json found — using ANTHROPIC_API_KEY env var`);
    }

    // Initialize the pi-agent with model and optional stream function
    this.piAgent = new Agent({
      initialState: {
        model: config.model,
        thinkingLevel: config.thinkingLevel ?? 'off',
        systemPrompt: config.systemPrompt ?? '',
      },
      streamFn,
      getApiKey: resolvedGetApiKey,
    });

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

    // Initialize RLM (recursive sub-agent) if configured
    if (config.rlm) {
      this.rlmState = initRlmState(config.rlm);

      // Wire native execution: child agents use the same stream function,
      // model, and tools as the parent (tools evaluated lazily each call).
      this.rlmState.streamFn = streamFn;
      this.rlmState.parentModel = config.model;
      this.rlmState.getApiKey = this.piAgent.getApiKey;
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

    // 1. Convert VEIL state to messages
    const messages = this.contextAdapter.renderToMessages(veilState, streamRef);

    // 2. Build system prompt (base + ambient facets + skills + RLM)
    const systemPrompt =
      this.contextAdapter.getSystemPrompt(veilState, streamRef) + this.skillPromptFragment + this.rlmPromptFragment;

    // Pass composed system prompt to RLM so child agents inherit it
    if (this.rlmState) this.rlmState.parentSystemPrompt = systemPrompt;

    // 3. Configure pi-agent for this cycle
    this.piAgent.setSystemPrompt(systemPrompt);
    this.piAgent.setModel(this.config.model);

    if (this.config.thinkingLevel) {
      this.piAgent.setThinkingLevel(this.config.thinkingLevel);
    }

    // Combine VEIL-discovered tools, converted handler tools, RLM tools, and extra tools
    const veilTools = this.toolBridge.getAllTools();
    const extraTools = this.config.extraTools ?? [];
    this.piAgent.setTools([...veilTools, ...this.convertedHandlerTools, ...this.rlmTools, ...extraTools]);

    // Separate the latest user message from the conversation history.
    // pi-agent.prompt() expects the new input message(s) to be passed as
    // an argument, while prior context should already be in piAgent.state.messages.
    // We split messages into history (all but the last user message) and
    // the prompt (the trailing user message).
    const { history, userMessage } = this.splitMessages(messages);

    // Set the conversation history (everything before the latest user input)
    this.piAgent.replaceMessages(history);

    // Record message count before prompting so we can extract new output
    const messageCountBefore = this.piAgent.state.messages.length;

    // 4. Prompt the agent — this runs the full tool loop and resolves when done
    if (userMessage) {
      await this.piAgent.prompt(userMessage);
    } else {
      // No user message found — pass the full messages as history and
      // use continue() or prompt with an empty nudge
      this.piAgent.replaceMessages(messages);
      await this.piAgent.prompt('Continue.');
    }

    // 5. Wait for idle (should already be done since prompt() is async, but just in case)
    await this.piAgent.waitForIdle();

    // Check for errors caught internally by pi-agent
    if (this.piAgent.state.error) {
      throw new Error(this.piAgent.state.error);
    }

    // 6. Extract new messages
    const allMessages = this.piAgent.state.messages;
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

    // Configure pi-agent (append skill descriptions + RLM to system prompt)
    const composedPrompt = systemPrompt + this.skillPromptFragment + this.rlmPromptFragment;
    this.piAgent.setSystemPrompt(composedPrompt);
    this.piAgent.setModel(this.config.model);

    // Pass composed system prompt to RLM so child agents inherit it
    if (this.rlmState) this.rlmState.parentSystemPrompt = composedPrompt;

    if (this.config.thinkingLevel) {
      this.piAgent.setThinkingLevel(this.config.thinkingLevel);
    }

    const veilTools = this.toolBridge.getAllTools();
    const extraTools = this.config.extraTools ?? [];
    this.piAgent.setTools([...veilTools, ...this.convertedHandlerTools, ...this.rlmTools, ...extraTools]);

    if (continuation) {
      // Continuation mode: load all messages and call continue() so the model
      // resumes from its last assistant turn (prefill/completion style).
      // The context already contains the bot's previous speech as assistant messages,
      // so the model sees its own output and continues naturally.
      this.piAgent.replaceMessages(messages);
      const messageCountBefore = this.piAgent.state.messages.length;
      console.log(`[ConnectomeAgent:${this.name}] Continuation mode — resuming from ${messages.length} messages`);
      await this.piAgent.continue();
      await this.piAgent.waitForIdle();
      if (this.piAgent.state.error) {
        throw new Error(this.piAgent.state.error);
      }
      const allMessages = this.piAgent.state.messages;
      const newMessages = allMessages.slice(messageCountBefore);
      const content = this.extractTextContent(newMessages);
      const tokensUsed = this.extractTokenUsage(newMessages);
      const operations = this.contextAdapter.messagesToVEILOps(newMessages, streamRef);
      return { content, operations, messages: newMessages, tokensUsed };
    }

    // Normal mode: split into history + latest user message
    const { history, userMessage } = this.splitMessages(messages);
    this.piAgent.replaceMessages(history);

    const messageCountBefore = this.piAgent.state.messages.length;

    if (userMessage) {
      await this.piAgent.prompt(userMessage);
    } else {
      this.piAgent.replaceMessages(messages);
      await this.piAgent.prompt('Continue.');
    }

    await this.piAgent.waitForIdle();

    // Check for errors caught internally by pi-agent
    if (this.piAgent.state.error) {
      throw new Error(this.piAgent.state.error);
    }

    // Extract new messages
    const allMessages = this.piAgent.state.messages;
    const newMessages = allMessages.slice(messageCountBefore);

    const content = this.extractTextContent(newMessages);
    const tokensUsed = this.extractTokenUsage(newMessages);

    // Convert to VEIL operations
    const operations = this.contextAdapter.messagesToVEILOps(newMessages, streamRef);

    return { content, operations, messages: newMessages, tokensUsed };
  }

  // ---------------------------------------------------------------------------
  // Agent control (steering, abort, subscribe)
  // ---------------------------------------------------------------------------

  /**
   * Steer the agent mid-run (inject a user message into the conversation).
   * The steering message is delivered after the current tool execution completes,
   * skipping any remaining tool calls in the current batch.
   */
  steer(message: string): void {
    this.piAgent.steer({
      role: 'user',
      content: [{ type: 'text', text: message }],
      timestamp: Date.now(),
    });
  }

  /**
   * Abort the current cycle. The pi-agent will stop streaming and tool execution.
   */
  abort(): void {
    this.piAgent.abort();
  }

  /**
   * Subscribe to pi-agent events (for streaming UI updates, logging, etc.)
   * Returns an unsubscribe function.
   */
  subscribe(fn: (e: AgentEvent) => void): () => void {
    return this.piAgent.subscribe(fn);
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
   * Access the underlying pi-agent Agent (for advanced usage — streaming
   * subscriptions, direct message manipulation, etc.)
   */
  getPiAgent(): Agent {
    return this.piAgent;
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
