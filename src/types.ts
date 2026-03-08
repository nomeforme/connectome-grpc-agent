/**
 * Shared types for @connectome/agent-core
 *
 * Bridges between connectome-ts VEIL types and pi-agent types.
 */

import type { AgentMessage, AgentTool, AgentEvent, ThinkingLevel } from '@mariozechner/pi-agent-core';
import type { Model } from '@mariozechner/pi-ai';
import type { Frame, OutgoingVEILOperation, VEILState, StreamRef, Facet } from '@connectome/connectome-ts';
import type { AgentConfig, AgentState as ConnectomeAgentState, AgentCommand } from '@connectome/connectome-ts';
import type { RenderedContext } from '@connectome/connectome-ts';
import type { RlmConfig } from './rlm/types.js';

// Re-export key types for convenience
export type {
  AgentMessage,
  AgentTool,
  AgentEvent,
  ThinkingLevel,
  Model,
  Frame,
  OutgoingVEILOperation,
  VEILState,
  StreamRef,
  Facet,
  AgentConfig,
  AgentCommand,
  RenderedContext,
};

export type { ConnectomeAgentState };

// ---------------------------------------------------------------------------
// VEILContextAdapter
// ---------------------------------------------------------------------------

/**
 * Configuration for the VEILContextAdapter
 */
export interface VEILContextAdapterConfig {
  /** Agent's unique ID in the space */
  agentId: string;
  /** Agent's display name */
  agentName: string;
  /** Base system prompt (skills/context appended dynamically) */
  systemPrompt: string;
  /** Maximum frames to include in context */
  maxFrames?: number;
  /** Token budget for context window */
  contextTokenBudget?: number;
}

// ---------------------------------------------------------------------------
// ToolHandler (MCP / platform tools)
// ---------------------------------------------------------------------------

/**
 * Simple tool interface used by MCPManager and platform tool factories.
 * ConnectomeAgent converts these to pi-agent AgentTool[] internally.
 */
export interface ToolHandler {
  name: string;
  description: string;
  parameters: Record<string, any>;
  /** Optional list of required parameter names (if not set, all are assumed required) */
  required?: string[];
  handler: (input: Record<string, any>) => Promise<string>;
}

// ---------------------------------------------------------------------------
// ConnectomeAgent
// ---------------------------------------------------------------------------

/**
 * Configuration for the ConnectomeAgent
 */
export interface ConnectomeAgentConfig extends AgentConfig {
  /** Pi-ai model to use for LLM calls */
  model: Model<any>;
  /** Thinking level for models that support it */
  thinkingLevel?: ThinkingLevel;
  /** Maximum tool execution rounds before stopping */
  maxToolRounds?: number;
  /** Paths to skill directories to load */
  skillPaths?: string[];
  /** Additional pi-agent tools beyond VEIL-discovered ones */
  extraTools?: AgentTool[];
  /** MCP/platform tools (auto-converted to AgentTool internally) */
  toolHandlers?: ToolHandler[];
  /** Custom stream function (for proxy backends) */
  streamFn?: (...args: any[]) => any;
  /** Enable prompt caching (default true). Set false for bedrock cross-region models that don't support it. */
  promptCaching?: boolean;
  /** Max output tokens per API call. Overrides model default (model.maxTokens / 3). */
  maxOutputTokens?: number;
  /** RLM (Recursive Sub-Agent) configuration. When set, rlm_query/rlm_check_job/rlm_cost tools are added. */
  rlm?: RlmConfig;
}

/**
 * Result from a ConnectomeAgent cycle
 */
export interface ConnectomeCycleResult {
  /** Final speech content (text output) */
  content: string;
  /** VEIL operations to apply (speech, thought, action facets) */
  operations: OutgoingVEILOperation[];
  /** All agent messages from this cycle */
  messages: AgentMessage[];
  /** Token usage from this cycle */
  tokensUsed: number;
}

// ---------------------------------------------------------------------------
// VEIL State Abstraction
// ---------------------------------------------------------------------------

/**
 * Minimal interface for accessing VEIL state without importing VEILStateManager.
 * Implemented by VEILStateManager directly (server-side).
 * Used by VEILContextAdapter.renderToMessages() and runPiCycle().
 */
export interface VEILStateLike {
  getFrameHistory(): readonly Frame[];
  getFacets(): ReadonlyMap<string, Facet>;
}

// ---------------------------------------------------------------------------
// Context Provider (for ConnectomeEffector)
// ---------------------------------------------------------------------------

/**
 * Pre-rendered context ready to feed to the agent. Contains pi-agent messages
 * and a system prompt. This is the common currency between the two context
 * paths:
 *
 * - **Server-side**: VEILContextAdapter converts raw frames/facets → AgentContext
 * - **Client-side**: gRPC getContext returns pre-rendered conversation → mapped to AgentContext
 */
export interface AgentContext {
  /** Conversation messages in pi-agent format */
  messages: AgentMessage[];
  /** System prompt for this cycle */
  systemPrompt: string;
}

/**
 * Provides context for a given stream. Two implementation patterns:
 *
 * 1. **Server-side** (VEILContextProvider): Uses VEILContextAdapter to convert
 *    raw VEIL state → AgentContext
 * 2. **Client-side** (GrpcContextProvider): Calls gRPC getContext → maps the
 *    pre-rendered conversation array to pi-agent messages
 */
export interface ContextProvider {
  getContext(
    streamId: string,
    options?: { maxFrames?: number },
  ): Promise<AgentContext>;
}

/**
 * Records agent speech back to the connectome server so it appears in
 * VEIL state as a speech facet. The axon-side implementation calls
 * grpcClient.emitEvent('agent:speech', ...).
 */
export interface SpeechRecorder {
  recordSpeech(
    content: string,
    metadata: {
      agentId: string;
      agentName: string;
      streamId: string;
      attachments?: Array<{
        id: string;
        contentType: string;
        data: string;          // base64
        filename?: string;
        sizeBytes?: number;
      }>;
    },
  ): Promise<void>;
}

// ---------------------------------------------------------------------------
// Platform Adapter
// ---------------------------------------------------------------------------

/**
 * Thin, stateless adapter for platform-specific delivery and formatting.
 *
 * Each platform (Discord, Signal, terminal, web) implements this interface.
 * The ConnectomeEffector orchestrates the cycle and delegates platform
 * concerns through this adapter.
 */
export interface PlatformAdapter {
  /** Platform identifier (e.g., 'discord', 'signal', 'terminal') */
  readonly platformType: string;

  /** Build a canonical stream ID from platform-specific identifiers */
  buildStreamId(platformData: Record<string, any>): string;

  /**
   * Deliver cleaned speech content to the platform.
   * The adapter handles mention resolution, message splitting, and sending.
   */
  deliverSpeech(content: string, context: PlatformContext): Promise<void>;

  /**
   * Format content for this platform (mention resolution, markdown adjustments).
   * May be async for platforms that require API lookups (e.g., Discord user ID resolution).
   */
  formatContent(content: string, context: PlatformContext): Promise<string>;

  /** Clean incoming content from platform (strip platform-specific formatting) */
  cleanIncoming(content: string, context: PlatformContext): string;

  /** Send typing/processing indicator */
  sendTypingIndicator(context: PlatformContext): Promise<void>;
}

/**
 * Platform-specific context passed through the effector pipeline
 */
export interface PlatformContext {
  streamId: string;
  streamType: string;
  /** Platform-specific data (channelId for Discord, groupId for Signal, etc.) */
  platformData: Record<string, any>;
}

/**
 * Unified activation event for ConnectomeEffector
 */
export interface UnifiedActivation {
  streamId: string;
  platformContext: PlatformContext;
  messageContent: string;
  authorName: string;
  /** When true, the agent continues from its last truncated output instead of starting a new response. */
  continuation?: boolean;
}

// ---------------------------------------------------------------------------
// ConnectomeEffector
// ---------------------------------------------------------------------------

/**
 * Minimal agent interface used by the ConnectomeEffector.
 * Implemented by ConnectomeAgent — avoids circular import with the class.
 */
export interface EffectorAgent {
  readonly id: string;
  readonly name: string;

  /**
   * Run a cycle from pre-rendered context (messages + system prompt).
   * This is the primary path used by the effector — works with both
   * server-side (VEILContextAdapter → AgentContext) and client-side
   * (gRPC → AgentContext) context sources.
   *
   * When continuation is true, the agent resumes from its last assistant
   * message (prefill/completion mode) instead of prompting with a new user message.
   */
  runWithContext(
    context: AgentContext,
    streamRef?: { streamId: string; streamType?: string },
    continuation?: boolean,
  ): Promise<ConnectomeCycleResult>;

  /**
   * Subscribe to pi-agent events (message_end, turn_end, etc.)
   * Returns an unsubscribe function. Optional — not all agents support this.
   */
  subscribe?(fn: (e: AgentEvent) => void): () => void;
}

/**
 * Configuration for the ConnectomeEffector
 */
export interface ConnectomeEffectorConfig {
  /** Agent to run cycles on */
  agent: EffectorAgent;
  /** Platform-specific adapter for delivery */
  adapter: PlatformAdapter;
  /** Provider for fetching VEIL state */
  contextProvider: ContextProvider;
  /** Optional recorder for persisting speech back to the server */
  speechRecorder?: SpeechRecorder;
  /** Interval for refreshing typing indicator (ms, default 8000) */
  typingRefreshMs?: number;
  /** Maximum frames to request from context provider (default 200) */
  maxFrames?: number;
  /** Called when an error occurs during a cycle */
  onError?: (error: Error, activation: UnifiedActivation) => void;
  /** Drain queued attachments after agent cycle completes (e.g. from attach_file tool) */
  drainAttachments?: () => Array<{
    id: string; contentType: string; data: string; filename?: string; sizeBytes?: number;
  }>;
}
