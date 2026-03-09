/**
 * Type definitions for RLM (Recursive Sub-Agent) capability.
 *
 * These types model ypi's rlm_query recursion machinery:
 * config, runtime state, async jobs, and cost tracking.
 */

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/**
 * Configuration for RLM recursion capability.
 * Added as an optional `rlm` field on ConnectomeAgentConfig.
 */
export interface RlmConfig {
  /** Maximum recursion depth (maps to RLM_MAX_DEPTH, default 3) */
  maxDepth?: number;
  /** Maximum total rlm_query invocations (maps to RLM_MAX_CALLS) */
  maxCalls?: number;
  /** Maximum dollar spend for entire recursive tree (maps to RLM_BUDGET) */
  budget?: number;
  /** Wall-clock timeout in seconds for the entire call chain (maps to RLM_TIMEOUT) */
  timeoutSeconds?: number;
  /** LLM provider name (maps to RLM_PROVIDER) */
  provider?: string;
  /** LLM model (maps to RLM_MODEL) */
  model?: string;
  /** Model for depth > 0 calls (maps to RLM_CHILD_MODEL) */
  childModel?: string;
  /** Provider for depth > 0 calls (maps to RLM_CHILD_PROVIDER) */
  childProvider?: string;
  /** Enable jj workspace isolation (maps to RLM_JJ, default true) */
  jjIsolation?: boolean;
  /** Working directory for child processes */
  cwd?: string;
  /** Path to custom system prompt file (maps to RLM_SYSTEM_PROMPT) */
  systemPromptFile?: string;
  /** Enable Pi extensions in children (maps to RLM_EXTENSIONS, default true) */
  extensions?: boolean;
  /** Override extensions for depth > 0 (maps to RLM_CHILD_EXTENSIONS) */
  childExtensions?: boolean;
  /** Enable shared sessions across children (maps to RLM_SHARED_SESSIONS, default true) */
  sharedSessions?: boolean;
  /** Pi session directory (maps to RLM_SESSION_DIR) */
  sessionDir?: string;
}

// ---------------------------------------------------------------------------
// Runtime State
// ---------------------------------------------------------------------------

/**
 * Mutable runtime state for an RLM-enabled agent.
 * Tracks depth, call count, timing, and async jobs across cycles.
 */
export interface RlmState {
  /** Current recursion depth (incremented for child calls) */
  depth: number;
  /** Total rlm_query invocations so far */
  callCount: number;
  /** Unique trace ID linking all sessions in this recursive tree */
  traceId: string;
  /** Epoch ms when the root call started */
  startTime: number;
  /** Path to JSONL cost ledger file (set when budget tracking is enabled) */
  costFilePath: string | null;
  /** Map of active async jobs by jobId */
  asyncJobs: Map<string, RlmAsyncJob>;
  /** Callback to get parent agent's current tools (for native RLM execution) */
  getParentTools?: () => any[];
  /** Stream function from parent agent (for native RLM execution) */
  streamFn?: (...args: any[]) => any;
  /** Parent agent's Model object (for native RLM child agents) */
  parentModel?: any;
  /** Parent agent's fully composed system prompt (base + skills + RLM), updated each cycle */
  parentSystemPrompt?: string;
  /** API key resolver inherited from parent (for OAuth / subscription auth) */
  getApiKey?: (provider: string) => Promise<string | undefined> | string | undefined;
}

// ---------------------------------------------------------------------------
// Async Jobs
// ---------------------------------------------------------------------------

/**
 * Metadata for an async rlm_query job.
 * Returned by `rlm_query --async` and tracked in RlmState.asyncJobs.
 */
export interface RlmAsyncJob {
  /** Unique job identifier */
  jobId: string;
  /** Path to the file where output is written */
  outputPath: string;
  /** Path to sentinel file (existence means job is done) */
  sentinelPath: string;
  /** PID of the background process */
  pid: number;
  /** The prompt that was sent */
  prompt: string;
}

// ---------------------------------------------------------------------------
// Cost Tracking
// ---------------------------------------------------------------------------

/**
 * Aggregated cost summary from the JSONL cost ledger.
 */
export interface RlmCostSummary {
  /** Total dollar cost across all calls */
  cost: number;
  /** Total tokens across all calls */
  tokens: number;
  /** Number of cost entries (calls that reported cost) */
  calls: number;
}

// ---------------------------------------------------------------------------
// Tool Result Details
// ---------------------------------------------------------------------------

/**
 * Details attached to AgentToolResult from RLM tools.
 */
export interface RlmToolDetails {
  /** Which RLM tool produced this result */
  rlmTool: 'rlm_query' | 'rlm_check_job' | 'rlm_cost';
  /** Whether this was an async invocation */
  async?: boolean;
  /** Job ID for async invocations */
  jobId?: string;
  /** Cost summary if available */
  cost?: RlmCostSummary;
  /** Error message if the tool failed */
  error?: string;
}
