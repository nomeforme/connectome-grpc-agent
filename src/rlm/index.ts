/**
 * RLM (Recursive Sub-Agent) — barrel exports.
 */

// Types
export type {
  RlmConfig,
  RlmState,
  RlmAsyncJob,
  RlmCostSummary,
  RlmToolDetails,
} from './types.js';

// State initialization
export { initRlmState, resetRlmStateForCycle } from './state.js';

// Guardrails
export { checkGuardrails, parseCostFile } from './guardrails.js';

// Subprocess
export { buildEnv, execRlmSync, execRlmAsync } from './subprocess.js';
export type { RlmSyncResult, RlmAsyncResult, ExecOptions } from './subprocess.js';

// Tool factories
export { createRlmQueryTool, createRlmCheckJobTool, createRlmCostTool } from './tools.js';

// System prompt
export { buildRlmSystemPromptFragment } from './system-prompt.js';
