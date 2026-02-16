/**
 * @connectome/agent-core
 *
 * Pi-agent wrapped in VEIL-native agents with platform abstraction.
 */

// Core classes
export { VEILContextAdapter } from './veil-context-adapter.js';
export { VEILToolBridge, toolHandlerToAgentTool } from './veil-tool-bridge.js';
export { ConnectomeAgent } from './connectome-agent.js';
export { ConnectomeEffector } from './connectome-effector.js';

// Multi-agent routing
export { ActivationRouter } from './activation-router.js';

// Context & model utilities
export { renderedContextToAgentContext } from './context-adapter.js';
export { resolveModel } from './model-resolver.js';

// Utilities
export { cleanSpeechContent, splitMessage } from './utils.js';

// Skill loader
export {
  loadSkillsFromPaths,
  loadSkillsFromDir,
  formatSkillsForPrompt,
  getSkillContent,
  stripFrontmatter,
} from './skill-loader.js';

// RLM (Recursive Sub-Agent)
export {
  initRlmState,
  checkGuardrails,
  parseCostFile,
  buildEnv,
  execRlmSync,
  execRlmAsync,
  createRlmQueryTool,
  createRlmCheckJobTool,
  createRlmCostTool,
  buildRlmSystemPromptFragment,
} from './rlm/index.js';

export type {
  RlmConfig,
  RlmState,
  RlmAsyncJob,
  RlmCostSummary,
  RlmToolDetails,
  RlmSyncResult,
  RlmAsyncResult,
  ExecOptions,
} from './rlm/index.js';

// Types
export type {
  // Agent config & results
  VEILContextAdapterConfig,
  ConnectomeAgentConfig,
  ConnectomeCycleResult,
  // Tool handler (MCP/platform tools)
  ToolHandler,
  // VEIL state abstraction
  VEILStateLike,
  AgentContext,
  ContextProvider,
  SpeechRecorder,
  // Platform abstraction
  PlatformAdapter,
  PlatformContext,
  UnifiedActivation,
  // Effector
  EffectorAgent,
  ConnectomeEffectorConfig,
} from './types.js';

// Context adapter types
export type {
  ContextMessage,
  RenderedContextLike,
} from './context-adapter.js';

// Skill types
export type {
  Skill,
  SkillFrontmatter,
  SkillLoadResult,
  SkillDiagnostic,
} from './skill-loader.js';

// Activation router types
export type {
  AgentRegistration,
  RoutingResult,
  RoutableActivation,
  ActivationRouterConfig,
} from './activation-router.js';

