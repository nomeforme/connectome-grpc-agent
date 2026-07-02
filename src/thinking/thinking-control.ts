/**
 * ThinkingControl — generic per-bot adapter for disabling chain-of-thought.
 *
 * Different model providers use different mechanisms to suppress "thinking" /
 * reasoning output:
 *
 *   - Qwen 3.x (llama-server, vLLM):   `/no_think` marker in system prompt
 *   - vLLM/llama-server Qwen:          extra_body.chat_template_kwargs.enable_thinking=false
 *   - Anthropic:                        thinkingEnabled: false at pi-agent config
 *   - OpenAI o-series/GPT-5:            reasoning_effort: 'minimal'
 *   - Z.ai (GLM):                       thinking: {type: 'disabled'} extra param
 *
 * This module gives bot-runtime a single dispatch point: when a bot config
 * has `disable_thinking: true`, resolve the right adapter for its model +
 * endpoint, and apply the adapter's intervention at the appropriate stage.
 *
 * Point of application varies by adapter:
 *   - `applyToSystemPrompt` runs before the system prompt reaches the model
 *     (prompt-append adapters). Applied in bot-runtime's ConnectomeBridge.
 *   - `applyToPiAgentOptions` (future) will run when constructing pi-agent
 *     request options (API-param adapters). Not yet plumbed — adapters that
 *     need it can define it, but only prompt-side adapters are active today.
 *
 * The plantoid + Qwen 3.6 case only needs prompt-side; API-side adapters can
 * be added as they become the second real data point.
 */

/** Runtime context passed to adapter.matches() for dispatch. */
export interface ThinkingControlContext {
  /** Model identifier (e.g. "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf", "claude-sonnet-4-5"). */
  model: string;
  /**
   * OpenAI-compatible endpoint URL if the bot points at a self-hosted server
   * (llama-server / vLLM / LM Studio). Undefined for Anthropic/Bedrock/etc.
   */
  endpoint?: string;
}

/**
 * Result of applying an adapter's system-prompt-side control.
 * `applied` = false means the adapter matched but there was nothing to
 * add (idempotency), OR the adapter is API-side only and should be
 * invoked via `applyToPiAgentOptions` instead.
 */
export interface ApplyResult {
  systemPrompt: string;
  applied: boolean;
}

/**
 * Adapter interface. Each adapter recognizes a specific model family /
 * provider shape and knows how to disable thinking there.
 */
export interface ThinkingControlAdapter {
  /** Short name for logging (e.g. "qwen-no-think", "anthropic"). */
  readonly name: string;
  /** Return true when this adapter is the right one for the given context. */
  matches(ctx: ThinkingControlContext): boolean;
  /**
   * Apply the prompt-side intervention. If the adapter is purely API-side,
   * return the input unchanged with `applied: false` and rely on
   * `applyToPiAgentOptions` (once plumbed).
   */
  applyToSystemPrompt?(systemPrompt: string): ApplyResult;
  /**
   * Future extension point — API-param adapters. Not yet invoked. When
   * added, this will receive the pi-agent request options object and
   * return a modified version. For now, adapters that need this should
   * document it in comments and their `applyToSystemPrompt` can no-op.
   */
  // applyToPiAgentOptions?(options: any): any;
}

// ---------------------------------------------------------------------------
// Registry — adapters are pushed in priority order (most-specific first)
// ---------------------------------------------------------------------------

const REGISTRY: ThinkingControlAdapter[] = [];

/** Register an adapter. First-matched-wins during resolve(). */
export function registerThinkingAdapter(adapter: ThinkingControlAdapter): void {
  REGISTRY.push(adapter);
}

/** Look up the adapter for a given model context. Returns undefined if none matches. */
export function resolveThinkingAdapter(
  ctx: ThinkingControlContext,
): ThinkingControlAdapter | undefined {
  for (const adapter of REGISTRY) {
    if (adapter.matches(ctx)) return adapter;
  }
  return undefined;
}

/**
 * Convenience: resolve + apply-to-system-prompt in one call.
 * If no adapter matches, returns the input unchanged with `applied: false`.
 * Callers should log the returned `adapterName` at boot time for observability.
 */
export function applyThinkingDisableToPrompt(
  ctx: ThinkingControlContext,
  systemPrompt: string,
): { systemPrompt: string; applied: boolean; adapterName: string | null } {
  const adapter = resolveThinkingAdapter(ctx);
  if (!adapter) {
    return { systemPrompt, applied: false, adapterName: null };
  }
  if (!adapter.applyToSystemPrompt) {
    return { systemPrompt, applied: false, adapterName: adapter.name };
  }
  const result = adapter.applyToSystemPrompt(systemPrompt);
  return { ...result, adapterName: adapter.name };
}

/** List all registered adapters (for debug/introspection). */
export function listThinkingAdapters(): string[] {
  return REGISTRY.map((a) => a.name);
}
