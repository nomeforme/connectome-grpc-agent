/**
 * Thinking-control barrel — interface, registry, and adapters.
 *
 * Adapter registration order matters (first-matched-wins during resolve).
 * Register more-specific adapters BEFORE more-generic ones.
 */

export type {
  ThinkingControlAdapter,
  ThinkingControlContext,
  ApplyResult,
} from './thinking-control.js';

export {
  registerThinkingAdapter,
  resolveThinkingAdapter,
  applyThinkingDisableToPrompt,
  listThinkingAdapters,
} from './thinking-control.js';

// Adapters
export { qwenNoThinkAdapter } from './adapters/qwen-no-think.js';

// ---------------------------------------------------------------------------
// Auto-registration — every adapter is registered on module import so
// consumers can just call applyThinkingDisableToPrompt() without wiring.
// ---------------------------------------------------------------------------

import { registerThinkingAdapter } from './thinking-control.js';
import { qwenNoThinkAdapter } from './adapters/qwen-no-think.js';

registerThinkingAdapter(qwenNoThinkAdapter);
// Future: register openai-reasoning, anthropic-thinking, zai-binary,
// vllm-template-kwargs here as they land.
