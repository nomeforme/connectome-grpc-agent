/**
 * RLM system prompt fragment — appended to the agent's system prompt
 * to teach it about recursive sub-agent decomposition.
 */

import type { RlmConfig, RlmState } from './types.js';
import { parseCostFile } from './guardrails.js';

/**
 * Build the recursion guidance section appended to the agent's system prompt.
 * Covers: when to recurse, decomposition patterns, async vs sync,
 * guardrail status, and depth awareness.
 */
export function buildRlmSystemPromptFragment(config: RlmConfig, state: RlmState): string {
  const maxDepth = config.maxDepth ?? 3;
  const remaining = maxDepth - state.depth;

  let budgetLine = '';
  if (config.budget != null && config.budget > 0 && state.costFilePath) {
    const spent = parseCostFile(state.costFilePath);
    budgetLine = `\n- Budget: $${spent.toFixed(4)} spent of $${config.budget.toFixed(2)} limit.`;
  }

  let callLimitLine = '';
  if (config.maxCalls != null) {
    callLimitLine = `\n- Call limit: ${state.callCount}/${config.maxCalls} calls used.`;
  }

  let timeoutLine = '';
  if (config.timeoutSeconds != null && config.timeoutSeconds > 0) {
    const elapsed = ((Date.now() - state.startTime) / 1000).toFixed(0);
    timeoutLine = `\n- Timeout: ${elapsed}s elapsed of ${config.timeoutSeconds}s limit.`;
  }

  return `

## Recursive Sub-Agent (RLM)

You have access to recursive sub-agents via the \`rlm_query\` tool. Each sub-agent gets a fresh context window and full tool access.

### Current Status
- Depth: ${state.depth}/${maxDepth} (${remaining} level${remaining !== 1 ? 's' : ''} remaining)${budgetLine}${callLimitLine}${timeoutLine}

### When to Use rlm_query
- **Decomposition**: Break complex tasks into independent subtasks that can be solved in isolation.
- **Fresh context**: When a subtask needs a clean context window (e.g., analyzing a different file, separate research).
- **Parallel work**: Use \`async: true\` to spawn multiple sub-agents, then collect results with \`rlm_check_job\`.

### Best Practices
- Be specific and self-contained in prompts — the child has no access to your conversation history (unless \`fork: true\`).
- Prefer sync mode for single subtasks; use async for 2+ independent subtasks.
- Check cost with \`rlm_cost\` if running many sub-agents.
${remaining <= 1 ? '- **WARNING**: You are near the recursion depth limit. Sub-agents at max depth cannot recurse further.\n' : ''}`;
}
