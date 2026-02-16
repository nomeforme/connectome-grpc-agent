/**
 * RLM guardrails — pre-validation before spawning rlm_query subprocesses.
 *
 * Checks depth, call count, budget, and timeout limits. Returns an error
 * message string if a guardrail is tripped, or null if safe to proceed.
 */

import { readFileSync } from 'node:fs';
import type { RlmConfig, RlmState } from './types.js';

/**
 * Parse the JSONL cost ledger and sum the cost field across all entries.
 */
export function parseCostFile(path: string): number {
  let totalCost = 0;
  try {
    const data = readFileSync(path, 'utf-8').trim();
    if (!data) return 0;
    for (const line of data.split('\n')) {
      if (!line.trim()) continue;
      try {
        const entry = JSON.parse(line);
        if (typeof entry.cost === 'number') {
          totalCost += entry.cost;
        }
      } catch {
        // Skip malformed lines
      }
    }
  } catch {
    // File doesn't exist or can't be read — cost is 0
  }
  return totalCost;
}

/**
 * Check all guardrails before spawning an rlm_query subprocess.
 * Returns an error message if any guardrail is tripped, or null if clear.
 */
export function checkGuardrails(config: RlmConfig, state: RlmState): string | null {
  // Depth guard
  const maxDepth = config.maxDepth ?? 3;
  if (state.depth >= maxDepth) {
    return `RLM depth limit reached (${state.depth}/${maxDepth}). Cannot recurse further.`;
  }

  // Call count guard
  if (config.maxCalls != null && state.callCount >= config.maxCalls) {
    return `RLM call limit reached (${state.callCount}/${config.maxCalls}). No more sub-agent calls allowed.`;
  }

  // Budget guard
  if (config.budget != null && config.budget > 0 && state.costFilePath) {
    const spent = parseCostFile(state.costFilePath);
    if (spent >= config.budget) {
      return `RLM budget exhausted ($${spent.toFixed(4)} >= $${config.budget.toFixed(4)}). No more sub-agent calls allowed.`;
    }
  }

  // Timeout guard
  if (config.timeoutSeconds != null && config.timeoutSeconds > 0) {
    const elapsedSeconds = (Date.now() - state.startTime) / 1000;
    if (elapsedSeconds >= config.timeoutSeconds) {
      return `RLM timeout reached (${elapsedSeconds.toFixed(0)}s >= ${config.timeoutSeconds}s). No more sub-agent calls allowed.`;
    }
  }

  return null;
}
