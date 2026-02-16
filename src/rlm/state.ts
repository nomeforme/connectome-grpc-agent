/**
 * RLM state initialization.
 *
 * Creates the runtime state for an RLM-enabled agent, including
 * trace ID generation and cost file setup.
 */

import { randomUUID } from 'node:crypto';
import { writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import type { RlmConfig, RlmState } from './types.js';

/**
 * Initialize RLM runtime state from config.
 * Called once during ConnectomeAgent construction when `config.rlm` is set.
 */
export function initRlmState(config: RlmConfig): RlmState {
  const traceId = randomUUID().replace(/-/g, '').slice(0, 16);
  const startTime = Date.now();

  // Create cost file if budget tracking is enabled
  let costFilePath: string | null = null;
  if (config.budget != null && config.budget > 0) {
    costFilePath = join(tmpdir(), `rlm_cost_${traceId}.jsonl`);
    writeFileSync(costFilePath, '', 'utf-8');
  }

  return {
    depth: 0,
    callCount: 0,
    traceId,
    startTime,
    costFilePath,
    asyncJobs: new Map(),
  };
}
