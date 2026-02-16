/**
 * RLM AgentTool factories — creates pi-agent tools for recursive sub-agent calls.
 *
 * Three tools:
 * 1. rlm_query — spawn a sub-agent (sync or async)
 * 2. rlm_check_job — check status of an async job
 * 3. rlm_cost — report cost/token usage
 */

import { Type } from '@sinclair/typebox';
import { existsSync, readFileSync } from 'node:fs';
import type { AgentTool, AgentToolResult } from '@mariozechner/pi-agent-core';
import type { RlmConfig, RlmState, RlmToolDetails } from './types.js';
import { checkGuardrails, parseCostFile } from './guardrails.js';
import { execRlmSync, execRlmAsync } from './subprocess.js';

// ---------------------------------------------------------------------------
// rlm_query — primary recursion tool
// ---------------------------------------------------------------------------

/**
 * Create the rlm_query tool for spawning sub-agents.
 */
export function createRlmQueryTool(config: RlmConfig, state: RlmState): AgentTool<any, RlmToolDetails> {
  const maxDepth = config.maxDepth ?? 3;

  return {
    name: 'rlm_query',
    label: 'Sub-Agent Query',
    get description() {
      const remaining = maxDepth - state.depth;
      const budgetInfo = config.budget != null && state.costFilePath
        ? ` Budget: $${config.budget.toFixed(2)} (spent: $${parseCostFile(state.costFilePath).toFixed(4)}).`
        : '';
      return (
        `Spawn a recursive sub-agent to handle a focused subtask. ` +
        `The child gets a fresh context window and full tool access. ` +
        `Current depth: ${state.depth}/${maxDepth} (${remaining} levels remaining).` +
        budgetInfo +
        ` Use this for tasks that benefit from decomposition: research, analysis, ` +
        `code generation, or any work that can be done independently.`
      );
    },
    parameters: Type.Object({
      prompt: Type.String({ description: 'The task or question for the sub-agent. Be specific and self-contained.' }),
      fork: Type.Optional(Type.Boolean({ description: 'Fork parent session into child, carrying conversation history. Default: false (fresh context).' })),
      async: Type.Optional(Type.Boolean({ description: 'Run in background, returning immediately with a job ID. Use rlm_check_job to poll for results.' })),
      context: Type.Optional(Type.String({ description: 'Additional context data to pipe to the sub-agent via stdin.' })),
    }),
    execute: async (
      _toolCallId: string,
      params: { prompt: string; fork?: boolean; async?: boolean; context?: string },
      signal?: AbortSignal,
      onUpdate?: (partialResult: AgentToolResult<RlmToolDetails>) => void,
    ): Promise<AgentToolResult<RlmToolDetails>> => {
      // Pre-flight guardrail check
      const guardrailError = checkGuardrails(config, state);
      if (guardrailError) {
        return {
          content: [{ type: 'text', text: guardrailError }],
          details: { rlmTool: 'rlm_query', error: guardrailError },
        };
      }

      try {
        // Async mode
        if (params.async) {
          const result = await execRlmAsync({
            config,
            state,
            prompt: params.prompt,
            fork: params.fork,
            context: params.context,
          });

          return {
            content: [{ type: 'text', text: `Async job started: ${result.job.jobId}\nOutput will be at: ${result.job.outputPath}\nUse rlm_check_job with this job_id to check status.` }],
            details: { rlmTool: 'rlm_query', async: true, jobId: result.job.jobId },
          };
        }

        // Sync mode — stream partial output via onUpdate
        const result = await execRlmSync({
          config,
          state,
          prompt: params.prompt,
          fork: params.fork,
          context: params.context,
          signal,
          onChunk: onUpdate
            ? (chunk) => {
                onUpdate({
                  content: [{ type: 'text', text: chunk }],
                  details: { rlmTool: 'rlm_query' },
                });
              }
            : undefined,
        });

        const details: RlmToolDetails = {
          rlmTool: 'rlm_query',
          cost: result.cost ?? undefined,
        };

        if (result.exitCode !== 0) {
          details.error = `Exit code ${result.exitCode}`;
        }

        return {
          content: [{ type: 'text', text: result.output || '(no output)' }],
          details,
        };
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return {
          content: [{ type: 'text', text: `rlm_query failed: ${message}` }],
          details: { rlmTool: 'rlm_query', error: message },
        };
      }
    },
  };
}

// ---------------------------------------------------------------------------
// rlm_check_job — async job status checker
// ---------------------------------------------------------------------------

/**
 * Create the rlm_check_job tool for checking async job status.
 */
export function createRlmCheckJobTool(_config: RlmConfig, state: RlmState): AgentTool<any, RlmToolDetails> {
  return {
    name: 'rlm_check_job',
    label: 'Check Async Job',
    description: 'Check the status of an async sub-agent job. Returns the output if the job is complete.',
    parameters: Type.Object({
      job_id: Type.String({ description: 'The job_id returned by a previous rlm_query with async=true.' }),
    }),
    execute: async (
      _toolCallId: string,
      params: { job_id: string },
    ): Promise<AgentToolResult<RlmToolDetails>> => {
      const job = state.asyncJobs.get(params.job_id);

      if (!job) {
        // List known jobs to help the agent
        const knownIds = Array.from(state.asyncJobs.keys());
        const hint = knownIds.length > 0
          ? ` Known job IDs: ${knownIds.join(', ')}`
          : ' No async jobs have been started.';
        return {
          content: [{ type: 'text', text: `Unknown job_id: ${params.job_id}.${hint}` }],
          details: { rlmTool: 'rlm_check_job', jobId: params.job_id, error: 'unknown job' },
        };
      }

      // Check sentinel file
      const done = existsSync(job.sentinelPath);

      if (!done) {
        return {
          content: [{ type: 'text', text: `Job ${params.job_id} is still running (PID ${job.pid}).` }],
          details: { rlmTool: 'rlm_check_job', jobId: params.job_id },
        };
      }

      // Job is done — read output
      try {
        const output = readFileSync(job.outputPath, 'utf-8');
        return {
          content: [{ type: 'text', text: output || '(no output)' }],
          details: { rlmTool: 'rlm_check_job', jobId: params.job_id },
        };
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return {
          content: [{ type: 'text', text: `Job ${params.job_id} completed but output could not be read: ${message}` }],
          details: { rlmTool: 'rlm_check_job', jobId: params.job_id, error: message },
        };
      }
    },
  };
}

// ---------------------------------------------------------------------------
// rlm_cost — cost reporter
// ---------------------------------------------------------------------------

/**
 * Create the rlm_cost tool for reporting cost/token usage.
 */
export function createRlmCostTool(config: RlmConfig, state: RlmState): AgentTool<any, RlmToolDetails> {
  return {
    name: 'rlm_cost',
    label: 'RLM Cost Report',
    description: 'Report the cumulative cost, token usage, and call count for this recursive agent tree.',
    parameters: Type.Object({}),
    execute: async (): Promise<AgentToolResult<RlmToolDetails>> => {
      if (!state.costFilePath) {
        return {
          content: [{ type: 'text', text: `Cost tracking not enabled (no budget configured). Calls so far: ${state.callCount}.` }],
          details: { rlmTool: 'rlm_cost' },
        };
      }

      // Parse the full cost file
      let totalCost = 0;
      let totalTokens = 0;
      let entries = 0;

      try {
        const data = readFileSync(state.costFilePath, 'utf-8').trim();
        if (data) {
          for (const line of data.split('\n')) {
            if (!line.trim()) continue;
            try {
              const entry = JSON.parse(line);
              if (typeof entry.cost === 'number') totalCost += entry.cost;
              if (typeof entry.tokens === 'number') totalTokens += entry.tokens;
              entries++;
            } catch { /* skip */ }
          }
        }
      } catch { /* ignore */ }

      const budgetStr = config.budget != null ? ` / $${config.budget.toFixed(2)} budget` : '';
      const summary = [
        `Cost: $${totalCost.toFixed(4)}${budgetStr}`,
        `Tokens: ${totalTokens}`,
        `Ledger entries: ${entries}`,
        `Total rlm_query calls: ${state.callCount}`,
        `Elapsed: ${((Date.now() - state.startTime) / 1000).toFixed(1)}s`,
      ].join('\n');

      const costSummary = { cost: totalCost, tokens: totalTokens, calls: entries };

      return {
        content: [{ type: 'text', text: summary }],
        details: { rlmTool: 'rlm_cost', cost: costSummary },
      };
    },
  };
}
