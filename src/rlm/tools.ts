/**
 * RLM AgentTool factories — creates pi-agent tools for recursive sub-agent calls.
 *
 * Three tools:
 * 1. rlm_query — spawn a sub-agent (native pi-agent-core Agent)
 * 2. rlm_check_job — check status of an async job
 * 3. rlm_cost — report cost/token usage
 */

import { Type } from '@sinclair/typebox';
import { readFileSync } from 'node:fs';
import { randomUUID } from 'node:crypto';
import { Agent } from '@mariozechner/pi-agent-core';
import type { AgentTool, AgentToolResult, AgentMessage } from '@mariozechner/pi-agent-core';
import type { RlmConfig, RlmState, RlmToolDetails, RlmAsyncJob } from './types.js';
import { checkGuardrails, parseCostFile } from './guardrails.js';

// ---------------------------------------------------------------------------
// rlm_query — primary recursion tool (native Agent execution)
// ---------------------------------------------------------------------------

/**
 * Build a system prompt for the child agent.
 * Inherits the parent's full system prompt (identity, skills, tool guidance)
 * and appends sub-agent context.
 */
function buildChildSystemPrompt(config: RlmConfig, state: RlmState): string {
  const parts: string[] = [];

  // Inherit parent's composed system prompt (base + skills + RLM guidance)
  if (state.parentSystemPrompt) {
    parts.push(state.parentSystemPrompt);
  }

  // Fall back to custom system prompt file if no parent prompt available
  if (!state.parentSystemPrompt && config.systemPromptFile) {
    try {
      parts.push(readFileSync(config.systemPromptFile, 'utf-8'));
    } catch {
      // File not found — skip
    }
  }

  // Add sub-agent context
  const childDepth = state.depth + 1;
  const maxDepth = config.maxDepth ?? 3;
  const remaining = maxDepth - childDepth;

  parts.push(`\n## Sub-Agent Context\nYou are a recursive sub-agent (depth ${childDepth}/${maxDepth}, ${remaining} level${remaining !== 1 ? 's' : ''} remaining). You have the same tools and capabilities as the parent agent. Always use your tools to perform actions — never simulate or describe performing an action without executing it.`);

  if (remaining > 0) {
    parts.push('You can spawn further sub-agents via the `rlm_query` tool if needed.');
  } else {
    parts.push('You are at the maximum recursion depth and cannot spawn further sub-agents.');
  }

  return parts.join('\n');
}

/**
 * Extract text content from agent messages.
 */
function extractText(messages: AgentMessage[]): string {
  const parts: string[] = [];
  for (const msg of messages) {
    if (msg.role !== 'assistant') continue;
    if (Array.isArray(msg.content)) {
      for (const block of msg.content) {
        if (block.type === 'text' && typeof (block as any).text === 'string') {
          parts.push((block as any).text);
        }
      }
    }
  }
  return parts.join('\n').trim();
}

/**
 * Extract total token usage from agent messages.
 */
function extractTokens(messages: AgentMessage[]): number {
  let total = 0;
  for (const msg of messages) {
    if (msg.role !== 'assistant') continue;
    const usage = (msg as any).usage;
    if (!usage) continue;
    if (typeof usage.totalTokens === 'number') total += usage.totalTokens;
    else if (typeof usage.input === 'number' && typeof usage.output === 'number') total += usage.input + usage.output;
  }
  return total;
}

/**
 * Create the rlm_query tool for spawning sub-agents.
 * Uses native pi-agent-core Agent instead of subprocess.
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
      context: Type.Optional(Type.String({ description: 'Additional context data provided to the sub-agent.' })),
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

      // Verify native execution is available
      if (!state.streamFn || !state.getParentTools) {
        return {
          content: [{ type: 'text', text: 'rlm_query failed: native execution not configured (streamFn or getParentTools missing on RlmState).' }],
          details: { rlmTool: 'rlm_query', error: 'native execution not configured' },
        };
      }

      try {
        const model = state.parentModel;
        const systemPrompt = buildChildSystemPrompt(config, state);

        // Create child Agent
        const childAgent = new Agent({
          initialState: {
            model,
            thinkingLevel: 'off',
            systemPrompt,
          },
          streamFn: state.streamFn,
          getApiKey: state.getApiKey,
        });

        // Get parent's tools, filter out rlm_query at max depth
        const parentTools = state.getParentTools() as AgentTool[];
        const childAtMaxDepth = state.depth + 1 >= maxDepth;
        const childTools = childAtMaxDepth
          ? parentTools.filter((t: AgentTool) => t.name !== 'rlm_query')
          : parentTools;
        childAgent.setTools(childTools);

        // Stream partial output if callback provided
        let unsub: (() => void) | undefined;
        if (onUpdate) {
          unsub = childAgent.subscribe((event: any) => {
            if (event.type === 'text' && event.text) {
              onUpdate({
                content: [{ type: 'text', text: event.text }],
                details: { rlmTool: 'rlm_query' },
              });
            }
          });
        }

        // Build prompt with optional context
        let fullPrompt = params.prompt;
        if (params.context) {
          fullPrompt = `<context>\n${params.context}\n</context>\n\n${params.prompt}`;
        }

        // Async mode — run in background, track as async job
        if (params.async) {
          const jobId = `rlm_native_${randomUUID().slice(0, 8)}`;
          const jobPromise = (async () => {
            const msgCountBefore = childAgent.state.messages.length;
            await childAgent.prompt(fullPrompt);
            await childAgent.waitForIdle();
            return childAgent.state.messages.slice(msgCountBefore);
          })();

          // Store as async job with a promise-based approach
          const job: RlmAsyncJob = {
            jobId,
            outputPath: '', // Not file-based in native mode
            sentinelPath: '', // Not file-based in native mode
            pid: 0, // No subprocess
            prompt: params.prompt,
          };
          // Attach the promise and unsub for later retrieval
          (job as any)._promise = jobPromise;
          (job as any)._unsub = unsub;
          (job as any)._childAgent = childAgent;
          state.asyncJobs.set(jobId, job);
          state.callCount++;

          return {
            content: [{ type: 'text', text: `Async job started: ${jobId}\nUse rlm_check_job with this job_id to check status.` }],
            details: { rlmTool: 'rlm_query', async: true, jobId },
          };
        }

        // Sync mode — run to completion
        const msgCountBefore = childAgent.state.messages.length;
        await childAgent.prompt(fullPrompt);
        await childAgent.waitForIdle();
        state.callCount++;

        if (unsub) unsub();

        const newMessages = childAgent.state.messages.slice(msgCountBefore);
        const output = extractText(newMessages);
        const totalTokens = extractTokens(newMessages);

        const details: RlmToolDetails = {
          rlmTool: 'rlm_query',
          cost: { cost: 0, tokens: totalTokens, calls: 1 },
        };

        if (childAgent.state.error) {
          details.error = childAgent.state.error;
        }

        return {
          content: [{ type: 'text', text: output || '(no output)' }],
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
        const knownIds = Array.from(state.asyncJobs.keys());
        const hint = knownIds.length > 0
          ? ` Known job IDs: ${knownIds.join(', ')}`
          : ' No async jobs have been started.';
        return {
          content: [{ type: 'text', text: `Unknown job_id: ${params.job_id}.${hint}` }],
          details: { rlmTool: 'rlm_check_job', jobId: params.job_id, error: 'unknown job' },
        };
      }

      // Native async jobs use a promise instead of sentinel files
      const promise = (job as any)._promise as Promise<AgentMessage[]> | undefined;
      if (promise) {
        // Check if the promise has resolved by racing with an immediate resolve
        const PENDING = Symbol('pending');
        const result = await Promise.race([promise, Promise.resolve(PENDING)]);

        if (result === PENDING) {
          return {
            content: [{ type: 'text', text: `Job ${params.job_id} is still running.` }],
            details: { rlmTool: 'rlm_check_job', jobId: params.job_id },
          };
        }

        // Job is done — extract output
        const messages = result as AgentMessage[];
        const output = extractText(messages);
        const totalTokens = extractTokens(messages);

        // Clean up
        const unsub = (job as any)._unsub as (() => void) | undefined;
        if (unsub) unsub();
        state.asyncJobs.delete(params.job_id);

        return {
          content: [{ type: 'text', text: output || '(no output)' }],
          details: { rlmTool: 'rlm_check_job', jobId: params.job_id, cost: { cost: 0, tokens: totalTokens, calls: 1 } },
        };
      }

      // Fallback: file-based sentinel (legacy subprocess mode)
      const { existsSync: exists } = await import('node:fs');
      if (!exists(job.sentinelPath)) {
        return {
          content: [{ type: 'text', text: `Job ${params.job_id} is still running (PID ${job.pid}).` }],
          details: { rlmTool: 'rlm_check_job', jobId: params.job_id },
        };
      }

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
