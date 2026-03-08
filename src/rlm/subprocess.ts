/**
 * RLM subprocess spawning — executes rlm_query as a child process.
 *
 * Keeps ypi as the source of truth for recursion mechanics. We call
 * `rlm_query` as a subprocess rather than porting the bash logic.
 */

import { spawn } from 'node:child_process';
import { readFileSync } from 'node:fs';
import type { RlmConfig, RlmState, RlmAsyncJob, RlmCostSummary } from './types.js';
import { parseCostFile } from './guardrails.js';

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

export interface RlmSyncResult {
  /** Full stdout output from the child */
  output: string;
  /** Exit code (0 = success) */
  exitCode: number;
  /** Cost summary from the ledger (if budget tracking enabled) */
  cost: RlmCostSummary | null;
}

export interface RlmAsyncResult {
  /** Parsed async job metadata */
  job: RlmAsyncJob;
}

export interface ExecOptions {
  config: RlmConfig;
  state: RlmState;
  prompt: string;
  fork?: boolean;
  context?: string;
  signal?: AbortSignal;
  onChunk?: (chunk: string) => void;
}

// ---------------------------------------------------------------------------
// Environment builder
// ---------------------------------------------------------------------------

/**
 * Build the RLM_* environment variables for the child rlm_query process.
 */
export function buildEnv(config: RlmConfig, state: RlmState): Record<string, string> {
  const env: Record<string, string> = {
    ...process.env as Record<string, string>,
    RLM_DEPTH: String(state.depth + 1),
    RLM_MAX_DEPTH: String(config.maxDepth ?? 3),
    RLM_CALL_COUNT: String(state.callCount),
    RLM_TRACE_ID: state.traceId,
    RLM_START_TIME: String(Math.floor(state.startTime / 1000)),
  };

  if (config.maxCalls != null) {
    env.RLM_MAX_CALLS = String(config.maxCalls);
  }
  if (config.budget != null && config.budget > 0) {
    env.RLM_BUDGET = String(config.budget);
  }
  if (state.costFilePath) {
    env.RLM_COST_FILE = state.costFilePath;
  }
  if (config.timeoutSeconds != null) {
    env.RLM_TIMEOUT = String(config.timeoutSeconds);
  }
  if (config.provider) {
    env.RLM_PROVIDER = config.provider;
  }
  if (config.model) {
    env.RLM_MODEL = config.model;
  }
  if (config.childModel) {
    env.RLM_CHILD_MODEL = config.childModel;
  }
  if (config.childProvider) {
    env.RLM_CHILD_PROVIDER = config.childProvider;
  }
  if (config.jjIsolation === false) {
    env.RLM_JJ = '0';
  }
  if (config.systemPromptFile) {
    env.RLM_SYSTEM_PROMPT = config.systemPromptFile;
  }
  if (config.extensions === false) {
    env.RLM_EXTENSIONS = '0';
  }
  if (config.childExtensions === false) {
    env.RLM_CHILD_EXTENSIONS = '0';
  }
  if (config.sharedSessions === false) {
    env.RLM_SHARED_SESSIONS = '0';
  }
  if (config.sessionDir) {
    env.RLM_SESSION_DIR = config.sessionDir;
  }

  return env;
}

// ---------------------------------------------------------------------------
// Sync execution
// ---------------------------------------------------------------------------

/**
 * Spawn `rlm_query` synchronously, streaming stdout via onChunk callback.
 * Returns the full output, exit code, and cost summary when done.
 */
export function execRlmSync(options: ExecOptions): Promise<RlmSyncResult> {
  const { config, state, prompt, fork, context, signal, onChunk } = options;

  return new Promise((resolve, reject) => {
    const args: string[] = [];
    if (fork) args.push('--fork');
    args.push(prompt);

    const env = buildEnv(config, state);
    const cwd = config.cwd ?? process.cwd();

    const child = spawn('rlm_query', args, {
      env,
      cwd,
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    // Track cost before this call for delta
    const costBefore = state.costFilePath ? parseCostFile(state.costFilePath) : 0;

    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (data: Buffer) => {
      const chunk = data.toString();
      stdout += chunk;
      if (onChunk) onChunk(chunk);
    });

    child.stderr.on('data', (data: Buffer) => {
      stderr += data.toString();
    });

    // Pipe context to stdin if provided
    if (context) {
      child.stdin.write(context);
      child.stdin.end();
    } else {
      child.stdin.end();
    }

    // Handle abort signal
    if (signal) {
      const onAbort = () => {
        child.kill('SIGTERM');
        setTimeout(() => {
          if (!child.killed) child.kill('SIGKILL');
        }, 5000);
      };
      if (signal.aborted) {
        onAbort();
      } else {
        signal.addEventListener('abort', onAbort, { once: true });
      }
    }

    child.on('error', (err) => {
      reject(new Error(`Failed to spawn rlm_query: ${err.message}`));
    });

    child.on('close', (code) => {
      state.callCount++;

      // Calculate cost delta
      let cost: RlmCostSummary | null = null;
      if (state.costFilePath) {
        const costAfter = parseCostFile(state.costFilePath);
        cost = {
          cost: costAfter - costBefore,
          tokens: 0, // Token count from ledger entries
          calls: 1,
        };
        // Re-parse for token totals
        try {
          const data = readFileSync(state.costFilePath, 'utf-8').trim();
          if (data) {
            let totalTokens = 0;
            for (const line of data.split('\n')) {
              try {
                const entry = JSON.parse(line);
                if (typeof entry.tokens === 'number') totalTokens += entry.tokens;
              } catch { /* skip */ }
            }
            cost.tokens = totalTokens;
          }
        } catch { /* ignore */ }
      }

      const exitCode = code ?? 1;
      if (exitCode !== 0 && !stdout.trim()) {
        // Include stderr in output on failure if stdout is empty
        resolve({
          output: stderr.trim() || `rlm_query exited with code ${exitCode}`,
          exitCode,
          cost,
        });
      } else {
        resolve({ output: stdout, exitCode, cost });
      }
    });
  });
}

// ---------------------------------------------------------------------------
// Async execution
// ---------------------------------------------------------------------------

/**
 * Spawn `rlm_query --async`, parse the JSON job metadata from stdout,
 * and store it in state.asyncJobs.
 */
export function execRlmAsync(options: ExecOptions): Promise<RlmAsyncResult> {
  const { config, state, prompt, fork, context } = options;

  return new Promise((resolve, reject) => {
    const args: string[] = [];
    if (fork) args.push('--fork');
    args.push('--async');
    args.push(prompt);

    const env = buildEnv(config, state);
    const cwd = config.cwd ?? process.cwd();

    const child = spawn('rlm_query', args, {
      env,
      cwd,
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (data: Buffer) => {
      stdout += data.toString();
    });

    child.stderr.on('data', (data: Buffer) => {
      stderr += data.toString();
    });

    if (context) {
      child.stdin.write(context);
      child.stdin.end();
    } else {
      child.stdin.end();
    }

    child.on('error', (err) => {
      reject(new Error(`Failed to spawn rlm_query --async: ${err.message}`));
    });

    child.on('close', (code) => {
      state.callCount++;

      if (code !== 0) {
        reject(new Error(`rlm_query --async failed (code ${code}): ${stderr || stdout}`));
        return;
      }

      // Parse the JSON job metadata from stdout
      try {
        const jobData = JSON.parse(stdout.trim());
        const job: RlmAsyncJob = {
          jobId: jobData.job_id,
          outputPath: jobData.output,
          sentinelPath: jobData.sentinel,
          pid: jobData.pid,
          prompt,
        };
        state.asyncJobs.set(job.jobId, job);
        resolve({ job });
      } catch (err) {
        reject(new Error(`Failed to parse rlm_query --async output: ${stdout.trim()}`));
      }
    });
  });
}
