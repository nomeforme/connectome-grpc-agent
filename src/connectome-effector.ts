/**
 * ConnectomeEffector — unified activation→cycle→delivery pipeline.
 *
 * Replaces the duplicated DiscordAgentEffector / SignalAgentEffector with a
 * single orchestrator that delegates platform-specific concerns to a
 * PlatformAdapter.
 *
 * Flow:
 *   1. Receive UnifiedActivation
 *   2. Deduplicate (skip if stream already processing)
 *   3. Start typing indicator (refresh on interval)
 *   4. Fetch context via ContextProvider
 *   5. Run agent cycle via EffectorAgent.runWithContext()
 *   6. Clean speech content (shared utility)
 *   7. Deliver via PlatformAdapter.deliverSpeech()
 *   8. Record speech via SpeechRecorder (if configured)
 *   9. Return ConnectomeCycleResult
 */

import type {
  ConnectomeEffectorConfig,
  ConnectomeCycleResult,
  EffectorAgent,
  PlatformAdapter,
  ContextProvider,
  SpeechRecorder,
  PlatformContext,
  UnifiedActivation,
  AgentEvent,
} from './types.js';
import { cleanSpeechContent } from './utils.js';

/**
 * Format an error message for display in a chat platform.
 * Extracts human-readable messages from JSON API errors when possible.
 */
function formatErrorForChat(message: string): string {
  // Try to extract a human-readable message from JSON API errors
  // e.g. '400 {"type":"error","error":{"type":"api_error","message":"Internal server error"}}'
  try {
    const jsonStart = message.indexOf('{');
    if (jsonStart >= 0) {
      const json = JSON.parse(message.slice(jsonStart));
      if (json.error?.message) {
        const errType = json.error.type || 'unknown';
        return `[Anthropic API ${errType}] ${json.error.message}`;
      }
    }
  } catch {}
  // Fallback: use raw message, truncated
  const truncated = message.length > 500 ? message.slice(0, 500) + '...' : message;
  return `[Error] ${truncated}`;
}

/**
 * Extract text content from a single assistant message's content blocks.
 */
function extractTurnText(message: any): string | null {
  if (!message || message.role !== 'assistant') return null;
  const content = message.content;
  if (!Array.isArray(content)) return null;
  const text = content
    .filter((b: any) => b.type === 'text')
    .map((b: any) => b.text)
    .join('\n')
    .trim();
  return text || null;
}

export class ConnectomeEffector {
  private readonly agent: EffectorAgent;
  private readonly adapter: PlatformAdapter;
  private readonly contextProvider: ContextProvider;
  private readonly speechRecorder: SpeechRecorder | undefined;
  private readonly typingRefreshMs: number;
  private readonly maxFrames: number;
  private readonly onError?: (error: Error, activation: UnifiedActivation) => void;
  private readonly drainAttachments?: () => Array<{
    id: string; contentType: string; data: string; filename?: string; sizeBytes?: number;
  }>;

  /** Streams currently being processed — prevents back-to-back activations on the
   *  SAME stream from racing the same per-stream pi-agent's `_state.isStreaming`.
   *  Cross-stream concurrency is fully native: each stream has its own pi-agent
   *  instance via the per-stream pool inside ConnectomeAgent. */
  private readonly processingStreams: Set<string> = new Set();

  /** Streams where abort was requested — prevents error recording for intentional stops. */
  private readonly abortedStreams: Set<string> = new Set();

  constructor(config: ConnectomeEffectorConfig) {
    this.agent = config.agent;
    this.adapter = config.adapter;
    this.contextProvider = config.contextProvider;
    this.speechRecorder = config.speechRecorder;
    this.typingRefreshMs = config.typingRefreshMs ?? 8000;
    this.maxFrames = config.maxFrames ?? 200;
    this.onError = config.onError;
    this.drainAttachments = config.drainAttachments;
  }

  // ---------------------------------------------------------------------------
  // Primary API
  // ---------------------------------------------------------------------------

  /**
   * Handle an activation: run the full agent cycle and deliver the result.
   *
   * Returns the cycle result on success, or null if the activation was
   * skipped (already processing) or the agent produced no speech.
   */
  async handleActivation(
    activation: UnifiedActivation,
  ): Promise<ConnectomeCycleResult | null> {
    const { streamId, platformContext } = activation;
    const prefix = `[ConnectomeEffector:${this.agent.name}]`;

    // Per-stream dedup — skip if THIS stream already has an active cycle.
    // Each stream owns its own pi-agent instance, so different streams can
    // run truly in parallel; this guard only suppresses literal same-stream
    // duplicates (e.g. multiple rapid activations on the same channel).
    if (this.processingStreams.has(streamId)) {
      console.log(`${prefix} Skipping duplicate activation on ${streamId} — already processing`);
      return null;
    }
    this.processingStreams.add(streamId);

    let typingInterval: ReturnType<typeof setInterval> | undefined;
    let unsub: (() => void) | undefined;
    const cycleStart = Date.now();

    try {
      // Start typing indicator
      await this.adapter.sendTypingIndicator(platformContext).catch(() => {});
      typingInterval = setInterval(() => {
        this.adapter.sendTypingIndicator(platformContext).catch(() => {});
      }, this.typingRefreshMs);

      // Fetch context (messages + system prompt)
      const context = await this.contextProvider.getContext(streamId, {
        maxFrames: this.maxFrames,
      });
      console.log(`${prefix} Context: ${context.messages.length} messages, sysprompt ${context.systemPrompt.length} chars`);

      // Build stream ref for the agent cycle
      const streamRef = {
        streamId,
        streamType: platformContext.streamType,
      };

      // Per-turn speech: subscribe to message_end events BEFORE running the agent
      let turnEmitted = false;
      let turnCount = 0;

      if (this.agent.subscribe && this.speechRecorder) {
        // Subscribe to the per-stream pi-agent so we only see events for THIS
        // stream's cycle, even if the bot is concurrently running cycles on
        // other streams.
        unsub = this.agent.subscribe((event: AgentEvent) => {
          if (event.type === 'message_end') {
            turnCount++;
            const text = extractTurnText(event.message);
            const textLen = text?.length ?? 0;
            const contentTypes = Array.isArray((event.message as any)?.content)
              ? (event.message as any).content.map((b: any) => b.type).join(',')
              : 'none';
            console.log(`${prefix} message_end #${turnCount}: role=${(event.message as any)?.role} contentTypes=[${contentTypes}] textLen=${textLen}`);
            if (text) {
              const cleaned = cleanSpeechContent(text);
              if (cleaned) {
                turnEmitted = true;
                console.log(`${prefix} Per-turn speech #${turnCount}: ${cleaned.length} chars`);
                // Fire-and-forget: don't block the agent loop
                this.speechRecorder!.recordSpeech(cleaned, {
                  agentId: this.agent.id,
                  agentName: this.agent.name,
                  streamId,
                  cyclePending: true,
                }).catch(err => console.error(`${prefix} Per-turn speech failed:`, err));
              }
            }
          }
        }, streamId);
      }

      // Run the agent cycle
      console.log(`${prefix} Running agent cycle...`);
      const result = await this.agent.runWithContext(context, streamRef, activation.continuation);
      const elapsed = ((Date.now() - cycleStart) / 1000).toFixed(1);

      console.log(`${prefix} Cycle complete: ${elapsed}s, ${result.messages?.length ?? 0} new messages, ${result.tokensUsed ?? 0} tokens, content=${result.content?.length ?? 0} chars, turnEmitted=${turnEmitted}, turns=${turnCount}`);

      // Drain queued attachments (e.g. from attach_file tool)
      const attachments = this.drainAttachments?.() ?? [];

      // Deliver speech if the agent produced any
      if (result.content) {
        const cleaned = cleanSpeechContent(result.content);

        if (cleaned) {
          // Deliver to platform (adapter handles formatting, splitting, sending)
          await this.adapter.deliverSpeech(cleaned, platformContext);

          // Record on server only if per-turn didn't already emit
          // (avoids duplicating the full concatenated output)
          if (this.speechRecorder && !turnEmitted) {
            console.log(`${prefix} Recording final speech: ${cleaned.length} chars`);
            await this.speechRecorder.recordSpeech(cleaned, {
              agentId: this.agent.id,
              agentName: this.agent.name,
              streamId,
              attachments: attachments.length > 0 ? attachments : undefined,
            });
          } else if (this.speechRecorder && attachments.length > 0) {
            console.log(`${prefix} Per-turn already emitted, recording attachments only`);
            // Per-turn already emitted text, but we have attachments to send
            await this.speechRecorder.recordSpeech('', {
              agentId: this.agent.id,
              agentName: this.agent.name,
              streamId,
              attachments,
            });
          } else if (turnEmitted) {
            console.log(`${prefix} Per-turn already emitted ${turnCount} turn(s), skipping final record`);
          }
        } else {
          console.log(`${prefix} Content produced but cleaned to empty`);
        }
      } else {
        console.log(`${prefix} No content produced`);
        if (this.speechRecorder && attachments.length > 0) {
          // Agent produced no text but queued attachments
          await this.speechRecorder.recordSpeech('', {
            agentId: this.agent.id,
            agentName: this.agent.name,
            streamId,
            attachments,
          });
        }
      }

      return result;
    } catch (err: unknown) {
      const error = err instanceof Error ? err : new Error(String(err));
      const elapsed = ((Date.now() - cycleStart) / 1000).toFixed(1);

      // Check if this was an intentional abort (!stop command)
      if (this.abortedStreams.has(streamId)) {
        console.log(`${prefix} Cycle stopped by user after ${elapsed}s`);
        if (this.speechRecorder) {
          this.speechRecorder.recordSpeech('[Cycle stopped]', {
            agentId: this.agent.id,
            agentName: this.agent.name,
            streamId,
          }).catch(() => {});
        }
        return null;
      }

      console.error(`${prefix} Cycle FAILED after ${elapsed}s: ${error.message}`);
      if (this.onError) {
        this.onError(error, activation);
      }
      // Record error as speech so it appears in the chat platform
      if (this.speechRecorder) {
        const errorMsg = formatErrorForChat(error.message);
        this.speechRecorder.recordSpeech(errorMsg, {
          agentId: this.agent.id,
          agentName: this.agent.name,
          streamId,
        }).catch(() => {});
      }
      return null;
    } finally {
      unsub?.();
      if (typingInterval) clearInterval(typingInterval);
      this.processingStreams.delete(streamId);
      // Reset only THIS stream's pi-agent — prevents stuck "isStreaming" on
      // failed/aborted cycles without disturbing other streams' in-flight work.
      try { this.agent.resetStream?.(streamId); } catch { /* ignore */ }
      this.abortedStreams.delete(streamId);
    }
  }

  // ---------------------------------------------------------------------------
  // Agent control (stop / steer)
  // ---------------------------------------------------------------------------

  /**
   * Abort the current agent cycle. Returns true if there was an active cycle.
   * The cycle's catch block will detect the abort and emit a clean confirmation
   * instead of an error message.
   */
  abort(): boolean {
    if (this.processingStreams.size === 0) return false;
    // Mark all active streams as intentionally aborted
    for (const sid of this.processingStreams) {
      this.abortedStreams.add(sid);
    }
    if (this.agent.abort) {
      this.agent.abort();
      console.log(`[ConnectomeEffector:${this.agent.name}] Abort requested (${this.processingStreams.size} active stream(s))`);
    }
    return true;
  }

  /**
   * Steer the agent mid-run by injecting a user message. Returns true if
   * there was an active cycle to steer.
   */
  steer(message: string): boolean {
    if (this.processingStreams.size === 0) return false;
    if (this.agent.steer) {
      this.agent.steer(message);
      console.log(`[ConnectomeEffector:${this.agent.name}] Steer injected: ${message.substring(0, 80)}`);
    }
    return true;
  }

  // ---------------------------------------------------------------------------
  // Accessors
  // ---------------------------------------------------------------------------

  /** Check if a stream is currently being processed. */
  isProcessing(streamId: string): boolean {
    return this.processingStreams.has(streamId);
  }

  /** Get the number of streams currently being processed. */
  get activeCount(): number {
    return this.processingStreams.size;
  }

  /** Access the underlying platform adapter. */
  getAdapter(): PlatformAdapter {
    return this.adapter;
  }

  /** Access the underlying agent. */
  getAgent(): EffectorAgent {
    return this.agent;
  }
}
