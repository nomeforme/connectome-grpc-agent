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
  // e.g. '400 {"type":"error","error":{"type":"...","message":"..."}}'
  try {
    const jsonStart = message.indexOf('{');
    if (jsonStart >= 0) {
      const json = JSON.parse(message.slice(jsonStart));
      if (json.error?.message) {
        return `[Error] ${json.error.message}`;
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

  /** Streams currently being processed — prevents parallel cycles on the same stream. */
  private readonly processingStreams: Set<string> = new Set();

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

    // Deduplicate — don't run two cycles on the same stream concurrently
    if (this.processingStreams.has(streamId)) {
      return null;
    }
    this.processingStreams.add(streamId);

    let typingInterval: ReturnType<typeof setInterval> | undefined;

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

      // Build stream ref for the agent cycle
      const streamRef = {
        streamId,
        streamType: platformContext.streamType,
      };

      // Per-turn speech: subscribe to message_end events BEFORE running the agent
      let turnEmitted = false;
      let unsub: (() => void) | undefined;

      if (this.agent.subscribe && this.speechRecorder) {
        unsub = this.agent.subscribe((event: AgentEvent) => {
          if (event.type === 'message_end') {
            const text = extractTurnText(event.message);
            if (text) {
              const cleaned = cleanSpeechContent(text);
              if (cleaned) {
                turnEmitted = true;
                // Fire-and-forget: don't block the agent loop
                this.speechRecorder!.recordSpeech(cleaned, {
                  agentId: this.agent.id,
                  agentName: this.agent.name,
                  streamId,
                }).catch(err => console.error('[ConnectomeEffector] Per-turn speech failed:', err));
              }
            }
          }
        });
      }

      // Run the agent cycle
      const result = await this.agent.runWithContext(context, streamRef, activation.continuation);
      unsub?.();

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
            await this.speechRecorder.recordSpeech(cleaned, {
              agentId: this.agent.id,
              agentName: this.agent.name,
              streamId,
              attachments: attachments.length > 0 ? attachments : undefined,
            });
          } else if (this.speechRecorder && attachments.length > 0) {
            // Per-turn already emitted text, but we have attachments to send
            await this.speechRecorder.recordSpeech('', {
              agentId: this.agent.id,
              agentName: this.agent.name,
              streamId,
              attachments,
            });
          }
        }
      } else if (this.speechRecorder && attachments.length > 0) {
        // Agent produced no text but queued attachments
        await this.speechRecorder.recordSpeech('', {
          agentId: this.agent.id,
          agentName: this.agent.name,
          streamId,
          attachments,
        });
      }

      return result;
    } catch (err: unknown) {
      const error = err instanceof Error ? err : new Error(String(err));
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
      if (typingInterval) clearInterval(typingInterval);
      this.processingStreams.delete(streamId);
    }
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
