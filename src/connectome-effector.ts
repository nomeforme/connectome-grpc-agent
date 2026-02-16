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
} from './types.js';
import { cleanSpeechContent } from './utils.js';

export class ConnectomeEffector {
  private readonly agent: EffectorAgent;
  private readonly adapter: PlatformAdapter;
  private readonly contextProvider: ContextProvider;
  private readonly speechRecorder: SpeechRecorder | undefined;
  private readonly typingRefreshMs: number;
  private readonly maxFrames: number;
  private readonly onError?: (error: Error, activation: UnifiedActivation) => void;

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

      // Run the agent cycle
      const result = await this.agent.runWithContext(context, streamRef);

      // Deliver speech if the agent produced any
      if (result.content) {
        const cleaned = cleanSpeechContent(result.content);

        if (cleaned) {
          // Deliver to platform (adapter handles formatting, splitting, sending)
          await this.adapter.deliverSpeech(cleaned, platformContext);

          // Record on server (so it appears in VEIL state for all participants)
          if (this.speechRecorder) {
            await this.speechRecorder.recordSpeech(cleaned, {
              agentId: this.agent.id,
              agentName: this.agent.name,
              streamId,
            });
          }
        }
      }

      return result;
    } catch (err: unknown) {
      const error = err instanceof Error ? err : new Error(String(err));
      if (this.onError) {
        this.onError(error, activation);
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
