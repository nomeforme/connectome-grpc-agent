/**
 * TTSProvider — abstract interface for text-to-speech synthesis.
 *
 * Providers convert cleaned agent speech text into an audio attachment that
 * the ConnectomeEffector piggybacks onto the standard `recordSpeech` path.
 * Once attached, axon speech effectors (discord/signal/whatsapp) resolve the
 * audio blob and deliver it as a file attachment on the same message.
 *
 * Design: the provider is bot-scoped (voice is tied to the bot's identity,
 * not the platform), synthesis is per-final-message (per-turn is text-only),
 * failure is non-fatal (text always ships, audio is best-effort).
 */

/** Synthesized audio result — ready to attach to a speech facet. */
export interface TTSSynthesisResult {
  /** Raw audio bytes (mp3/opus/wav/etc). */
  data: Buffer;
  /** MIME type, e.g. "audio/mpeg" for mp3, "audio/ogg" for opus. */
  contentType: string;
  /** Suggested filename with extension, e.g. "voice-1704067200000.mp3". */
  filename: string;
  /** Duration in ms if known (informational — may be undefined). */
  durationMs?: number;
}

/** Optional per-call overrides (usually left unset — bot config decides). */
export interface TTSSynthesisOptions {
  /** Override the provider's default voice for this call. */
  voice?: string;
  /** Override output format (e.g. "mp3", "opus"). */
  format?: string;
  /** Override speaking speed (0.5–2.0). */
  speed?: number;
  /** Abort signal for cancellation. */
  signal?: AbortSignal;
}

/** Provider interface. */
export interface TTSProvider {
  /** Provider name for logging (e.g. "omnivoice"). */
  readonly name: string;
  /**
   * Synthesize text into audio. Should throw on unrecoverable errors so the
   * effector can log and ship text without audio. Should honor `opts.signal`.
   */
  synthesize(text: string, opts?: TTSSynthesisOptions): Promise<TTSSynthesisResult>;
}

/**
 * Format → MIME type + file extension mapping. Handled here so providers
 * don't need to reinvent it.
 */
export function contentTypeForFormat(format: string): { contentType: string; extension: string } {
  switch (format.toLowerCase()) {
    case 'mp3':
      return { contentType: 'audio/mpeg', extension: 'mp3' };
    case 'opus':
      return { contentType: 'audio/ogg', extension: 'opus' };
    case 'aac':
      return { contentType: 'audio/aac', extension: 'aac' };
    case 'flac':
      return { contentType: 'audio/flac', extension: 'flac' };
    case 'wav':
      return { contentType: 'audio/wav', extension: 'wav' };
    case 'pcm':
      return { contentType: 'audio/L16', extension: 'pcm' };
    default:
      return { contentType: 'application/octet-stream', extension: format };
  }
}
