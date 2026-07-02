/**
 * OmniVoiceTTSProvider — TTSProvider backed by an OmniVoice API instance.
 *
 * OmniVoice speaks the OpenAI TTS shape (`POST /v1/audio/speech`), so this
 * provider is a thin fetch wrapper. Runs against a self-hosted endpoint
 * (typically over Tailscale, e.g. `http://REDACTED-IP:8000`).
 *
 * Failure semantics: throws on HTTP error, timeout, or abort. Caller
 * (ConnectomeEffector) catches and ships text-only.
 */

import type { TTSProvider, TTSSynthesisOptions, TTSSynthesisResult } from './tts-provider.js';
import { contentTypeForFormat } from './tts-provider.js';

export interface OmniVoiceProviderConfig {
  /** Base URL of the OmniVoice API (no trailing `/v1`). */
  endpoint: string;
  /** Voice ID — e.g. "auto", "alloy", "clone:plantony", or design string. */
  voice: string;
  /** Output format (default: "mp3"). */
  format?: string;
  /** Speaking speed 0.5–2.0 (default: 1.0). */
  speed?: number;
  /** Request timeout in ms (default: 30000). */
  timeoutMs?: number;
  /** Max input length in chars — synth is skipped over this (default: 4000). */
  maxInputChars?: number;
  /** Model identifier (default: "omnivoice"). */
  model?: string;
}

export class OmniVoiceTTSProvider implements TTSProvider {
  readonly name = 'omnivoice';

  private readonly endpoint: string;
  private readonly voice: string;
  private readonly format: string;
  private readonly speed: number;
  private readonly timeoutMs: number;
  private readonly maxInputChars: number;
  private readonly model: string;

  constructor(config: OmniVoiceProviderConfig) {
    // Strip any trailing slash / `/v1` — we append `/v1/audio/speech` ourselves.
    this.endpoint = config.endpoint.replace(/\/+$/, '').replace(/\/v1$/, '');
    this.voice = config.voice;
    this.format = config.format ?? 'mp3';
    this.speed = config.speed ?? 1.0;
    this.timeoutMs = config.timeoutMs ?? 30000;
    this.maxInputChars = config.maxInputChars ?? 4000;
    this.model = config.model ?? 'omnivoice';
  }

  async synthesize(text: string, opts?: TTSSynthesisOptions): Promise<TTSSynthesisResult> {
    const trimmed = text.trim();
    if (!trimmed) {
      throw new Error('OmniVoice: empty input text');
    }
    if (trimmed.length > this.maxInputChars) {
      throw new Error(
        `OmniVoice: input ${trimmed.length} chars exceeds max_input_chars ${this.maxInputChars}`,
      );
    }

    const voice = opts?.voice ?? this.voice;
    const format = (opts?.format ?? this.format).toLowerCase();
    const speed = opts?.speed ?? this.speed;

    const body = {
      model: this.model,
      input: trimmed,
      voice,
      response_format: format,
      speed,
    };

    const url = `${this.endpoint}/v1/audio/speech`;

    // Combine caller signal (if any) with our own timeout signal.
    const timeoutController = new AbortController();
    const timeoutHandle = setTimeout(() => timeoutController.abort(), this.timeoutMs);
    const combined = opts?.signal
      ? anySignal(opts.signal, timeoutController.signal)
      : timeoutController.signal;

    const startedAt = Date.now();
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(body),
        signal: combined,
      });

      if (!response.ok) {
        const errText = await response.text().catch(() => '');
        throw new Error(
          `OmniVoice HTTP ${response.status}: ${errText.slice(0, 300)}`,
        );
      }

      const arrayBuffer = await response.arrayBuffer();
      const data = Buffer.from(arrayBuffer);
      const elapsed = Date.now() - startedAt;

      const { contentType, extension } = contentTypeForFormat(format);
      const filename = `voice-${Date.now()}.${extension}`;

      console.log(
        `[OmniVoiceTTSProvider] Synthesized ${data.length} bytes (${format}) in ${elapsed}ms — voice=${voice}, chars=${trimmed.length}`,
      );

      return { data, contentType, filename };
    } finally {
      clearTimeout(timeoutHandle);
    }
  }
}

/**
 * AbortSignal.any() polyfill — Node 20+ has it natively but the type may
 * not be in every @types/node version. Merges multiple signals into one.
 */
function anySignal(...signals: AbortSignal[]): AbortSignal {
  if (typeof (AbortSignal as any).any === 'function') {
    return (AbortSignal as any).any(signals);
  }
  const controller = new AbortController();
  for (const s of signals) {
    if (s.aborted) {
      controller.abort(s.reason);
      break;
    }
    s.addEventListener('abort', () => controller.abort(s.reason), { once: true });
  }
  return controller.signal;
}
