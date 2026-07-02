/**
 * TTS barrel — exports provider interface + implementations + factory.
 */

export type {
  TTSProvider,
  TTSSynthesisOptions,
  TTSSynthesisResult,
} from './tts-provider.js';
export { contentTypeForFormat } from './tts-provider.js';

export { OmniVoiceTTSProvider } from './omnivoice-provider.js';
export type { OmniVoiceProviderConfig } from './omnivoice-provider.js';

import { OmniVoiceTTSProvider } from './omnivoice-provider.js';
import type { TTSProvider } from './tts-provider.js';

/** Provider config discriminated by `provider` field. */
export type TTSProviderConfig =
  | {
      provider: 'omnivoice';
      endpoint: string;
      voice: string;
      format?: string;
      speed?: number;
      timeout_ms?: number;
      max_input_chars?: number;
      model?: string;
    };

/**
 * Factory — instantiate a provider from a bot-config-shaped object.
 * Returns `undefined` if the config is missing or the provider is unknown
 * (so callers can silently fall through to text-only).
 */
export function createTTSProvider(config: TTSProviderConfig | undefined): TTSProvider | undefined {
  if (!config) return undefined;
  switch (config.provider) {
    case 'omnivoice':
      return new OmniVoiceTTSProvider({
        endpoint: config.endpoint,
        voice: config.voice,
        format: config.format,
        speed: config.speed,
        timeoutMs: config.timeout_ms,
        maxInputChars: config.max_input_chars,
        model: config.model,
      });
    default:
      console.warn(`[TTS] Unknown provider: ${(config as any).provider}`);
      return undefined;
  }
}
