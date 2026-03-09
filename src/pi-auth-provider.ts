/**
 * Lightweight pi auth provider — reads ~/.pi/agent/auth.json and provides
 * OAuth tokens for pi-agent's getApiKey callback.
 *
 * This lets bot-runtime use a Claude Max/Pro subscription instead of API keys.
 * Run `pi login` once on the host, then volume-mount ~/.pi/agent into containers.
 *
 * Token refresh is handled by pi-ai's OAuth providers (auto-refresh on expiry).
 * Updated credentials are written back to auth.json with atomic rename.
 */

import { readFileSync, writeFileSync, mkdirSync, renameSync } from 'fs';
import { dirname, join } from 'path';
import { getOAuthApiKey, type OAuthCredentials } from '@mariozechner/pi-ai';

/** Shape of ~/.pi/agent/auth.json — keyed by provider ID */
type AuthFile = Record<string, OAuthCredentials>;

export class PiAuthProvider {
  private authPath: string;
  private credentials: AuthFile | null = null;
  private lastLoadTime = 0;
  private readonly reloadIntervalMs = 30_000; // re-read file every 30s

  constructor(authPath?: string) {
    this.authPath = authPath ?? join(
      process.env.PI_AUTH_DIR ?? join(process.env.HOME ?? '/root', '.pi', 'agent'),
      'auth.json',
    );
  }

  /**
   * getApiKey callback — pass this to pi-agent's Agent constructor.
   * Returns an API key string for the given provider, or undefined to
   * fall through to env var resolution.
   */
  getApiKey = async (provider: string): Promise<string | undefined> => {
    const creds = this.loadCredentials();
    if (!creds || !creds[provider]) {
      return undefined; // no OAuth creds → fall through to ANTHROPIC_API_KEY env var
    }

    try {
      const result = await getOAuthApiKey(provider, creds);
      if (!result) return undefined;

      // Write back refreshed credentials if they changed
      if (result.newCredentials !== creds[provider]) {
        creds[provider] = result.newCredentials;
        this.saveCredentials(creds);
      }

      return result.apiKey;
    } catch (err) {
      console.warn(`[PiAuthProvider] OAuth token refresh failed for ${provider}: ${err}`);
      return undefined; // fall through to env var
    }
  };

  /**
   * Check if auth.json exists and has credentials for a provider.
   */
  hasCredentials(provider: string): boolean {
    const creds = this.loadCredentials();
    return !!(creds && creds[provider]);
  }

  // ---------------------------------------------------------------------------
  // Private
  // ---------------------------------------------------------------------------

  private loadCredentials(): AuthFile | null {
    // Cache: don't re-read file on every API call
    if (this.credentials && Date.now() - this.lastLoadTime < this.reloadIntervalMs) {
      return this.credentials;
    }

    try {
      const raw = readFileSync(this.authPath, 'utf-8');
      this.credentials = JSON.parse(raw) as AuthFile;
      this.lastLoadTime = Date.now();
      return this.credentials;
    } catch {
      // File doesn't exist or is invalid — that's fine, fall through to env var
      this.credentials = null;
      this.lastLoadTime = Date.now();
      return null;
    }
  }

  private saveCredentials(creds: AuthFile): void {
    try {
      const dir = dirname(this.authPath);
      mkdirSync(dir, { recursive: true });
      // Atomic write: write to temp file then rename
      const tmpPath = this.authPath + '.tmp';
      writeFileSync(tmpPath, JSON.stringify(creds, null, 2), 'utf-8');
      renameSync(tmpPath, this.authPath);
      this.credentials = creds;
    } catch (err) {
      console.warn(`[PiAuthProvider] Failed to save updated credentials: ${err}`);
    }
  }
}
