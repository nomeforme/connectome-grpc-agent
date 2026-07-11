/**
 * Lightweight pi auth provider — reads ~/.pi/agent/auth.json and provides
 * OAuth tokens for pi-agent's getApiKey callback.
 *
 * This lets bot-runtime use a Claude Max/Pro subscription instead of API keys.
 * Run `pi login` once on the host, then volume-mount ~/.pi/agent into containers.
 *
 * Token refresh is handled by pi-ai's OAuth providers (auto-refresh on expiry).
 * Updated credentials are written back to auth.json with atomic rename.
 *
 * STRICT MODE (`strict: true`): for bots that are genuinely configured to run on
 * the Claude subscription, an invalid/expired/unrefreshable OAuth credential is a
 * FATAL error, not a soft fallback. Previously getApiKey() returned `undefined` on
 * failure, which made pi fall through to the ANTHROPIC_API_KEY env var — silently
 * billing the API account for months while the boot log still claimed "Using pi
 * OAuth auth". In strict mode we throw instead, so the failure is loud and the bot
 * never quietly spends API credit it wasn't meant to.
 *
 * Strict mode is scoped to the OAuth provider only ('anthropic'). Any other
 * provider still returns undefined, so a strict instance is harmless if it is ever
 * consulted for bedrock/gateway/local models.
 */

import { readFileSync, writeFileSync, mkdirSync, renameSync } from 'fs';
import { dirname, join } from 'path';
// pi-ai 0.73 moved the OAuth *runtime* exports off the root barrel onto the
// `/oauth` subpath; the types still live on the root.
import { getOAuthApiKey } from '@earendil-works/pi-ai/oauth';
import type { OAuthCredentials } from '@earendil-works/pi-ai';

/** Shape of ~/.pi/agent/auth.json — keyed by provider ID */
type AuthFile = Record<string, OAuthCredentials>;

export interface PiAuthProviderOptions {
  /** Path to auth.json. Defaults to $PI_AUTH_DIR/auth.json or ~/.pi/agent/auth.json. */
  authPath?: string;
  /**
   * Throw instead of returning undefined when OAuth for `oauthProvider` is
   * missing or unrefreshable. Use for bots that rely on the subscription, so
   * they fail fast rather than silently degrading to ANTHROPIC_API_KEY.
   */
  strict?: boolean;
  /** Provider that strict mode applies to. Default 'anthropic'. */
  oauthProvider?: string;
}

/** Thrown in strict mode when subscription OAuth cannot produce a usable token. */
export class OAuthUnavailableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'OAuthUnavailableError';
  }
}

export class PiAuthProvider {
  private authPath: string;
  private strict: boolean;
  private oauthProvider: string;
  private credentials: AuthFile | null = null;
  private lastLoadTime = 0;
  private readonly reloadIntervalMs = 30_000; // re-read file every 30s

  constructor(options?: PiAuthProviderOptions | string) {
    // Back-compat: the old constructor took a bare authPath string.
    const opts: PiAuthProviderOptions =
      typeof options === 'string' ? { authPath: options } : (options ?? {});
    this.authPath = opts.authPath ?? join(
      process.env.PI_AUTH_DIR ?? join(process.env.HOME ?? '/root', '.pi', 'agent'),
      'auth.json',
    );
    this.strict = opts.strict ?? false;
    this.oauthProvider = opts.oauthProvider ?? 'anthropic';
  }

  /**
   * getApiKey callback — pass this to pi-agent's Agent constructor.
   *
   * Non-strict: returns an API key string, or undefined to fall through to env
   * var resolution (pi's default behaviour).
   *
   * Strict: for `this.oauthProvider`, throws OAuthUnavailableError rather than
   * returning undefined, so the bot cannot silently fall back to the API key.
   *
   * NB pi-agent-core 0.73 documents getApiKey as "must not throw". We
   * deliberately violate that in strict mode: an auth failure SHOULD tear down
   * the cycle loudly. Boot-time validation via ConnectomeAgent.assertAuthReady()
   * is what normally catches this first, so a throw here is the backstop for a
   * token that dies mid-process.
   */
  getApiKey = async (provider: string): Promise<string | undefined> => {
    const strict = this.strict && provider === this.oauthProvider;

    const creds = this.loadCredentials();
    if (!creds || !creds[provider]) {
      if (strict) {
        throw new OAuthUnavailableError(
          `No OAuth credentials for '${provider}' in ${this.authPath}. ` +
            `This bot is configured to use the Claude subscription — refusing to fall back to ` +
            `ANTHROPIC_API_KEY. Run \`pi login\` on the host, or set "use_api_key": true for this bot.`,
        );
      }
      return undefined; // no OAuth creds → fall through to ANTHROPIC_API_KEY env var
    }

    let result: Awaited<ReturnType<typeof getOAuthApiKey>>;
    try {
      result = await getOAuthApiKey(provider, creds);
    } catch (err) {
      const detail = err instanceof Error ? err.message : String(err);
      if (strict) {
        throw new OAuthUnavailableError(
          `OAuth token refresh failed for '${provider}': ${detail}. ` +
            `This bot is configured to use the Claude subscription — refusing to fall back to ` +
            `ANTHROPIC_API_KEY. Run \`pi login\` on the host to re-authenticate.`,
        );
      }
      console.warn(`[PiAuthProvider] OAuth token refresh failed for ${provider}: ${detail}`);
      return undefined; // fall through to env var
    }

    if (!result) {
      if (strict) {
        throw new OAuthUnavailableError(
          `OAuth provider '${provider}' returned no token (unknown provider or unusable credentials). ` +
            `Refusing to fall back to ANTHROPIC_API_KEY. Run \`pi login\` on the host.`,
        );
      }
      return undefined;
    }

    // Write back refreshed credentials if they changed
    if (result.newCredentials !== creds[provider]) {
      creds[provider] = result.newCredentials;
      this.saveCredentials(creds);
    }

    return result.apiKey;
  };

  /**
   * Check if auth.json has credentials for a provider.
   *
   * NB presence only — this does NOT prove the credential is usable. An expired
   * access token with a live refresh token is normal and recoverable, so the only
   * authoritative check is an actual refresh attempt (see validate()). Callers
   * that need the truth (e.g. the boot log) must use validate(), not this.
   */
  hasCredentials(provider: string): boolean {
    const creds = this.loadCredentials();
    return !!(creds && creds[provider]);
  }

  /**
   * Authoritative auth check: attempt a real token resolution (which performs a
   * refresh if the access token is expired). Returns the reason on failure.
   *
   * Deliberately does NOT infer validity from `expires` alone — an expired access
   * token with a valid refresh token is the normal steady state.
   */
  async validate(provider = this.oauthProvider): Promise<{ ok: true } | { ok: false; reason: string }> {
    try {
      const key = await this.getApiKey(provider);
      if (!key) return { ok: false, reason: `no OAuth token available for '${provider}'` };
      return { ok: true };
    } catch (err) {
      return { ok: false, reason: err instanceof Error ? err.message : String(err) };
    }
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
    } catch (err: any) {
      // Missing file is the expected "no OAuth configured" case — stay quiet.
      // A present-but-unreadable/corrupt file is NOT: it silently downgraded us
      // to the API key with no trace. Say so (strict callers turn this fatal).
      if (err?.code !== 'ENOENT') {
        console.warn(
          `[PiAuthProvider] Could not read ${this.authPath}: ${err?.message ?? err} — ` +
            `treating as no OAuth credentials`,
        );
      }
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
