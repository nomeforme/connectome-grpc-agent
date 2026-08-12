#!/usr/bin/env node
/**
 * pi-oauth-login.mjs — headless Anthropic OAuth re-auth, no TUI required.
 *
 * Drives the exact same PKCE flow `pi`'s `/login` command uses
 * (@earendil-works/pi-ai/oauth → loginAnthropic), but as a ~90-line script
 * instead of the full interactive TUI. Useful because:
 *   - This box is headless: no browser, so the flow always needs a human to
 *     open the printed URL elsewhere and paste back the result. A skill/agent
 *     can't do that step — but everything *around* it (starting the flow,
 *     writing the resulting credentials to disk in the right shape) can be
 *     scripted, which this does.
 *   - `~/.pi/agent/auth.json` is bind-mounted into every OAuth-mode bot
 *     container (see docker-compose.yml), so a single run here re-authenticates
 *     all of them — no container restart needed, PiAuthProvider re-reads the
 *     file on refresh failure ("auth.json changed on disk" in bot logs).
 *
 * Usage:
 *   node scripts/pi-oauth-login.mjs               # login/refresh anthropic
 *   node scripts/pi-oauth-login.mjs --provider anthropic
 *
 * Run from inside connectome-agent-core (or `pnpm --filter connectome-agent-core exec`)
 * so `@earendil-works/pi-ai` resolves — it's a direct dependency here.
 */
import { loginAnthropic } from '@earendil-works/pi-ai/oauth';
import { createInterface } from 'node:readline/promises';
import { readFileSync, writeFileSync, existsSync, mkdirSync, copyFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { join, dirname } from 'node:path';

const provider = (process.argv.includes('--provider')
  ? process.argv[process.argv.indexOf('--provider') + 1]
  : 'anthropic');

if (provider !== 'anthropic') {
  console.error(`Only 'anthropic' is wired up in this script today (got --provider ${provider}).`);
  process.exit(1);
}

const authDir = process.env.PI_CODING_AGENT_DIR || join(homedir(), '.pi', 'agent');
const authPath = join(authDir, 'auth.json');

const rl = createInterface({ input: process.stdin, output: process.stdout });
const ask = (message) => rl.question(`${message}\n> `);

console.log(`[pi-oauth-login] Re-authenticating '${provider}' — credentials will be written to ${authPath}`);
console.log(`[pi-oauth-login] That file is bind-mounted into every OAuth-mode bot container, so this fixes all of them at once. No restart needed.\n`);

let credentials;
try {
  credentials = await loginAnthropic({
    onAuth: ({ url, instructions }) => {
      console.log('Open this URL in ANY browser (does not need to be on this machine) and complete login:\n');
      console.log(`  ${url}\n`);
      if (instructions) console.log(`${instructions}\n`);
    },
    onProgress: (message) => console.log(`[pi-oauth-login] ${message}`),
    // Races against the local callback server (127.0.0.1:53692). On a headless
    // box nothing will ever hit that port, so this is effectively the only
    // path that resolves — paste the code, or the full redirect URL, or the
    // "connection refused" URL your browser lands on after authorizing.
    onManualCodeInput: () => ask('Paste the authorization code or the final redirect URL:'),
    // Fallback the library falls through to if onManualCodeInput resolved
    // empty (e.g. accidental blank paste) — same prompt, asked again.
    onPrompt: async (prompt) => ask(prompt.message || 'Paste the authorization code or the final redirect URL:'),
  });
} catch (err) {
  console.error(`\n[pi-oauth-login] Login failed: ${err.message}`);
  rl.close();
  process.exit(1);
}

rl.close();

// Preserve the on-disk shape exactly as bot-runtime's PiAuthProvider expects
// it today: { "<provider>": { refresh, access, expires } } — no "type"
// wrapper (that's a newer pi-coding-agent AuthStorage convention this repo's
// reader doesn't use).
mkdirSync(authDir, { recursive: true, mode: 0o700 });
let existing = {};
if (existsSync(authPath)) {
  copyFileSync(authPath, `${authPath}.bak`);
  try {
    existing = JSON.parse(readFileSync(authPath, 'utf-8'));
  } catch {
    console.warn(`[pi-oauth-login] Existing ${authPath} was not valid JSON — replacing it (backup saved to ${authPath}.bak).`);
  }
}
existing[provider] = {
  refresh: credentials.refresh,
  access: credentials.access,
  expires: credentials.expires,
};
writeFileSync(authPath, JSON.stringify(existing, null, 2), { encoding: 'utf-8', mode: 0o600 });

const expiresIn = Math.round((credentials.expires - Date.now()) / 60000);
console.log(`\n[pi-oauth-login] Wrote ${authPath}. Access token valid ~${expiresIn} min; refresh token used for silent renewal after that.`);
console.log(`[pi-oauth-login] Done — bots pick this up automatically on their next OAuth refresh attempt.`);
