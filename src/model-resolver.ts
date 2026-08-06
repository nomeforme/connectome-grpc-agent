/**
 * Model resolver — maps config model names to pi-ai Model objects.
 *
 * Tries anthropic provider first, then amazon-bedrock. Config should use
 * exact pi-ai model IDs:
 *   - "claude-sonnet-4-20250514"                     → anthropic
 *   - "anthropic.claude-3-opus-20240229-v1:0"        → amazon-bedrock
 *   - "us.anthropic.claude-3-opus-20240229-v1:0"     → amazon-bedrock (cross-region)
 *   - "eu.anthropic.claude-sonnet-4-20250514-v1:0"   → amazon-bedrock (cross-region)
 *
 * For cross-region prefixed IDs (us./eu./global.) not in pi-ai's registry,
 * the resolver strips the prefix, finds the base model, and clones it with
 * the prefixed ID so AWS receives the correct cross-region model identifier.
 *
 * For models routed through Vercel AI Gateway (e.g. when the direct/Bedrock
 * routes are deprecated but Vertex/another upstream still serves), use
 * `resolveGatewayModel(slug, { only: ['vertex'] })` — pi-ai's
 * openai-completions provider auto-detects Vercel and applies the routing.
 */

// pi-ai 0.80 moved the static-catalog API (getModels/stream/streamSimple) off the
// package root onto the `/compat` subpath. compat is a strict superset of the root
// and is what pi-agent-core itself imports, so it is a safe bridge — but upstream
// intends to delete it once the ModelManager migration lands, at which point this
// should move to `createModels()` / `Models.getModels()`.
import { getModels } from '@earendil-works/pi-ai/compat';
import type { Model, Api } from '@earendil-works/pi-ai';

const REGION_PREFIX_RE = /^(us|eu|global|apac)\./;

/**
 * Models the pi-ai catalog no longer ships, which we deliberately keep running.
 *
 * pi-ai 0.80 gutted the legacy Claude catalog (anthropic went 23 -> 14 models):
 * every Claude 3.x entry and several Claude 4 dated snapshots were deleted, on
 * both `anthropic` and `amazon-bedrock`. The models are still SERVED by the
 * providers — they were merely dropped from pi's built-in list. Without pinning,
 * resolveModel() returns undefined and bot-runtime throws "Model not found" at
 * startup, crash-looping 8 of our bots (claude-3-opus among them, which we keep
 * intentionally).
 *
 * These literals are captured verbatim from pi-ai 0.53's registry — the version
 * these bots have actually been running on — so cost/context/capability metadata
 * stays truthful rather than being approximated from a newer sibling.
 *
 * `Model` is a plain interface and ProviderId stays open (KnownProvider | string),
 * so hand-constructing is a supported pattern — it is what resolveGatewayModel and
 * resolveLocalModel already do.
 */
const PINNED_MODELS: Record<string, Model<Api>> = {
  // ---- anthropic (direct) ----
  // claude-opus-5: the OPPOSITE case from the elders below — too NEW for the pi
  // version we pin (0.80.6), not too old. It landed in pi-ai's registry in 0.82.x,
  // but 0.82's auth layer was rebuilt (getOAuthApiKey -> resolveProviderAuth /
  // CredentialStore), which would force a rewrite + re-verification of the OAuth
  // path all 11 subscription bots depend on. Pinning the model instead gets Opus 5
  // running on our validated 0.80.6 baseline with zero auth risk — the same trick
  // used for fable-5 / sonnet-5 / opus-4-8 before they entered the registry.
  //
  // Object captured verbatim from pi-ai 0.82.1's registry, so cost/context/compat
  // are authoritative. 0.80.6's anthropic-messages provider honors the correctness-
  // relevant compat flags (supportsTemperature, forceAdaptiveThinking) and
  // thinkingLevelMap; supportsStrictTools is a 0.82 addition it simply ignores
  // (tools aren't sent strict — a safe no-op, not a break). Carry it anyway so the
  // pin stays truthful when we eventually move to the 0.82 auth model.
  'claude-opus-5': {
    id: 'claude-opus-5',
    name: 'Claude Opus 5',
    api: 'anthropic-messages',
    provider: 'anthropic',
    baseUrl: 'https://api.anthropic.com',
    reasoning: true,
    input: ['text', 'image'],
    cost: { input: 5, output: 25, cacheRead: 0.5, cacheWrite: 6.25 },
    contextWindow: 1000000,
    maxTokens: 128000,
    thinkingLevelMap: { xhigh: 'xhigh', max: 'max' },
    compat: {
      forceAdaptiveThinking: true,
      supportsTemperature: false,
      supportsStrictTools: true,
    },
  } as Model<Api>,
  'claude-3-opus-20240229': {
    id: 'claude-3-opus-20240229',
    name: 'Claude Opus 3',
    api: 'anthropic-messages',
    provider: 'anthropic',
    baseUrl: 'https://api.anthropic.com',
    reasoning: false,
    input: ['text', 'image'],
    cost: { input: 15, output: 75, cacheRead: 1.5, cacheWrite: 18.75 },
    contextWindow: 200000,
    maxTokens: 4096,
  } as Model<Api>,
  'claude-3-haiku-20240307': {
    id: 'claude-3-haiku-20240307',
    name: 'Claude Haiku 3',
    api: 'anthropic-messages',
    provider: 'anthropic',
    baseUrl: 'https://api.anthropic.com',
    reasoning: false,
    input: ['text', 'image'],
    cost: { input: 0.25, output: 1.25, cacheRead: 0.03, cacheWrite: 0.3 },
    contextWindow: 200000,
    maxTokens: 4096,
  } as Model<Api>,
  'claude-sonnet-4-20250514': {
    id: 'claude-sonnet-4-20250514',
    name: 'Claude Sonnet 4',
    api: 'anthropic-messages',
    provider: 'anthropic',
    baseUrl: 'https://api.anthropic.com',
    reasoning: true,
    input: ['text', 'image'],
    cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
    contextWindow: 200000,
    maxTokens: 64000,
  } as Model<Api>,

  // ---- amazon-bedrock ----
  'anthropic.claude-3-sonnet-20240229-v1:0': {
    id: 'anthropic.claude-3-sonnet-20240229-v1:0',
    name: 'Claude Sonnet 3',
    api: 'bedrock-converse-stream',
    provider: 'amazon-bedrock',
    baseUrl: 'https://bedrock-runtime.us-east-1.amazonaws.com',
    reasoning: false,
    input: ['text', 'image'],
    cost: { input: 3, output: 15, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 200000,
    maxTokens: 4096,
  } as Model<Api>,
  'anthropic.claude-3-5-sonnet-20240620-v1:0': {
    id: 'anthropic.claude-3-5-sonnet-20240620-v1:0',
    name: 'Claude Sonnet 3.5',
    api: 'bedrock-converse-stream',
    provider: 'amazon-bedrock',
    baseUrl: 'https://bedrock-runtime.us-east-1.amazonaws.com',
    reasoning: false,
    input: ['text', 'image'],
    cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
    contextWindow: 200000,
    maxTokens: 8192,
  } as Model<Api>,
  'anthropic.claude-3-5-sonnet-20241022-v2:0': {
    id: 'anthropic.claude-3-5-sonnet-20241022-v2:0',
    name: 'Claude Sonnet 3.5 v2',
    api: 'bedrock-converse-stream',
    provider: 'amazon-bedrock',
    baseUrl: 'https://bedrock-runtime.us-east-1.amazonaws.com',
    reasoning: false,
    input: ['text', 'image'],
    cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
    contextWindow: 200000,
    maxTokens: 8192,
  } as Model<Api>,
  'anthropic.claude-3-5-haiku-20241022-v1:0': {
    id: 'anthropic.claude-3-5-haiku-20241022-v1:0',
    name: 'Claude Haiku 3.5',
    api: 'bedrock-converse-stream',
    provider: 'amazon-bedrock',
    baseUrl: 'https://bedrock-runtime.us-east-1.amazonaws.com',
    reasoning: false,
    input: ['text', 'image'],
    cost: { input: 0.8, output: 4, cacheRead: 0.08, cacheWrite: 1 },
    contextWindow: 200000,
    maxTokens: 8192,
  } as Model<Api>,
  'anthropic.claude-3-7-sonnet-20250219-v1:0': {
    id: 'anthropic.claude-3-7-sonnet-20250219-v1:0',
    name: 'Claude Sonnet 3.7',
    api: 'bedrock-converse-stream',
    provider: 'amazon-bedrock',
    baseUrl: 'https://bedrock-runtime.us-east-1.amazonaws.com',
    reasoning: false,
    input: ['text', 'image'],
    cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
    contextWindow: 200000,
    maxTokens: 8192,
  } as Model<Api>,
};

/**
 * Registry lookup with the pinned fallback.
 *
 * NB the registry is checked FIRST, so if upstream ever restores a model we pick
 * up their (authoritative) entry — including compat flags we'd have no way to
 * know about — and the pin quietly becomes dead weight rather than shadowing it.
 */
function findModel(provider: 'anthropic' | 'amazon-bedrock', id: string): Model<Api> | undefined {
  const fromRegistry = getModels(provider).find((m) => m.id === id) as Model<Api> | undefined;
  if (fromRegistry) return fromRegistry;
  const pinned = PINNED_MODELS[id];
  // Guard the pin by provider so an anthropic id can't satisfy a bedrock lookup.
  return pinned && pinned.provider === provider ? pinned : undefined;
}

/**
 * Resolve a model name to a pi-ai Model object.
 *
 * Searches anthropic provider first, then amazon-bedrock. For region-prefixed
 * bedrock IDs (us.anthropic.*, eu.anthropic.*) not found in the registry,
 * falls back to the unprefixed base model and clones it with the prefixed ID.
 */
export function resolveModel(modelName: string): Model<Api> | undefined {
  // 1. Exact match — registry first, then our pinned entries for models pi-ai dropped.
  const exact = findModel('anthropic', modelName) ?? findModel('amazon-bedrock', modelName);
  if (exact) return exact;

  // 2. Cross-region prefix fallback: strip us./eu./apac./global., find the base
  //    bedrock model, clone it with the prefixed ID so AWS gets the right identifier.
  const prefixMatch = modelName.match(REGION_PREFIX_RE);
  if (prefixMatch) {
    const baseId = modelName.slice(prefixMatch[0].length);
    const baseModel = findModel('amazon-bedrock', baseId);
    if (baseModel) {
      return { ...baseModel, id: modelName } as Model<Api>;
    }
  }

  // NB there is no longer a MANUAL_MODELS clone step: claude-opus-4-7/4-8,
  // claude-fable-5 and claude-sonnet-5 are all real entries in pi-ai 0.80's
  // catalog now, so they resolve at step 1 — and crucially they carry compat
  // flags (e.g. forceAdaptiveThinking, supportsTemperature: false on 4-8/Fable)
  // that a hand-rolled clone of an older sibling would have silently omitted.
  return undefined;
}

/**
 * Options for resolveGatewayModel.
 */
export interface GatewayModelOptions {
  /** Hard-pin a provider set — request fails if none of these can serve. */
  only?: string[];
  /** Try providers in this order (with fallback to defaults if all fail). */
  order?: string[];
  /** Override the cost row used for accounting (rare — registry usually has it). */
  cost?: { input: number; output: number; cacheRead: number; cacheWrite: number };
}

/**
 * Build a pi-ai Model object that routes through Vercel AI Gateway's
 * OpenAI-compatible /v1/chat/completions endpoint with vercelGatewayRouting
 * pre-baked.
 *
 * Why openai-completions and not anthropic-messages: pi-ai's openai-completions
 * provider is the one wired to translate `compat.vercelGatewayRouting` into the
 * request body's `providerOptions.gateway.{only,order}` fields. The
 * anthropic-messages path ignores it.
 *
 * @param slug   — Vercel model slug, e.g. "anthropic/claude-opus-4"
 * @param opts   — provider routing pins (only / order) and optional cost override
 *
 * Looks up cost + capabilities from pi-ai's registry by trying the existing
 * vercel-ai-gateway entry first (anthropic-messages flavor) then OpenRouter,
 * since the registry sometimes only has one or the other for a given slug.
 */
export function resolveGatewayModel(
  slug: string,
  opts: GatewayModelOptions = {},
): Model<'openai-completions'> {
  // Lift cost + context info from whichever registry entry has it.
  const registryEntry =
    getModels('vercel-ai-gateway').find((m) => m.id === slug) ??
    getModels('openrouter').find((m) => m.id === slug);

  const cost =
    opts.cost ?? registryEntry?.cost ?? { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };
  const contextWindow = registryEntry?.contextWindow ?? 200_000;
  const maxTokens = registryEntry?.maxTokens ?? 8192;
  const reasoning = registryEntry?.reasoning ?? false;
  const input = (registryEntry?.input as ('text' | 'image')[] | undefined) ?? ['text'];

  return {
    id: slug,
    name: registryEntry?.name ?? slug,
    api: 'openai-completions',
    provider: 'vercel-ai-gateway',
    baseUrl: 'https://ai-gateway.vercel.sh/v1',
    reasoning,
    input,
    cost,
    contextWindow,
    maxTokens,
    compat: {
      // Pi-ai's openai-completions provider detects ai-gateway.vercel.sh and
      // translates this into `providerOptions.gateway.{only,order}` in the body.
      vercelGatewayRouting: {
        only: opts.only,
        order: opts.order,
      },
    },
  };
}

/**
 * Options for resolveLocalModel.
 */
export interface LocalModelOptions {
  /** Context window of the served model (llama-server -c ÷ --parallel slots). */
  contextWindow?: number;
  /** Max output tokens per response. */
  maxTokens?: number;
  /** Whether the model emits reasoning (Qwen3 thinking → reasoning_content). */
  reasoning?: boolean;
  /** Modalities the model accepts. */
  input?: ('text' | 'image')[];
}

/**
 * Build a pi-ai Model that targets a self-hosted, OpenAI-compatible endpoint
 * (llama.cpp / llama-server, LM Studio, vLLM, Ollama's /v1, …).
 *
 * For the connectome fleet this is a llama-server on the plantoidz GPU box,
 * addressed by its Tailscale IP (e.g. http://REDACTED-IP:1234/v1). Container →
 * Tailscale-IP routing works through the host's tailscale interface — use the
 * raw IP, NOT the MagicDNS name (which does not resolve inside bot containers).
 *
 * The provider is tagged `local-llm` so ConnectomeAgent's streamFn injects a
 * throwaway API key (llama-server ignores auth, but pi-ai's openai-completions
 * provider requires a non-empty key). Reasoning models that return
 * `reasoning_content` (Qwen3, DeepSeek-R1) are parsed natively by pi-ai's
 * openai-completions provider into thinking blocks — no extra handling needed.
 *
 * @param id       — model id sent in the request body; match the served model
 *                   (llama-server /v1/models), e.g. "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
 * @param baseUrl  — OpenAI-compatible base URL, including the /v1 suffix
 * @param opts     — context window, max tokens, reasoning flag, input modalities
 */
export function resolveLocalModel(
  id: string,
  baseUrl: string,
  opts: LocalModelOptions = {},
): Model<'openai-completions'> {
  return {
    id,
    name: id,
    api: 'openai-completions',
    provider: 'local-llm',
    baseUrl,
    reasoning: opts.reasoning ?? true,
    input: opts.input ?? ['text'],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: opts.contextWindow ?? 32_768,
    maxTokens: opts.maxTokens ?? 8_192,
  };
}
