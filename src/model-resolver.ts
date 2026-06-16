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

import { getModels } from '@mariozechner/pi-ai';
import type { Model, Api } from '@mariozechner/pi-ai';

const REGION_PREFIX_RE = /^(us|eu|global|apac)\./;

/** Models not yet in pi-ai's registry — cloned from a base model with overridden ID. */
const MANUAL_MODELS: Record<string, string> = {
  'claude-opus-4-7': 'claude-opus-4-6',
  'claude-opus-4-8': 'claude-opus-4-6',
  'claude-fable-5': 'claude-opus-4-6',
};

/**
 * Resolve a model name to a pi-ai Model object.
 *
 * Searches anthropic provider first, then amazon-bedrock. For region-prefixed
 * bedrock IDs (us.anthropic.*, eu.anthropic.*) not found in the registry,
 * falls back to the unprefixed base model and clones it with the prefixed ID.
 */
export function resolveModel(modelName: string): Model<Api> | undefined {
  // 1. Exact match in anthropic or bedrock registries
  const exact = getModels('anthropic').find((m) => m.id === modelName)
    ?? getModels('amazon-bedrock').find((m) => m.id === modelName);
  if (exact) return exact;

  // 2. Cross-region prefix fallback: strip us./eu./global., find base, clone with prefixed ID
  const prefixMatch = modelName.match(REGION_PREFIX_RE);
  if (prefixMatch) {
    const baseId = modelName.slice(prefixMatch[0].length);
    const baseModel = getModels('amazon-bedrock').find((m) => m.id === baseId);
    if (baseModel) {
      return { ...baseModel, id: modelName } as Model<Api>;
    }
  }

  // 3. Manual overrides for models not yet in pi-ai
  const baseModelName = MANUAL_MODELS[modelName];
  if (baseModelName) {
    const base = getModels('anthropic').find((m) => m.id === baseModelName);
    if (base) return { ...base, id: modelName } as Model<Api>;
  }

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
