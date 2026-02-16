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
 */

import { getModels } from '@mariozechner/pi-ai';
import type { Model, Api } from '@mariozechner/pi-ai';

const REGION_PREFIX_RE = /^(us|eu|global)\./;

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

  return undefined;
}
