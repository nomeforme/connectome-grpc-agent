/**
 * Qwen /no_think adapter.
 *
 * Qwen 3.x chat templates recognize the literal string `/no_think` when it
 * appears in the system prompt (or a user message) and skip emitting the
 * `<think>...</think>` reasoning block entirely. This is a tokenizer-side
 * feature — no server change, no API param, just a prompt marker.
 *
 * Matches on model IDs beginning with `Qwen` (case-insensitive), covering
 * the local llama-server / vLLM / LM Studio deployments. If a Qwen-family
 * model is served through a different vendor (e.g. Alibaba Cloud API) the
 * same marker still works.
 */

import type { ThinkingControlAdapter, ThinkingControlContext, ApplyResult } from '../thinking-control.js';

const NO_THINK_MARKER = '/no_think';

export const qwenNoThinkAdapter: ThinkingControlAdapter = {
  name: 'qwen-no-think',

  matches(ctx: ThinkingControlContext): boolean {
    // Qwen model IDs: "Qwen3-...", "Qwen3.6-...", "qwen-2.5", etc.
    // Match on prefix (case-insensitive) — covers local ggufs and API models.
    return /^qwen/i.test(ctx.model);
  },

  applyToSystemPrompt(systemPrompt: string): ApplyResult {
    // Idempotent — if the marker is already present, don't add it again.
    if (systemPrompt.includes(NO_THINK_MARKER)) {
      return { systemPrompt, applied: false };
    }
    // Prepend so the marker is guaranteed to reach the chat-template
    // renderer before any persona content that might contain something
    // that gets confused for a directive.
    return {
      systemPrompt: `${NO_THINK_MARKER}\n\n${systemPrompt}`,
      applied: true,
    };
  },
};
