/**
 * Context adapter — converts rendered context (from gRPC or VEILContextAdapter)
 * to pi-agent's AgentContext (used by ConnectomeAgent.runWithContext).
 *
 * Accepts any context whose messages have { role, content, metadata? } — works
 * with both connectome-ts RenderedContext and axon-local RenderedContext types.
 *
 * Handles:
 * - System message extraction into systemPrompt
 * - Image attachments → pi-ai ImageContent
 * - Consecutive same-role message merging (required by Bedrock Converse API)
 * - First-message-must-be-user constraint (required by Bedrock / Claude 3 Sonnet)
 */

import type { Message, UserMessage, AssistantMessage, TextContent, ImageContent } from '@mariozechner/pi-ai';
import type { AgentContext } from './types.js';

// ---------------------------------------------------------------------------
// Minimal input types (structurally compatible with both RenderedContext variants)
// ---------------------------------------------------------------------------

/**
 * Minimal attachment shape accepted by the context adapter.
 *
 * Three transport modes (mutually exclusive in practice):
 *  - `blobId`: sha256 ref into the content-addressed blob store. Must be
 *    pre-resolved (via {@link resolveAttachmentRefs}) before the message
 *    reaches the LLM — once resolved, `data` is populated alongside.
 *  - `data`: inline base64 (legacy, kept for back-compat with historical
 *    facets that still carry `inline_data`).
 *  - `url`: external URL (rare).
 */
export interface ContextAttachment {
  id?: string;
  url?: string;
  blobId?: string;
  contentType?: string;
  name?: string;
  filename?: string;
  size?: number;
  sizeBytes?: number;
  data?: string;  // base64 encoded
}

/**
 * Minimal message shape accepted by the context adapter.
 * Both connectome-ts RenderedContext and axon-local RenderedContext satisfy this.
 */
export interface ContextMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
  metadata?: {
    attachments?: ContextAttachment[];
    [key: string]: any;
  };
}

/**
 * Minimal rendered context shape. Any object with a `messages` array of
 * ContextMessage satisfies this — both connectome-ts and axon-local types.
 */
export interface RenderedContextLike {
  messages: ContextMessage[];
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

type UserContent = string | (TextContent | ImageContent)[];

function contextMsgToUserContent(msg: ContextMessage): UserContent {
  const attachments = msg.metadata?.attachments;
  if (!attachments || attachments.length === 0) return msg.content;

  const hasImageAttachments = attachments.some(
    (a) => a.data && a.contentType?.startsWith('image/')
  );
  const fileAttachments = attachments.filter(
    (a) => a.data && !a.contentType?.startsWith('image/')
  );

  if (hasImageAttachments || fileAttachments.length > 0) {
    const content: (TextContent | ImageContent)[] = [];
    if (msg.content) {
      content.push({ type: 'text', text: msg.content });
    }
    for (const att of attachments) {
      if (att.data && att.contentType?.startsWith('image/')) {
        content.push({ type: 'image', data: att.data, mimeType: att.contentType });
      }
    }
    // Annotate non-image file attachments so the agent knows they're available
    if (fileAttachments.length > 0) {
      const lines = fileAttachments.map((att) => {
        const name = att.name || att.id || 'unnamed';
        const size = att.size ? `${(att.size / 1024).toFixed(1)}KB` : 'unknown size';
        return `  - ${name} (${att.contentType || 'unknown type'}, ${size})`;
      });
      content.push({
        type: 'text',
        text: `[Attached files — use save_attachment to save to workspace:\n${lines.join('\n')}]`,
      });
    }
    return content;
  }

  return msg.content;
}

function mergeUserContent(a: UserContent, b: UserContent): (TextContent | ImageContent)[] {
  const normalize = (c: UserContent): (TextContent | ImageContent)[] =>
    typeof c === 'string' ? [{ type: 'text', text: c }] : c;
  return [...normalize(a), ...normalize(b)];
}

const STUB_USAGE = {
  input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Convert a rendered context to a pi-agent AgentContext.
 *
 * - Extracts the system prompt from system-role messages
 * - Converts user messages to pi-ai UserMessage (with image attachments as ImageContent)
 * - Converts assistant messages to pi-ai AssistantMessage (with stub metadata)
 * - Merges consecutive same-role messages (required by Bedrock Converse API)
 * - Ensures first message is user role (required by Bedrock / Claude 3 Sonnet)
 */
export function renderedContextToAgentContext(rendered: RenderedContextLike): AgentContext {
  let systemPrompt = '';
  const messages: Message[] = [];

  for (const msg of rendered.messages) {
    if (msg.role === 'system') {
      systemPrompt += (systemPrompt ? '\n\n' : '') + msg.content;
      continue;
    }

    const prev = messages.length > 0 ? messages[messages.length - 1] : null;

    if (msg.role === 'user') {
      const content = contextMsgToUserContent(msg);

      // Merge with previous user message if consecutive
      if (prev && prev.role === 'user') {
        const prevUser = prev as UserMessage;
        prevUser.content = mergeUserContent(prevUser.content, content);
        continue;
      }

      const userMsg: UserMessage = { role: 'user', content, timestamp: Date.now() };
      messages.push(userMsg);
      continue;
    }

    if (msg.role === 'assistant') {
      // Merge with previous assistant message if consecutive
      if (prev && prev.role === 'assistant') {
        const prevAssist = prev as AssistantMessage;
        prevAssist.content.push({ type: 'text', text: msg.content });
        continue;
      }

      const assistantMsg: AssistantMessage = {
        role: 'assistant',
        content: [{ type: 'text', text: msg.content }],
        api: 'anthropic-messages',
        provider: 'anthropic',
        model: 'unknown',
        usage: STUB_USAGE,
        stopReason: 'stop',
        timestamp: Date.now(),
      };
      messages.push(assistantMsg);
    }
  }

  // Ensure first message is user role (required by Bedrock / Claude 3 Sonnet)
  if (messages.length > 0 && messages[0].role === 'assistant') {
    const placeholder: UserMessage = {
      role: 'user',
      content: '[conversation history]',
      timestamp: Date.now(),
    };
    messages.unshift(placeholder);
  }

  return { messages, systemPrompt };
}

// ---------------------------------------------------------------------------
// Blob ref resolution
// ---------------------------------------------------------------------------

/**
 * Function that resolves a blob id to bytes.
 * Typically a thin wrapper around ConnectomeClient.getBlob().
 */
export type BlobFetcher = (blobId: string) => Promise<{
  bytes: Uint8Array;
  contentType: string;
  filename?: string;
}>;

/**
 * Walk a list of messages and resolve any blob-ref attachments to inline data.
 *
 * Attachments with `blobId` set and no `data` are fetched via `fetchBlob` and
 * populated with `data` (base64) so the downstream context-adapter can inline
 * them as `ImageContent` exactly like the legacy inline path.
 *
 * Failed fetches are logged and the attachment is dropped (better than failing
 * the whole cycle on a missing blob — the LLM just doesn't see that one).
 *
 * Caches resolved blobs by id within a single call so the same blob referenced
 * by multiple messages (e.g. a quoted image) is fetched once.
 *
 * @returns A new array of messages with resolved attachments. Input is not mutated.
 */
export async function resolveAttachmentRefs(
  messages: ContextMessage[],
  fetchBlob: BlobFetcher
): Promise<ContextMessage[]> {
  const cache = new Map<string, { bytes: Uint8Array; contentType: string; filename?: string }>();

  const fetchCached = async (blobId: string) => {
    let result = cache.get(blobId);
    if (!result) {
      result = await fetchBlob(blobId);
      cache.set(blobId, result);
    }
    return result;
  };

  const resolved: ContextMessage[] = [];

  for (const msg of messages) {
    const attachments = msg.metadata?.attachments;
    if (!attachments || attachments.length === 0) {
      resolved.push(msg);
      continue;
    }

    const newAttachments: ContextAttachment[] = [];
    let mutated = false;

    for (const att of attachments) {
      // Already has inline data — pass through unchanged (legacy path)
      if (att.data) {
        newAttachments.push(att);
        continue;
      }

      // Has blobId but no data — resolve
      if (att.blobId) {
        try {
          const blob = await fetchCached(att.blobId);
          mutated = true;
          newAttachments.push({
            ...att,
            contentType: att.contentType || blob.contentType,
            filename: att.filename || att.name || blob.filename,
            // Convert bytes → base64 for the inline pipeline
            data: Buffer.from(blob.bytes).toString('base64'),
            sizeBytes: att.sizeBytes ?? blob.bytes.length,
          });
        } catch (err: any) {
          console.warn(
            `[context-adapter] Failed to resolve blob ${att.blobId.substring(0, 12)}...: ${err.message} — dropping attachment`
          );
          mutated = true;
          // Drop unresolvable attachment rather than failing the cycle
        }
        continue;
      }

      // URL-only or no transport mode — pass through
      newAttachments.push(att);
    }

    if (mutated) {
      resolved.push({
        ...msg,
        metadata: { ...msg.metadata, attachments: newAttachments }
      });
    } else {
      resolved.push(msg);
    }
  }

  return resolved;
}
