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
 * Minimal message shape accepted by the context adapter.
 * Both connectome-ts RenderedContext and axon-local RenderedContext satisfy this.
 */
export interface ContextMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
  metadata?: {
    attachments?: Array<{
      id?: string;
      url?: string;
      contentType?: string;
      name?: string;
      size?: number;
      data?: string;  // base64 encoded
    }>;
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
