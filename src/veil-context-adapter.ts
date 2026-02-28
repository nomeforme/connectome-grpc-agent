/**
 * VEILContextAdapter — Bridge between VEIL frame/facet model and pi-agent Message[] model.
 *
 * Converts VEIL state (frames containing facets) into the Message[] sequence
 * that pi-agent feeds to the LLM, and converts pi-agent assistant output back
 * into OutgoingVEILOperation[] that can be applied to the VEIL state.
 */

import type {
  Message,
  UserMessage,
  AssistantMessage,
  ToolResultMessage,
  TextContent,
  ImageContent,
  ThinkingContent,
  ToolCall,
} from '@mariozechner/pi-ai';

import type {
  Frame,
  Facet,
  OutgoingVEILOperation,
  VEILDelta,
  StreamRef,
  SpeechFacet,
  ThoughtFacet,
  ActionFacet,
  EventFacet,
  AmbientFacet,
} from '@connectome/connectome-ts';

import {
  hasContentAspect,
  hasAgentGeneratedAspect,
  hasStreamAspect,
  friendlyId,
} from '@connectome/connectome-ts';

import type { VEILContextAdapterConfig, VEILStateLike } from './types.js';

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/** Re-export VEILStateLike locally for use in method signatures. */
type VEILStateManagerLike = VEILStateLike;

/** Sentinel assistant message fields — we don't have real API metadata for
 *  reconstructed history, so we use placeholders. */
const PLACEHOLDER_ASSISTANT_FIELDS = {
  api: 'anthropic' as const,
  provider: 'anthropic' as const,
  model: 'reconstructed',
  usage: {
    input: 0,
    output: 0,
    cacheRead: 0,
    cacheWrite: 0,
    totalTokens: 0,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
  },
  stopReason: 'stop' as const,
};

/**
 * Parse a Frame/Facet timestamp (ISO-8601 string or unix-ms number) into a
 * unix-ms number suitable for pi-ai Message.timestamp.
 */
function toTimestampMs(ts: string | number | undefined): number {
  if (typeof ts === 'number') return ts;
  if (typeof ts === 'string') {
    const parsed = Date.parse(ts);
    if (!Number.isNaN(parsed)) return parsed;
  }
  return Date.now();
}

/**
 * Try to extract author name from an event facet.
 * Event facets store metadata under `state.metadata`.
 */
function getAuthorName(facet: Facet): string | undefined {
  const meta = (facet as any).state?.metadata;
  if (meta?.authorName) return meta.authorName as string;
  if (meta?.author) return meta.author as string;
  return undefined;
}

/**
 * Try to extract image attachments from an event facet.
 * Returns ImageContent[] for any base64-encoded image data found.
 */
function getImageAttachments(facet: Facet): ImageContent[] {
  const meta = (facet as any).state?.metadata;
  if (!meta?.attachments || !Array.isArray(meta.attachments)) return [];

  const images: ImageContent[] = [];
  for (const att of meta.attachments) {
    if (att.data && att.mimeType && att.mimeType.startsWith('image/')) {
      images.push({
        type: 'image',
        data: att.data,
        mimeType: att.mimeType,
      });
    }
  }
  return images;
}

/**
 * Stream reference with optional parent linkage for hierarchy-aware filtering.
 */
export interface StreamRefWithParent {
  streamId: string;
  streamType?: string;
  parentId?: string;
  forkSequence?: number;
}

/**
 * Check whether a frame's activeStream matches the desired streamRef.
 * If no streamRef filter is provided, every frame matches.
 * Supports hierarchy: parent stream frames before forkSequence are included.
 */
function frameMatchesStream(
  frame: Frame,
  streamRef?: StreamRefWithParent,
): boolean {
  if (!streamRef) return true;
  if (!frame.activeStream) return true; // frames with no stream are always included

  const fStreamId = frame.activeStream.streamId;
  if (fStreamId === streamRef.streamId) return true;

  // Hierarchy: include parent stream frames before fork point
  if (streamRef.parentId && fStreamId === streamRef.parentId
      && streamRef.forkSequence != null && frame.sequence <= streamRef.forkSequence) {
    return true;
  }

  return false;
}

/**
 * Check whether a facet belongs to the target stream.
 */
function facetMatchesStream(
  facet: Facet,
  streamRef?: StreamRefWithParent,
): boolean {
  if (!streamRef) return true;
  if (!hasStreamAspect(facet)) return true; // facets without stream aspect are ambient
  return (facet as any).streamId === streamRef.streamId;
}

// ---------------------------------------------------------------------------
// VEILContextAdapter
// ---------------------------------------------------------------------------

export class VEILContextAdapter {
  private readonly config: VEILContextAdapterConfig;
  private readonly maxFrames: number;

  constructor(config: VEILContextAdapterConfig) {
    this.config = config;
    this.maxFrames = config.maxFrames ?? 200;
  }

  // -----------------------------------------------------------------------
  // Public API: VEIL → Messages
  // -----------------------------------------------------------------------

  /**
   * Convert VEIL state into a pi-agent Message[] sequence.
   *
   * The system prompt is NOT included in the returned array — call
   * `getSystemPrompt()` separately and pass it to the agent loop config.
   */
  renderToMessages(
    veilState: VEILStateManagerLike,
    streamRef?: { streamId: string; streamType?: string },
  ): Message[] {
    const allFrames = veilState.getFrameHistory();

    // 1. Filter by stream
    let frames = streamRef
      ? allFrames.filter((f) => frameMatchesStream(f, streamRef))
      : Array.from(allFrames);

    // 2. Limit to maxFrames (take from end = most recent)
    if (frames.length > this.maxFrames) {
      frames = frames.slice(frames.length - this.maxFrames);
    }

    // 3. Walk frames chronologically and build messages
    const messages: Message[] = [];

    for (const frame of frames) {
      const ts = toTimestampMs(frame.timestamp);

      for (const delta of frame.deltas) {
        if (delta.type !== 'addFacet') continue;

        const facet = delta.facet;

        // Skip facets that don't match the target stream
        if (!facetMatchesStream(facet, streamRef)) continue;

        const msg = this.facetToMessage(facet, ts);
        if (msg) {
          messages.push(msg);
        }
      }
    }

    return messages;
  }

  /**
   * Build the full system prompt, incorporating ambient facets from VEIL state.
   */
  getSystemPrompt(
    veilState: VEILStateManagerLike,
    streamRef?: { streamId: string; streamType?: string },
  ): string {
    const parts: string[] = [this.config.systemPrompt];

    // Gather ambient facets
    const facets = veilState.getFacets();
    const ambientParts: string[] = [];

    for (const [, facet] of facets) {
      if (facet.type !== 'ambient') continue;
      if (!facetMatchesStream(facet, streamRef)) continue;
      if (hasContentAspect(facet) && (facet as any).content) {
        ambientParts.push((facet as any).content as string);
      }
    }

    if (ambientParts.length > 0) {
      parts.push('');
      parts.push('## Current Context');
      for (const ambient of ambientParts) {
        parts.push(ambient);
      }
    }

    return parts.join('\n');
  }

  // -----------------------------------------------------------------------
  // Public API: Messages → VEIL Operations
  // -----------------------------------------------------------------------

  /**
   * Convert pi-agent output messages back into VEIL operations.
   *
   * Typically you pass the newly produced messages from a single agent cycle
   * (the AssistantMessage and any ToolResultMessages).
   */
  messagesToVEILOps(
    messages: Message[],
    streamRef?: { streamId: string; streamType?: string },
  ): OutgoingVEILOperation[] {
    const ops: OutgoingVEILOperation[] = [];
    const streamId = streamRef?.streamId ?? 'default';
    const streamType = streamRef?.streamType;

    for (const msg of messages) {
      switch (msg.role) {
        case 'assistant':
          ops.push(...this.assistantMessageToOps(msg, streamId, streamType));
          break;

        case 'toolResult':
          ops.push(...this.toolResultToOps(msg, streamId, streamType));
          break;

        // UserMessages are inbound — they don't produce VEIL ops from the agent side.
        case 'user':
          break;
      }
    }

    return ops;
  }

  // -----------------------------------------------------------------------
  // Private: Facet → Message conversion
  // -----------------------------------------------------------------------

  /**
   * Convert a single facet into a Message (or null if not representable).
   */
  private facetToMessage(facet: Facet, frameTimestamp: number): Message | null {
    switch (facet.type) {
      case 'event':
        return this.eventFacetToMessage(facet, frameTimestamp);

      case 'speech':
        return this.speechFacetToMessage(facet, frameTimestamp);

      case 'action':
        return this.actionFacetToMessage(facet, frameTimestamp);

      case 'thought':
        // Thought facets are internal reasoning — skip for context.
        // They were already seen by the agent that produced them.
        return null;

      default:
        // Other facet types (state, config, ephemeral, etc.) are not
        // conversational and don't map to Message[].
        return null;
    }
  }

  /**
   * Event facets → UserMessage.
   *
   * Events represent external happenings (user messages, system events, etc.).
   * They always become UserMessages regardless of content.
   */
  private eventFacetToMessage(facet: Facet, ts: number): UserMessage | null {
    if (!hasContentAspect(facet)) return null;

    const textContent = (facet as any).content as string;
    if (!textContent && getImageAttachments(facet).length === 0) return null;

    const authorName = getAuthorName(facet);
    const images = getImageAttachments(facet);

    // If there are image attachments, use the multi-block content form
    if (images.length > 0) {
      const contentBlocks: (TextContent | ImageContent)[] = [];

      if (textContent) {
        const label = authorName ? `${authorName}: ${textContent}` : textContent;
        contentBlocks.push({ type: 'text', text: label });
      }

      contentBlocks.push(...images);

      return {
        role: 'user',
        content: contentBlocks,
        timestamp: ts,
      };
    }

    // Plain text event
    const label = authorName ? `${authorName}: ${textContent}` : textContent;
    return {
      role: 'user',
      content: label,
      timestamp: ts,
    };
  }

  /**
   * Speech facets → AssistantMessage (if ours) or UserMessage (if from another agent).
   */
  private speechFacetToMessage(facet: Facet, ts: number): Message | null {
    if (!hasContentAspect(facet)) return null;

    const textContent = (facet as any).content as string;
    if (!textContent) return null;

    const isOurSpeech =
      hasAgentGeneratedAspect(facet) &&
      (facet as any).agentId === this.config.agentId;

    if (isOurSpeech) {
      // This agent's prior output → AssistantMessage
      return {
        role: 'assistant',
        content: [{ type: 'text', text: textContent }],
        ...PLACEHOLDER_ASSISTANT_FIELDS,
        timestamp: ts,
      } as AssistantMessage;
    }

    // Another agent's speech → UserMessage prefixed with agent name
    const agentName = (facet as any).agentName ?? (facet as any).agentId ?? 'agent';
    return {
      role: 'user',
      content: `[${agentName}] ${textContent}`,
      timestamp: ts,
    };
  }

  /**
   * Action facets → AssistantMessage with ToolCall content block.
   *
   * Only this agent's actions are represented (other agents' actions are opaque).
   */
  private actionFacetToMessage(facet: Facet, ts: number): Message | null {
    if (!hasAgentGeneratedAspect(facet)) return null;
    if ((facet as any).agentId !== this.config.agentId) return null;

    const state = (facet as any).state;
    if (!state?.toolName) return null;

    const toolCall: ToolCall = {
      type: 'toolCall',
      id: facet.id,
      name: state.toolName,
      arguments: state.parameters ?? {},
    };

    return {
      role: 'assistant',
      content: [toolCall],
      ...PLACEHOLDER_ASSISTANT_FIELDS,
      timestamp: ts,
    } as AssistantMessage;
  }

  // -----------------------------------------------------------------------
  // Private: Message → VEIL Operation conversion
  // -----------------------------------------------------------------------

  /**
   * Convert an AssistantMessage into addFacet operations.
   */
  private assistantMessageToOps(
    msg: AssistantMessage,
    streamId: string,
    streamType?: string,
  ): VEILDelta[] {
    const ops: VEILDelta[] = [];
    const streamFields: Record<string, string> = { streamId };
    if (streamType) streamFields.streamType = streamType;

    // Accumulate consecutive text blocks into a single speech facet
    let pendingText: string[] = [];

    const flushText = () => {
      if (pendingText.length === 0) return;
      const content = pendingText.join('');
      pendingText = [];

      const speechFacet: SpeechFacet = {
        id: friendlyId('speech'),
        type: 'speech',
        content,
        agentId: this.config.agentId,
        agentName: this.config.agentName,
        ...streamFields,
      } as SpeechFacet;

      ops.push({ type: 'addFacet', facet: speechFacet });
    };

    for (const block of msg.content) {
      switch (block.type) {
        case 'text':
          pendingText.push(block.text);
          break;

        case 'thinking': {
          // Flush any accumulated text before the thinking block
          flushText();

          const thoughtFacet: ThoughtFacet = {
            id: friendlyId('thought'),
            type: 'thought',
            content: block.thinking,
            agentId: this.config.agentId,
            agentName: this.config.agentName,
            ...streamFields,
          } as ThoughtFacet;

          ops.push({ type: 'addFacet', facet: thoughtFacet });
          break;
        }

        case 'toolCall': {
          // Flush any accumulated text before the tool call
          flushText();

          const actionFacet: ActionFacet = {
            id: friendlyId('action'),
            type: 'action',
            content: `@${block.name}`,
            state: {
              toolName: block.name,
              parameters: block.arguments,
            },
            agentId: this.config.agentId,
            agentName: this.config.agentName,
            ...streamFields,
          } as ActionFacet;

          ops.push({ type: 'addFacet', facet: actionFacet });
          break;
        }
      }
    }

    // Flush any trailing text
    flushText();

    return ops;
  }

  /**
   * Convert a ToolResultMessage into an addFacet operation (action-result event).
   */
  private toolResultToOps(
    msg: ToolResultMessage,
    streamId: string,
    streamType?: string,
  ): VEILDelta[] {
    const streamFields: Record<string, string> = { streamId };
    if (streamType) streamFields.streamType = streamType;

    // Extract text content from the tool result
    const textParts: string[] = [];
    for (const block of msg.content) {
      if (block.type === 'text') {
        textParts.push(block.text);
      }
    }

    const content = textParts.join('\n') || (msg.isError ? '[error]' : '[no output]');

    const resultFacet: EventFacet = {
      id: friendlyId('action-result'),
      type: 'event',
      content,
      state: {
        source: this.config.agentId,
        eventType: 'action-result',
        metadata: {
          toolCallId: msg.toolCallId,
          toolName: msg.toolName,
          isError: msg.isError,
        },
      },
      ...streamFields,
    } as EventFacet;

    return [{ type: 'addFacet', facet: resultFacet }];
  }
}
