/**
 * Shared utilities for @connectome/agent-core
 *
 * Platform-agnostic speech cleanup and message splitting, merged from the
 * nearly-identical implementations in discord-axon and signal-axon.
 */

// ---------------------------------------------------------------------------
// Speech Cleanup
// ---------------------------------------------------------------------------

/**
 * Clean LLM speech output for delivery to any platform.
 *
 * Handles:
 * - Extraction of content from tool-call-as-text patterns
 * - Removal of XML-style wrapper/reasoning tags
 * - Whitespace normalization
 */
export function cleanSpeechContent(content: string): string {
  if (!content) return '';

  let cleaned = content;

  // Extract content from tool-as-text patterns like:
  // @discord-control.send_message({"content": "..."})
  // @signal-control.send_message({"content": "..."})
  const toolTextMatch = cleaned.match(
    /@[\w-]+\.send_message\s*\(\s*(\{[\s\S]*?\})\s*\)/,
  );
  if (toolTextMatch) {
    try {
      const parsed = JSON.parse(toolTextMatch[1]);
      if (parsed.content) {
        cleaned = parsed.content;
      }
    } catch {
      // JSON parse failed — try regex extraction for the content field
      const contentMatch = toolTextMatch[1].match(
        /"content"\s*:\s*"((?:[^"\\]|\\.)*)"/,
      );
      if (contentMatch) {
        cleaned = contentMatch[1]
          .replace(/\\n/g, '\n')
          .replace(/\\"/g, '"')
          .replace(/\\\\/g, '\\');
      }
    }
  }

  // Remove XML-style wrapper and reasoning tags
  cleaned = cleaned.replace(
    /<\/?(?:response|reply|message|output|answer|thinking|thought|inner_monologue|reasoning|reflection)[^>]*>/gi,
    '',
  );

  // Remove turn markers
  cleaned = cleaned.replace(/<\/?(?:my_turn|their_turn|turn)[^>]*>/gi, '');

  // Remove tool result wrappers entirely
  cleaned = cleaned.replace(/<tool_result[^>]*>[\s\S]*?<\/tool_result>/gi, '');

  // Remove thinking/reasoning blocks entirely
  cleaned = cleaned.replace(/<thinking>[\s\S]*?<\/thinking>/gi, '');
  cleaned = cleaned.replace(/<reasoning>[\s\S]*?<\/reasoning>/gi, '');
  cleaned = cleaned.replace(/<reflection>[\s\S]*?<\/reflection>/gi, '');

  // Strip remaining XML-like tags, but preserve platform mention syntax:
  // Discord: <@id>, <@!id>, <#channel>, <@&role>
  // These all start with <@ or <#, so we skip those.
  cleaned = cleaned.replace(/<(?!@|#)[^>]+>/g, '');

  // Normalize whitespace
  cleaned = cleaned.replace(/\n{4,}/g, '\n\n\n'); // max 3 consecutive newlines
  cleaned = cleaned.replace(/[ \t]+/g, ' '); // collapse horizontal whitespace
  cleaned = cleaned.replace(/\n[ \t]+/g, '\n'); // strip leading whitespace on lines
  cleaned = cleaned.replace(/[ \t]+\n/g, '\n'); // strip trailing whitespace on lines

  return cleaned.trim();
}

// ---------------------------------------------------------------------------
// Message Splitting
// ---------------------------------------------------------------------------

/**
 * Split a long message into chunks that respect a maximum length.
 *
 * Tries to break at natural boundaries (paragraph → sentence → word → forced).
 *
 * @param content    The message content to split
 * @param maxLength  Maximum characters per chunk (default 2000)
 */
export function splitMessage(
  content: string,
  maxLength: number = 2000,
): string[] {
  if (content.length <= maxLength) {
    return [content];
  }

  const chunks: string[] = [];
  let remaining = content;

  while (remaining.length > 0) {
    if (remaining.length <= maxLength) {
      chunks.push(remaining);
      break;
    }

    let breakPoint = maxLength;

    // 1. Try paragraph break (double newline)
    const paragraphBreak = remaining.lastIndexOf('\n\n', maxLength);
    if (paragraphBreak > maxLength * 0.4) {
      breakPoint = paragraphBreak + 2;
    } else {
      // 2. Try sentence break
      const sentenceEndings = ['. ', '! ', '? ', '.\n', '!\n', '?\n'];
      let bestSentence = -1;
      for (const sep of sentenceEndings) {
        const idx = remaining.lastIndexOf(sep, maxLength);
        if (idx > bestSentence && idx > maxLength * 0.3) {
          bestSentence = idx + sep.length;
        }
      }

      if (bestSentence > 0) {
        breakPoint = bestSentence;
      } else {
        // 3. Try word break
        const spaceBreak = remaining.lastIndexOf(' ', maxLength);
        if (spaceBreak > maxLength * 0.3) {
          breakPoint = spaceBreak + 1;
        }
        // 4. Otherwise force-break at maxLength
      }
    }

    chunks.push(remaining.substring(0, breakPoint).trim());
    remaining = remaining.substring(breakPoint).trim();
  }

  return chunks.filter((chunk) => chunk.length > 0);
}
