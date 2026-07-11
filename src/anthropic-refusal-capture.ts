/**
 * Anthropic refusal capture — enriches pi-ai's stripped "An unknown error
 * occurred" throw with the real `stop_details.category` + explanation from
 * Anthropic's messages API.
 *
 * Background: when Anthropic's mid-stream safety classifier trips, the API
 * responds with `stop_reason: "refusal"` and a `stop_details` object carrying
 * a human-readable `category` and `explanation`. Pi-ai's provider maps this
 * stop reason to `"error"` and throws a generic Error before either piece of
 * detail can reach any downstream consumer. Bumping pi-ai does not help —
 * the failing code path is identical in all versions up through 0.73.1.
 *
 * Strategy: wrap the streamFn. Watch pi-ai's event stream. If the stream
 * emits `type: "error"` with `errorMessage === "An unknown error occurred"`
 * AND no content blocks were produced, replay a lightweight non-streaming
 * `messages.create` call using `@anthropic-ai/sdk` directly to fetch
 * `stop_reason` + `stop_details`. Rewrite the error event's `errorMessage`
 * with `[Refused: <category>] <explanation>` before forwarding.
 *
 * Non-invasive: no patches to pi-ai. Uses only pi-ai's public exports
 * (`createAssistantMessageEventStream`) and the `@anthropic-ai/sdk` we
 * already depend on. Params are captured via pi-ai's `onPayload` hook, so
 * the replay reuses the exact same request pi-ai built (cache control,
 * messages, tools, thinking, everything).
 */

import Anthropic from '@anthropic-ai/sdk';
import {
  createAssistantMessageEventStream,
  type AssistantMessageEvent,
  type AssistantMessageEventStream,
  type Model,
  type Context,
  type StreamOptions,
} from '@earendil-works/pi-ai';

type StreamFn = (
  model: Model<any>,
  context: Context,
  options?: StreamOptions & Record<string, unknown>,
) => AssistantMessageEventStream;

const STRIPPED_ERROR = 'An unknown error occurred';

function isOAuthToken(apiKey: string): boolean {
  return apiKey.includes('sk-ant-oat');
}

function buildAnthropicClient(model: Model<any>, apiKey: string): Anthropic {
  const oauth = isOAuthToken(apiKey);
  if (oauth) {
    return new Anthropic({
      apiKey: null as unknown as string,
      authToken: apiKey,
      baseURL: model.baseUrl,
      dangerouslyAllowBrowser: true,
      defaultHeaders: {
        accept: 'application/json',
        'anthropic-dangerous-direct-browser-access': 'true',
        'anthropic-beta': 'claude-code-20250219,oauth-2025-04-20',
      },
    });
  }
  return new Anthropic({
    apiKey,
    baseURL: model.baseUrl,
    dangerouslyAllowBrowser: true,
    defaultHeaders: {
      accept: 'application/json',
      'anthropic-dangerous-direct-browser-access': 'true',
    },
  });
}

/**
 * Wrap a streamFn so that when the underlying anthropic provider throws its
 * stripped "An unknown error occurred" (which almost always means a mid-stream
 * classifier trip), we enrich the error event with the real category and
 * explanation before it propagates further.
 *
 * @param baseFn The streamFn to wrap (pi-ai's streamSimple, or a chained wrapper).
 * @param getApiKey Callback to resolve the current API key (OAuth or bearer).
 *                  If omitted, falls back to options.apiKey then ANTHROPIC_API_KEY env.
 */
export function wrapAnthropicWithRefusalCapture(
  baseFn: StreamFn,
  getApiKey?: (provider: string) => Promise<string | undefined> | string | undefined,
): StreamFn {
  return (model, context, options) => {
    if (model.provider !== 'anthropic') {
      return baseFn(model, context, options);
    }

    let capturedParams: any = undefined;
    const originalOnPayload = options?.onPayload;
    const wrappedOptions: any = {
      ...(options ?? {}),
      // pi-ai 0.73 changed this hook: it now receives (payload, model), and a
      // NON-UNDEFINED return value REPLACES the outgoing provider payload.
      // We only observe — never rewrite — so we must not return anything of our
      // own. Delegating to the upstream hook preserves any replacement IT makes
      // (and yields undefined when there is no upstream hook).
      onPayload: (payload: unknown, payloadModel: any) => {
        capturedParams = payload;
        return originalOnPayload?.(payload, payloadModel);
      },
    };

    const inputStream = baseFn(model, context, wrappedOptions);
    const outputStream = createAssistantMessageEventStream();

    (async () => {
      let hadContent = false;
      let pendingErrorEvent: AssistantMessageEvent | undefined;

      try {
        for await (const event of inputStream) {
          if (
            event.type === 'text_delta' ||
            event.type === 'thinking_delta' ||
            event.type === 'toolcall_delta' ||
            event.type === 'text_end' ||
            event.type === 'thinking_end' ||
            event.type === 'toolcall_end'
          ) {
            hadContent = true;
          }

          if (
            event.type === 'error' &&
            (event as any).error?.errorMessage === STRIPPED_ERROR
          ) {
            // Hold the error event — we may enrich it below.
            pendingErrorEvent = event;
            continue;
          }

          outputStream.push(event);
        }

        if (pendingErrorEvent && !hadContent && capturedParams) {
          try {
            const optApiKey = (wrappedOptions as any).apiKey as string | undefined;
            const resolved =
              optApiKey ??
              (getApiKey ? await getApiKey(model.provider) : undefined) ??
              process.env.ANTHROPIC_API_KEY;

            if (resolved) {
              const client = buildAnthropicClient(model, resolved);
              // capturedParams is the exact anthropic-messages-shaped payload
              // pi-ai just built. Reuse everything (messages, cache control,
              // tools, thinking) but force non-streaming + tiny max_tokens.
              const replayParams: any = { ...capturedParams };
              delete replayParams.stream;
              replayParams.max_tokens = 8;

              const resp: any = await client.messages.create(replayParams);
              const stopReason = resp?.stop_reason;
              const stopDetails = resp?.stop_details;

              if (stopReason === 'refusal' && stopDetails) {
                const category = stopDetails.category ?? 'unknown';
                const explanation = (stopDetails.explanation ?? '').split(' To learn more,')[0];
                const enriched = `[Refused: ${category}] ${explanation}`;
                (pendingErrorEvent as any).error.errorMessage = enriched;
              } else if (stopReason) {
                (pendingErrorEvent as any).error.errorMessage =
                  `[Model stop_reason=${stopReason}] (no additional detail from replay)`;
              }
            }
          } catch (replayErr) {
            // Enrichment failed — leave the original "An unknown error occurred"
            // so downstream still surfaces something. Log so we can see why.
            console.error(
              '[refusal-capture] replay failed:',
              (replayErr as Error)?.message ?? String(replayErr),
            );
          }
        }

        if (pendingErrorEvent) outputStream.push(pendingErrorEvent);
        outputStream.end();
      } catch (err) {
        // If something in our wrapper explodes, emit a synthetic error event
        // so downstream doesn't hang.
        try {
          outputStream.push({
            type: 'error',
            reason: 'error',
            error: {
              role: 'assistant',
              content: [],
              api: model.api,
              provider: model.provider,
              model: model.id,
              usage: {
                input: 0,
                output: 0,
                cacheRead: 0,
                cacheWrite: 0,
                totalTokens: 0,
                cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
              },
              stopReason: 'error',
              errorMessage: (err as Error)?.message ?? String(err),
              timestamp: Date.now(),
            },
          } as AssistantMessageEvent);
        } catch { /* nothing more we can do */ }
        outputStream.end();
      }
    })();

    return outputStream;
  };
}
