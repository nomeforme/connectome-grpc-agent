/**
 * VEILToolBridge
 *
 * Bridges VEIL action-definition facets and static tools into pi-agent's
 * AgentTool interface. Static tools (registered by platform adapters or the
 * ConnectomeAgent itself) live alongside dynamically-discovered VEIL tools
 * so that getAllTools() returns one unified set the LLM can invoke.
 */

import type { AgentTool, AgentToolResult } from '@mariozechner/pi-agent-core';
// pi-ai 0.73 swapped @sinclair/typebox for its successor `typebox` v1. Tool
// schemas MUST come from the same package pi validates against — see rlm/tools.ts.
import { Type, type TObject, type TSchema } from 'typebox';
import type { Facet, ActionDefinitionFacet } from '@connectome/connectome-ts';
import type { ToolHandler } from './types.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Sanitise a tool name so it is safe for LLM tool-calling conventions:
 * lowercase, only alphanumerics and underscores, no leading digits.
 */
function sanitizeToolName(raw: string): string {
  let name = raw
    .toLowerCase()
    .replace(/[^a-z0-9_]/g, '_')  // replace non-alphanum/underscore
    .replace(/_+/g, '_')          // collapse consecutive underscores
    .replace(/^_|_$/g, '');       // trim leading/trailing underscores

  // Tool names must not start with a digit
  if (/^[0-9]/.test(name)) {
    name = `action_${name}`;
  }

  return name || 'unnamed_action';
}

/**
 * Convert a single VEIL parameter definition to a TypeBox schema property.
 *
 * VEIL action-definition facets may express parameters in two styles:
 *
 * 1. Simple string description – `{ "query": "The search query" }`
 *    → `Type.String({ description: "The search query" })`
 *
 * 2. JSON-schema-ish object – `{ "count": { type: "number", description: "..." } }`
 *    → `Type.Number({ description: "..." })`
 */
function veilParamToTypebox(value: unknown): TSchema {
  // Style 1: bare string → String parameter with that description
  if (typeof value === 'string') {
    return Type.String({ description: value });
  }

  // Style 2: object with optional `type` / `description`
  if (value !== null && typeof value === 'object') {
    const spec = value as Record<string, any>;
    const desc = spec.description as string | undefined;
    const opts = desc ? { description: desc } : {};

    switch (spec.type) {
      case 'string':
        return Type.String(opts);
      case 'number':
      case 'integer':
        return Type.Number(opts);
      case 'boolean':
        return Type.Boolean(opts);
      case 'array':
        return Type.Array(Type.Any(), opts);
      case 'object':
        return Type.Record(Type.String(), Type.Any(), opts);
      default:
        // Unknown or missing type
        return Type.Any(opts);
    }
  }

  return Type.Any();
}

/**
 * Build a TypeBox TObject from the parameter list found in an
 * ActionDefinitionFacet.
 *
 * The facet's `state.parameters` is typed as `string[]` in the facet schema,
 * but at runtime components may supply richer objects.  We handle both.
 */
function buildParameterSchema(
  parameters: string[] | Record<string, any> | undefined,
  requiredFields?: string[],
): TObject {
  if (!parameters) {
    return Type.Object({});
  }

  // Array of parameter names (ActionDefinitionFacet's declared type)
  if (Array.isArray(parameters)) {
    const props: Record<string, TSchema> = {};
    for (const paramName of parameters) {
      if (typeof paramName === 'string') {
        props[paramName] = Type.String({ description: paramName });
      }
    }
    return Type.Object(props);
  }

  // Record<string, string | SchemaObject>
  if (typeof parameters === 'object') {
    const props: Record<string, TSchema> = {};
    for (const [key, value] of Object.entries(parameters)) {
      const schema = veilParamToTypebox(value);
      // If requiredFields is specified, wrap non-required params as Optional
      if (requiredFields && !requiredFields.includes(key)) {
        props[key] = Type.Optional(schema);
      } else {
        props[key] = schema;
      }
    }
    return Type.Object(props);
  }

  return Type.Object({});
}

// ---------------------------------------------------------------------------
// ToolHandler → AgentTool conversion
// ---------------------------------------------------------------------------

/**
 * Convert a ToolHandler (MCP/platform tool) to a pi-agent AgentTool.
 *
 * ToolHandler is the simple interface used by MCPManager and platform tool
 * factories (fetch, etc.). This bridges it into pi-agent's richer AgentTool
 * format with TypeBox schemas and structured results.
 */
export function toolHandlerToAgentTool(handler: ToolHandler): AgentTool {
  const parameterSchema = buildParameterSchema(handler.parameters, handler.required);

  return {
    name: handler.name,
    label: handler.name,
    description: handler.description,
    parameters: parameterSchema,
    execute: async (
      _toolCallId: string,
      params: unknown,
      _signal?: AbortSignal,
      _onUpdate?: any,
    ): Promise<AgentToolResult<any>> => {
      const resolvedParams = (params ?? {}) as Record<string, any>;
      try {
        const result = await handler.handler(resolvedParams);
        return {
          content: [{ type: 'text', text: result }],
          details: { toolHandler: true, name: handler.name },
        };
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : String(err);
        return {
          content: [{ type: 'text', text: `Error executing tool "${handler.name}": ${message}` }],
          details: { toolHandler: true, name: handler.name, error: message },
        };
      }
    },
  };
}

// ---------------------------------------------------------------------------
// VEILToolBridge
// ---------------------------------------------------------------------------

export class VEILToolBridge {
  private staticTools: Map<string, AgentTool> = new Map();
  private veilTools: Map<string, AgentTool> = new Map();

  // -------------------------------------------------------------------
  // Static tool management
  // -------------------------------------------------------------------

  /**
   * Register a static tool (not from VEIL). Used for platform-level tools
   * such as "send_message", "search_web", etc.
   */
  registerStaticTool(tool: AgentTool): void {
    this.staticTools.set(tool.name, tool);
  }

  /**
   * Remove a previously registered static tool by name.
   */
  removeStaticTool(name: string): void {
    this.staticTools.delete(name);
  }

  // -------------------------------------------------------------------
  // VEIL tool discovery
  // -------------------------------------------------------------------

  /**
   * Scan VEIL state for action-definition facets and create AgentTool
   * wrappers for each one.
   *
   * For every `action-definition` facet found in the supplied facet map:
   *   1. Extract actionName, description, parameters from facet.state
   *   2. Convert parameter definitions into a TypeBox schema
   *   3. Create an AgentTool whose execute() delegates to `executeAction`
   *
   * Existing VEIL tools are cleared first so that repeated calls during
   * successive agent cycles reflect the latest VEIL state.
   *
   * @param facets        The current VEIL facet map (`VEILState.facets`).
   * @param executeAction Callback that the ConnectomeAgent supplies.
   *                      Responsible for emitting the action as a VEIL event,
   *                      waiting for the result (e.g. via continuation facets),
   *                      and returning a result string.
   */
  discoverVEILTools(
    facets: ReadonlyMap<string, any>,
    executeAction: (toolName: string, params: Record<string, any>) => Promise<string>,
  ): void {
    this.veilTools.clear();

    for (const [_id, facet] of facets) {
      if (!facet || facet.type !== 'action-definition') continue;

      const actionDef = facet as ActionDefinitionFacet;
      const state = actionDef.state;
      if (!state || !state.actionName) continue;

      const toolName = sanitizeToolName(state.actionName);
      const description = state.description || `VEIL action: ${state.actionName}`;
      const parameterSchema = buildParameterSchema(state.parameters as any);

      const tool: AgentTool = {
        name: toolName,
        label: state.actionName,
        description,
        parameters: parameterSchema,
        execute: async (
          _toolCallId: string,
          params: unknown,
          _signal?: AbortSignal,
          _onUpdate?: any,
        ): Promise<AgentToolResult<any>> => {
          const resolvedParams = (params ?? {}) as Record<string, any>;
          try {
            const result = await executeAction(toolName, resolvedParams);
            return {
              content: [{ type: 'text', text: result }],
              details: { veilAction: true, actionName: state.actionName },
            };
          } catch (err: unknown) {
            const message =
              err instanceof Error ? err.message : String(err);
            return {
              content: [{ type: 'text', text: `Error executing VEIL action "${state.actionName}": ${message}` }],
              details: { veilAction: true, actionName: state.actionName, error: message },
            };
          }
        },
      };

      this.veilTools.set(toolName, tool);
    }
  }

  // -------------------------------------------------------------------
  // Combined access
  // -------------------------------------------------------------------

  /**
   * Return every registered tool: static tools first, then VEIL-discovered
   * tools. If a VEIL tool has the same name as a static tool, the static
   * tool takes precedence (platform tools are authoritative).
   */
  getAllTools(): AgentTool[] {
    const combined = new Map<string, AgentTool>();

    // VEIL tools first so static tools can override
    for (const [name, tool] of this.veilTools) {
      combined.set(name, tool);
    }
    for (const [name, tool] of this.staticTools) {
      combined.set(name, tool);
    }

    return Array.from(combined.values());
  }

  /**
   * Clear all VEIL-discovered tools. Called before re-discovery so stale
   * definitions don't linger across cycles.
   */
  clearVEILTools(): void {
    this.veilTools.clear();
  }
}
