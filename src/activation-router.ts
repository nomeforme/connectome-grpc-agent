/**
 * ActivationRouter — routes activations to the best-fit agent in a multi-agent setup.
 *
 * When multiple ConnectomeAgents are running (e.g. a coding agent + a search
 * agent + a creative agent), the router determines which agent(s) should handle
 * an incoming activation.
 *
 * Routing priority:
 *   1. Explicit targeting (activation.targetAgentId or targetAgentName)
 *   2. Capability matching (activation tags vs agent capabilities)
 *   3. Load balancing (prefer idle agents over busy ones)
 *   4. Default agent (fallback if no specific match)
 *
 * Usage:
 *   const router = new ActivationRouter();
 *   router.registerAgent({ id: 'coder', name: 'Coder', capabilities: ['code', 'debug'] }, agent);
 *   router.registerAgent({ id: 'search', name: 'Searcher', capabilities: ['search', 'web'] }, agent);
 *   const result = router.routeActivation(activation);
 *   if (result.agent) await effector.handleActivation(activation);
 */

import type { EffectorAgent, UnifiedActivation, ConnectomeCycleResult } from './types.js';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Registration info for an agent in the router. */
export interface AgentRegistration {
  /** Unique agent ID */
  id: string;
  /** Display name */
  name: string;
  /** Capability tags this agent handles (e.g., 'code', 'search', 'creative') */
  capabilities: string[];
  /** Priority weight — higher values are preferred when capabilities tie (default 0) */
  priority?: number;
  /** If true, this is the default agent for untargeted, unmatched activations */
  isDefault?: boolean;
}

/** Result from routing an activation. */
export interface RoutingResult {
  /** The selected agent, or null if no match */
  agent: EffectorAgent | null;
  /** Registration info for the selected agent */
  registration: AgentRegistration | null;
  /** How the agent was selected */
  reason: 'targeted' | 'capability' | 'default' | 'none';
  /** All agents that were considered (for debugging) */
  candidates: AgentRegistration[];
}

/** Activation with optional routing hints. */
export interface RoutableActivation extends UnifiedActivation {
  /** Target a specific agent by ID */
  targetAgentId?: string;
  /** Target a specific agent by name */
  targetAgentName?: string;
  /** Capability tags to match against (e.g., ['code', 'debug']) */
  capabilities?: string[];
}

/** Configuration for the ActivationRouter. */
export interface ActivationRouterConfig {
  /** Called when an activation cannot be routed to any agent */
  onUnroutable?: (activation: RoutableActivation) => void;
}

// ---------------------------------------------------------------------------
// ActivationRouter
// ---------------------------------------------------------------------------

export class ActivationRouter {
  private readonly agents: Map<string, { registration: AgentRegistration; agent: EffectorAgent }> =
    new Map();
  private readonly config: ActivationRouterConfig;

  constructor(config: ActivationRouterConfig = {}) {
    this.config = config;
  }

  // ---------------------------------------------------------------------------
  // Agent registry
  // ---------------------------------------------------------------------------

  /** Register an agent with its capabilities. */
  registerAgent(registration: AgentRegistration, agent: EffectorAgent): void {
    this.agents.set(registration.id, { registration, agent });
  }

  /** Remove an agent from the registry. */
  unregisterAgent(id: string): void {
    this.agents.delete(id);
  }

  /** Get registration info for a specific agent. */
  getRegistration(id: string): AgentRegistration | undefined {
    return this.agents.get(id)?.registration;
  }

  /** Get all registered agent IDs. */
  getRegisteredIds(): string[] {
    return Array.from(this.agents.keys());
  }

  /** Get all registrations. */
  getAllRegistrations(): AgentRegistration[] {
    return Array.from(this.agents.values()).map((e) => e.registration);
  }

  /** Number of registered agents. */
  get size(): number {
    return this.agents.size;
  }

  // ---------------------------------------------------------------------------
  // Routing
  // ---------------------------------------------------------------------------

  /**
   * Route an activation to the best-fit agent.
   *
   * Returns a RoutingResult with the selected agent and the reason for selection.
   */
  routeActivation(activation: RoutableActivation): RoutingResult {
    const candidates = this.getAllRegistrations();

    // 1. Explicit targeting by ID
    if (activation.targetAgentId) {
      const entry = this.agents.get(activation.targetAgentId);
      if (entry) {
        return {
          agent: entry.agent,
          registration: entry.registration,
          reason: 'targeted',
          candidates,
        };
      }
    }

    // 2. Explicit targeting by name
    if (activation.targetAgentName) {
      const targetName = activation.targetAgentName.toLowerCase();
      for (const [, entry] of this.agents) {
        if (
          entry.registration.name.toLowerCase() === targetName ||
          entry.registration.id.toLowerCase() === targetName
        ) {
          return {
            agent: entry.agent,
            registration: entry.registration,
            reason: 'targeted',
            candidates,
          };
        }
      }
    }

    // 3. Capability matching
    if (activation.capabilities && activation.capabilities.length > 0) {
      const scored = this.scoreByCapabilities(activation.capabilities);
      if (scored) {
        return {
          agent: scored.agent,
          registration: scored.registration,
          reason: 'capability',
          candidates,
        };
      }
    }

    // 4. Default agent
    const defaultEntry = this.findDefaultAgent();
    if (defaultEntry) {
      return {
        agent: defaultEntry.agent,
        registration: defaultEntry.registration,
        reason: 'default',
        candidates,
      };
    }

    // 5. Single agent fallback — if only one agent registered, use it
    if (this.agents.size === 1) {
      const [, entry] = this.agents.entries().next().value!;
      return {
        agent: entry.agent,
        registration: entry.registration,
        reason: 'default',
        candidates,
      };
    }

    // No match
    if (this.config.onUnroutable) {
      this.config.onUnroutable(activation);
    }

    return { agent: null, registration: null, reason: 'none', candidates };
  }

  // ---------------------------------------------------------------------------
  // Private
  // ---------------------------------------------------------------------------

  /**
   * Score agents by capability overlap and priority.
   * Returns the best match, or null if no agent has any matching capability.
   */
  private scoreByCapabilities(
    requestedCapabilities: string[],
  ): { registration: AgentRegistration; agent: EffectorAgent } | null {
    let bestEntry: { registration: AgentRegistration; agent: EffectorAgent } | null = null;
    let bestScore = 0;

    const requested = new Set(requestedCapabilities.map((c) => c.toLowerCase()));

    for (const [, entry] of this.agents) {
      const agentCaps = new Set(entry.registration.capabilities.map((c) => c.toLowerCase()));
      let overlap = 0;
      for (const cap of requested) {
        if (agentCaps.has(cap)) overlap++;
      }

      if (overlap === 0) continue;

      // Score = overlap count + priority tiebreaker (normalized to <1)
      const priority = entry.registration.priority ?? 0;
      const score = overlap + priority * 0.01;

      if (score > bestScore) {
        bestScore = score;
        bestEntry = entry;
      }
    }

    return bestEntry;
  }

  /** Find the agent marked as default (first one wins if multiple). */
  private findDefaultAgent(): { registration: AgentRegistration; agent: EffectorAgent } | null {
    for (const [, entry] of this.agents) {
      if (entry.registration.isDefault) {
        return entry;
      }
    }
    return null;
  }
}
