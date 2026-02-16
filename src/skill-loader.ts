/**
 * SkillLoader — discovers and loads Agent Skills from directories and files.
 *
 * Ported from pi-mono's coding-agent skill loader, simplified for
 * connectome-agent-core's needs. Skills are SKILL.md files with YAML
 * frontmatter that provide specialized instructions for the agent.
 *
 * Skills follow the Agent Skills standard: https://agentskills.io
 */

import { existsSync, readdirSync, readFileSync, statSync } from 'fs';
import { basename, dirname, join, resolve } from 'path';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface Skill {
  /** Skill name (must match parent directory name) */
  name: string;
  /** Brief description — injected into system prompt */
  description: string;
  /** Absolute path to the SKILL.md file */
  filePath: string;
  /** Absolute path to the skill directory (parent of SKILL.md) */
  baseDir: string;
  /** Where this skill was loaded from */
  source: string;
  /** If true, hidden from system prompt (only invocable via /skill:name) */
  disableModelInvocation: boolean;
}

export interface SkillFrontmatter {
  name?: string;
  description?: string;
  'disable-model-invocation'?: boolean;
  [key: string]: unknown;
}

export interface SkillLoadResult {
  skills: Skill[];
  diagnostics: SkillDiagnostic[];
}

export interface SkillDiagnostic {
  type: 'warning' | 'collision';
  message: string;
  path: string;
}

// ---------------------------------------------------------------------------
// Frontmatter parsing (minimal — no yaml dependency needed for flat k/v)
// ---------------------------------------------------------------------------

/**
 * Parse YAML frontmatter from a markdown file.
 *
 * Handles the simple flat key-value pairs used by SKILL.md files.
 * Does NOT support nested YAML, arrays, or multi-line values.
 */
function parseFrontmatter(content: string): { frontmatter: SkillFrontmatter; body: string } {
  const normalized = content.replace(/\r\n/g, '\n').replace(/\r/g, '\n');

  if (!normalized.startsWith('---')) {
    return { frontmatter: {}, body: normalized };
  }

  const endIndex = normalized.indexOf('\n---', 3);
  if (endIndex === -1) {
    return { frontmatter: {}, body: normalized };
  }

  const yamlString = normalized.slice(4, endIndex);
  const body = normalized.slice(endIndex + 4).trim();

  const frontmatter: Record<string, unknown> = {};

  for (const line of yamlString.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;

    const colonIdx = trimmed.indexOf(':');
    if (colonIdx === -1) continue;

    const key = trimmed.slice(0, colonIdx).trim();
    let rawValue = trimmed.slice(colonIdx + 1).trim();

    // Strip surrounding quotes
    if (
      (rawValue.startsWith('"') && rawValue.endsWith('"')) ||
      (rawValue.startsWith("'") && rawValue.endsWith("'"))
    ) {
      rawValue = rawValue.slice(1, -1);
    }

    // Parse booleans
    if (rawValue === 'true') {
      frontmatter[key] = true;
    } else if (rawValue === 'false') {
      frontmatter[key] = false;
    } else {
      frontmatter[key] = rawValue;
    }
  }

  return { frontmatter: frontmatter as SkillFrontmatter, body };
}

/**
 * Strip frontmatter from markdown content, returning just the body.
 */
export function stripFrontmatter(content: string): string {
  return parseFrontmatter(content).body;
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

const MAX_NAME_LENGTH = 64;
const MAX_DESCRIPTION_LENGTH = 1024;

function validateName(name: string, parentDirName: string): string[] {
  const errors: string[] = [];

  if (name !== parentDirName) {
    errors.push(`name "${name}" does not match parent directory "${parentDirName}"`);
  }
  if (name.length > MAX_NAME_LENGTH) {
    errors.push(`name exceeds ${MAX_NAME_LENGTH} characters (${name.length})`);
  }
  if (!/^[a-z0-9-]+$/.test(name)) {
    errors.push('name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)');
  }
  if (name.startsWith('-') || name.endsWith('-')) {
    errors.push('name must not start or end with a hyphen');
  }
  if (name.includes('--')) {
    errors.push('name must not contain consecutive hyphens');
  }

  return errors;
}

function validateDescription(description: string | undefined): string[] {
  if (!description || description.trim() === '') {
    return ['description is required'];
  }
  if (description.length > MAX_DESCRIPTION_LENGTH) {
    return [`description exceeds ${MAX_DESCRIPTION_LENGTH} characters (${description.length})`];
  }
  return [];
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/**
 * Load a single skill from a SKILL.md file.
 */
function loadSkillFromFile(
  filePath: string,
  source: string,
): { skill: Skill | null; diagnostics: SkillDiagnostic[] } {
  const diagnostics: SkillDiagnostic[] = [];

  try {
    const rawContent = readFileSync(filePath, 'utf-8');
    const { frontmatter } = parseFrontmatter(rawContent);
    const skillDir = dirname(filePath);
    const parentDirName = basename(skillDir);

    // Validate description
    for (const error of validateDescription(frontmatter.description)) {
      diagnostics.push({ type: 'warning', message: error, path: filePath });
    }

    // Use name from frontmatter, or fall back to parent directory name
    const name = frontmatter.name || parentDirName;

    // Validate name
    for (const error of validateName(name, parentDirName)) {
      diagnostics.push({ type: 'warning', message: error, path: filePath });
    }

    // Still load even with warnings — unless description is completely missing
    if (!frontmatter.description || frontmatter.description.trim() === '') {
      return { skill: null, diagnostics };
    }

    return {
      skill: {
        name,
        description: frontmatter.description,
        filePath,
        baseDir: skillDir,
        source,
        disableModelInvocation: frontmatter['disable-model-invocation'] === true,
      },
      diagnostics,
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : 'failed to parse skill file';
    diagnostics.push({ type: 'warning', message, path: filePath });
    return { skill: null, diagnostics };
  }
}

/**
 * Recursively load skills from a directory.
 *
 * Discovery rules:
 * - Direct .md files in the root directory
 * - Recursive SKILL.md files under subdirectories
 */
function loadSkillsFromDirInternal(
  dir: string,
  source: string,
  includeRootFiles: boolean,
): SkillLoadResult {
  const skills: Skill[] = [];
  const diagnostics: SkillDiagnostic[] = [];

  if (!existsSync(dir)) {
    return { skills, diagnostics };
  }

  try {
    const entries = readdirSync(dir, { withFileTypes: true });

    for (const entry of entries) {
      // Skip hidden files and node_modules
      if (entry.name.startsWith('.') || entry.name === 'node_modules') {
        continue;
      }

      const fullPath = join(dir, entry.name);

      // Follow symlinks
      let isDirectory = entry.isDirectory();
      let isFile = entry.isFile();
      if (entry.isSymbolicLink()) {
        try {
          const stats = statSync(fullPath);
          isDirectory = stats.isDirectory();
          isFile = stats.isFile();
        } catch {
          continue; // broken symlink
        }
      }

      if (isDirectory) {
        const subResult = loadSkillsFromDirInternal(fullPath, source, false);
        skills.push(...subResult.skills);
        diagnostics.push(...subResult.diagnostics);
        continue;
      }

      if (!isFile) continue;

      // Root: any .md file; subdirectories: only SKILL.md
      const isRootMd = includeRootFiles && entry.name.endsWith('.md');
      const isSkillMd = !includeRootFiles && entry.name === 'SKILL.md';
      if (!isRootMd && !isSkillMd) continue;

      const result = loadSkillFromFile(fullPath, source);
      if (result.skill) {
        skills.push(result.skill);
      }
      diagnostics.push(...result.diagnostics);
    }
  } catch {
    // Directory read failed — skip silently
  }

  return { skills, diagnostics };
}

/**
 * Load skills from a directory.
 *
 * Top-level .md files and recursive SKILL.md files are both discovered.
 */
export function loadSkillsFromDir(dir: string, source: string = 'path'): SkillLoadResult {
  return loadSkillsFromDirInternal(dir, source, true);
}

/**
 * Load skills from an array of paths (files or directories).
 *
 * This is the primary entry point for ConnectomeAgent, which receives
 * `skillPaths` from its config.
 *
 * Deduplicates by skill name (first loaded wins). Returns loaded skills
 * and any validation diagnostics.
 */
export function loadSkillsFromPaths(paths: string[]): SkillLoadResult {
  const skillMap = new Map<string, Skill>();
  const allDiagnostics: SkillDiagnostic[] = [];

  for (const rawPath of paths) {
    const resolvedPath = resolve(rawPath);

    if (!existsSync(resolvedPath)) {
      allDiagnostics.push({
        type: 'warning',
        message: 'skill path does not exist',
        path: resolvedPath,
      });
      continue;
    }

    try {
      const stats = statSync(resolvedPath);

      if (stats.isDirectory()) {
        const result = loadSkillsFromDirInternal(resolvedPath, 'path', true);
        allDiagnostics.push(...result.diagnostics);
        for (const skill of result.skills) {
          addSkill(skillMap, skill, allDiagnostics);
        }
      } else if (stats.isFile() && resolvedPath.endsWith('.md')) {
        const result = loadSkillFromFile(resolvedPath, 'path');
        allDiagnostics.push(...result.diagnostics);
        if (result.skill) {
          addSkill(skillMap, result.skill, allDiagnostics);
        }
      } else {
        allDiagnostics.push({
          type: 'warning',
          message: 'skill path is not a directory or markdown file',
          path: resolvedPath,
        });
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'failed to read skill path';
      allDiagnostics.push({ type: 'warning', message, path: resolvedPath });
    }
  }

  return {
    skills: Array.from(skillMap.values()),
    diagnostics: allDiagnostics,
  };
}

function addSkill(
  map: Map<string, Skill>,
  skill: Skill,
  diagnostics: SkillDiagnostic[],
): void {
  const existing = map.get(skill.name);
  if (existing) {
    diagnostics.push({
      type: 'collision',
      message: `name "${skill.name}" collision — keeping ${existing.filePath}, skipping ${skill.filePath}`,
      path: skill.filePath,
    });
  } else {
    map.set(skill.name, skill);
  }
}

// ---------------------------------------------------------------------------
// Formatting for system prompt
// ---------------------------------------------------------------------------

function escapeXml(str: string): string {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

/**
 * Format skills for inclusion in a system prompt.
 *
 * Uses XML format per the Agent Skills standard.
 * Skills with disableModelInvocation=true are excluded.
 */
export function formatSkillsForPrompt(skills: Skill[]): string {
  const visible = skills.filter((s) => !s.disableModelInvocation);

  if (visible.length === 0) {
    return '';
  }

  const lines = [
    '',
    '',
    'The following skills provide specialized instructions for specific tasks.',
    'Use the read tool to load a skill\'s file when the task matches its description.',
    'When a skill file references a relative path, resolve it against the skill directory (parent of SKILL.md) and use that absolute path in tool commands.',
    '',
    '<available_skills>',
  ];

  for (const skill of visible) {
    lines.push('  <skill>');
    lines.push(`    <name>${escapeXml(skill.name)}</name>`);
    lines.push(`    <description>${escapeXml(skill.description)}</description>`);
    lines.push(`    <location>${escapeXml(skill.filePath)}</location>`);
    lines.push('  </skill>');
  }

  lines.push('</available_skills>');

  return lines.join('\n');
}

// ---------------------------------------------------------------------------
// Skill content access (for /skill:name expansion)
// ---------------------------------------------------------------------------

/**
 * Read a skill's full content, stripping frontmatter.
 *
 * Returns the markdown body wrapped in an XML `<skill>` tag with metadata,
 * ready for injection into a conversation.
 */
export function getSkillContent(skill: Skill): string {
  const rawContent = readFileSync(skill.filePath, 'utf-8');
  const body = stripFrontmatter(rawContent);

  return [
    `<skill name="${escapeXml(skill.name)}" location="${escapeXml(skill.filePath)}">`,
    `References are relative to ${skill.baseDir}/`,
    '',
    body,
    '</skill>',
  ].join('\n');
}
