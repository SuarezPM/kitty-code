# Progressive Skill Disclosure

## Purpose
Reduce system prompt size by listing skills by name+description only, loading full content on-demand via the Read tool.

## Requirements

### R1: Skill Listing in Prompt
- GIVEN skills are available in the system
- WHEN the system prompt is built
- THEN skills are listed as XML:
  ```xml
  <available_skills>
  <skill>
      <name>skill-name</name>
      <description>Brief description...</description>
      <location>/path/to/SKILL.md</location>
    </skill>
  </available_skills>
  ```
- AND full skill content is NOT included in the prompt

### R2: On-Demand Loading
- GIVEN the model encounters a task matching a skill description
- WHEN the model uses the Read tool on the skill location
- THEN the full SKILL.md content is loaded and available

### R3: Skill Discovery
- GIVEN skills exist in `.kitty/skills/` or `~/.kitty/skills/`
- WHEN the harness starts
- THEN skills are discovered by scanning for `SKILL.md` files with valid frontmatter (name, description)
- AND duplicate names are resolved (first wins, warning on collision)

### R4: Token Savings Measurement
- GIVEN the progressive disclosure system
- WHEN comparing prompt sizes
- THEN the token count reduction is measurable via telemetry

### R5: Unit Tests
- GIVEN the skill system
- WHEN tests are run
- THEN skill discovery, XML formatting, and on-demand loading work correctly

## Non-Goals
- No skill versioning or dependency management
- No dynamic skill creation at runtime
