# Tool-Gated Prompt Sections

## Purpose
Only include prompt guidelines for tools that are currently active in the session, reducing prompt size and improving relevance.

## Requirements

### R1: Active Tool Detection
- GIVEN the current tool set for a session
- WHEN the system prompt is built
- THEN only guidelines for active tools are included

### R2: Conditional Sections
- GIVEN tool-specific guidelines (file exploration, bash safety, edit patterns)
- WHEN a tool is not in the active set
- THEN its guidelines are excluded from the prompt

### R3: Example Gating
- GIVEN bash-specific examples and warnings
- WHEN bash is not in the active tool set
- THEN bash guidelines are excluded
- AND when bash IS active, bash guidelines are included

### R4: Integration with ToolRouter
- GIVEN the `ToolRouter` classification from router.rs
- WHEN tools are selected for a prompt
- THEN prompt sections are filtered to match

### R5: Unit Tests
- GIVEN the prompt builder
- WHEN tests are run
- THEN correct sections are included/excluded based on active tools

## Non-Goals
- No dynamic guideline generation (static sections per tool)
- No per-model guideline customization (same guidelines for all models)
