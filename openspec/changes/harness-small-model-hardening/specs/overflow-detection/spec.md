# Context Overflow Detection

## Purpose
Detect context overflow errors from 25+ provider patterns and automatically compact + retry.

## Requirements

### R1: Overflow Pattern Matching
- GIVEN an error response from any provider
- WHEN `detect_overflow()` is called
- THEN the error is checked against 25+ regex patterns for provider-specific overflow messages
- AND patterns cover: Ollama, llama.cpp, OpenAI, Anthropic, vLLM, SGLang, and others

### R2: Silent Overflow Detection
- GIVEN a response with zero output tokens and a large input
- WHEN overflow detection runs
- THEN it's classified as silent overflow (provider returned empty response due to context limit)

### R3: Auto-Compact + Retry
- GIVEN overflow is detected
- WHEN overflow recovery triggers
- THEN compact context
- THEN retry the failed turn exactly once
- AND if retry also fails, report the error

### R4: Stale Overflow Filtering
- GIVEN a pre-compaction overflow error
- WHEN a new request is made after compaction
- THEN the old overflow error is ignored (timestamp comparison)

### R5: Unit Tests
- GIVEN the overflow module
- WHEN tests are run
- THEN all overflow patterns match correctly, silent overflow is detected, stale errors are filtered

## Non-Goals
- No provider-specific overflow handling (generic patterns only)
- No proactive overflow prevention (reactive only)
