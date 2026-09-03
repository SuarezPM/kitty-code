# Multi-Tier Retry

## Purpose
Enhance the basic retry mechanism with 3-tier error classification and 60+ patterns for transient vs permanent error detection.

## Requirements

### R1: Error Classification
- GIVEN an error from any provider or tool
- WHEN `classify_error()` is called
- THEN the error is classified as one of: transient, permanent, overflow, rate-limit, auth, quota

### R2: Provider HTTP Retry (Tier 1)
- GIVEN an HTTP error (429, 5xx, timeout, connection reset)
- WHEN a chat request fails
- THEN retry with exponential backoff: `min(0.5 * 2^attempt, 8)` seconds + jitter
- AND max 3 retries

### R3: Application Retry (Tier 2)
- GIVEN a non-HTTP error (malformed response, stream drop, tool failure)
- WHEN the error is classified as transient
- THEN retry with backoff: `base_delay * 2^(attempt-1)` where base_delay=2s
- AND max 3 retries
- AND emit retry events (on_retry_scheduled, on_retry_attempt_start, on_retry_finished)

### R4: Overflow Recovery (Tier 3)
- GIVEN a context overflow error
- WHEN overflow is detected
- THEN remove the failed message from agent state
- THEN compact context
- THEN retry the turn exactly once
- AND set `_overflow_recovery_attempted` flag to prevent infinite loops

### R5: Error Pattern Database
- GIVEN 60+ regex patterns for error classification
- WHEN matching against error messages
- THEN patterns cover: overloaded, rate limit, 5xx, network errors, stream drops, quota exhaustion, billing, subscription limits

### R6: Unit Tests
- GIVEN the retry module
- WHEN tests are run
- THEN all classification scenarios pass: transient, permanent, overflow, rate-limit, auth, quota

## Non-Goals
- No circuit breaker pattern (future enhancement)
- No cross-request retry persistence (session-scoped only)
