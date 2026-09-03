# JSON Repair Pipeline

## Purpose
Automatically fix malformed tool call arguments produced by small local models before validation and execution.

## Requirements

### R1: JSON String Repair
- GIVEN a tool call argument string containing raw control characters or invalid escapes
- WHEN `repair_json()` is called
- THEN the string is cleaned: raw control chars escaped, backslashes doubled before invalid escape targets
- AND the result is valid JSON or the original string if unrepairable

### R2: Schema-Based Type Coercion
- GIVEN a JSON object and a TypeBox/JSON Schema
- WHEN `coerce_with_schema()` is called
- THEN string values are coerced to target types: `"1"` → `1` for integers, `"true"` → `true` for booleans, `null` → `""` for strings
- AND union types (`anyOf`, `oneOf`, `allOf`) are resolved recursively
- AND nested objects and arrays are coerced recursively

### R3: Optional Null Normalization
- GIVEN a JSON object with `null` values for optional properties
- WHEN coercion completes
- THEN `null` is removed for optional properties that don't accept null

### R4: Integration Point
- GIVEN a tool call parsed from LLM response
- WHEN the JSON arguments fail schema validation
- THEN `repair_json()` is attempted first
- THEN `coerce_with_schema()` is attempted on the repaired result
- AND only if both fail, the original validation error is returned

### R5: Unit Tests
- GIVEN the repair pipeline
- WHEN tests are run
- THEN all repair scenarios pass: raw control chars, invalid escapes, string-to-int coercion, string-to-bool coercion, union types, nested objects, optional null removal

## Non-Goals
- No streaming JSON repair (handled separately by partial-json library)
- No semantic validation (only structural JSON + schema types)
