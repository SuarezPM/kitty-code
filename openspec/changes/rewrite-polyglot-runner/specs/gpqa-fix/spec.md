# GPQA Diamond Fix

## Purpose
Fix the GPQA runner to produce results comparable to the official leaderboard: add CoT, align prompt format, strict scoring.

## Requirements

### R1: Chain-of-Thought Prompt
- WHEN the GPQA prompt is built
- THEN it includes "Think step by step before answering" before the question
- AND the model is expected to reason before selecting A/B/C/D

### R2: Official Prompt Format
- WHEN the GPQA prompt is presented
- THEN it uses the official format: "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of A,B,C,D."

### R3: Strict Format Scoring
- WHEN the model response is graded
- THEN only responses matching `ANSWER: X` pattern receive credit
- AND responses without the ANSWER prefix receive 0 points

### R4: Multiple Epochs
- WHEN running the full benchmark
- THEN support configurable epochs (default: 1 for local, 4 for official)

### R5: No Permutation (or Documented)
- WHEN options are presented
- THEN either use the original dataset order (A/B/C/D fixed) OR clearly document that permutation is applied and track the correct index

### R6: Unit Tests
- GIVEN the GPQA fixes
- WHEN tests are run
- THEN CoT prompt is correct, strict scoring works, format detection is accurate
