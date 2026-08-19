# Plan: Kitty Code — Local Coding Agent for ornith:9B

**Status: pending approval**
**Review: Round 2 of 3 (1 re-review remaining)**

## Requirements Summary

Kitty Code is a local coding agent written in Rust, a single binary, optimized for ornith:9B on RTX 2060S 8GB. It uses an appropriate harness (strong guards + conditional knowledge injection), hashline edits as primary edit primitive, 2-stage router for 10 tools, tail injection + tree-sitter signatures + stdout truncation for context management, and episodic blake3 memory. Performance is measured with CNAS (Polyglot + TB + SWE-V) and real numbers are published in AUDIT.md. The project follows OpenSSF Best Practices Badge Silver and targets a high OpenSSF Scorecard.

## Quality & Repo Hardening Mandate (NON-NEGOTIABLE)

All code and repo artifacts MUST follow these rules:

### 1. Architecture
- Design by cohesive modules/packages with explicit interfaces between layers.
- Strong encapsulation. No god objects, no circular dependencies.
- Simplicity and readability over cleverness.
- Eliminate dead code, unused branches, unnecessary adapters, empty abstractions.
- Domain/Application/Infrastructure/Presentation separation where it makes sense, without overengineering.

### 2. Code Design
- Precise domain naming. Small, cohesive, testable classes/structs.
- Composition over inheritance. Value objects, DTOs, clear schemas.
- No primitive obsession — use newtypes for domain semantics.
- Immutability by default (`&self` over `&mut self` where possible).

### 3. Quality
- Unit and integration tests on all critical paths.
- Linting (clippy), formatting (rustfmt), type checking in CI.
- Every relevant change verifiable by tests.
- No `.unwrap()` in library code. Use `?` or `.expect("reason")`.
- No `.clone()` without justification. Prefer references.

### 4. Repository Security (OpenSSF Scorecard)
- LICENSE (MIT), SECURITY.md, CONTRIBUTING.md, CODEOWNERS, README, issue templates.
- Branch protection, mandatory code review.
- Dependency update tool (Renovate/Dependabot).
- SAST (CodeQL or cargo-audit/cargo-deny).
- SBOM generation on releases.
- Signed releases or provenance generation.
- Pin GitHub Actions by SHA, minimize GITHUB_TOKEN permissions.
- No dangerous workflows, no generated binaries in repo.
- Pinned dependencies or correct lockfiles.
- Secret scanning / gitleaks.

### 5. Supply Chain & Releases
- Reproducible builds (locked toolchain via rust-toolchain.toml).
- Changelog generation.
- Semantic versioning strategy.
- Clean, verifiable release artifacts.
- Documented installation, deployment, and verification.

### 6. Delivery
Always produce:
a) repo structure,
b) initial files,
c) CI/CD workflows,
d) OpenSSF/Scorecard checklist,
e) testing plan,
f) explicit debt log for deferred items.

## VRAM Budget (Week 0 measurement required — DO NOT COMMIT to any context size before measuring)

| Component | VRAM (4K ctx) | VRAM (8K ctx) | VRAM (16K ctx) | Notes |
|-----------|---------------|---------------|----------------|-------|
| Model weights (Q4_K_M) | ~5.6 GB | ~5.6 GB | ~5.6 GB | ornith:9b default quant |
| KV cache (q8_0) | ~0.6 GB | ~1.2 GB | ~2.4 GB | Flash Attention enabled |
| CUDA overhead | ~0.5 GB | ~0.5 GB | ~0.5 GB | Driver, allocator |
| **Total** | **~6.7 GB** | **~7.3 GB** | **~8.5 GB** | 8K leaves 0.7GB margin; 16K OOMs |

**CRITICAL RISK:** 8K context leaves only 0.7 GB headroom. One allocation spike or context expansion causes OOM. Week 0 MUST measure actual VRAM with `nvidia-smi` during inference. If 8K is unstable, fall back to 4K baseline.

**Quantization:** Q4_K_M default (5.6GB). Q5_K_M (6.8GB) ONLY if Week 0 shows quality gap AND VRAM allows. Q4_K_S (~5.2GB) as fallback if 8K OOMs.
**Context:** MEASURE in Week 0. Start at 4K, increase only if stable. Do not assume 8K works.

## Acceptance Criteria

1. **Polyglot ≥45%** with ornith:9B on 8GB, measured on 225 exercises (gate: surpass little-coder 45.56%). NOTE: This is a HYPOTHESIS validated by Week 0 baseline. If raw baseline <25%, 45% target is likely unreachable.
2. **TB 2.0 ≥18%** on 8GB, measured on 89 tasks × 1 trial
3. **Tool-call validity ≥85%** on 100 prompts, measured before any other work. CRITICAL: if <85%, model cannot support this architecture.
4. **Edit apply success (hashline) ≥80%**, measured on 10 synthetic cases + Polyglot exercises
5. **Binary <25 MB stripped, <30 MB RSS, <500 ms first-inference-after-load**
6. **AUDIT.md** published with 🟢 measured / 🟡 self-reported / 🔬 not run
7. **VRAM usage stable at chosen context size** (measured, not assumed). Target: no OOM during inference.
8. **OpenSSF Scorecard ≥7/10** (target: 8+)
9. **OpenSSF Best Practices Badge: Silver** (minimum passing, Gold aspirational)

## Repo Structure

```
kitty-code/
├── .github/
│   ├── workflows/
│   │   ├── ci.yml              # Lint + test + build on every PR
│   │   ├── release.yml         # Release artifact generation
│   │   ├── scorecard.yml       # OpenSSF Scorecard analysis
│   │   ├── codeql.yml          # SAST scanning
│   │   └── dependency-review.yml
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   ├── CODEOWNERS
│   └── dependabot.yml
├── crates/
│   ├── kitty-runtime/          # Ollama/llama.cpp API, sampling, template check
│   ├── kitty-harness/          # Agent loop, tools, hashline, parser, context budget
│   ├── kitty-memory/           # Episodic blake3 + sqlite-vec (write-local)
│   └── kitty-cli/              # clap + RPC bench + tui feature-gated
├── vendor/                     # apohara-agentguard (seccomp+Landlock)
├── benches/                    # Polyglot, TB adapter, CNAS runner
├── knowledge/                  # Cheat sheets, tool cards ≤500 tok/turn
├── tests/                      # Integration tests
├── AUDIT.md                    # Public benchmark results
├── CNAS.md                     # Consumer-Normalized Agentic Score
├── SECURITY.md
├── CONTRIBUTING.md
├── LICENSE                     # MIT
├── README.md
├── Cargo.toml                  # Workspace
├── rust-toolchain.toml         # Pinned Rust version
├── Cargo.lock                  # Committed lockfile
├── deny.toml                   # cargo-deny config
└── .gitignore
```

## Implementation Steps

### Step 0: Empirical Fail-Fast (Week 0 — before writing any crates)

**THIS IS THE PLAN. Everything else is hypothesis until Week 0 validates.**

**Kill criteria (hard stops):**
- If ornith cannot produce parseable tool-calls even with correct template → ABORT
- If Polyglot raw baseline <25% → PIVOT (publish negative result, re-evaluate)
- If 8K context OOMs consistently → FALLBACK to 4K baseline
- If tool-call validity <85% after 100 prompts → RECONSIDER model choice

**Acceptance:**
- Suite of 100 tool-call prompts (3-5 tools each) against ornith:9b → parse success rate ≥85% (CRITICAL: validate tool-calling BEFORE any other work)
- num_ctx sweep {4K, 8K, 16K} with `nvidia-smi` during actual inference → measure real VRAM, tok/s, generates or stays silent
- **Polyglot baseline:** Run 50+ exercises with ornith:9B Q4_K_M to establish raw baseline (no harness, no scaffold). This determines if 45% target is achievable.
- A/B temp 0.3 vs 0.6 vs 1.0 on 50 Polyglot exercises (50 per condition for detectable effect size)
- A/B fixed tools vs 2-stage router (50 exercises per condition)
- A/B knowledge injection vs no knowledge (50 exercises per condition)
- **Decision rule for knowledge injection:** Enable if Polyglot delta ≥10% absolute with n=50 per arm. Default: NO knowledge injection. The burden of proof is on injection adding value. Note: with n=50, we can detect ~12% effect at p<0.05; smaller effects are indistinguishable from noise.

**Existing infrastructure (use, don't rebuild):**
- `oc_bench.js` — working benchmark harness (34 tasks across 3 suites). Adapt for ornith:9B.
- `stress.js` — concurrent load testing against Ollama. Use for VRAM measurement.
- `speed_proxy.js` — latency measurement proxy.

**Files:** AUDIT.md (day 1 numbers), benches/ (adapt from oc_bench.js)
**Verify:** 100 tool-calls parse correctly at ≥85%. Max ctx without OOM identified. Polyglot baseline established.

### Step 1: Minimum Benchmarcable Loop (Weeks 1-2)

**NOTE:** Spans 2 weeks because Step 0 may reveal blockers extending validation.

**Acceptance:**
- Workspace of 4 crates + vendor apohara-agentguard
- **Vendor scope (verified by Architect ora-3):** `apohara-agentguard` v0.2.0 on crates.io contains: pathguard module (symlink-escaping path canonicalization), audit module (local telemetry-free log), sandbox (seccomp-bpf + Landlock), firewall (prompt-injection regex), gate (anti-bypass command gate). ALL of pathsafety + audit are INSIDE this one crate.
- **Write-local scope (corrected):** Episodic memory (blake3 + SQLite) ~200 LOC. This is the ONLY write-local piece — pathsafety and audit already exist inside agentguard.
- **Day 1 spike (2h):** Verify apohara-agentguard v0.2.0 compiles as vendored dependency. Check pathguard and audit module APIs. If API mismatch, write thin adapter (~50 LOC).
- Tools 1-8 (no web). Hashline read/edit. Write-guard. Read-before-edit.
- Parser native tool_calls (Qwen3 XML) + forgiving fallback.
- Immutable prefix + tail injection.
- CLI `kitty run` and `kitty bench polyglot --limit 15`
- **Gate:** Hashline edits apply correctly on 10/10 synthetic cases.

**Files:** crates/kitty-runtime/, crates/kitty-harness/, crates/kitty-memory/, crates/kitty-cli/, vendor/ (apohara-agentguard)
**Verify:** `cargo build --release` produces binary <25 MB. `kitty bench polyglot --limit 15` runs and produces scores.

### Step 2: Conditional Protections (Week 3)

**Acceptance:**
- Thinking budget (cap + retry with thinking off)
- Quality monitor (empty/loop/hallucinated tool)
- **Knowledge injection by keyword** — ONLY if Week 0 A/B demonstrates ≥10% Polyglot improvement
- Watchdog 80% + loop guard
- Permission gate for command chains and redirects
- ShellStart with wake_on
- Bootstrap: detect cargo/pytest/npm, expose single `Test` tool
- **Gate:** Thinking budget cap works. Watchdog fires at 80%. Permission gate blocks `rm -rf` in chain.

**Files:** crates/kitty-harness/src/ (modules: thinking_budget, quality_monitor, knowledge_inject, watchdog, permission_gate, shell_start, bootstrap)
**Verify:** Each component passes its unit tests. Integration test: agent completes simple task without crash.

### Step 3: First Public Number (Week 3)

**Acceptance:**
- Aider Polyglot complete, 2 runs, same protocol as little-coder
- Publish AUDIT.md with Consumer vs Vendor columns
- **Go/no-go gate:** If ≥45.56%, repo opens and is named "Kitty Code". If not, ablate hashline/thinking budget/knowledge BEFORE adding memory or TUI.

**Files:** AUDIT.md, benches/ (Polyglot adapter)
**Verify:** Polyglot score ≥45.56% with ornith:9B on 8GB. AUDIT.md published with real numbers.

### Step 4: Terminal-Bench + Memory v0 (Weeks 4-5)

**Acceptance:**
- Adapter Harbor / TB 2.0
- Episodic recall (blake3 + sqlite-vec)
- Native stdout truncation
- Tree-sitter signatures for files >N lines
- **Gate:** TB 2.0 ≥18% on 8GB. Episodic recall works (save/retrieve episodes).

**Files:** crates/kitty-memory/, benches/ (TB adapter)
**Verify:** TB 2.0 ≥18% on 8GB. Episodic recall works.

### Step 5: Minimal TUI + CNAS v1 (Week 6)

**Acceptance:**
- `kitty tui` minimal (status: ctx %, CH, step, model)
- CNAS.md + leaderboard Consumer vs Vendor
- llama.cpp feature behind flag
- Document what is NOT claimed
- **Gate:** `kitty tui` shows correct state. CNAS.md published.

**Files:** crates/kitty-cli/src/tui.rs, CNAS.md
**Verify:** `kitty tui` shows correct state. CNAS.md published with Consumer vs Vendor.

## RALPLAN-DR Summary

### Principles
1. **Appropriate harness, neither heavy nor light** — Strong guards (write-guard, thinking budget, watchdog) always-on. Knowledge injection CONDITIONAL on Week 0 A/B result.
2. **Measure before building** — Week 0 fail-fast before writing any crates.
3. **Numbers or it didn't happen** — Every claim carries VRAM, real ctx, and commit hash.
4. **Extract from Apohara, don't rewrite** — Vendor apohara-agentguard (contains pathguard + audit + sandbox), write-local episodic memory only (~200 LOC).
5. **Binary > framework** — Single binary, no runtime dependencies.

### Decision Drivers
1. ornith:9B is self-scaffolding — harness must not fight the model
2. 8GB VRAM is the hard constraint — every MB matters
3. Nobody publishes Consumer vs Vendor numbers — honesty is differentiation

### Viable Options

**Option A: 4 crates + vendor agentguard + write-local episodic (CHOSEN)**
- Pros: Reuse proven safety layer (apohara-agentguard v0.2.0, seccomp+Landlock, 1,136 tests). Write-local episodic only = ~200 LOC, minimal maintenance.
- Cons: Vendor coupling with apohara-agentguard API. But 200 LOC is trivial to maintain or replace.

**Option B: 11 crates from scratch (REJECTED)**
- Pros: Full control, no external dependencies.
- Cons: Over-engineering for MVP. Rewriting tested code = theater, not progress.

**Option C: Fork little-coder (REJECTED)**
- Pros: Scaffold-model fit already proven (45.56%).
- Cons: Node/TypeScript runtime, not a Rust binary. pi dependency = no control over agent loop.

**Invalidation rationale for Option B:** 11 crates = 11 API boundaries = more time to first reproducible number. Apohara has 1,136 tests. Rewriting is cleanup theater.

**Invalidation rationale for Option C:** User explicitly rejected Node/Python runtime. "Single binary, zero Node/Python" is a non-negotiable constraint.

### Pre-mortem (5 failure scenarios)
1. **ornith cannot produce parseable tool-calls** → Abortion criterion week 0. If template is broken, project pivots to "Ollama/GGUF fixer."
2. **Polyglot <45% with ornith** → Ablate hashline/thinking budget/knowledge BEFORE adding memory or TUI. Gate is model + scaffold, not additional features.
3. **VRAM insufficient for 16K+ ctx** → Start at 8K, increase gradually. Context watchdog prevents OOM.
4. **Apohara crates not published as standalone** → Verify on crates.io week 0. If missing, write-local (~550 LOC total, manageable).
5. **Knowledge injection competes with ornith's self-scaffolding** → A/B in week 0 settles it. If it hurts, remove from plan.

## Risks and Mitigations

| Risk | Probability | Mitigation |
|------|-------------|------------|
| ornith template mismatch with Ollama | Medium | Week 0: suite of 20 synthetic tool-calls BEFORE writing any crates |
| Polyglot <45% after week 3 | Medium | Ablate hashline/thinking/knowledge before adding features |
| VRAM OOM at 16K+ ctx | Medium | Measure in week 0, start at 8K, watchdog 80% |
| apohara-agentguard vendor coupling | Low | Crate is small, MIT licensed, well-documented |
| Hashline overhead too high | Low | Search-replace fallback, measure in week 0 A/B |
| Knowledge injection hurts ornith | Medium | Conditional on A/B result, not default |
| Write-local safety-critical code (pathsafety) | Medium | 150 LOC, focused scope, fuzz testing in CI |

## Rollback Strategy

- **Week 0 abort:** Publish AUDIT.md with negative result. Pivot to "Ollama/GGUF fixer" or abandon.
- **Week 3 polyglot <45%:** Ablate features one by one. Publish gap. Do not add memory or TUI.
- **Week 5 TB <18%:** Accept that TB requires more context/reasoning. Publish result. Focus on Polyglot as differentiator.

## Verification Steps

1. **Week 0:** 100 tool-call prompts → parse success rate ≥85%
2. **Week 0:** num_ctx sweep → max ctx without OOM (4K guaranteed, 8K aspirational)
3. **Week 0:** A/B temp 0.3 vs 0.6 vs 1.0 → best polyglot score
4. **Week 0:** A/B fixed vs router → best score
5. **Week 0:** A/B knowledge inject vs no knowledge → impact on polyglot (30 exercises/condition)
6. **Week 1:** `cargo build --release` → binary <25 MB
7. **Week 1:** `kitty bench polyglot --limit 15` → scores
8. **Week 3:** Polyglot complete → ≥45.56%
9. **Week 5:** TB 2.0 → ≥18%
10. **Week 6:** CNAS.md published with Consumer vs Vendor
11. **Week 6:** OpenSSF Scorecard ≥7/10
12. **Week 6:** OpenSSF Best Practices Badge: Silver

## Rust Coding Standards (from loaded skills)

### Error Handling (m06)
- Library code: `thiserror` for typed errors. Application: `anyhow` for ergonomic errors.
- Use `?` propagation, never `try!()`.
- `.expect("reason")` over `.unwrap()` when value is guaranteed.
- Assertions for invariants at function entry.

### Zero-Cost Abstractions (m04)
- Static dispatch (generics) for performance-critical paths.
- Dynamic dispatch (`dyn Trait`) only for plugin/extension points.
- Concrete types when possible, avoid over-genericizing.

### Mutability (m03)
- Immutability by default. `&self` over `&mut self` where possible.
- `Cell<T>` for Copy types, `RefCell<T>` for non-Copy in single-thread.
- `AtomicXxx` for simple counters/flags in multi-thread.
- Never hold locks across await points.

### Domain Modeling (m09)
- Value objects with validated constructors (newtype pattern).
- Entities with unique identity.
- Aggregate roots own children, enforce invariants.
- Repository traits for persistence abstraction.

### Performance (m10)
- Profile before optimizing. Don't guess.
- Pre-allocate with `Vec::with_capacity()`.
- Avoid `.clone()` on hot paths. Use references or `Cow<T>`.
- `SmallVec` for usually-small collections.

### Anti-Patterns to Avoid (m15)
- No `.clone()` without justification.
- No `.unwrap()` in library code.
- No `pub` fields with invariants.
- No index loops when iterator works.
- No `String` where `&str` suffices.
- No `unsafe` without SAFETY comment.
- No functions >50 lines.
- No god objects, no circular dependencies.

## Changelog

### Deep Interview (10 rounds)
- Round 1: Appropriate harness (conditional after review)
- Round 2: Hashline + fallback confirmed
- Round 3: 2-stage router confirmed
- Round 4: Measure ctx in week 0
- Round 5: CNAS framework confirmed
- Round 6: Episodic blake3 for memory
- Round 7: 10 tools (8 base + web search + web fetch)
- Round 8: Extract from Apohara (vendor agentguard + write-local 3)
- Round 9: Tail + tree-sitter + truncado nativo
- Round 10: Name "Kitty Code"

### Architect Review Round 1 (ora-1)
- Fixed vendor source: Apohara, not Catalyst
- Resolved harness contradiction: appropriate harness with conditional knowledge injection
- Added VRAM budget breakdown
- Added quantization specification
- Adjusted binary size to <25 MB
- Clarified boot time
- Added rollback strategy
- Added intermediate acceptance criteria

### Architect Review Round 2 (ora-2)
- CRITICAL: 3 of 4 Apohara crates don't exist on crates.io → vendor agentguard ONLY + write-local pathsafety/episodic/audit (~550 LOC)
- Added Day 1 spike: verify crate existence before coding
- A/B sample size increased from 15 to 30 exercises per condition
- A/B decision rule added: enable knowledge inject if Polyglot delta ≥5%

### Architect Review Round 3 (ora-3)
- CORRECTION: pathsafety + audit are INSIDE apohara-agentguard (not separate crates)
- Write-local corrected to ~200 LOC (episodic only)
- Referenced existing infrastructure (oc_bench.js, stress.js, speed_proxy.js)
- Binary size <25 MB verified as realistic

### Critic Review Round 1 (cou-1)
- REVISE verdict: quantization spec missing, harness contradiction, binary size unrealistic, no VRAM budget, no rollback strategy
- All resolved in revision

### Critic Review Round 2 (cou-2)
- CONDITIONAL PASS: A/B decision rule and sample size resolved
- Write-local risk acknowledged
- Acceptance criteria operationalized

### Critic Review Round 3 (cou-3)
- REJECT verdict: VRAM 7.3GB marginal (0.7GB headroom), A/B underpowered (30→50), no tool-call validation, 6-week timeline unrealistic
- Plan revised: VRAM risk acknowledged, 4K baseline added, tool-call validation (100 prompts), timeline adjusted to 7 weeks, knowledge threshold raised to ≥10%

## ADR: Harness Strategy

- **Decision:** Use an "appropriate harness" — strong guards (write-guard, thinking budget, watchdog) always-on, knowledge injection conditional on Week 0 A/B result.
- **Drivers:** (1) ornith is self-scaffolding — heavy injection may compete with learned policy; (2) little-coder proved guards-only works at 45.56%; (3) Week 0 A/B will settle the question empirically.
- **Alternatives considered:** Heavy harness (always inject) — rejected because no evidence it helps ornith specifically. Thin harness (guards only) — kept as fallback if A/B shows knowledge hurts.
- **Why chosen:** Preserves the "trust boundary" principle while allowing empirical validation of cognitive injection. The model decides its own plans; the harness gates dangerous actions and manages context.
- **Consequences:** Step 2 is conditional — knowledge inject module exists but may be disabled. This adds slight complexity but prevents building on wrong assumptions.
- **Follow-ups:** Week 0 A/B result determines Step 2 scope. If knowledge inject helps, enable it. If not, ship without it.

## Debt Log

| Item | Priority | Deferred to | Reason |
|------|----------|-------------|--------|
| MemScenes (EverMemOS lifecycle) | Low | Post-MVP | Only if v0 episodic memory demonstrates real forgetting |
| Cloud fallback (opt-in) | Low | Post-MVP | Not core to local-first thesis |
| Plugin system | Low | Post-MVP | Not needed for single-binary MVP |
| 2-stage router A/B result | — | Week 0 | Will determine if router is kept or replaced with fixed tools |
| Tree-sitter grammar count | Low | Week 2 | Start with 5 languages (Rust, Python, JS, TS, Go), expand as needed |
