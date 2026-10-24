//! Safety guards for the agent loop.
//!
//! Provides permission gates for shell commands and chain analysis.
//! Split (audit 2026-09 Fase 1.2): pattern tables live in `patterns`,
//! parsing/write-target analysis in `shell_parse`, and the interpreter
//! table in `interpreters`.

mod interpreters;
mod patterns;
mod shell_parse;

use interpreters::INTERPRETERS;
pub(crate) use patterns::*;
pub(crate) use shell_parse::*;
pub use shell_parse::{detect_write_targets, is_write_via_redirect};

/// Nombre de env que marca un run con leak (tripwire BenchPolicy).
/// Lo escribe `bench_policy_check` al denegar (deny ⇒ flag leak) y lo lee
/// `bench/ppc.rs` (rechazo pre-LLM + `RunProvenance.leaked` real). Nunca se
/// limpia en el proceso: un deny contamina el run entero (exit 3).
pub const ENV_RUN_LEAKED: &str = "KITTY_RUN_LEAKED";

/// Marca el run actual como contaminado por leak (best-effort, nunca panica).
pub fn flag_run_leaked() {
    // `set_var` es `unsafe` en ediciones recientes: el deny nunca debe
    // abortar el run por el flag; si falla, el deny ya bloqueó la IO.
    unsafe {
        std::env::set_var(ENV_RUN_LEAKED, "1");
    }
}

/// ¿El run actual está marcado con leak?
pub fn is_run_leaked() -> bool {
    matches!(
        std::env::var(ENV_RUN_LEAKED).as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE")
    )
}

/// BenchPolicy (Capa 0, Lane B): deny pre-dispatch sobre `bench/**` y
/// `benches/**` canonicalizados. Expuesto como `pub` para lanes C/D
/// (`pub fn bench_policy_check`): devuelve `Some(reason)` si la llamada
/// debe denegarse ANTES de cualquier IO, `None` si pasa.
///
/// Cubre Read(path) / Grep(path) / Glob(pattern) / Bash(command). Para Bash
/// inspecciona tokens + redirects + `of=` (incl. `cat bench/...`: estar en
/// `SAFE_OUTPUT_COMMANDS` NO salta esta política — corre antes que Guard).
///
/// Scope (B1): solo raíces bench (`bench/…`, `benches/…` top-level o ruta
/// absoluta a ellas). NO bloquea fuentes `src/**/bench/**`
/// (`crates/…/src/bench/mod.rs`) ni el subcomando `cargo bench` fuera de
/// bench (solo la palabra `bench` pelada tras `cargo`; `cargo bench` con
/// args `bench/…` sí deniega). Denegar ⇒ [`flag_run_leaked`] (tripwire).
pub fn bench_policy_check(tool: &str, args: &serde_json::Value) -> Option<String> {
    let denied: Option<String> = match tool {
        "Read" => args
            .get("path")
            .and_then(|v| v.as_str())
            .filter(|p| is_bench_path(p))
            .map(|p| format!("BenchPolicy: Read denied sobre bench/** ({p})")),
        "Grep" => {
            // B1.1: normaliza `./`/`..` ANTES de decidir; deniega si el root
            // efectivo es el cwd, un ancestro del cwd, o contiene la raíz
            // bench (`src/..` ⇒ cwd, `..` ⇒ ancestro, abs del repo ⇒ cwd).
            // Sin `path` el walker cae a "." y barre el repo (incluye bench).
            if let Some(p) = args.get("path").and_then(|v| v.as_str()) {
                if is_bench_path(p) {
                    return Some(format!("BenchPolicy: Grep denied sobre bench/** ({p})"));
                }
            }
            let raw_opt = args.get("path").and_then(|v| v.as_str());
            if grep_root_sweeps_bench(raw_opt) {
                Some(
                    "BenchPolicy: Grep denied: sin path efectivo barre repo (incluye bench/**)"
                        .to_string(),
                )
            } else {
                None
            }
        }
        "Glob" => args
            .get("pattern")
            .and_then(|v| v.as_str())
            .filter(|p| is_bench_path(p))
            .map(|p| format!("BenchPolicy: Glob denied sobre bench/** ({p})")),
        "Bash" => args
            .get("command")
            .and_then(|v| v.as_str())
            .filter(|c| bash_refs_bench(c))
            .map(|_| "BenchPolicy: Bash denied: referencia a bench/** o benches/**".to_string()),
        _ => None,
    };
    if denied.is_some() {
        flag_run_leaked();
    }
    denied
}

/// ¿La ruta/patrón cae bajo `bench/**` o `benches/**`?
/// Canonicalización léxica (sin IO): separadores `\\`→`/`, split por `/`,
/// cada segmento se limpia de magia glob (`*?[]{}!`) y se compara exacto.
/// `benchmark/...` o `mybench` NO matchean; `bench/../bench/x` SÍ.
///
/// Scope (B1): fuentes `src/**/bench/**` (p. ej.
/// `crates/kitty-cli/src/bench/mod.rs`, `src/bench/foo`) NO matchean: son
/// código, no raíces bench. La detección es: top-level `bench/…` primero,
/// si no, excepción `src/bench`, si no, cualquier segmento `bench`
/// (cubre absolutas `/abs/repo/benches/…`).
pub fn is_bench_path(raw: &str) -> bool {
    let t = raw.trim().trim_matches(|c| c == '"' || c == '\'').trim();
    if t.is_empty() {
        return false;
    }
    let norm = t.replace('\\', "/");
    let is_abs = norm.starts_with('/');
    // Normalización léxica (sin IO): resuelve `.` y `..` dentro del propio
    // path ANTES de decidir (B1.3). `..` más allá de la raíz relativa se
    // preserva como `..`; en absoluta se ignora (queda en raíz).
    let mut stack: Vec<String> = Vec::new();
    for seg in norm.split('/') {
        let s = seg.trim();
        if s.is_empty() || s == "." {
            continue;
        }
        if s == ".." {
            if let Some(top) = stack.last() {
                if top != ".." {
                    stack.pop();
                    continue;
                }
            }
            if !is_abs {
                stack.push("..".to_string());
            }
            continue;
        }
        let cleaned: String = s
            .chars()
            .filter(|c| !matches!(c, '*' | '?' | '[' | ']' | '{' | '}' | '!'))
            .collect();
        let cleaned = cleaned.trim().to_string();
        if cleaned.is_empty() || cleaned == "." {
            continue;
        }
        if cleaned == ".." {
            if let Some(top) = stack.last() {
                if top != ".." {
                    stack.pop();
                    continue;
                }
            }
            if !is_abs {
                stack.push("..".to_string());
            }
            continue;
        }
        stack.push(cleaned);
    }
    if stack.is_empty() {
        return false;
    }
    let is_deny_seg = |s: &str| BENCH_DENY_COMPONENTS.contains(&s);
    // First-bench-wins (B1.3): el marcador `bench` más externo decide.
    // - `bench/…` top-level o `/abs/repo/bench/…` ⇒ deny aunque más adentro
    //   aparezca `src/bench` (`/abs/repo/bench/src/bench/x` ⇒ true).
    // - `src/bench` (código) ⇒ allow: todo el subárbol es fuente, aunque un
    //   segmento anidado se llame `bench` (`src/bench/bench/x` ⇒ false).
    // Tras normalizar `..`, `src/bench/../../bench/x` ya es `bench/x` ⇒ true.
    for (i, s) in stack.iter().enumerate() {
        if is_deny_seg(s) {
            if i > 0 && stack[i - 1] == "src" {
                return false;
            }
            return true;
        }
    }
    false
}

/// Normaliza léxicamente un path absoluto (sin IO): resuelve `.` y `..`
/// por stack. Usado por el check Grep B1.1 (root efectivo vs cwd).
fn lexical_normalize_abs(p: &std::path::Path) -> std::path::PathBuf {
    use std::path::Component;
    let mut stack: Vec<std::ffi::OsString> = Vec::new();
    let mut is_abs = false;
    for comp in p.components() {
        match comp {
            Component::Prefix(pref) => {
                stack.push(pref.as_os_str().to_owned());
            }
            Component::RootDir => {
                stack.clear();
                is_abs = true;
            }
            Component::CurDir => {}
            Component::ParentDir => {
                if let Some(top) = stack.last() {
                    // No hacer pop del prefijo Windows ni de un `..` previo.
                    let top_is_parent =
                        top.to_string_lossy().as_ref() == "..";
                    if !top_is_parent {
                        stack.pop();
                        continue;
                    }
                }
                if !is_abs {
                    stack.push("..".into());
                }
            }
            Component::Normal(s) => stack.push(s.to_owned()),
        }
    }
    let mut out = std::path::PathBuf::new();
    if is_abs {
        out.push("/");
    }
    for s in stack {
        // Evita duplicar el prefijo raíz ya empujado como `/`.
        if is_abs && out.as_os_str() == "/" && s.to_string_lossy().as_ref() == "/"
        {
            continue;
        }
        out.push(s);
    }
    if out.as_os_str().is_empty() {
        out.push(".");
    }
    out
}

/// ¿El root efectivo de Grep barre bench/**? (B1.1)
/// Normaliza `./`/`..` ANTES de decidir; deniega si el root efectivo es el
/// cwd, un ancestro del cwd, o contiene la raíz bench (`cwd/bench` o
/// `cwd/benches`). `None`/vacío ⇒ cwd (walker a `.`) ⇒ true.
fn grep_root_sweeps_bench(raw_opt: Option<&str>) -> bool {
    let cwd = std::env::current_dir()
        .map(|p| lexical_normalize_abs(&p))
        .unwrap_or_else(|_| std::path::PathBuf::from("."));
    let eff = match raw_opt {
        None => cwd.clone(),
        Some(raw) => {
            let t = raw.trim().trim_matches(|c| c == '"' || c == '\'').trim();
            if t.is_empty() {
                cwd.clone()
            } else {
                let norm = t.replace('\\', "/");
                if norm.starts_with('/') {
                    lexical_normalize_abs(std::path::Path::new(&norm))
                } else {
                    lexical_normalize_abs(&cwd.join(norm))
                }
            }
        }
    };
    if eff == cwd {
        return true;
    }
    // Ancestro del cwd: el cwd cuelga del efectivo (p.ej. `..`, `/`, abs padre).
    if cwd.starts_with(&eff) {
        return true;
    }
    // Contiene la raíz bench: bench/ cuelga del efectivo (p.ej. `src/..` ⇒
    // cwd, `.` ⇒ cwd; `src` no contiene bench ⇒ false).
    for bench in ["bench", "benches"] {
        let root = cwd.join(bench);
        if eff == root || root.starts_with(&eff) {
            return true;
        }
    }
    false
}

/// ¿El comando Bash referencia `bench/**` o `benches/**`?
/// Corre ANTES que `Guard::check_command`: `cat bench/...` debe denegarse
/// aunque `cat` esté en `SAFE_OUTPUT_COMMANDS`. Inspecciona words (con
/// quotes peladas) + redirect targets + `of=`; ignora heredoc bodies.
///
/// B1: además inspecciona payloads de intérpretes (`bash -c 'cat bench/x'`,
/// `python3 -c 'open("bench/x")'`) por recursión acotada sobre el contenido
/// quoteado, y `grep -r … .` recursivo desde `.` (barre bench/** ⇒ deny).
/// `cargo bench` pelado (subcomando, sin `/`) NO deniega fuera de bench.
pub(crate) fn bash_refs_bench(cmd: &str) -> bool {
    bash_refs_bench_depth(cmd, 0)
}

fn bash_refs_bench_depth(cmd: &str, depth: usize) -> bool {
    if depth > 2 {
        return false;
    }
    let stripped = strip_heredoc_bodies(cmd);
    // 1. Redirect targets (`> bench/x`, `tee bench/x`, `dd of=bench/x`).
    for t in detect_write_targets(&stripped) {
        if is_bench_path(&t) {
            return true;
        }
    }
    // 2. Words del chain (incl. `cat bench/...`, `grep -r x bench/`).
    for segment in split_command_chain(&stripped) {
        let words = split_words(&segment);
        if words.is_empty() {
            continue;
        }
        let first = unquote(&words[0]).to_lowercase();
        let is_cargo = first == "cargo";
        // `grep -r … .` recursivo barre bench/** ⇒ deny (si pisa bench).
        if is_grep_recursive_sweep(&words) {
            return true;
        }
        for (idx, w) in words.iter().enumerate() {
            let u = unquote(w);
            if u.is_empty() || u.starts_with('-') {
                continue;
            }
            // `cargo bench` pelado: el subcomando no es path bench/**.
            if is_cargo && idx == 1 && (u == "bench" || u == "benches") {
                continue;
            }
            // `of=bench/...` (dd por palabra, sin redirect).
            if let Some(path) = u.strip_prefix("of=") {
                if is_bench_path(path) {
                    return true;
                }
                continue;
            }
            // Flags con `=` (`--out=bench/x`).
            if let Some((_, val)) = u.split_once('=') {
                if is_bench_path(val) {
                    return true;
                }
                // No `continue`: el valor ya se juzgó, pero la palabra
                // completa podría ser payload con espacios (ver abajo).
            }
            if is_bench_path(&u) {
                return true;
            }
            // Payload quoteado con espacios (`'cat bench/x'` tras unquote):
            // parte en sub-palabras y juzga cada una (cubre `bash -c` sin
            // necesidad de parsear el flag; la recursión de abajo cubre el
            // caso general con quotes anidadas).
            if u.contains(' ') || u.contains('\t') {
                for sub in u.split_whitespace() {
                    let s = sub.trim().trim_matches(|c| c == '"' || c == '\'');
                    if s.is_empty() || s.starts_with('-') {
                        continue;
                    }
                    if let Some((_, val)) = s.split_once('=') {
                        if is_bench_path(val) {
                            return true;
                        }
                    }
                    if is_bench_path(s) {
                        return true;
                    }
                }
            }
        }
        // 3. Payloads quoteados (intérpretes `bash -c '<cmd>'`,
        // `python3 -c "open(\"bench/x\")"` y anidados): extrae contenidos
        // `'...'` / `"..."` y re-inspecciónalos (acotada, depth≤2). Se hace
        // para TODO segmento (no solo intérpretes): `open("bench/x")`
        // esconde `bench/x` entre paréntesis+quotes sin espacios.
        for payload in extract_quoted_payloads(&segment) {
            // Payload directo con path (`bench/x` pelado entre quotes).
            let p = payload.trim();
            if !p.is_empty() {
                // Chequeo léxico directo: parte en tokens no-alfanuméricos
                // (`open("bench/x")` ⇒ `bench`, `x`) y juzga cada token-path.
                // Rápido antes de la recursión completa.
                for tok in p
                    .split(|c: char| !(c.is_alphanumeric() || c == '/' || c == '.' || c == '_' || c == '-' || c == '*'))
                    .filter(|s| !s.is_empty())
                {
                    if is_bench_path(tok) {
                        return true;
                    }
                }
            }
            if bash_refs_bench_depth(&payload, depth + 1) {
                return true;
            }
        }
    }
    false
}

/// ¿Es un `grep` recursivo cuyo root barre bench/**? (B1.2)
/// `grep -r <pat> .` / `grep -R … ./` / `rg … .` sin path bench-explícito:
/// el walker barre el repo ⇒ pisa bench/** ⇒ deny. `grep -r … src/` no
/// barre bench ⇒ allow. Sin path explícito el root es implícito `.`
/// (`grep -r SECRET`, `rg SECRET` ⇒ deny); `rg`/`ag`/`ack` son recursivos
/// por defecto. Solo el primer posicional es patrón; los roots son el resto:
/// un `.` como patrón no cuenta como root (solo roots con `.` deniegan).
fn is_grep_recursive_sweep(words: &[String]) -> bool {
    if words.is_empty() {
        return false;
    }
    let first = unquote(&words[0]).to_lowercase();
    let is_grep_like = matches!(
        first.as_str(),
        "grep" | "egrep" | "fgrep" | "rg" | "ripgrep" | "ag" | "ack"
    );
    if !is_grep_like {
        return false;
    }
    // `rg`/`ag`/`ack` barren recursivo sin flag; `grep` exige `-r`/`-R`.
    let is_rg_family = matches!(first.as_str(), "rg" | "ripgrep" | "ag" | "ack");
    let mut recursive = is_rg_family;
    let mut positionals: Vec<String> = Vec::new();
    // Flags con valor separado cuyo siguiente word es el patrón/valor (no un
    // root): se consume sin contarse como root independiente.
    const VALUE_FLAGS: &[&str] = &[
        "-e", "-f", "-m", "-A", "-B", "-C", "--regexp", "--include", "--exclude",
        "--include-dir", "--exclude-dir", "--max-count",
    ];
    let mut skip_next_is_value = false;
    for w in words.iter().skip(1) {
        let u = unquote(w);
        if u.is_empty() {
            continue;
        }
        if skip_next_is_value {
            positionals.push(u);
            skip_next_is_value = false;
            continue;
        }
        if u.starts_with('-') {
            if u == "--" {
                continue;
            }
            if u.starts_with("--") {
                if u == "--recursive" {
                    recursive = true;
                }
                if VALUE_FLAGS.contains(&u.as_str()) {
                    skip_next_is_value = true;
                }
                continue;
            }
            // Flags cortos combinados (`-ri`, `-R`): `r`/`R` ⇒ recursivo.
            if u.contains('r') || u.contains('R') {
                recursive = true;
            }
            if VALUE_FLAGS.contains(&u.as_str()) {
                skip_next_is_value = true;
            }
            continue;
        }
        positionals.push(u);
    }
    if !recursive {
        return false;
    }
    // Sin root explícito (solo patrón o nada) el root es implícito `.` ⇒ deny.
    // `grep -r SECRET` (1 posicional) ⇒ deny; `grep -r` (0) ⇒ deny.
    if positionals.len() <= 1 {
        return true;
    }
    // Con roots explícitos: `.` domina (`grep -r pat . src/` barre bench).
    // Solo roots (tras el patrón) cuentan; `grep -r . src/` (patrón `.`)
    // con root `src/` ⇒ allow.
    let is_dot = |s: &str| s == "." || s == "./" || s == "./." || s == "/";
    for root in positionals.iter().skip(1) {
        if is_dot(root) {
            return true;
        }
    }
    // `grep -r … bench/` ya lo caza el loop general vía `is_bench_path`;
    // `grep -r … src/` (sin `.`) ⇒ allow.
    false
}

/// Extrae contenidos dentro de `'...'` / `"..."` de un segmento (sin
/// procesar escapes más allá de `\"` dentro de dobles). Acotado a 8
/// payloads por segmento para no amplificar comandos gigantes.
fn extract_quoted_payloads(segment: &str) -> Vec<String> {
    let bytes = segment.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() && out.len() < 8 {
        let b = bytes[i];
        if b == b'\'' || b == b'"' {
            let quote = b;
            i += 1;
            let start = i;
            while i < bytes.len() {
                if quote == b'"' && bytes[i] == b'\\' && i + 1 < bytes.len() {
                    i += 2;
                    continue;
                }
                if bytes[i] == quote {
                    break;
                }
                i += 1;
            }
            out.push(segment[start..i.min(segment.len())].to_string());
            if i < bytes.len() {
                i += 1;
            }
        } else {
            i += 1;
        }
    }
    out
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GuardVerdict {
    /// Command is safe to run.
    Allow,
    /// Command is dangerous, refuse to execute.
    Block(String),
    /// Command is risky, requires user confirmation.
    Warn(String),
}

/// Detect command substitution in the raw command string.
///
/// `$(` / backticks / process substitution `<(`/`>(` all make the shell
/// **execute** whatever they wrap, even when the outer command is a "safe
/// output" command like `echo` or `cat` (`echo $(rm -rf /)` really runs
/// `rm -rf /`). Variable expansion (`$VAR`, `$5`, `${x}`) is NOT execution and
/// must not trigger this path — only `$(` (which includes `${x:=$(cmd)}` and
/// `$((...))` arithmetic forms that can contain commands) and backticks do.
/// Safety guard that analyzes shell commands.
pub struct Guard;

impl Guard {
    /// Analyze a single shell command for safety.
    ///
    /// Layers in order: 0 raw pre-scan, 1 always-blocked, 1b shell-redirect
    /// write detection, 2 safe output, 3 interpreters, 4 general scan.
    #[allow(clippy::doc_markdown)]
    pub fn check_command(cmd: &str) -> GuardVerdict {
        let name = command_name(cmd).to_lowercase();

        // Layer 0: command substitution anywhere in the raw command means
        // arguments get executed, no matter who the command is.
        if has_command_substitution(cmd) {
            return Self::scan(cmd);
        }

        // Layer 1: hard-blocked commands (eval).
        if ALWAYS_BLOCKED_COMMANDS.contains(&name.as_str()) {
            return GuardVerdict::Block(format!(
                "Blocked: '{name}' is always blocked (arbitrary code execution)"
            ));
        }

        // Layer 1b: shell-redirect write detection (port of _shared/shell-write.ts)
        // Must run BEFORE safe-output allow, because `cat`/`echo` with `>` is a write.
        let targets = detect_write_targets(cmd);
        if !targets.is_empty() {
            return GuardVerdict::Block(format!(
                "use Write for new file / Edit for existing — do not redirect (writes to {})",
                targets.join(", ")
            ));
        }

        // Layer 2: safe output commands.
        if SAFE_OUTPUT_COMMANDS.contains(&name.as_str()) {
            return GuardVerdict::Allow;
        }

        // Layer 3: interpreters.
        if INTERPRETERS.contains(&name.as_str()) {
            return Self::scan(cmd);
        }

        // Layer 4: general command — strip quotes (literal data args) then scan.
        Self::scan(&strip_quotes(cmd))
    }

    /// Scan a command string for blocked/risky patterns (case-insensitive).
    fn scan(cmd: &str) -> GuardVerdict {
        let lowered = cmd.to_lowercase();

        for pattern in BLOCKED_PATTERNS {
            // Refine `dd if=` to not block benign cases:
            // - dd if=/dev/null (reading from null is safe)
            // - dd ... of=/dev/null etc. (writing to null device is safe, not a file)
            if *pattern == "dd if="
                && (lowered.contains("dd if=/dev/null")
                    || lowered.contains("of=/dev/null")
                    || lowered.contains("of=/dev/stdout")
                    || lowered.contains("of=/dev/stderr")
                    || lowered.contains("of=/dev/stdin")
                    || lowered.contains("of=/dev/tty")
                    || lowered.contains("of=/dev/zero"))
            {
                continue;
            }
            if lowered.contains(&pattern.to_lowercase()) {
                return GuardVerdict::Block(format!(
                    "Blocked: command contains dangerous pattern '{pattern}'"
                ));
            }
        }

        for pattern in RISKY_PATTERNS {
            if lowered.contains(&pattern.to_lowercase()) {
                return GuardVerdict::Warn(format!(
                    "Warning: command contains risky pattern '{pattern}'"
                ));
            }
        }

        GuardVerdict::Allow
    }

    /// Analyze a shell chain (commands separated by &&, ||, |, ;).
    /// Each command is judged independently.
    pub fn check_chain(chain: &str) -> Vec<GuardVerdict> {
        let mut results = Vec::new();
        for cmd in split_command_chain(chain) {
            let cmd = cmd.trim();
            if !cmd.is_empty() {
                results.push(Self::check_command(cmd));
            }
        }
        results
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_command_allows() {
        assert_eq!(Guard::check_command("ls -la"), GuardVerdict::Allow);
        assert_eq!(Guard::check_command("cat file.rs"), GuardVerdict::Allow);
        assert_eq!(Guard::check_command("cargo build"), GuardVerdict::Allow);
    }

    #[test]
    fn blocked_command_detected() {
        match Guard::check_command("rm -rf /") {
            GuardVerdict::Block(_) => {}
            other => panic!("Expected Block, got {other:?}"),
        }
        match Guard::check_command("dd if=/dev/zero of=/dev/sda") {
            GuardVerdict::Block(_) => {}
            other => panic!("Expected Block, got {other:?}"),
        }
    }

    #[test]
    fn risky_command_warns() {
        match Guard::check_command("rm -rf target/") {
            GuardVerdict::Warn(_) => {}
            other => panic!("Expected Warn, got {other:?}"),
        }
        match Guard::check_command("git push --force") {
            GuardVerdict::Warn(_) => {}
            other => panic!("Expected Warn, got {other:?}"),
        }
    }

    #[test]
    fn chain_split_works() {
        let results = Guard::check_chain("ls && rm -rf / && pwd");
        assert_eq!(results.len(), 3);
        assert_eq!(results[0], GuardVerdict::Allow);
        assert!(matches!(results[1], GuardVerdict::Block(_)));
        assert_eq!(results[2], GuardVerdict::Allow);
    }

    #[test]
    fn pipe_chain_split() {
        let results = Guard::check_chain("cat file.txt | grep pattern | wc -l");
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|r| *r == GuardVerdict::Allow));
    }

    #[test]
    fn echo_quoted_dangerous_not_blocked() {
        // Arguments to echo are just strings, not executed
        assert_eq!(Guard::check_command("echo 'rm -rf /'"), GuardVerdict::Allow);
        assert_eq!(
            Guard::check_command("echo \"rm -rf /\""),
            GuardVerdict::Allow
        );
    }

    #[test]
    fn cat_quoted_dangerous_not_blocked() {
        // cat's arguments are file paths / strings, not executed
        assert_eq!(Guard::check_command("cat 'rm -rf /'"), GuardVerdict::Allow);
        assert_eq!(
            Guard::check_command("cat \"rm -rf /\""),
            GuardVerdict::Allow
        );
    }

    #[test]
    fn grep_quoted_dangerous_not_blocked() {
        // grep's arguments are patterns, not executed
        assert_eq!(
            Guard::check_command("grep 'rm -rf' file.txt"),
            GuardVerdict::Allow
        );
        assert_eq!(
            Guard::check_command("grep \"rm -rf\" file.txt"),
            GuardVerdict::Allow
        );
    }

    #[test]
    fn echo_without_quotes_dangerous_not_blocked() {
        // echo's arguments are strings even without quotes
        assert_eq!(Guard::check_command("echo rm -rf /"), GuardVerdict::Allow);
    }

    #[test]
    fn sed_quoted_dangerous_not_blocked() {
        assert_eq!(
            Guard::check_command("sed 's/rm -rf/foo/' file"),
            GuardVerdict::Allow
        );
    }

    #[test]
    fn sh_c_with_dangerous_payload_is_blocked() {
        // Regression: the quote-stripping bypass used to allow this.
        match Guard::check_command("sh -c 'rm -rf /'") {
            GuardVerdict::Block(_) => {}
            other => panic!("Expected Block, got {other:?}"),
        }
        match Guard::check_command("bash -c \"rm -rf /\"") {
            GuardVerdict::Block(_) => {}
            other => panic!("Expected Block, got {other:?}"),
        }
    }

    #[test]
    fn python_c_with_system_is_blocked() {
        match Guard::check_command("python3 -c 'import os; os.system(\"rm -rf /\")'") {
            GuardVerdict::Block(_) => {}
            other => panic!("Expected Block, got {other:?}"),
        }
    }

    #[test]
    fn awk_system_is_blocked() {
        // awk was previously in SAFE_OUTPUT_COMMANDS and silently allowed this.
        match Guard::check_command("awk 'BEGIN{system(\"rm -rf /\")}'") {
            GuardVerdict::Block(_) => {}
            other => panic!("Expected Block, got {other:?}"),
        }
    }

    #[test]
    fn benign_interpreter_commands_still_allowed() {
        assert_eq!(
            Guard::check_command("python3 build.py"),
            GuardVerdict::Allow
        );
        assert_eq!(
            Guard::check_command("sh -c 'cargo build'"),
            GuardVerdict::Allow
        );
        assert_eq!(Guard::check_command("node index.js"), GuardVerdict::Allow);
    }

    #[test]
    fn dangerous_patterns_still_blocked() {
        // Actual dangerous commands should still be blocked
        match Guard::check_command("rm -rf /") {
            GuardVerdict::Block(_) => {}
            other => panic!("Expected Block, got {other:?}"),
        }
        match Guard::check_command("dd if=/dev/zero of=/dev/sda") {
            GuardVerdict::Block(_) => {}
            other => panic!("Expected Block, got {other:?}"),
        }
    }

    #[test]
    fn dangerous_after_chain_operator_still_blocked() {
        // Dangerous command after && should be blocked
        let results = Guard::check_chain("echo foo && rm -rf /");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], GuardVerdict::Allow);
        assert!(matches!(results[1], GuardVerdict::Block(_)));
    }

    #[test]
    fn strip_quotes_removes_single_quotes() {
        assert_eq!(strip_quotes("echo 'rm -rf /'"), "echo ");
        assert_eq!(strip_quotes("echo 'hello' world"), "echo  world");
    }

    #[test]
    fn strip_quotes_removes_double_quotes() {
        assert_eq!(strip_quotes("echo \"rm -rf /\""), "echo ");
        assert_eq!(strip_quotes("echo \"hello\" world"), "echo  world");
    }

    #[test]
    fn strip_quotes_handles_escaped_double_quotes() {
        // Escaped quotes inside double quotes don't close the quote
        // We just need to strip the quoted content, not preserve it
        assert_eq!(strip_quotes("echo \"say \\\"hi\\\"\" world"), "echo  world");
    }

    #[test]
    fn strip_quotes_preserves_single_char() {
        assert_eq!(strip_quotes("ls -la"), "ls -la");
        assert_eq!(strip_quotes("rm -rf /"), "rm -rf /");
    }

    #[test]
    fn command_name_extracts_first_word() {
        assert_eq!(command_name("rm -rf /"), "rm");
        assert_eq!(command_name("echo hello"), "echo");
        assert_eq!(command_name(""), "");
    }

    // === F1.1 regression: command substitution bypasses ===

    #[test]
    fn command_substitution_in_echo_blocked() {
        // echo is a "safe output" command, but $(...) executes its payload.
        match Guard::check_command("echo $(rm -rf /)") {
            GuardVerdict::Block(_) => {}
            other => panic!("Expected Block, got {other:?}"),
        }
    }

    #[test]
    fn command_substitution_in_printf_blocked() {
        match Guard::check_command("printf \"$(rm -rf /)\"") {
            GuardVerdict::Block(_) => {}
            other => panic!("Expected Block, got {other:?}"),
        }
    }

    #[test]
    fn command_substitution_backticks_blocked() {
        // Backticks are command substitution too.
        match Guard::check_command("echo `rm -rf /`") {
            GuardVerdict::Block(_) => {}
            other => panic!("Expected Block, got {other:?}"),
        }
    }

    #[test]
    fn process_substitution_blocked() {
        match Guard::check_command("cat <(rm -rf /)") {
            GuardVerdict::Block(_) => {}
            other => panic!("Expected Block, got {other:?}"),
        }
    }

    #[test]
    fn eval_always_blocked() {
        match Guard::check_command("eval \"rm -rf /\"") {
            GuardVerdict::Block(_) => {}
            other => panic!("Expected Block, got {other:?}"),
        }
        // Even benign-looking eval is blocked: it can execute anything.
        match Guard::check_command("eval 'x=1'") {
            GuardVerdict::Block(_) => {}
            other => panic!("Expected Block, got {other:?}"),
        }
    }

    #[test]
    fn substitution_inside_sh_c_blocked() {
        // The shell executes $(...) inside its -c argument.
        match Guard::check_command("sh -c 'echo $(rm -rf /)'") {
            GuardVerdict::Block(_) => {}
            other => panic!("Expected Block, got {other:?}"),
        }
    }

    #[test]
    fn variable_expansion_not_blocked() {
        // $VAR / $5 / ${x} are expansion, not execution.
        assert_eq!(
            Guard::check_command("echo \"cost: $5\""),
            GuardVerdict::Allow
        );
        assert_eq!(Guard::check_command("echo $HOME/path"), GuardVerdict::Allow);
    }

    #[test]
    fn literal_dangerous_text_not_blocked() {
        // Plain literal text with no substitution is data, not execution.
        assert_eq!(
            Guard::check_command("echo 'rm -rf / is dangerous'"),
            GuardVerdict::Allow
        );
    }

    #[test]
    fn backup_tarball_allowed() {
        // F1.1.3 regression (sh-backup false positive): tarball creation must
        // not be blocked by the "rm -rf" / backup-content heuristics.
        assert_eq!(
            Guard::check_command("tar -czf backup.tar.gz dir/"),
            GuardVerdict::Allow
        );
    }

    #[test]
    fn find_exec_rm_documented_decision() {
        // Decide explicitly: `find ... -exec rm {} \;` on a *named pattern*
        // (e.g. '*.tmp') is a legitimate cleanup operation, not a recursive
        // wipe. Documented Allow (matching the plan's "Allow o Warn
        // documentado"). A recursive wipe of a root path would still hit the
        // Risky `rm -r` pattern if present; here the target is a temp glob.
        assert_eq!(
            Guard::check_command("find . -name '*.tmp' -exec rm {} \\;"),
            GuardVerdict::Allow
        );
    }

    #[test]
    fn source_and_env_raw_scan() {
        // source / . / env / xargs execute their arguments: raw scan applies.
        match Guard::check_command("source backup.sh") {
            GuardVerdict::Allow => {}
            other => panic!("Expected Allow for benign source, got {other:?}"),
        }
        match Guard::check_command(". myscript.sh") {
            GuardVerdict::Allow => {}
            other => panic!("Expected Allow for benign dot-source, got {other:?}"),
        }
        match Guard::check_command("env VAR=1 sh -c 'rm -rf /'") {
            GuardVerdict::Block(_) => {}
            other => panic!("Expected Block for env-wrapped danger, got {other:?}"),
        }
    }

    // === P0c: shell-redirect write detection (port of _shared/shell-write.ts) ===

    #[test]
    fn redirect_cat_to_file_blocked() {
        match Guard::check_command("cat > file.txt") {
            GuardVerdict::Block(msg) => {
                assert!(msg.contains("do not redirect") || msg.contains("Write"))
            }
            other => panic!("Expected Block for cat > file, got {other:?}"),
        }
        assert!(is_write_via_redirect("cat > file.txt"));
        assert_eq!(detect_write_targets("cat > file.txt"), vec!["file.txt"]);
    }

    #[test]
    fn redirect_echo_append_blocked() {
        match Guard::check_command("echo hi >> file.txt") {
            GuardVerdict::Block(_) => {}
            other => panic!("Expected Block for echo >>, got {other:?}"),
        }
        assert!(is_write_via_redirect("echo hi >> file.txt"));
    }

    #[test]
    fn redirect_no_space_blocked() {
        assert_eq!(detect_write_targets("cat>main.py"), vec!["main.py"]);
        match Guard::check_command("cat>main.py") {
            GuardVerdict::Block(_) => {}
            other => panic!("Expected Block for cat>main.py, got {other:?}"),
        }
    }

    #[test]
    fn tee_with_file_blocked() {
        match Guard::check_command("echo x | tee out.log") {
            GuardVerdict::Block(_) => {}
            other => panic!("Expected Block for tee out.log, got {other:?}"),
        }
        assert_eq!(
            detect_write_targets("echo x | tee -a out.log"),
            vec!["out.log"]
        );
    }

    #[test]
    fn tee_without_args_allowed() {
        assert_eq!(Guard::check_command("echo hi | tee"), GuardVerdict::Allow);
        assert!(!is_write_via_redirect("echo hi | tee"));
        assert_eq!(detect_write_targets("echo hi | tee"), Vec::<String>::new());
    }

    #[test]
    fn tee_to_dev_null_allowed() {
        assert_eq!(
            Guard::check_command("echo x | tee /dev/null"),
            GuardVerdict::Allow
        );
        assert!(!is_write_via_redirect("echo x | tee /dev/null"));
    }

    #[test]
    fn redirect_to_dev_null_allowed() {
        assert_eq!(Guard::check_command("cat > /dev/null"), GuardVerdict::Allow);
        assert_eq!(
            Guard::check_command("cmd > /dev/null 2>&1"),
            GuardVerdict::Allow
        );
        assert!(!is_write_via_redirect("cat > /dev/null"));
        assert!(!is_write_via_redirect("cmd > /dev/null 2>&1"));
    }

    #[test]
    fn redirect_stderr_to_dev_null_allowed() {
        assert_eq!(
            Guard::check_command("grep -ri \"x\" ~/.config/ 2>/dev/null"),
            GuardVerdict::Allow
        );
        assert_eq!(
            Guard::check_command("make 2>/dev/null"),
            GuardVerdict::Allow
        );
        assert!(!is_write_via_redirect("make 2>/dev/null"));
    }

    #[test]
    fn redirect_to_dev_stdout_allowed() {
        assert_eq!(
            Guard::check_command("cmd > /dev/stdout"),
            GuardVerdict::Allow
        );
        assert_eq!(
            Guard::check_command("cmd 2> /dev/stderr"),
            GuardVerdict::Allow
        );
        assert!(!is_write_via_redirect("cmd > /dev/stdout"));
    }

    #[test]
    fn fd_duplication_not_flagged() {
        assert_eq!(detect_write_targets("make 2>&1"), Vec::<String>::new());
        assert_eq!(detect_write_targets("echo err >&2"), Vec::<String>::new());
        assert_eq!(Guard::check_command("make 2>&1"), GuardVerdict::Allow);
    }

    #[test]
    fn tar_czf_still_allowed() {
        assert_eq!(
            Guard::check_command("tar -czf backup.tar.gz dir/"),
            GuardVerdict::Allow
        );
        assert!(!is_write_via_redirect("tar -czf backup.tar.gz dir/"));
    }

    #[test]
    fn heredoc_strip_preserves_redirect() {
        let cmd = "cat > backend/main.py << 'ENDOFFILE'\nfrom fastapi import FastAPI\napp = FastAPI()\nENDOFFILE";
        assert_eq!(detect_write_targets(cmd), vec!["backend/main.py"]);
        match Guard::check_command(cmd) {
            GuardVerdict::Block(_) => {}
            other => panic!("Expected Block for heredoc > file, got {other:?}"),
        }
        // Heredoc body with `>` should not be mistaken for redirect
        let cmd2 = "cat > cmp.py << 'EOF'\nif a > b:\n    pass  # don't compare like this\nEOF";
        assert_eq!(detect_write_targets(cmd2), vec!["cmp.py"]);
    }

    #[test]
    fn heredoc_body_not_split_as_commands() {
        assert_eq!(
            split_command_chain("cat > f.sh << 'EOF'\nrm -rf /\nEOF"),
            vec!["cat > f.sh"]
        );
    }

    #[test]
    fn dedup_write_targets() {
        let writes = detect_write_targets("echo a > one.txt && echo b >> one.txt");
        assert_eq!(writes, vec!["one.txt"]);
        let writes2 = detect_write_targets("echo a > one.txt && echo b >> two.txt");
        assert_eq!(writes2, vec!["one.txt", "two.txt"]);
    }

    #[test]
    fn dd_of_file_blocked() {
        assert_eq!(
            detect_write_targets("dd if=/dev/zero of=disk.img bs=1M"),
            vec!["disk.img"]
        );
        match Guard::check_command("dd if=/dev/zero of=disk.img bs=1M") {
            GuardVerdict::Block(_) => {}
            other => panic!("Expected Block for dd of=, got {other:?}"),
        }
    }

    #[test]
    fn dd_of_dev_null_allowed() {
        assert_eq!(
            detect_write_targets("dd if=big.img of=/dev/null"),
            Vec::<String>::new()
        );
        assert_eq!(
            Guard::check_command("dd if=big.img of=/dev/null"),
            GuardVerdict::Allow
        );
    }

    #[test]
    fn dd_if_dev_null_not_blocked() {
        // Refined dd if= : benign input should not be blocked
        assert_eq!(
            Guard::check_command("dd if=/dev/null of=/dev/null"),
            GuardVerdict::Allow
        );
        assert_eq!(
            Guard::check_command("dd if=/dev/null bs=1 count=1"),
            GuardVerdict::Allow
        );
    }

    #[test]
    fn redirect_inside_quotes_not_flagged() {
        assert_eq!(
            detect_write_targets("grep \"a > b\" file.txt"),
            Vec::<String>::new()
        );
        assert_eq!(detect_write_targets("echo 'x >> y'"), Vec::<String>::new());
        assert_eq!(
            Guard::check_command("grep \"a > b\" file.txt"),
            GuardVerdict::Allow
        );
    }

    #[test]
    fn input_redirect_not_flagged() {
        assert_eq!(
            detect_write_targets("wc -l < input.txt"),
            Vec::<String>::new()
        );
        assert_eq!(
            detect_write_targets("grep foo <<< \"$var\""),
            Vec::<String>::new()
        );
        assert_eq!(
            Guard::check_command("wc -l < input.txt"),
            GuardVerdict::Allow
        );
    }

    #[test]
    fn chain_each_segment_judged() {
        // ls && rm -rf / should be refused on rm, not admitted on ls
        let results = Guard::check_chain("ls && cat > evil.txt");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], GuardVerdict::Allow);
        assert!(matches!(results[1], GuardVerdict::Block(_)));
    }

    #[test]
    fn extract_redirect_targets_works() {
        assert_eq!(extract_redirect_targets("cat > file.txt"), vec!["file.txt"]);
        assert_eq!(
            extract_redirect_targets("echo hi >> file.txt"),
            vec!["file.txt"]
        );
        assert_eq!(
            extract_redirect_targets("cmd > /dev/null"),
            Vec::<String>::new()
        );
        assert_eq!(extract_redirect_targets("cmd 2>&1"), Vec::<String>::new());
    }

    #[test]
    fn is_non_destructive_target_works() {
        assert!(is_non_destructive_target("/dev/null"));
        assert!(is_non_destructive_target("/dev/stdout"));
        assert!(is_non_destructive_target("/dev/stderr"));
        assert!(is_non_destructive_target("/dev/fd/2"));
        assert!(!is_non_destructive_target("file.txt"));
        assert!(!is_non_destructive_target("/dev/null.bak"));
        assert!(is_non_destructive_target("&1"));
    }

    #[test]
    fn split_command_chain_quote_aware() {
        assert_eq!(
            split_command_chain("echo 'a && b' ; ls"),
            vec!["echo 'a && b'", "ls"]
        );
        assert_eq!(
            split_command_chain("ls && rm -rf / ; echo done | wc -l"),
            vec!["ls", "rm -rf /", "echo done", "wc -l"]
        );
    }

    #[test]
    fn strip_heredoc_bodies_works() {
        assert_eq!(
            strip_heredoc_bodies("cat > f << 'EOF'\nbody\nEOF"),
            "cat > f "
        );
        assert_eq!(
            strip_heredoc_bodies("cat > f << EOF\nbody without terminator"),
            "cat > f "
        );
        assert_eq!(
            strip_heredoc_bodies("grep foo <<< \"bar\""),
            "grep foo <<< \"bar\""
        );
    }

    #[test]
    fn strip_heredoc_bodies_multiline_kills_arithmetic_mutant() {
        // Two body lines + trailing chain pin body_end precisely (overshoot is masked by the clamp without trailing content).
        assert_eq!(
            strip_heredoc_bodies("cat > f << EOF\na\nb\nEOF\necho hi"),
            "cat > f echo hi"
        );
    }

    #[test]
    fn strip_heredoc_bodies_double_heredoc_rescans() {
        // Two heredocs in one command exercise the re-scan path
        // (`search_from = at` after the first strip).
        assert_eq!(
            strip_heredoc_bodies("cat > a << EOF\nx\nEOF\ncat > b << END\ny\nEND\necho done"),
            "cat > a cat > b echo done"
        );
    }

    #[test]
    fn split_command_chain_or_longest_match() {
        assert_eq!(split_command_chain("a || b"), vec!["a", "b"]);
        assert_eq!(split_command_chain("a || b || c"), vec!["a", "b", "c"]);
    }

    #[test]
    fn split_command_chain_mixed_operators_and_edges() {
        assert_eq!(
            split_command_chain("ls; rm -rf / || echo done"),
            vec!["ls", "rm -rf /", "echo done"]
        );
        assert_eq!(split_command_chain("&& ls &&"), vec!["ls"]);
    }

    #[test]
    fn split_command_chain_heredoc_body_never_command() {
        assert_eq!(
            split_command_chain("cat > f << EOF\nrm -rf /\nEOF\n&& echo hi"),
            vec!["cat > f", "echo hi"]
        );
    }

    // === P1 fix-mutants: Guard::scan dd-refinement isolates ===
    // Each benign `of=/dev/*` target in isolation must Allow. Before this,
    // only `dd if=/dev/null` / `of=/dev/null` were covered, so the tail
    // `||` mutants (of=/dev/stderr|stdin|tty|zero) survived masked by the
    // early-true disjuncts. One isolated Allow per disjunct kills them.

    #[test]
    fn dd_of_dev_stdout_allowed_isolated() {
        assert_eq!(
            Guard::check_command("dd if=/dev/zero of=/dev/stdout"),
            GuardVerdict::Allow
        );
    }

    #[test]
    fn dd_of_dev_stderr_allowed_isolated() {
        assert_eq!(
            Guard::check_command("dd if=/dev/zero of=/dev/stderr"),
            GuardVerdict::Allow
        );
    }

    #[test]
    fn dd_of_dev_stdin_allowed_isolated() {
        assert_eq!(
            Guard::check_command("dd if=/dev/zero of=/dev/stdin"),
            GuardVerdict::Allow
        );
    }

    #[test]
    fn dd_of_dev_tty_allowed_isolated() {
        assert_eq!(
            Guard::check_command("dd if=/dev/zero of=/dev/tty"),
            GuardVerdict::Allow
        );
    }

    #[test]
    fn dd_of_dev_zero_allowed_isolated() {
        assert_eq!(
            Guard::check_command("dd if=/dev/zero of=/dev/zero"),
            GuardVerdict::Allow
        );
    }

    #[test]
    fn dd_refinement_needs_dd_prefix() {
        // Outer `&&` killer (mod.rs:93): a non-dd blocked pattern that
        // happens to contain a benign `of=` must still Block. Under
        // `&&`→`||` this would `continue` past the `rm -rf /` pattern.
        match Guard::check_command("rm -rf / of=/dev/null") {
            GuardVerdict::Block(_) => {}
            other => panic!("Expected Block, got {other:?}"),
        }
    }

    #[test]
    fn process_substitution_output_blocked() {
        // `>(...)` killer for `has_command_substitution` tail `||`→`&&`:
        // only `>(` is present here (`$(`, backtick, `<(` absent).
        match Guard::check_command("cat >(rm -rf /)") {
            GuardVerdict::Block(_) => {}
            other => panic!("Expected Block, got {other:?}"),
        }
    }

    #[test]
    fn is_non_destructive_target_edge_cases() {
        // Kills `&&`→`||` / `||`→`&&` survivors in shell_parse.rs:
        // fd-number spellings, plumbing spellings, and empty/non-numeric
        // `/dev/fd/` which must stay non-destructive=false.
        assert!(is_non_destructive_target("1"));
        assert!(is_non_destructive_target("2"));
        assert!(is_non_destructive_target("&2"));
        assert!(is_non_destructive_target("2>"));
        assert!(is_non_destructive_target("&>"));
        assert!(is_non_destructive_target(">&"));
        assert!(is_non_destructive_target("2>&1"));
        assert!(is_non_destructive_target(">&2"));
        assert!(!is_non_destructive_target("/dev/fd/"));
        assert!(!is_non_destructive_target("/dev/fd/abc"));
        assert!(is_non_destructive_target("/dev/fd/2"));
    }

    // === Capa 0 F1: BenchPolicy pre-dispatch (Lane B) ===

    #[test]
    fn bench_policy_is_bench_path_components() {
        // Limpia tripwire entre tests (bench_policy_check flaggea leak).
        unsafe {
            std::env::remove_var(ENV_RUN_LEAKED);
        }
        assert!(is_bench_path("bench/demo-12/demo-12.json"));
        assert!(is_bench_path("benches/baseline.json"));
        assert!(is_bench_path("./bench/gpqa.json"));
        assert!(is_bench_path("bench/../bench/demo-12.json"));
        assert!(is_bench_path("/abs/repo/benches/ab/result.json"));
        assert!(is_bench_path("bench/**"));
        assert!(is_bench_path("**/bench/**"));
        assert!(is_bench_path("benches/*.json"));
        assert!(!is_bench_path("benchmark/foo.json"));
        assert!(!is_bench_path("mybench/x.json"));
        assert!(!is_bench_path("src/main.rs"));
        assert!(!is_bench_path(""));
        // B1 scope: fuentes `src/**/bench/**` son código, no raíces bench.
        assert!(!is_bench_path("src/bench/foo.rs"));
        assert!(!is_bench_path("crates/kitty-cli/src/bench/mod.rs"));
        assert!(!is_bench_path("crates/kitty-cli/src/bench/ppc.rs"));
        unsafe {
            std::env::remove_var(ENV_RUN_LEAKED);
        }
    }

    #[test]
    fn bench_policy_cat_does_not_bypass() {
        unsafe {
            std::env::remove_var(ENV_RUN_LEAKED);
        }
        // `cat` está en SAFE_OUTPUT_COMMANDS (Guard Allow), pero BenchPolicy
        // corre antes y debe denegar bench/** igual.
        assert_eq!(
            Guard::check_command("cat bench/demo-12/demo-12.json"),
            GuardVerdict::Allow,
            "Guard solo ve SAFE_OUTPUT"
        );
        assert!(
            bench_policy_check(
                "Bash",
                &serde_json::json!({"command": "cat bench/demo-12/demo-12.json"})
            )
            .is_some(),
            "BenchPolicy debe denegar cat bench/... pre-dispatch"
        );
        // deny ⇒ flag leak (tripwire).
        assert!(is_run_leaked(), "deny debe flaggear KITTY_RUN_LEAKED");
        unsafe {
            std::env::remove_var(ENV_RUN_LEAKED);
        }
    }

    #[test]
    fn bench_policy_adversary_20_prompts_zero_reads() {
        unsafe {
            std::env::remove_var(ENV_RUN_LEAKED);
        }
        // F1: 20 prompts adversarios (Read/Grep/cat bench) ⇒ 0 lecturas:
        // todos denegados por `bench_policy_check` antes de cualquier IO.
        let cases: Vec<(&str, serde_json::Value)> = vec![
            ("Read", serde_json::json!({"path": "bench/demo-12/demo-12.json"})),
            ("Read", serde_json::json!({"path": "bench/gpqa.json"})),
            ("Read", serde_json::json!({"path": "./bench/demo-12/demo-12.json"})),
            ("Read", serde_json::json!({"path": "benches/baseline.json"})),
            ("Read", serde_json::json!({"path": "benches/exercises_tb2.json"})),
            ("Grep", serde_json::json!({"pattern": "answer", "path": "bench/demo-12"})),
            ("Grep", serde_json::json!({"pattern": "sha256", "path": "benches"})),
            ("Grep", serde_json::json!({"pattern": "fairness", "path": "bench"})),
            ("Glob", serde_json::json!({"pattern": "bench/**/*.json"})),
            ("Glob", serde_json::json!({"pattern": "benches/*.json"})),
            ("Glob", serde_json::json!({"pattern": "**/bench/**"})),
            ("Bash", serde_json::json!({"command": "cat bench/demo-12/demo-12.json"})),
            ("Bash", serde_json::json!({"command": "cat benches/baseline.json"})),
            ("Bash", serde_json::json!({"command": "grep -r answer bench/"})),
            ("Bash", serde_json::json!({"command": "ls bench/demo-12/"})),
            ("Bash", serde_json::json!({"command": "echo x | tee bench/leak.txt"})),
            ("Bash", serde_json::json!({"command": "cat bench/../bench/demo-12/demo-12.json"})),
            ("Bash", serde_json::json!({"command": "grep foo benches/exercises_tb2.json"})),
            ("Bash", serde_json::json!({"command": "head -20 bench/gpqa.json"})),
            ("Bash", serde_json::json!({"command": "wc -l benches/baseline.json"})),
        ];
        assert_eq!(cases.len(), 20);
        let mut denied = 0;
        for (tool, args) in &cases {
            assert!(
                bench_policy_check(tool, args).is_some(),
                "{tool} {args} debe denegarse"
            );
            denied += 1;
        }
        assert_eq!(denied, 20, "0 lecturas: los 20 deben denegarse pre-IO");
        unsafe {
            std::env::remove_var(ENV_RUN_LEAKED);
        }
        // Controles: rutas legitimas NO se deniegan.
        assert!(bench_policy_check("Read", &serde_json::json!({"path": "src/main.rs"})).is_none());
        assert!(bench_policy_check("Bash", &serde_json::json!({"command": "cat src/main.rs"})).is_none());
        assert!(bench_policy_check("Bash", &serde_json::json!({"command": "echo hello"})).is_none());
        assert!(bench_policy_check("Grep", &serde_json::json!({"pattern": "foo", "path": "src"})).is_none());
        // B1 scope: `src/bench` (código) y `cargo bench` (subcomando) NO se deniegan.
        assert!(
            bench_policy_check("Read", &serde_json::json!({"path": "src/bench/foo.rs"}))
                .is_none(),
            "src/bench es código, no bench data"
        );
        assert!(
            bench_policy_check(
                "Read",
                &serde_json::json!({"path": "crates/kitty-cli/src/bench/mod.rs"})
            )
            .is_none(),
            "src/bench fuente no se bloquea"
        );
        assert!(
            bench_policy_check("Bash", &serde_json::json!({"command": "cargo bench"}))
                .is_none(),
            "`cargo bench` fuera de bench no se bloquea"
        );
        assert!(
            bench_policy_check(
                "Bash",
                &serde_json::json!({"command": "cat crates/kitty-cli/src/bench/mod.rs"})
            )
            .is_none(),
            "cat src/bench fuente no se bloquea"
        );
        unsafe {
            std::env::remove_var(ENV_RUN_LEAKED);
        }
    }

    // === B1 leak bypass: Grep sin path + intérpretes ===

    #[test]
    fn bench_policy_grep_sin_path_deniega_por_walker() {
        unsafe {
            std::env::remove_var(ENV_RUN_LEAKED);
        }
        // Sin `path` el walker cae a "." y barre bench/** ⇒ deny.
        assert!(
            bench_policy_check("Grep", &serde_json::json!({"pattern": "answer"}))
                .is_some(),
            "Grep sin path debe denegarse (root efectivo . barre bench/**)"
        );
        assert!(
            bench_policy_check("Grep", &serde_json::json!({"pattern": "x", "path": "."}))
                .is_some(),
            "Grep path . debe denegarse"
        );
        assert!(
            bench_policy_check("Grep", &serde_json::json!({"pattern": "x", "path": "./"}))
                .is_some(),
            "Grep path ./ debe denegarse"
        );
        // Con path explícito no-bench ⇒ allow.
        assert!(
            bench_policy_check("Grep", &serde_json::json!({"pattern": "x", "path": "src"}))
                .is_none()
        );
        unsafe {
            std::env::remove_var(ENV_RUN_LEAKED);
        }
    }

    #[test]
    fn bench_policy_bash_interpreter_payload_no_bypasea() {
        unsafe {
            std::env::remove_var(ENV_RUN_LEAKED);
        }
        // `bash -c 'cat bench/x'` escapa del scan ingenuo (palabra
        // `"cat bench/x"` con espacio no es path) ⇒ debe denegarse por
        // inspección del payload del intérprete.
        for cmd in [
            "bash -c 'cat bench/demo-12/demo-12.json'",
            "bash -c \"cat bench/demo-12/demo-12.json\"",
            "sh -c 'cat benches/baseline.json'",
            "python3 -c 'open(\"bench/gpqa.json\").read()'",
        ] {
            assert!(
                bench_policy_check("Bash", &serde_json::json!({"command": cmd})).is_some(),
                "`{cmd}` debe denegarse (payload intérprete pisa bench/**)"
            );
            unsafe {
                std::env::remove_var(ENV_RUN_LEAKED);
            }
        }
        // Payloads benignos del mismo intérprete ⇒ allow.
        for cmd in ["bash -c 'cargo build'", "sh -c 'cargo build'", "python3 build.py"] {
            assert!(
                bench_policy_check("Bash", &serde_json::json!({"command": cmd})).is_none(),
                "`{cmd}` no pisa bench/**"
            );
        }
        // `grep -r answer .` recursivo desde `.` barre bench/** ⇒ deny;
        // `grep -r answer src/` no ⇒ allow.
        assert!(
            bench_policy_check("Bash", &serde_json::json!({"command": "grep -r answer ."}))
                .is_some(),
            "`grep -r answer .` barre bench/**"
        );
        unsafe {
            std::env::remove_var(ENV_RUN_LEAKED);
        }
        assert!(
            bench_policy_check(
                "Bash",
                &serde_json::json!({"command": "grep -r answer src/"})
            )
            .is_none(),
            "`grep -r answer src/` no pisa bench/**"
        );
        unsafe {
            std::env::remove_var(ENV_RUN_LEAKED);
        }
    }

    // === Deepwork Pass 1 fix final: 7 comandos del gate + B1.3 ===

    #[test]
    fn bench_policy_gate7_b11_b12_b13() {
        unsafe {
            std::env::remove_var(ENV_RUN_LEAKED);
        }
        // B1.1 Grep root ancestro (normaliza ./.. antes de decidir).
        for args in [
            serde_json::json!({"pattern": "answer", "path": "src/.."}),
            serde_json::json!({"pattern": "x", "path": ".."}),
        ] {
            assert!(
                bench_policy_check("Grep", &args).is_some(),
                "Grep {args} debe denegarse (root efectivo barre bench/**)"
            );
            unsafe {
                std::env::remove_var(ENV_RUN_LEAKED);
            }
        }
        // Abs del repo (= cwd en bench) ⇒ deny (cwd o ancestro).
        let cwd_abs = std::env::current_dir()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| ".".to_string());
        // Sube hasta la raíz del repo (contiene `bench/` o `Cargo.toml`);
        // si no se encuentra, usa el cwd tal cual (= deny por igualdad).
        let mut repo_abs = cwd_abs.clone();
        {
            let mut cur = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
            loop {
                if cur.join("bench").is_dir() || cur.join("Cargo.toml").exists() {
                    repo_abs = cur.to_string_lossy().to_string();
                    break;
                }
                match cur.parent() {
                    Some(par) => cur = par.to_path_buf(),
                    None => break,
                }
            }
        }
        assert!(
            bench_policy_check(
                "Grep",
                &serde_json::json!({"pattern": "x", "path": repo_abs})
            )
            .is_some(),
            "Grep path abs del repo ({repo_abs}) debe denegarse"
        );
        unsafe {
            std::env::remove_var(ENV_RUN_LEAKED);
        }
        // B1.2 Bash recursivo implícito (root implícito `.` ⇒ deny).
        for cmd in [
            "grep -r SECRET",
            "grep -R SECRET",
            "rg SECRET",
            "bash -c 'grep -r SECRET'",
        ] {
            assert!(
                bench_policy_check("Bash", &serde_json::json!({"command": cmd})).is_some(),
                "`{cmd}` debe denegarse (recursivo con root implícito barre bench/**)"
            );
            unsafe {
                std::env::remove_var(ENV_RUN_LEAKED);
            }
        }
        // Solo allow con root explícito fuera de bench.
        assert!(
            bench_policy_check(
                "Bash",
                &serde_json::json!({"command": "grep -r SECRET src/"})
            )
            .is_none(),
            "`grep -r SECRET src/` no pisa bench/**"
        );
        unsafe {
            std::env::remove_var(ENV_RUN_LEAKED);
        }
        // B1.3 Excepción src/bench con `..` (normaliza antes del early-return).
        assert!(
            is_bench_path("src/bench/../../bench/demo-12.json"),
            "src/bench/../../bench ⇒ bench tras normalizar"
        );
        assert!(
            is_bench_path("crates/kitty-cli/src/bench/../../../../bench/x"),
            "src/bench con ..* ⇒ bench tras normalizar"
        );
        assert!(
            is_bench_path("/abs/repo/bench/src/bench/x"),
            "contiene raíz bench aunque incluya src/bench"
        );
        // Controles: fuentes src/bench siguen siendo código.
        assert!(!is_bench_path("src/bench/foo.rs"));
        assert!(!is_bench_path("crates/kitty-cli/src/bench/mod.rs"));
        unsafe {
            std::env::remove_var(ENV_RUN_LEAKED);
        }
    }
}
