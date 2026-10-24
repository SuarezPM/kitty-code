//! `kitty doctor` — environment probe (W1, plan SDD-ESTADO-ARTE §3 W-INSTALAR).
//!
//! Checks, in order:
//! 1. Ollama reachability via `GET /api/tags` (reuses
//!    [`OllamaClient::health_check`]) and exact model presence via
//!    [`OllamaClient::list_models`] — comparison is byte-exact equality,
//!    never a regex or prefix match (HU-PRIMER-RUN: propone `pull` exacto).
//! 2. Model metadata via [`OllamaClient::show_model`] (`/api/show` one-shot,
//!    same probe as the `num_ctx` resolver in `main.rs`).
//! 3. Host RAM (`/proc/meminfo`), GPU VRAM (`nvidia-smi`, best-effort) and
//!    free disk (`df`, best-effort).
//!
//! Every host probe degrades to `None` loudly instead of failing: doctor must
//! run on machines without GPU or without Ollama and still print actionable
//! next steps.

use kitty_runtime::OllamaClient;
use serde::{Deserialize, Serialize};

/// Doctor findings, serializable for `--json` output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoctorReport {
    /// Ollama base URL that was probed.
    pub ollama_url: String,
    /// `GET /api/tags` answered 2xx.
    pub ollama_reachable: bool,
    /// Exact model names returned by `/api/tags` (unsorted server order).
    pub models: Vec<String>,
    /// Model the user asked for (`--model`).
    pub model_requested: String,
    /// Exact-equality presence of `model_requested` in `models`.
    pub model_present: bool,
    /// `/api/show` answered 2xx for `model_requested`.
    pub model_details_ok: bool,
    /// Host RAM in GiB (from `/proc/meminfo`, `None` off-Linux).
    pub ram_total_gb: Option<f64>,
    /// GPU VRAM in GiB (first `nvidia-smi` line, `None` without NVIDIA GPU).
    pub gpu_vram_gb: Option<f64>,
    /// `ollama ps` runs CPU-only with resident models (`None` when `ps`
    /// failed or no model is resident — nothing to judge).
    pub ollama_cpu_only: Option<bool>,
    /// `ollama ps` summary (`"empty"`, `"N model row(s)"`, `"ollama ps failed"`).
    pub ollama_ps_summary: String,
    /// Profile recommended from measured VRAM (`recommend_profile_by_vram`).
    pub perfil_recomendado: String,
    /// Free disk in GiB at the current directory (`None` when `df` fails).
    pub disk_free_gb: Option<f64>,
    /// Recommended backend: `"ollama"` (or the `kitty-infer` env override).
    pub backend_recomendado: String,
    /// Actionable next steps, most urgent first. Pull commands always carry
    /// the EXACT model id (`ollama pull <id>`), never a pattern.
    pub acciones: Vec<String>,
}

/// Exact-equality model presence — NEVER a regex or prefix match.
///
/// Rationale (HU-PRIMER-RUN): `ornith:9b` present must not satisfy a request
/// for `ornith` and vice versa; a fuzzy match would propose the wrong `pull`.
#[must_use]
pub fn model_present_exact(models: &[String], requested: &str) -> bool {
    models.iter().any(|m| m == requested)
}

/// Backend recommendation. `kitty-infer` wins only via the explicit
/// `KITTY_BACKEND=kitty-infer` env (never auto-selected); otherwise Ollama is
/// the recommended backend and the report says how to start it when down.
#[must_use]
pub fn recommend_backend(ollama_reachable: bool) -> String {
    let infer =
        kitty_runtime::backend_selected(std::env::var(kitty_runtime::ENV_BACKEND).ok().as_deref());
    if infer {
        return "kitty-infer".to_string();
    }
    if ollama_reachable {
        "ollama".to_string()
    } else {
        "ollama (no disponible — inicia con `ollama serve`)".to_string()
    }
}

/// Parse `MemTotal:       16384000 kB` from `/proc/meminfo` into GiB.
#[must_use]
pub fn parse_memtotal_gb(meminfo: &str) -> Option<f64> {
    for line in meminfo.lines() {
        let Some(rest) = line.strip_prefix("MemTotal:") else {
            continue;
        };
        let kb: f64 = rest.split_whitespace().next()?.parse::<f64>().ok()?;
        if kb <= 0.0 {
            return None;
        }
        return Some(kb / 1024.0 / 1024.0);
    }
    None
}

/// Parse the first `nvidia-smi --query-gpu=memory.total` line (MiB) into GiB.
#[must_use]
pub fn parse_nvidia_smi_mib(output: &str) -> Option<f64> {
    let first = output.lines().map(str::trim).find(|l| !l.is_empty())?;
    let mib: f64 = first.split_whitespace().next()?.parse::<f64>().ok()?;
    if mib <= 0.0 {
        return None;
    }
    Some(mib / 1024.0)
}

/// Parse `df -k --output=avail <dir>` output (last line, KiB) into GiB.
#[must_use]
pub fn parse_df_avail_gb(output: &str) -> Option<f64> {
    let last = output.lines().map(str::trim).rfind(|l| !l.is_empty())?;
    // Header line contains "Avail"; skip it when it is the only line.
    if last.bytes().any(|b| b.is_ascii_alphabetic()) {
        return None;
    }
    let kb: f64 = last.split_whitespace().next()?.parse::<f64>().ok()?;
    if kb < 0.0 {
        return None;
    }
    Some(kb / 1024.0 / 1024.0)
}

/// Is Ollama running CPU-only? Parses `ollama ps` output (table with a
/// `PROCESSOR` column like `100% GPU`, `100% CPU` or `86% CPU/14% GPU`).
/// - Header-only / empty ⇒ `None` (no resident model, nothing to judge).
/// - Any row mentioning GPU ⇒ `Some(false)` (offload active).
/// - Rows only CPU ⇒ `Some(true)`.
#[must_use]
pub fn parse_ollama_ps_cpu_only(output: &str) -> Option<bool> {
    let mut rows: u32 = 0;
    let mut gpu_seen = false;
    for line in output.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        // Header (`NAME … PROCESSOR …`); data rows never carry that word.
        if trimmed.to_ascii_uppercase().contains("PROCESSOR") {
            continue;
        }
        rows += 1;
        if trimmed.to_ascii_uppercase().contains("GPU") {
            gpu_seen = true;
        }
    }
    if rows == 0 {
        None
    } else {
        Some(!gpu_seen)
    }
}

/// Profile recommended from measured VRAM. Conservative thresholds aligned
/// with the w6-vram series (9B peaks ≈6.5 GiB on 8 GiB; medium needs ≥8 GiB).
/// `None` (no NVIDIA GPU) ⇒ `tiny` (CPU-friendly ≤4B models).
#[must_use]
pub fn recommend_profile_by_vram(vram_gb: Option<f64>) -> &'static str {
    match vram_gb {
        None => "tiny",
        Some(v) if v < 6.0 => "tiny",
        Some(v) if v < 8.0 => "small",
        Some(v) if v < 20.0 => "medium",
        _ => "large",
    }
}

/// Pure action builder — unit-tested without network or subprocesses.
#[must_use]
pub fn build_actions(
    ollama_reachable: bool,
    model_requested: &str,
    model_present: bool,
    ram_total_gb: Option<f64>,
    disk_free_gb: Option<f64>,
    gpu_vram_gb: Option<f64>,
    ollama_cpu_only: Option<bool>,
) -> Vec<String> {
    let mut acciones = Vec::new();
    if !ollama_reachable {
        acciones.push(
            "Ollama no responde. Inicia el servidor con `ollama serve` y re-ejecuta `kitty doctor`."
                .to_string(),
        );
    }
    if ollama_reachable && !model_present {
        acciones.push(format!(
            "Modelo ausente. Descárgalo con el comando exacto: `ollama pull {model_requested}`"
        ));
    }
    if let Some(ram) = ram_total_gb {
        if ram < 8.0 {
            acciones.push(format!(
                "RAM total {ram:.1} GiB (< 8 GiB): usa `--profile tiny` y modelos ≤4B."
            ));
        }
    }
    if let Some(free) = disk_free_gb {
        if free < 10.0 {
            acciones.push(format!(
                "Disco libre {free:.1} GiB (< 10 GiB): un Q4_K_M 8-9B ocupa ~6 GiB; libera espacio antes del pull."
            ));
        }
    }
    if ollama_cpu_only == Some(true) {
        acciones.push(
            "Ollama corre CPU-only (`ollama ps` sin GPU): espera latencias altas; usa `--profile tiny` y modelos ≤4B, o revisa el offload a GPU."
                .to_string(),
        );
    }
    let perfil = recommend_profile_by_vram(gpu_vram_gb);
    match gpu_vram_gb {
        None => acciones.push(
            "Sin VRAM detectable (sin nvidia-smi): si no hay GPU, usa `--profile tiny` y modelos ≤4B."
                .to_string(),
        ),
        Some(v) if perfil == "tiny" || perfil == "small" => acciones.push(format!(
            "VRAM {v:.1} GiB: usa `--profile {perfil}` y modelos acordes (9B ≈6.5 GiB pico)."
        )),
        Some(_) => {}
    }
    if acciones.is_empty() {
        acciones.push("Todo listo: Ollama responde y el modelo está presente.".to_string());
    }
    acciones
}

fn read_host_ram_gb() -> Option<f64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    parse_memtotal_gb(&meminfo)
}

fn read_gpu_vram_gb() -> Option<f64> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    parse_nvidia_smi_mib(&String::from_utf8_lossy(&out.stdout))
}

fn read_disk_free_gb() -> Option<f64> {
    let cwd = std::env::current_dir().ok()?;
    let out = std::process::Command::new("df")
        .args(["-k", "--output=avail"])
        .arg(&cwd)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    parse_df_avail_gb(&String::from_utf8_lossy(&out.stdout))
}

/// `ollama ps` best-effort: (`cpu_only`, summary). Nunca falla el doctor:
/// fallo o `ps` vacío degradan a (`None`, resumen honesto).
fn read_ollama_ps() -> (Option<bool>, String) {
    let Ok(out) = std::process::Command::new("ollama").arg("ps").output() else {
        return (None, "ollama ps failed".to_string());
    };
    if !out.status.success() {
        return (None, "ollama ps failed".to_string());
    }
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    let cpu_only = parse_ollama_ps_cpu_only(&text);
    let rows = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.to_ascii_uppercase().contains("PROCESSOR"))
        .count();
    let summary = if rows == 0 {
        "empty".to_string()
    } else {
        format!("{rows} model row(s)")
    };
    (cpu_only, summary)
}

/// Run the full probe. Network/subprocess failures degrade to `false`/`None`
/// fields — this function never returns `Err` for a missing server or GPU.
pub async fn run_doctor(url: &str, model: &str) -> DoctorReport {
    let client = OllamaClient::new(url);
    let ollama_reachable = client.health_check().await.unwrap_or(false);

    let mut models: Vec<String> = Vec::new();
    let mut model_details_ok = false;
    if ollama_reachable {
        if let Ok(list) = client.list_models().await {
            models = list.iter().map(|m| m.name.clone()).collect();
        }
        // One-shot `/api/show`: proves the exact id resolves server-side
        // (same call the `num_ctx` resolver makes at startup).
        model_details_ok = client.show_model(model).await.is_ok();
    }

    let model_present = model_present_exact(&models, model);
    let ram_total_gb = read_host_ram_gb();
    let gpu_vram_gb = read_gpu_vram_gb();
    let disk_free_gb = read_disk_free_gb();
    let (ollama_cpu_only, ollama_ps_summary) = read_ollama_ps();
    let perfil_recomendado = recommend_profile_by_vram(gpu_vram_gb).to_string();
    let backend_recomendado = recommend_backend(ollama_reachable);
    let acciones = build_actions(
        ollama_reachable,
        model,
        model_present,
        ram_total_gb,
        disk_free_gb,
        gpu_vram_gb,
        ollama_cpu_only,
    );

    DoctorReport {
        ollama_url: url.to_string(),
        ollama_reachable,
        models,
        model_requested: model.to_string(),
        model_present,
        model_details_ok,
        ram_total_gb,
        disk_free_gb,
        gpu_vram_gb,
        ollama_cpu_only,
        ollama_ps_summary,
        perfil_recomendado,
        backend_recomendado,
        acciones,
    }
}

/// Human-readable rendering (default `kitty doctor` output).
#[must_use]
pub fn format_human(r: &DoctorReport) -> String {
    let mut out = String::from("kitty doctor\n");
    out.push_str(&format!(
        "Ollama: {} ({})\n",
        r.ollama_url,
        if r.ollama_reachable {
            "responde"
        } else {
            "NO responde"
        }
    ));
    out.push_str(&format!(
        "Modelo '{}': {}\n",
        r.model_requested,
        if r.model_present {
            "presente"
        } else {
            "AUSENTE"
        }
    ));
    if r.ollama_reachable {
        out.push_str(&format!(
            "Modelos locales ({}): {}\n",
            r.models.len(),
            if r.models.is_empty() {
                "(vacío)".to_string()
            } else {
                r.models.join(", ")
            }
        ));
    }
    if let Some(ram) = r.ram_total_gb {
        out.push_str(&format!("RAM total: {ram:.1} GiB\n"));
    }
    match r.gpu_vram_gb {
        Some(v) => out.push_str(&format!("VRAM GPU: {v:.1} GiB\n")),
        None => out.push_str("VRAM GPU: no detectada (sin nvidia-smi)\n"),
    }
    match r.disk_free_gb {
        Some(d) => out.push_str(&format!("Disco libre: {d:.1} GiB\n")),
        None => out.push_str("Disco libre: desconocido\n"),
    }
    let ps_nota = match r.ollama_cpu_only {
        Some(true) => " (CPU-only, sin GPU)",
        Some(false) => " (con GPU)",
        None => "",
    };
    out.push_str(&format!("Ollama ps: {}{ps_nota}\n", r.ollama_ps_summary));
    out.push_str(&format!(
        "Perfil recomendado (por VRAM): --profile {}\n",
        r.perfil_recomendado
    ));
    out.push_str(&format!("Backend recomendado: {}\n", r.backend_recomendado));
    out.push_str("Acciones:\n");
    for (i, a) in r.acciones.iter().enumerate() {
        out.push_str(&format!("  {}. {a}\n", i + 1));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_match_never_fuzzy() {
        let models = vec!["ornith:9b".to_string(), "qwen3:4b".to_string()];
        assert!(model_present_exact(&models, "ornith:9b"));
        // Prefix / regex-style queries must NOT match.
        assert!(!model_present_exact(&models, "ornith"));
        assert!(!model_present_exact(&models, "ornith:.*"));
        assert!(!model_present_exact(&models, "ornith:9b-extra"));
        assert!(!model_present_exact(&models, ""));
        assert!(!model_present_exact(&[], "ornith:9b"));
    }

    #[test]
    fn meminfo_parses_gb() {
        let meminfo = "MemTotal:        8000000 kB\nMemFree: 1000 kB\n";
        let gb = parse_memtotal_gb(meminfo).expect("should parse");
        assert!((gb - 7.629).abs() < 0.01, "got {gb}");
        // MemTotal not on the first line (real /proc/meminfo has headers).
        let meminfo2 = "Name: foo\nMemTotal:        8000000 kB\nMemFree: 1000 kB\n";
        let gb2 = parse_memtotal_gb(meminfo2).expect("should skip non-matching lines");
        assert!((gb2 - 7.629).abs() < 0.01, "got {gb2}");
        assert_eq!(parse_memtotal_gb("nada útil\n"), None);
        assert_eq!(parse_memtotal_gb("MemTotal: 0 kB\n"), None);
    }

    #[test]
    fn nvidia_smi_parses_first_line_mib() {
        assert_eq!(parse_nvidia_smi_mib("8192\n"), Some(8.0));
        assert_eq!(parse_nvidia_smi_mib("  6144  \n8192\n"), Some(6.0));
        assert_eq!(parse_nvidia_smi_mib(""), None);
        assert_eq!(parse_nvidia_smi_mib("N/A\n"), None);
    }

    #[test]
    fn df_parses_avail_last_line() {
        let out = "     Avail\n  20971520\n";
        let gb = parse_df_avail_gb(out).expect("should parse");
        assert!((gb - 20.0).abs() < 0.01, "got {gb}");
        assert_eq!(parse_df_avail_gb("     Avail\n"), None);
    }

    #[test]
    fn missing_model_proposes_exact_pull() {
        let acciones = build_actions(
            true,
            "ornith:9b",
            false,
            Some(16.0),
            Some(50.0),
            Some(16.0),
            Some(false),
        );
        assert!(
            acciones.iter().any(|a| a.contains("ollama pull ornith:9b")),
            "got {acciones:?}"
        );
    }

    #[test]
    fn unreachable_ollama_asks_to_serve_first() {
        let acciones = build_actions(false, "ornith:9b", false, None, None, None, None);
        assert!(
            acciones.iter().any(|a| a.contains("ollama serve")),
            "got {acciones:?}"
        );
        // Sin servidor no tiene sentido proponer el pull todavía.
        assert!(
            !acciones.iter().any(|a| a.contains("ollama pull")),
            "got {acciones:?}"
        );
    }

    #[test]
    fn all_good_reports_ready() {
        // VRAM de sobra (medium) + ps con GPU: sin avisos nuevos.
        let acciones = build_actions(
            true,
            "ornith:9b",
            true,
            Some(16.0),
            Some(50.0),
            Some(16.0),
            Some(false),
        );
        assert_eq!(acciones.len(), 1);
        assert!(acciones[0].contains("Todo listo"));
    }

    #[test]
    fn low_ram_warns_tiny_profile() {
        let acciones = build_actions(
            true,
            "qwen3:4b",
            true,
            Some(4.0),
            Some(50.0),
            Some(16.0),
            Some(false),
        );
        assert!(
            acciones.iter().any(|a| a.contains("--profile tiny")),
            "got {acciones:?}"
        );
    }

    #[test]
    fn ollama_ps_cpu_only_parsing() {
        assert_eq!(parse_ollama_ps_cpu_only(""), None);
        assert_eq!(
            parse_ollama_ps_cpu_only("NAME\tID\tSIZE\tPROCESSOR\tCONTEXT\tUNTIL\n"),
            None,
            "solo header ⇒ sin residentes"
        );
        assert_eq!(
            parse_ollama_ps_cpu_only(
                "NAME\tID\tSIZE\tPROCESSOR\tCONTEXT\tUNTIL\nornith:9b\te5df7dc\t5.5 GB\t100% CPU\t16384\t5 minutes from now\n"
            ),
            Some(true)
        );
        assert_eq!(
            parse_ollama_ps_cpu_only(
                "NAME ID SIZE PROCESSOR CONTEXT UNTIL\nornith:9b e5df7dc 5.5 GB 100% GPU 16384 5m\n"
            ),
            Some(false)
        );
        assert_eq!(
            parse_ollama_ps_cpu_only(
                "NAME ID SIZE PROCESSOR CONTEXT UNTIL\nornith:9b e5df7dc 5.5 GB 86% CPU/14% GPU 16384 5m\n"
            ),
            Some(false),
            "offload parcial cuenta como GPU"
        );
        assert_eq!(
            parse_ollama_ps_cpu_only(
                "NAME ID SIZE PROCESSOR CONTEXT UNTIL\na x 1 GB 100% CPU 4096 1m\nb y 2 GB 100% GPU 4096 1m\n"
            ),
            Some(false),
            "una fila con GPU basta"
        );
    }

    #[test]
    fn vram_profile_thresholds() {
        assert_eq!(recommend_profile_by_vram(None), "tiny");
        assert_eq!(recommend_profile_by_vram(Some(4.0)), "tiny");
        assert_eq!(recommend_profile_by_vram(Some(6.0)), "small");
        assert_eq!(recommend_profile_by_vram(Some(7.9)), "small");
        assert_eq!(recommend_profile_by_vram(Some(8.0)), "medium");
        assert_eq!(recommend_profile_by_vram(Some(16.0)), "medium");
        assert_eq!(recommend_profile_by_vram(Some(24.0)), "large");
    }

    #[test]
    fn cpu_only_warns_tiny_and_offload() {
        let acciones = build_actions(
            true,
            "ornith:9b",
            true,
            Some(16.0),
            Some(50.0),
            Some(8.0),
            Some(true),
        );
        assert!(
            acciones
                .iter()
                .any(|a| a.contains("CPU-only") && a.contains("--profile tiny")),
            "got {acciones:?}"
        );
    }

    #[test]
    fn low_vram_recommends_matching_profile() {
        let tiny = build_actions(
            true,
            "m",
            true,
            Some(16.0),
            Some(50.0),
            Some(5.0),
            Some(false),
        );
        assert!(
            tiny.iter().any(|a| a.contains("--profile tiny")),
            "got {tiny:?}"
        );
        let small = build_actions(
            true,
            "m",
            true,
            Some(16.0),
            Some(50.0),
            Some(7.0),
            Some(false),
        );
        assert!(
            small.iter().any(|a| a.contains("--profile small")),
            "got {small:?}"
        );
    }

    #[test]
    fn missing_vram_suggests_tiny() {
        let acciones = build_actions(true, "m", true, Some(16.0), Some(50.0), None, None);
        assert!(
            acciones
                .iter()
                .any(|a| a.contains("Sin VRAM") && a.contains("--profile tiny")),
            "got {acciones:?}"
        );
    }
}
