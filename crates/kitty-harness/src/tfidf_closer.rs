//! W2 TF-IDF baseline closers (plan híbrido 2026-10-07, HU-2).
//!
//! Clasificador TF-IDF puro-Rust, 0 dependencias nuevas (solo `std`):
//! decide done-gate/abstain en microsegundos sin quemar el SLM.
//!
//! TRAMPA CONOCIDA: está OFF en bench por defecto
//! ([`CloserConfig::enabled`] = `false`). Activar un gate adaptativo dentro
//! del bench sesga la medida; el bench debe medir el SLM-entero, no el gate.

use std::collections::{HashMap, HashSet};

/// Etiqueta del closer para una salida de tool / segmento final.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloserLabel {
    Done,
    NotDone,
    Abstain,
}

/// Veredicto con confianza recalibrada en datos propios (no umbrales ajenos).
#[derive(Debug, Clone, Copy)]
pub struct CloserVerdict {
    pub label: CloserLabel,
    /// Coseno ganador en [0,1].
    pub conf: f32,
    /// |cos_done − cos_not|; margen estrecho ⇒ abstención.
    pub margin: f32,
}

/// Config del closer. `enabled=false` por defecto: OFF en bench.
#[derive(Debug, Clone, Copy)]
pub struct CloserConfig {
    pub enabled: bool,
    pub min_conf: f32,
    pub min_margin: f32,
}

impl Default for CloserConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            min_conf: 0.30,
            min_margin: 0.05,
        }
    }
}

fn tokenize(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_string())
        .collect()
}

/// Clasificador TF-IDF de dos centroides (done / not-done).
/// Estados permanentes prohibidos: no hay TTL aquí porque el modelo es
/// inmutable tras `fit`; la recalibración es re-`fit` con datos kitty.
#[derive(Debug, Clone, Default)]
pub struct TfidfCloser {
    idf: HashMap<String, f32>,
    done_centroid: HashMap<String, f32>,
    not_centroid: HashMap<String, f32>,
    n_docs: usize,
}

impl TfidfCloser {
    /// Ajusta con docs etiquetados: `(texto, es_done)`.
    /// Vacío ⇒ predictor que siempre se abstiene (fail-open).
    pub fn fit(docs: &[(String, bool)]) -> Self {
        let n = docs.len();
        if n == 0 {
            return Self::default();
        }
        let tokenized: Vec<Vec<String>> = docs.iter().map(|(t, _)| tokenize(t)).collect();
        let mut df: HashMap<String, usize> = HashMap::new();
        for toks in &tokenized {
            let seen: HashSet<&str> = toks.iter().map(|s| s.as_str()).collect();
            for t in seen {
                *df.entry(t.to_string()).or_insert(0) += 1;
            }
        }
        let n_f = n as f32;
        let idf: HashMap<String, f32> = df
            .iter()
            .map(|(t, d)| {
                let d = *d as f32;
                (t.clone(), (1.0 + (n_f - d + 0.5) / (d + 0.5)).ln())
            })
            .collect();

        let mut done_acc: HashMap<String, f32> = HashMap::new();
        let mut not_acc: HashMap<String, f32> = HashMap::new();
        let (mut n_done, mut n_not) = (0_f32, 0_f32);
        for (toks, (_, is_done)) in tokenized.iter().zip(docs.iter()) {
            let mut tf: HashMap<&str, usize> = HashMap::new();
            for t in toks {
                *tf.entry(t.as_str()).or_insert(0) += 1;
            }
            let acc = if *is_done {
                n_done += 1.0;
                &mut done_acc
            } else {
                n_not += 1.0;
                &mut not_acc
            };
            for (t, c) in tf {
                let w = c as f32 * idf.get(t).copied().unwrap_or(0.0);
                *acc.entry(t.to_string()).or_insert(0.0) += w;
            }
        }
        for v in done_acc.values_mut() {
            *v /= n_done.max(1.0);
        }
        for v in not_acc.values_mut() {
            *v /= n_not.max(1.0);
        }
        Self {
            idf,
            done_centroid: done_acc,
            not_centroid: not_acc,
            n_docs: n,
        }
    }

    fn cosine(&self, text_vec: &HashMap<String, f32>, centroid: &HashMap<String, f32>) -> f32 {
        if text_vec.is_empty() || centroid.is_empty() {
            return 0.0;
        }
        let mut dot = 0.0_f32;
        for (t, w) in text_vec {
            if let Some(c) = centroid.get(t) {
                dot += w * c;
            }
        }
        if dot <= 0.0 {
            return 0.0;
        }
        let norm_t: f32 = text_vec.values().map(|w| w * w).sum::<f32>().sqrt();
        let norm_c: f32 = centroid.values().map(|w| w * w).sum::<f32>().sqrt();
        if norm_t <= 0.0 || norm_c <= 0.0 {
            0.0
        } else {
            (dot / (norm_t * norm_c)).clamp(0.0, 1.0)
        }
    }

    /// Predice. Sin modelo (`fit` vacío) o confianza/margen bajo ⇒ `Abstain`
    /// (fail-open: decide el SLM).
    pub fn predict(&self, text: &str, cfg: &CloserConfig) -> CloserVerdict {
        if self.n_docs == 0 {
            return CloserVerdict {
                label: CloserLabel::Abstain,
                conf: 0.0,
                margin: 0.0,
            };
        }
        let toks = tokenize(text);
        let mut counts: HashMap<String, usize> = HashMap::new();
        for t in &toks {
            *counts.entry(t.clone()).or_insert(0) += 1;
        }
        let mut vec: HashMap<String, f32> = HashMap::new();
        for (t, c) in counts {
            if let Some(idf) = self.idf.get(&t) {
                vec.insert(t, c as f32 * idf);
            }
        }
        let s_done = self.cosine(&vec, &self.done_centroid);
        let s_not = self.cosine(&vec, &self.not_centroid);
        let margin = (s_done - s_not).abs();
        let (label, conf) = if s_done >= s_not {
            (CloserLabel::Done, s_done)
        } else {
            (CloserLabel::NotDone, s_not)
        };
        if conf < cfg.min_conf || margin < cfg.min_margin {
            CloserVerdict {
                label: CloserLabel::Abstain,
                conf,
                margin,
            }
        } else {
            CloserVerdict {
                label,
                conf,
                margin,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Bench vs SLM-entero en 1 tarea fija (W2)
// ---------------------------------------------------------------------------

/// Tarea fija: 8 salidas de tool etiquetadas done / not-done.
/// Mini-fixture de humo, NO benchmark general: solo evita quemar el SLM
/// en lo que un clasificador de microsegundos resuelve.
pub const FIXED_TASK: &[(&str, bool)] = &[
    ("task complete all tests pass solution saved", true),
    ("done implementation finished verified ok", true),
    ("finished successfully all checks passed", true),
    ("completed output written tests green", true),
    ("error failed traceback exception occurred", false),
    ("timeout killed after retries exhausted", false),
    ("partial progress blocked waiting input", false),
    ("failed assertion mismatch needs fix", false),
];

/// Referencia externa SLM-entero en la tarea fija (acierto medido fuera de
/// este bench; pendiente de re-medida con SLM local — hasta entonces el
/// bench declara PARIDAD salvo superación clara, sin maquillaje).
pub const SLM_REFERENCE_ACC: f32 = 0.875;

/// Veredicto del bench: solo `BeatsSlm` supera la referencia; empate o
/// peor ⇒ `Parity` (se sigue con el SLM-entero, no se adopta el gate).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TfidfBenchVerdict {
    BeatsSlm,
    Parity,
}

#[derive(Debug, Clone, Copy)]
pub struct TfidfBenchReport {
    pub accuracy: f32,
    pub avg_predict_micros: u128,
    pub verdict: TfidfBenchVerdict,
}

/// Bench honesto: leave-one-out sobre la tarea fija (ajusta con N−1,
/// evalúa el restante). La abstención cuenta como fallo: fail-open deriva
/// al SLM, no suma acierto al baseline.
pub fn bench_fixed_task(cfg: &CloserConfig) -> TfidfBenchReport {
    let n = FIXED_TASK.len();
    let mut correct = 0_usize;
    let t0 = std::time::Instant::now();
    for (i, (text, expected)) in FIXED_TASK.iter().enumerate() {
        let train: Vec<(String, bool)> = FIXED_TASK
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, (t, d))| (t.to_string(), *d))
            .collect();
        let closer = TfidfCloser::fit(&train);
        let v = closer.predict(text, cfg);
        let got_done = v.label == CloserLabel::Done;
        if got_done == *expected {
            correct += 1;
        }
    }
    let elapsed = t0.elapsed().as_micros();
    let accuracy = correct as f32 / n as f32;
    let verdict = if accuracy > SLM_REFERENCE_ACC + 1e-6 {
        TfidfBenchVerdict::BeatsSlm
    } else {
        TfidfBenchVerdict::Parity
    };
    TfidfBenchReport {
        accuracy,
        avg_predict_micros: elapsed / n.max(1) as u128,
        verdict,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> CloserConfig {
        CloserConfig {
            enabled: true,
            ..CloserConfig::default()
        }
    }

    #[test]
    fn off_en_bench_por_defecto() {
        assert!(
            !CloserConfig::default().enabled,
            "el closer debe estar OFF por defecto (trampa conocida en bench)"
        );
    }

    #[test]
    fn fit_vacio_siempre_abstiene() {
        let c = TfidfCloser::fit(&[]);
        let v = c.predict("anything at all", &cfg());
        assert_eq!(v.label, CloserLabel::Abstain);
    }

    #[test]
    fn done_claro_se_detecta() {
        let train: Vec<(String, bool)> = FIXED_TASK
            .iter()
            .map(|(t, d)| (t.to_string(), *d))
            .collect();
        let c = TfidfCloser::fit(&train);
        // Recalibración en datos propios: umbrales explícitos del test.
        let loose = CloserConfig {
            enabled: true,
            min_conf: 0.1,
            min_margin: 0.01,
        };
        // Texto in-sample + vocabularios done/not disjuntos ⇒ s_not = 0.
        let v = c.predict(FIXED_TASK[0].0, &loose);
        assert_eq!(v.label, CloserLabel::Done);
        assert_eq!(v.margin, v.conf, "sin solape rival el margen es la conf");
    }

    #[test]
    fn texto_ambiguo_abstiene() {
        let train: Vec<(String, bool)> = FIXED_TASK
            .iter()
            .map(|(t, d)| (t.to_string(), *d))
            .collect();
        let c = TfidfCloser::fit(&train);
        // Sin solape de vocabulario ⇒ conf 0 ⇒ abstención fail-open.
        let v = c.predict("xyzzy qwerty blorpt", &cfg());
        assert_eq!(v.label, CloserLabel::Abstain);
    }

    #[test]
    fn bench_tarea_fija_reporta_acierto_y_tiempo() {
        let r = bench_fixed_task(&cfg());
        assert!((0.0..=1.0).contains(&r.accuracy));
        assert!(
            r.avg_predict_micros < 10_000,
            "el baseline debe resolver en microsegundos, no milisegundos: {}µs",
            r.avg_predict_micros
        );
        // Regla sin maquillaje: solo supera ⇒ BeatsSlm, si no ⇒ Parity.
        assert_eq!(
            r.verdict,
            if r.accuracy > SLM_REFERENCE_ACC + 1e-6 {
                TfidfBenchVerdict::BeatsSlm
            } else {
                TfidfBenchVerdict::Parity
            }
        );
    }
}
