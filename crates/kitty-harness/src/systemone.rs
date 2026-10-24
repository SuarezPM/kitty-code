//! Capa 1 — System-One online, contrato mínimo (Lane C, Pass 1).
//!
//! Solo trait + TF-IDF. Fuera de este fichero quedan: Laya runtime,
//! `client.rs` / logprobs y `router.rs` (vetados en este lane).
//!
//! Contrato:
//! - [`SystemOne::decide`] es pura, sin I/O y sin bloqueo: clasifica en
//!   microsegundos sin quemar el SLM.
//! - Fail-open: sin evidencia (modelo vacío, deshabilitado, texto vacío,
//!   confianza/margen bajo) ⇒ [`Decision`] `Noul` con `abstained = true`
//!   (decide el SLM).
//! - OFF en bench: [`systemone_status`] devuelve `("systemone", "off")`
//!   cuando `KITTY_BENCH_MODE=1`; `decide()` nunca se llama en bench
//!   scored (Lane B la apaga).
//! - Laya sin calibrar: 0/200 etiquetas sigue HELD (ver
//!   `laya_classifier::fit`); aquí solo se deja el hueco, sin calibrar.

use crate::tfidf_closer::{CloserConfig, CloserLabel, TfidfCloser};

/// Entrada mínima para una decisión System-One.
#[derive(Debug, Clone, Default)]
pub struct DecisionRequest {
    /// Texto a clasificar (salida de tool / segmento final).
    pub text: String,
}

impl DecisionRequest {
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

/// Las tres salidas tipadas del wire (choice / score / noul).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    /// Elección discreta (p. ej. closer-Done: parar).
    Choice,
    /// Puntuación / evidencia parcial (p. ej. closer-NotDone: seguir).
    Score,
    /// Sin decisión: fail-open, decide el SLM.
    Noul,
}

/// Decisión con metadatos para manifiesto / diagnóstico.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoredDecision {
    /// Salida tipada.
    pub decision: Decision,
    /// Origen del veredicto (`"off"` | `"tfidf"`).
    pub source: &'static str,
    /// Id del modelo (`"none"` | `"tfidf-closer"`; hueco Laya: sin calibrar).
    pub model_id: &'static str,
    /// Confianza en [0,1] (0.0 cuando hay abstención).
    pub confidence: f32,
    /// `true` ⇒ fail-open: el SLM decide.
    pub abstained: bool,
}

impl ScoredDecision {
    /// Fail-open canónico: Noul abstinente.
    #[must_use]
    pub fn noul(source: &'static str, model_id: &'static str) -> Self {
        Self {
            decision: Decision::Noul,
            source,
            model_id,
            confidence: 0.0,
            abstained: true,
        }
    }
}

/// Contrato System-One: 1 forward tipado, 0 output tokens.
///
/// Implementaciones: puras, sin I/O, sin bloqueo, fail-open a Noul.
pub trait SystemOne {
    /// Decide sobre `req`. Nunca falla: sin evidencia devuelve Noul
    /// abstinente.
    fn decide(&self, req: &DecisionRequest) -> ScoredDecision;
}

/// System-One apagado: siempre Noul abstinente (bench + default).
#[derive(Debug, Clone, Copy, Default)]
pub struct OffSystemOne;

impl SystemOne for OffSystemOne {
    fn decide(&self, _req: &DecisionRequest) -> ScoredDecision {
        ScoredDecision::noul("off", "none")
    }
}

/// Baseline TF-IDF como System-One: envuelve [`TfidfCloser`].
///
/// Mapeo: `Done` ⇒ `Choice`, `NotDone` ⇒ `Score`, `Abstain` (o closer
/// deshabilitado / modelo vacío) ⇒ `Noul` abstinente. Puro `std`, sin I/O.
#[derive(Debug, Clone)]
pub struct TfidfSystemOne {
    closer: TfidfCloser,
    cfg: CloserConfig,
}

impl TfidfSystemOne {
    #[must_use]
    pub fn new(closer: TfidfCloser, cfg: CloserConfig) -> Self {
        Self { closer, cfg }
    }

    /// Baseline default: modelo vacío ⇒ siempre abstiene (fail-open).
    /// Existe para dejar el hueco sin calibrar nada.
    #[must_use]
    pub fn baseline() -> Self {
        Self {
            closer: TfidfCloser::fit(&[]),
            cfg: CloserConfig::default(),
        }
    }
}

impl SystemOne for TfidfSystemOne {
    fn decide(&self, req: &DecisionRequest) -> ScoredDecision {
        if !self.cfg.enabled {
            return ScoredDecision::noul("tfidf", "tfidf-closer");
        }
        if req.text.trim().is_empty() {
            return ScoredDecision::noul("tfidf", "tfidf-closer");
        }
        let v = self.closer.predict(&req.text, &self.cfg);
        match v.label {
            CloserLabel::Done => ScoredDecision {
                decision: Decision::Choice,
                source: "tfidf",
                model_id: "tfidf-closer",
                confidence: v.conf.clamp(0.0, 1.0),
                abstained: false,
            },
            CloserLabel::NotDone => ScoredDecision {
                decision: Decision::Score,
                source: "tfidf",
                model_id: "tfidf-closer",
                confidence: v.conf.clamp(0.0, 1.0),
                abstained: false,
            },
            CloserLabel::Abstain => ScoredDecision::noul("tfidf", "tfidf-closer"),
        }
    }
}

/// Estado para volcado en runmark (Lane B): `("systemone", "off" | "tfidf")`.
///
/// Bench (`KITTY_BENCH_MODE=1`) siempre devuelve `"off"`.
#[must_use]
pub fn systemone_status() -> (&'static str, &'static str) {
    ("systemone", crate::config::systemone_mode())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tfidf_closer::FIXED_TASK;

    fn enabled_cfg() -> CloserConfig {
        CloserConfig {
            enabled: true,
            ..CloserConfig::default()
        }
    }

    #[test]
    fn off_siempre_noul_abstinente_fail_open() {
        let s = OffSystemOne;
        for text in ["", "   ", "task complete all tests pass", "error failed"] {
            let d = s.decide(&DecisionRequest::new(text));
            assert_eq!(d.decision, Decision::Noul);
            assert!(d.abstained);
            assert_eq!(d.confidence, 0.0);
            assert_eq!(d.source, "off");
        }
    }

    #[test]
    fn tfidf_sin_modelo_abstiene_fail_open() {
        // Modelo vacío ⇒ predict Abstain ⇒ Noul, aunque esté enabled.
        let s = TfidfSystemOne::new(TfidfCloser::fit(&[]), enabled_cfg());
        let d = s.decide(&DecisionRequest::new("task complete all tests pass"));
        assert_eq!(d.decision, Decision::Noul);
        assert!(d.abstained);
        // Texto vacío también es fail-open, nunca pánico.
        let d = s.decide(&DecisionRequest::new(""));
        assert_eq!(d.decision, Decision::Noul);
        assert!(d.abstained);
    }

    #[test]
    fn tfidf_deshabilitado_abstiene_aunque_haya_modelo() {
        let train: Vec<(String, bool)> = FIXED_TASK
            .iter()
            .map(|(t, d)| (t.to_string(), *d))
            .collect();
        // Default = OFF en bench: baseline default abstiene.
        let s = TfidfSystemOne::new(TfidfCloser::fit(&train), CloserConfig::default());
        assert!(!CloserConfig::default().enabled);
        let d = s.decide(&DecisionRequest::new(FIXED_TASK[0].0));
        assert_eq!(d.decision, Decision::Noul);
        assert!(d.abstained);
    }

    #[test]
    fn tfidf_habilitado_no_abstiene_en_done_claro() {
        let train: Vec<(String, bool)> = FIXED_TASK
            .iter()
            .map(|(t, d)| (t.to_string(), *d))
            .collect();
        let loose = CloserConfig {
            enabled: true,
            min_conf: 0.1,
            min_margin: 0.01,
        };
        let s = TfidfSystemOne::new(TfidfCloser::fit(&train), loose);
        let d = s.decide(&DecisionRequest::new(FIXED_TASK[0].0));
        assert_eq!(d.decision, Decision::Choice);
        assert!(!d.abstained);
        assert_eq!(d.source, "tfidf");
        assert_eq!(d.model_id, "tfidf-closer");
        assert!((0.0..=1.0).contains(&d.confidence));
    }

    #[test]
    fn bench_siempre_off() {
        // Núcleo puro: bench fuerza off aunque el flag pida tfidf.
        assert_eq!(
            crate::config::systemone_mode_with(Some("tfidf"), true),
            "off"
        );
        // El volcado expone la clave que Lane B escribe en runmark.
        let (k, _) = systemone_status();
        assert_eq!(k, "systemone");
    }

    #[test]
    fn laya_sigue_held_sin_calibrar() {
        // 0/200 etiquetas: el hueco existe pero no se calibra en este lane.
        assert_eq!(
            crate::laya_classifier::fit(&[]),
            crate::laya_classifier::FitOutcome::HeldDatos { have: 0, need: 200 }
        );
    }
}
