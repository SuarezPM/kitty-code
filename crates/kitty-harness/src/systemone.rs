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
//!   Sin manifiestos: los manifiestos no derivan `(confidence, correct)`,
//!   así que Laya nunca se calibra en este lane.
//! - Selección única: [`select_kind`] (pura) + [`systemone_status_for`];
//!   [`systemone_status`] sale de esa selección, no relee el env por
//!   un camino distinto.

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
    /// Id del modelo (`"none"` | `"tfidf-closer"` | `"laya-held"`).
    /// Laya sigue HELD en este lane: sin [`FitOutcome::Fitted`](crate::laya_classifier::FitOutcome)
    /// no hay id calibrado que exponer.
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
/// Puro y síncrono por diseño (rust-router/m04): `decide` no hace I/O,
/// no bloquea y nunca falla (sin evidencia ⇒ Noul abstinente).
pub trait SystemOne: Send + Sync {
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

/// Selector System-One (API factory de este lane).
///
/// - `Off`: siempre Noul (`none`).
/// - `Tfidf`: baseline TF-IDF cuando se aporta modelo.
/// - `Laya`: hueco HELD; solo un
///   [`FitOutcome::Fitted`](crate::laya_classifier::FitOutcome) lo habilitaría.
///   Sin evidencia ⇒ held (`laya-held`), fail-open.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SystemOneKind {
    Off,
    Tfidf,
    Laya,
}

impl SystemOneKind {
    /// Mapeo crudo `KITTY_SYSTEMONE`: `"tfidf"` ⇒ [`SystemOneKind::Tfidf`],
    /// `"laya"` ⇒ [`SystemOneKind::Laya`]; ausente o desconocido ⇒
    /// [`SystemOneKind::Off`] (fail-open).
    #[must_use]
    pub fn from_raw(raw: Option<&str>) -> Self {
        match raw {
            Some("tfidf") => Self::Tfidf,
            Some("laya") => Self::Laya,
            _ => Self::Off,
        }
    }
}

/// Selección pura: bench siempre fuerza [`SystemOneKind::Off`]
/// (los gates nunca puntúan en bench); si no, [`SystemOneKind::from_raw`].
#[must_use]
pub fn select_kind(raw: Option<&str>, bench: bool) -> SystemOneKind {
    if bench {
        SystemOneKind::Off
    } else {
        SystemOneKind::from_raw(raw)
    }
}

/// Estado para volcado en runmark (Lane B) desde la selección:
/// `("systemone", "off" | "tfidf" | "laya")`. No relee el env: deriva
/// del [`SystemOneKind`] ya seleccionado.
#[must_use]
pub fn systemone_status_for(kind: &SystemOneKind) -> (&'static str, &'static str) {
    match kind {
        SystemOneKind::Off => ("systemone", "off"),
        SystemOneKind::Tfidf => ("systemone", "tfidf"),
        SystemOneKind::Laya => ("systemone", "laya"),
    }
}

/// Estado para volcado en runmark (Lane B).
///
/// Sale de la misma selección ([`select_kind`] + [`systemone_status_for`]):
/// una sola lectura de `KITTY_SYSTEMONE` / `KITTY_BENCH_MODE`, sin segundo
/// camino que relea el env. Bench (`KITTY_BENCH_MODE=1`) siempre `"off"`.
#[must_use]
pub fn systemone_status() -> (&'static str, &'static str) {
    let raw = std::env::var(crate::config::ENV_SYSTEMONE).ok();
    let kind = select_kind(raw.as_deref(), crate::config::bench_mode());
    systemone_status_for(&kind)
}

/// Laya HELD: sin [`FitOutcome::Fitted`](crate::laya_classifier::FitOutcome)
/// no hay modelo calibrado; siempre Noul fail-open con
/// `source = "laya"`, `model_id = "laya-held"`. Puro, sin I/O.
///
/// El modelo de decisión vive fuera del trait: este tipo solo marca el
/// hueco. Una futura calibración aportaría el `CalibratedLaya` por
/// [`from_fit`](LayaSystemOne::from_fit); en este lane ese camino sigue
/// HELD (0/200).
#[derive(Debug, Clone, Copy, Default)]
pub struct LayaSystemOne;

impl LayaSystemOne {
    /// Hueco held: Noul `("laya", "laya-held")`.
    #[must_use]
    pub fn held() -> Self {
        Self
    }

    /// Gate HeldDatos: solo
    /// [`FitOutcome::Fitted`](crate::laya_classifier::FitOutcome) habilita
    /// Laya; [`FitOutcome::HeldDatos`](crate::laya_classifier::FitOutcome)
    /// ⇒ [`OffSystemOne`] (Noul `("off", "none")`). Puro, sin I/O.
    #[must_use]
    pub fn from_fit(
        outcome: crate::laya_classifier::FitOutcome,
    ) -> Box<dyn SystemOne + Send + Sync> {
        if crate::laya_classifier::is_fitted(&outcome) {
            Box::new(Self)
        } else {
            Box::new(OffSystemOne)
        }
    }
}

impl SystemOne for LayaSystemOne {
    fn decide(&self, _req: &DecisionRequest) -> ScoredDecision {
        ScoredDecision::noul("laya", "laya-held")
    }
}

/// Núcleo puro de la factory (testeable sin tocar el env):
/// bench fuerza Off aunque `kind` pida otra cosa.
fn build_with_bench(
    kind: SystemOneKind,
    tfidf: Option<TfidfCloser>,
    bench: bool,
) -> Box<dyn SystemOne + Send + Sync> {
    if bench {
        return Box::new(OffSystemOne);
    }
    match kind {
        SystemOneKind::Off => Box::new(OffSystemOne),
        SystemOneKind::Tfidf => match tfidf {
            Some(closer) => Box::new(TfidfSystemOne::new(
                closer,
                CloserConfig {
                    enabled: true,
                    ..CloserConfig::default()
                },
            )),
            // Sin modelo no hay evidencia: fail-open a Off/Noul.
            None => Box::new(OffSystemOne),
        },
        // Sin FitOutcome::Fitted en este lane ⇒ held (Noul laya-held).
        SystemOneKind::Laya => Box::new(LayaSystemOne::held()),
    }
}

/// Factory System-One (API exacta de este lane).
///
/// - Bench (`KITTY_BENCH_MODE=1`) fuerza [`OffSystemOne`].
/// - Desconocido ya viene como [`SystemOneKind::Off`] vía
///   [`SystemOneKind::from_raw`] ⇒ Off/Noul.
/// - `Tfidf` sin modelo (`None`) ⇒ Off/Noul (fail-open).
/// - `Laya` sin [`FitOutcome::Fitted`](crate::laya_classifier::FitOutcome)
///   ⇒ held: Noul `("laya", "laya-held")`.
/// - `model_id`: `none` (off) | `tfidf-closer` | `laya-held`.
///
/// Pura salvo la lectura de bench-mode; `decide` sigue puro/síncrono/sin I/O.
#[must_use]
pub fn build_systemone(
    kind: SystemOneKind,
    tfidf: Option<TfidfCloser>,
) -> Box<dyn SystemOne + Send + Sync> {
    build_with_bench(kind, tfidf, crate::config::bench_mode())
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

    #[test]
    fn seleccion_desconocido_es_off() {
        assert_eq!(SystemOneKind::from_raw(None), SystemOneKind::Off);
        assert_eq!(SystemOneKind::from_raw(Some("off")), SystemOneKind::Off);
        assert_eq!(
            SystemOneKind::from_raw(Some("wire-v1")),
            SystemOneKind::Off
        );
        assert_eq!(
            SystemOneKind::from_raw(Some("tfidf")),
            SystemOneKind::Tfidf
        );
        assert_eq!(SystemOneKind::from_raw(Some("laya")), SystemOneKind::Laya);
    }

    #[test]
    fn seleccion_bench_fuerza_off() {
        // Núcleo puro: bench fuerza Off aunque el flag pida tfidf/laya.
        assert_eq!(
            select_kind(Some("tfidf"), true),
            SystemOneKind::Off
        );
        assert_eq!(select_kind(Some("laya"), true), SystemOneKind::Off);
        assert_eq!(
            select_kind(Some("tfidf"), false),
            SystemOneKind::Tfidf
        );
        assert_eq!(select_kind(Some("laya"), false), SystemOneKind::Laya);
        assert_eq!(select_kind(None, false), SystemOneKind::Off);
    }

    #[test]
    fn status_sale_de_la_seleccion() {
        assert_eq!(
            systemone_status_for(&SystemOneKind::Off),
            ("systemone", "off")
        );
        assert_eq!(
            systemone_status_for(&SystemOneKind::Tfidf),
            ("systemone", "tfidf")
        );
        assert_eq!(
            systemone_status_for(&SystemOneKind::Laya),
            ("systemone", "laya")
        );
        // Bench ya viene forzado como Off en la selección.
        assert_eq!(
            systemone_status_for(&select_kind(Some("tfidf"), true)),
            ("systemone", "off")
        );
    }

    #[test]
    fn factory_off_es_noul_none() {
        let s = build_systemone(SystemOneKind::Off, None);
        let d = s.decide(&DecisionRequest::new("task complete all tests pass"));
        assert_eq!(d.decision, Decision::Noul);
        assert!(d.abstained);
        assert_eq!(d.source, "off");
        assert_eq!(d.model_id, "none");
    }

    #[test]
    fn factory_tfidf_con_modelo_clasifica() {
        // Núcleo puro con bench=false: hermético aunque el proceso corra con
        // `KITTY_BENCH_MODE=1` (build_systemone lee el env y forzaría Off).
        let train: Vec<(String, bool)> = FIXED_TASK
            .iter()
            .map(|(t, d)| (t.to_string(), *d))
            .collect();
        let s = build_with_bench(
            SystemOneKind::Tfidf,
            Some(TfidfCloser::fit(&train)),
            false,
        );
        let d = s.decide(&DecisionRequest::new(FIXED_TASK[0].0));
        assert_eq!(d.source, "tfidf");
        assert_eq!(d.model_id, "tfidf-closer");
        // Umbrales default (0.30/0.05): el Done claro de la fixture supera.
        assert_eq!(d.decision, Decision::Choice);
        assert!(!d.abstained);
    }

    #[test]
    fn factory_tfidf_sin_modelo_es_off() {
        let s = build_systemone(SystemOneKind::Tfidf, None);
        let d = s.decide(&DecisionRequest::new("task complete all tests pass"));
        assert_eq!(d.decision, Decision::Noul);
        assert!(d.abstained);
        assert_eq!(d.model_id, "none");
    }

    #[test]
    fn factory_laya_held_es_noul() {
        // Sin FitOutcome::Fitted en este lane ⇒ held laya-held.
        // Núcleo puro con bench=false: hermético con `KITTY_BENCH_MODE=1`.
        let s = build_with_bench(SystemOneKind::Laya, None, false);
        let d = s.decide(&DecisionRequest::new("task complete all tests pass"));
        assert_eq!(d.decision, Decision::Noul);
        assert!(d.abstained);
        assert_eq!(d.source, "laya");
        assert_eq!(d.model_id, "laya-held");
    }

    #[test]
    fn factory_bench_fuerza_off() {
        // Núcleo puro con bench=true: tfidf/laya colapsan a Off/Noul.
        let train: Vec<(String, bool)> = FIXED_TASK
            .iter()
            .map(|(t, d)| (t.to_string(), *d))
            .collect();
        let s = build_with_bench(
            SystemOneKind::Tfidf,
            Some(TfidfCloser::fit(&train)),
            true,
        );
        let d = s.decide(&DecisionRequest::new(FIXED_TASK[0].0));
        assert_eq!(d.decision, Decision::Noul);
        assert_eq!(d.model_id, "none");
        let s = build_with_bench(SystemOneKind::Laya, None, true);
        let d = s.decide(&DecisionRequest::new("anything"));
        assert_eq!(d.decision, Decision::Noul);
        assert_eq!(d.model_id, "none");
    }

    #[test]
    fn laya_gate_held_es_off() {
        // Gate HeldDatos: sin Fitted ⇒ Off/Noul.
        let held = crate::laya_classifier::fit(&[]);
        assert!(!crate::laya_classifier::is_fitted(&held));
        let s = LayaSystemOne::from_fit(held);
        let d = s.decide(&DecisionRequest::new("task complete"));
        assert_eq!(d.decision, Decision::Noul);
        assert_eq!(d.source, "off");
        assert_eq!(d.model_id, "none");
    }
}
