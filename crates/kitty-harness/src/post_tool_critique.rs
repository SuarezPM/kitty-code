//! W6 critique textual post-tool (plan híbrido 2026-10-07, HU-3/W6).
//!
//! Tras un tool-result, el agente PUEDE pedir al SLM una crítica textual
//! de una línea (qué salió mal / qué falta) antes del siguiente tool-call.
//!
//! COSTE DOCUMENTADO: **+1 roundtrip SLM por tool-call criticado** (una
//! inferencia extra completa). Por eso es SOLO opt-in con presupuesto:
//! [`CritiqueConfig::enabled`] = `false` por defecto y
//! [`CritiqueBudget`] acota los roundtrips extra por sesión. Sin presupuesto
//! restante ⇒ no se critica (fail-closed en coste, fail-open en loop).

/// Config del critique. Todo OFF por defecto.
#[derive(Debug, Clone, Copy)]
pub struct CritiqueConfig {
    /// Opt-in explícito. `false` ⇒ el loop jamás pide críticas.
    pub enabled: bool,
    /// Máx. roundtrips extra por sesión (cada crítica = +1 roundtrip).
    pub max_extra_roundtrips: usize,
    /// Solo criticar tool-calls fallidos (defecto: sí — el éxito no paga +1).
    pub only_on_failure: bool,
}

impl Default for CritiqueConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_extra_roundtrips: 2,
            only_on_failure: true,
        }
    }
}

/// Presupuesto consumido de críticas en la sesión.
#[derive(Debug, Clone, Copy, Default)]
pub struct CritiqueBudget {
    pub spent: usize,
}

impl CritiqueBudget {
    /// ¿Cabe una crítica más? Pura y testeable.
    pub fn should_critique(&self, cfg: &CritiqueConfig, tool_succeeded: bool) -> bool {
        if !cfg.enabled {
            return false;
        }
        if cfg.only_on_failure && tool_succeeded {
            return false;
        }
        self.spent < cfg.max_extra_roundtrips
    }

    /// Registra una crítica consumida (llamar SOLO tras gastar el roundtrip).
    pub fn spend(&mut self) {
        self.spent = self.spent.saturating_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn off_por_defecto_aunque_falle_y_haya_presupuesto() {
        let b = CritiqueBudget::default();
        assert!(!b.should_critique(&CritiqueConfig::default(), false));
    }

    #[test]
    fn exito_no_gasta_roundtrip_por_defecto() {
        let cfg = CritiqueConfig {
            enabled: true,
            ..CritiqueConfig::default()
        };
        let b = CritiqueBudget::default();
        assert!(!b.should_critique(&cfg, true));
        assert!(b.should_critique(&cfg, false));
    }

    #[test]
    fn presupuesto_acotado_fail_closed_en_coste() {
        let cfg = CritiqueConfig {
            enabled: true,
            max_extra_roundtrips: 1,
            only_on_failure: false,
        };
        let mut b = CritiqueBudget::default();
        assert!(b.should_critique(&cfg, true));
        b.spend();
        assert!(!b.should_critique(&cfg, true));
    }
}
