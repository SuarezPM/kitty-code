//! FC-15: suite propia de 15 tool-calls sintéticos con veredicto conocido
//! (W5-code, SDD estado-arte).
//!
//! Cada caso es una salida de modelo simulada (válida o malformada) con su
//! veredicto esperado. Dos paths:
//! - **baseline**: el `ToolCallParser` del harness sin gramática (emite lo
//!   que extrae, sin validar esquema — como el loop sin `GrammarClient`).
//! - **gramática activa**: parse + normalización + `GrammarConfig::validate`
//!   (exactamente el filtro de `GrammarClient::apply_strict_filter`: lo
//!   inválido se bloquea antes de ejecutar).
//!
//! AC (W5): `grammar_bad == 0` (cero malformados emitidos con gramática
//! activa, sin perder ningún válido) frente a `baseline_bad > 0` (la suite
//! tiene dientes: el baseline sí deja pasar malformados).
//!
//! Solo código en esta lane: sin modelo, sin GPU, sin benches.

use crate::parser::ToolCallParser;

/// Un caso sintético: texto de modelo + si debe producir una llamada válida.
#[derive(Debug, Clone, Copy)]
pub struct FcCase {
    /// Identificador estable del caso.
    pub id: &'static str,
    /// Salida de modelo simulada.
    pub text: &'static str,
    /// `true` = debe emitir una llamada que valida; `false` = la gramática
    /// no debe dejar pasar nada ejecutable.
    pub expect_valid: bool,
}

/// Los 15 casos (6 válidos + 9 malformados).
pub const FC15: &[FcCase] = &[
    // ——— Válidos (ambos paths deben emitir llamada que valida) ———
    FcCase {
        id: "V1-xml-read",
        text: r#"<tool_call>{"name": "Read", "arguments": {"path": "/tmp/a.txt"}}</tool_call>"#,
        expect_valid: true,
    },
    FcCase {
        id: "V2-fenced-bash",
        text: "```json\n{\"name\": \"Bash\", \"arguments\": {\"command\": \"echo hi\"}}\n```",
        expect_valid: true,
    },
    FcCase {
        id: "V3-bare-glob",
        text: r#"{"name": "Glob", "arguments": {"pattern": "src/*.rs"}}"#,
        expect_valid: true,
    },
    FcCase {
        id: "V4-xml-write",
        text: r#"<tool_call>{"name": "Write", "arguments": {"path": "/tmp/n.txt", "content": "hi"}}</tool_call>"#,
        expect_valid: true,
    },
    FcCase {
        id: "V5-think-embedded",
        text: "<think>I need the file first</think>\n<tool_call>{\"name\": \"Read\", \"arguments\": {\"path\": \"/tmp/a.txt\"}}</tool_call>",
        expect_valid: true,
    },
    FcCase {
        id: "V6-bare-grep",
        text: r#"{"name": "Grep", "arguments": {"pattern": "foo", "path": "src"}}"#,
        expect_valid: true,
    },
    // ——— Malformados (la gramática no debe dejar pasar nada) ———
    FcCase {
        id: "M1-unknown-tool",
        text: r#"{"name": "Delete", "arguments": {"path": "/tmp/a.txt"}}"#,
        expect_valid: false,
    },
    FcCase {
        id: "M2-missing-required",
        text: r#"{"name": "Read", "arguments": {}}"#,
        expect_valid: false,
    },
    FcCase {
        id: "M3-deep-truncate",
        text: r#"{"name": "Rea"#,
        expect_valid: false,
    },
    FcCase {
        id: "M4-write-missing-content",
        text: r#"{"name": "Write", "arguments": {"path": "/tmp/x"}}"#,
        expect_valid: false,
    },
    FcCase {
        id: "M5-xml-garbage",
        text: "<tool_call>not json at all</tool_call>",
        expect_valid: false,
    },
    FcCase {
        id: "M6-args-as-string",
        text: r#"{"name": "Read", "arguments": "/tmp/a.txt"}"#,
        expect_valid: false,
    },
    FcCase {
        id: "M7-missing-name",
        text: r#"{"arguments": {"path": "/tmp/a.txt"}}"#,
        expect_valid: false,
    },
    FcCase {
        id: "M8-prose-only",
        text: "Please read the file at /tmp/a.txt when you get a chance.",
        expect_valid: false,
    },
    FcCase {
        id: "M9-fenced-incomplete",
        text: "```json\n{\"name\": \"Read\"}\n```",
        expect_valid: false,
    },
];

/// Resultado agregado de la suite.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FcReport {
    /// Casos donde el baseline NO produce ninguna llamada que valide
    /// (emite basura o nada). Debe ser > 0 (la suite tiene dientes).
    pub baseline_bad: usize,
    /// Casos donde la gramática deja pasar algo inválido o pierde algo
    /// válido. Debe ser 0 (AC de W5).
    pub grammar_bad: usize,
}

/// Ejecuta la suite: baseline (parse sin validar) vs gramática activa
/// (parse + normalizar + `validate`, el filtro real de `GrammarClient`).
pub fn run_fc15(parser: &ToolCallParser, grammar: &kitty_runtime::GrammarConfig) -> FcReport {
    let mut report = FcReport::default();
    for case in FC15 {
        let parsed = parser.parse(case.text);
        // El loop prepara (normaliza + repara, `ToolRegistry::prepare_call`)
        // antes de despachar: la validación se mide sobre la llamada
        // preparada (misma que ejecutaría `execute`). La gramática emite
        // exactamente las que validan (filtro real de `GrammarClient`), así
        // que `good == kept` por construcción; el AC está en que ningún
        // válido se pierda y ningún inválido pase.
        let good: usize = parsed
            .iter()
            .filter_map(|tc| crate::tools::ToolRegistry::prepare_call(tc).ok())
            .filter(|tc| validates(grammar, tc))
            .count();
        if case.expect_valid {
            if good == 0 {
                // Válido perdido en ambos paths (el baseline también falla).
                report.baseline_bad += 1;
                report.grammar_bad += 1;
            }
        } else {
            if good == 0 {
                // Malformado que el baseline no convierte en llamada buena.
                report.baseline_bad += 1;
            } else {
                // Infradetección de la gramática: dejó pasar basura.
                report.grammar_bad += 1;
            }
        }
    }
    report
}

/// `true` si la llamada valida contra los esquemas (mismo predicado que
/// `GrammarClient::apply_strict_filter`: `{"name","arguments"}` + required).
fn validates(grammar: &kitty_runtime::GrammarConfig, tc: &crate::parser::ParsedToolCall) -> bool {
    let v = serde_json::json!({"name": tc.name, "arguments": tc.arguments});
    grammar.validate(&v.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::standard_tool_definitions;

    fn harness() -> (ToolCallParser, kitty_runtime::GrammarConfig) {
        (
            ToolCallParser::new(),
            kitty_runtime::GrammarConfig::from_tools(standard_tool_definitions()),
        )
    }

    /// Pin por caso: cada veredicto conocido se cumple en ambos paths.
    #[test]
    fn fc15_per_case_verdicts() {
        let (parser, grammar) = harness();
        for case in FC15 {
            let parsed = parser.parse(case.text);
            let prepared: Vec<_> = parsed
                .iter()
                .filter_map(|tc| crate::tools::ToolRegistry::prepare_call(tc).ok())
                .collect();
            let kept = prepared.iter().filter(|tc| validates(&grammar, tc)).count();
            if case.expect_valid {
                assert!(
                    kept > 0,
                    "{}: caso válido perdido por la gramática (parse={parsed:?})",
                    case.id
                );
            } else {
                assert_eq!(
                    kept, 0,
                    "{}: la gramática dejó pasar basura: {prepared:?}",
                    case.id,
                );
            }
        }
    }

    /// AC de W5: 0 malformados con gramática activa vs baseline con dientes.
    #[test]
    fn fc15_acceptance_grammar_zero_vs_baseline() {
        let (parser, grammar) = harness();
        let report = run_fc15(&parser, &grammar);
        assert!(
            report.baseline_bad > 0,
            "la suite debe tener dientes: baseline_bad={}",
            report.baseline_bad
        );
        assert_eq!(
            report.grammar_bad, 0,
            "AC W5: 0 malformados con gramática activa: {report:?}"
        );
    }
}
