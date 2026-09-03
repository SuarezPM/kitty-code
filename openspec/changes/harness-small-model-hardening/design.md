## Context

Kitty Code es un harness Rust para modelos locales pequeños (1B-35B parámetros). El loop actual tiene perfiles, routing, guardas, sandbox, compaction, truncación y resultados JSON de benchmark. Las mejoras provienen de analizar patrones exitosos en agentes TypeScript y adaptarlos al ecosistema Rust con enfoque en Fiabilidad de modelos pequeños y Completitud de benchmarks.

## Goals / Non-Goals

**Goals:**
- Reducir fallos de tool call en modelos 9B mediante JSON repair automático
- Preservar contexto mejor con compaction estructurada
- Reducir tamaño de prompt mediante progressive skill disclosure y tool-gated sections
- Habilitar comparación con leaderboard público de Aider Polyglot
- Detectar y recuperar de overflow de contexto automáticamente

**Non-Goals:**
- No reimplementar graders de Aider (usar test runners existentes)
- No añadir extensiones tipo plugin (futuro)
- No modificar el formato de resultados JSON existente
- No añadir dependencias TypeScript/JavaScript al proyecto

## Decisions

### JSON Repair: Rust-native implementation
Implementar `repair_json()` y `coerce_with_schema()` en Rust puro usando `serde_json` y `regex`. No depende de librerías externas de repair. El pipeline es: intentar parse → repair → coerce → re-validar.

### Structured Compaction: Template-based
Usar formato template (Goal/Constraints/Progress/Decisions/Next Steps) en lugar de LLM-powered summarization. Razón: determinismo, sin costo de inferencia adicional, reproducible. La estructura es fija; el contenido viene de los mensajes recientes.

### Progressive Skills: XML in system prompt
Los skills se listan como XML en el system prompt con name+description+location. El modelo usa Read para cargar contenido completo. Esto reduce el prompt ~40% vs incluir todo el contenido.

### Aider Polyglot: Standalone loader
El loader carga ejercicios desde `bench/official/polyglot-benchmark/`, extrae prompts de `.meta/`, y ejecuta test runners nativos (pytest, go test, cargo test). No integra con el CLI de Aider. Resultados en formato comparable al leaderboard.

### Multi-tier Retry: Extend existing F3.3
Extender el `is_transient()` existente con clasificación de errores de 3 tiers. Tier 1: HTTP provider retry. Tier 2: Application retry con 60+ patrones. Tier 3: Overflow recovery con compact + retry once.

## Risks / Trade-offs

- [JSON repair overhead] → El repair pipeline añade ~1-2ms por tool call. Aceptable para la mejora en fiabilidad.
- [Compaction determinística] → Las summaries template pueden perder matices vs LLM-powered. Aceptable para el beneficio de determinismo y costo cero.
- [Aider Polyglot sin Docker] → Ejecutar código del modelo sin sandbox. Aceptable para evaluación local; documentar riesgo.
- [60+ error patterns] → Mantener patrones de error actualizados. Mitigar con tests que validen cada patrón.
