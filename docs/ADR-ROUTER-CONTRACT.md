# ADR: ToolRouter / Contract stay wired (no mechanical cleanup)

- **Status:** accepted.
- **Context:** `ToolRouter` (`router.rs`, afirmación interna de subconjunto
  por perfil) y `Contract` (`contract.rs`, Definition-of-Done + done-guard)
  viven cableados en `agent/mod.rs` (`router: ToolRouter` campo,
  `contract: Option<Contract>`, `check_done_guard` en el path de
  `process_response`). Parecen "morralla" solo si se leen como helpers
  sueltos; son invariantes de producto (orden de prompt, DoD honesto).
- **Decision:** retirada SOLO con unwire explícito (quitar campo + call
  sites + tests consumidores a la vez, con justificación). Prohibida la
  limpieza mecánica por "código muerto aparente".
- **Debt (no trivial, documentado, no implementado):**
  - `cargo nextest` como runner (0.9.146 actual): cambia binario runner y
    flags de CI; evaluar fuera de este gate.
  - `llama-vulkan`: sin Vulkan SDK en ubuntu-latest ni en el contenedor
    CUDA del job `cuda` (T5.7: fuera de alcance hasta runner con SDK).
  - Mutants en advisory nocturno (ver `mutants.yml`); promover a gate
    solo con baseline de mutantes muertos medida.
