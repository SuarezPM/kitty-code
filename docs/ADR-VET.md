# ADR: cargo-vet — NO (2026-09-24)

## Decisión
No adoptar `cargo-vet` en este ciclo. Se mantiene `cargo-deny + cargo-audit`
como gate de supply-chain (ya verdes en CI: `deny`, `audit`, `sbom`).

## Por qué
- `cargo-deny` ya cubre licencias/advisories/sources con gate en CI.
- `cargo-audit` cubre RustSec en cada PR + sweep semanal.
- SBOM CycloneDX por artefacto + cosign/SLSA (W10) cubren provenance.
- `cargo-vet` exige importar auditorías de terceros y mantener
  `supply-chain/` auditado por humano; sin owner dedicado deriva en
  `vet --locked` rojo o en `exemptions` vacías que no aportan señal.

## Revisit
Si aparece dependencia con `unsound`/maintainer único crítico, reabrir con
alcance acotado (`cargo vet init` + 1 crate piloto). T10.3 (deny gap
0BSD/Zlib/MPL-2.0 + paste) resuelto vía excepciones documentadas en
`deny.toml`, sin bajar el gate.

## T5.2 nextest — documentado, no swap
El swap `cargo test` → `cargo nextest run` es trivial en YAML pero
`.github/` está fuera de ownership P3 (PROHIBIDO). Se deja como follow-up:
instalar nextest (`taiki-e/install-nextest-action`), cambiar el step
`cargo test -p kitty-*` por `cargo nextest run -p kitty-*`, y archivar
`.config/nextest.toml` (perfil `slow-timeout`, retries 0). Sin cambio CI aquí.
