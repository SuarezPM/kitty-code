# INFORME EJECUCIÓN HÍBRIDO W0-W8 (2026-10-07, baseline vs final)

## Benchmark antes/después (protocolo idéntico, mismo HEAD `0a1fe1b`)

| Métrica | Baseline | Final | Delta / veredicto |
|---|---|---|---|
| Build `kitty-cli` | 5s | 3s | −2s, resolución gruesa, sin juicio |
| Tests 4 crates | 772 pass / 33s | 813 pass / 27s | +41 tests nuevos (TF-IDF, compact, settled, critique, espejos); −6s wall |
| fmt / clippy | 0 / 0 | 0 / 0 | idéntico |
| parser_clean | 1.9053µs | 1.9653µs | +3.15% (ruido) |
| parser_mixed | 1.2963µs | 1.3277µs | +2.42% (ruido) |
| tooloop_50t | 99.91ms | 102.09ms | +2.18% (criterion: within noise) |
| Smoke tb2-l1 | 33s PASS | 30s PASS | −9.9% una celda, sin juicio |

Manifiestos: `manifest-20261007T192215Z-baseline-hibrido.json` → `manifest-20261007T203113Z-final-hibrido.json` (mismo schema + comparison/W7/W8).

## W7/W8 (HELDs honestos)
- W7 sleep-QR: HELD-diseño (sin task_ids, sin runner `bench holdout`, sin gate QR; knowledge.rs = inyección directa).
- W8 recalibración: HELD-datos (have 0/200; 2089 TurnEvents sin etiqueta, 0 episodios persistidos).

## Veredicto
0 regresiones medibles; +41 tests; infraestructura lista (router cableado invariante, spike logprobs, TF-IDF, settled, compact-QR, manifests). Sin commit (a tu orden). Puerta a maratón y a 8.2 intacta.
