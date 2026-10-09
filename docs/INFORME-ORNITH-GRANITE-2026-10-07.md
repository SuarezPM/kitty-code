# INFORME — Ornith-1.5-9B y Granite-4 vía Kitty Code (2026-10-07)

## 1. Modelos: descarga, análisis e investigación

### Ornith-1.5-9B (link pedido: `adityak74/Ornith-1.5-9B-MLX-distil-oQ4`)
- Estructura exacta: `.gitattributes` + `README.md` + `chat_template.jinja` + `config.json` + `model.safetensors` (5.28GB oQ4) + `model.safetensors.index.json` + `tokenizer.json` (20MB). **Cero `.gguf`: MLX-only**, formato Apple Silicon. En Linux/NVIDIA no lo cargan ni kitty-infer ni Ollama. **Descargar 5.28GB habría sido inútil: no se descargó** (verificado por estructura, sin gastar un byte).
- GGUF oficial SÍ existe: `ornith-ai/Ornith-1.5-9B-GGUF` (BF16 18.4GB, Q4_K_M 5.78GB, Q5/Q6/Q8 + mmproj visión) + tokenizer 12.8MB en repo base. Arch real: **`qwen35`** (no `qwen3_5`), híbrido 32L (linear/full attention, GQA 16/4, vocab 248320).
- Vía Ollama: sin nombre oficial (el tag local `ornith-1.5:9b` 6.6GB es `Modelfile FROM` privado).

### Granite 4.x (docs `ibm.com/granite/docs/models/granite4-2` + `ollama pull ibm/granite4`)
- Ollama `granite4` = familia Granite 4.0 instruct (Apache-2.0): tags `3b` (2.1GB, 128K), `1b`, `350m`, variantes `-h` hybrid-mamba, micro/tiny/small. Docs granite4-2: densas razonadoras 3B/8B/30B, CoT con `enable_thinking/low_effort`, ctx 128K (30B→512K), `temp 1.0/top_p 0.95`; 4.0 híbrida Mamba-2/transformer 9:1 + MoE en tiny/small.
- GGUF oficial: `ibm-granite/granite-4.0-h-micro-GGUF` (Q4_K_M 1.94GB … Q8 3.40GB, BF16 6.39GB) + espejos bartowski/unsloth. Tokenizer dentro del GGUF.
- OJO desambiguación: `k2_horizon.rs` del engine es **IFM K2-Horizon-3.7B** (otro modelo), NO Kimi-K2 MoE de 1T (ese necesitaría 128GB+ RAM: fuera de este box).

### Compatibilidad kitty-infer (motor propio)
- Ornith `qwen35`: arch de primera clase en registry (QWEN35_LOADERS, torre híbrida, bindings) — el mapeo funciona (427 tensores, 32 bloques).
- Granite `granitehybrid`: arch de primera clase (registry, loader MoEHybrid, bindings).
- PERO ambos mueren en ejecución con el mismo error determinista: **`hybrid sequence 0/1 has no recurrent state slot`** — el engine mapea híbridos pero no ejecuta su estado recurrente. Limitación común verificada, no flake.

## 2. Ejecución (serie VRAM, time-boxes, sin commit)

| Modelo | kitty-infer | Ollama |
|---|---|---|
| ornith-1.5:9b | HELD (symlink→blob 5.63GB, Connected, FAIL 4.7s híbrido) | **PASS 1/1** (wall 59s, GATE≥50%) |
| granite4:3b | HELD (GGUF 1.94GB verificado 506T+sha, Connected, FAIL 1.7s mismo error) | **PASS 1/1** (wall 8s, GATE≥50%) |

Mini-test = `bench tb2 --limit 1 --no-memory` (tb2-bash-count), exits sin pipe, serie con `ollama stop` + `nvidia-smi`, manifiestos honestos (smokes como `measurement-record`, maratón de infer como pre-reg de carga — GO-carga, NO-ejercicio).

## 3. Veredicto maratón
- **Ornith-1.5 vía Ollama: GO** (celda 60s vigente). Vía infer: NO-GO hasta fix de `recurrent_state_idx` en `HybridCacheManager` + rebuild CUDA.
- **Granite4:3b vía Ollama: GO** (rápido, 8s). Vía infer: NO-GO (misma causa).
- **MLX-link: descartado con prueba** (formato, no opinión).
- Deuda: re-smokes infer con logs persistidos tras el fix; N de polyglot por fijar; celdas infer con wall medido cuando haya PASS.

Puerta al fix real: asignar `recurrent_state_idx` a hybrid sequences en el engine (trabajo de engine, estimado medio-día + rebuild CUDA + re-smoke).

## 4. Addendum — fix aplicado y re-smokes (programa hybrid-plugplay-fix, Gates 1-3 PASS)
- Fix: `acquire` Zeroed en `set_none_cache` + antes del restore (`add_request.rs`), gemelo de `release`; suite engine 1074/3 ajenos; CUDA HELD (cudarc vs toolkit 13.4).
- Resultado: slot-error **desaparecido en ambos**; Granite SMOKE-OK turno real (1244s CPU); Ornith carga (13s, 32/32 CPU) pero **tok/s~0** (CPU 547%, RSS 19GB — cómputo saturado, no deadlock): turnos 9B-híbridos en CPU inviables → dueño `maratón-GPU`.
- Tabla `docs/SUPPORTED-MODELS.md` (27 archs) + early-check en `kitty_infer.rs` (fail-open header ilegible) + conversores documentados.
- Evidencia: `benches/ab/manifests/evidence-p3b/` (6 piezas + SHA256SUMS) y `evidence-infer-mk/`.
- Veredicto actualizado: infer GO-carga ambos; GO-ejercicio solo GPU; Ollama GO ambos. Sin commit (a tu orden).
