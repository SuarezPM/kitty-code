# SUPPORTED-MODELS — kitty-infer (embedded GGUF)

Fuente única: `general.architecture` del header GGUF vs
`CanonicalGgufArchitecture` en
`engine/mistralrs-core/src/gguf/normal_registry.rs` (27 archs).
El early-check (`crates/kitty-runtime/src/kitty_infer.rs`: `check_gguf_arch`,
llamado desde `from_path` + `health_check`) acepta exactamente esa lista;
arch desconocido → bail accionable (arch + esta tabla + "portar torre o
usar Ollama"). Header ilegible → pasa (el engine falla honesto en carga).

Embebido = SOLO GGUF (`from_path` rechaza no-`.gguf`). Safetensors nunca
carga en kitty-infer; la columna ST dice la vía (siempre Ollama/HF).

## Tabla por familia

| arch (`general.architecture`) | GGUF embebido | ST embebido | Estado | Nota |
|---|---|---|---|---|
| `llama` | ✓ | ✗ (Ollama ✓) | registrado, smoke pendiente | loader Llama/Mistral/Mixtral según inventario |
| `qwen2` | ✓ | ✗ (Ollama ✓) | testeado-parcial | humo F1: `qwen2.5-coder*` fuerza F32 en CUDA sm_75 (NaN F16 aislado) |
| `qwen3` | ✓ | ✗ (Ollama ✓) | testeado | Qwen3-4B humo-verificado (turnos OK, paridad Gate 2 verde 2026-09-22) |
| `qwen3moe` | ✓ | ✗ (Ollama ✓) | registrado, smoke pendiente | — |
| `qwen3next` | ✓ | ✗ (Ollama ✓) | registrado, smoke pendiente | híbrido SSM; también resuelve `qwen35moe` |
| `qwen35` | ✓ | ✗ (Ollama ✓) | HELD-diagnóstico | P3b 2026-10-07: slot-error recurrente GONE (ausente en logs) pero Ornith-1.5-9B TIMEOUT 1501s sin primer token en CPU 32/32; diagnóstico en `benches/ab/manifests/evidence-p3b/`; vía Ollama GO (ornith-1.5:9b PASS 1/1) |
| `qwen35moe` | ✓ | ✗ (Ollama ✓) | registrado, smoke pendiente | resuelve al loader Qwen3Next |
| `k2-horizon` (`k2horizon` alias) | ✓ | ✗ (Ollama ✓) | testeado-parcial | spill-12 estable sin OOM en 8GB, turno <10min pendiente; requiere sidecar `tokenizer.json` (+`KITTY_INFER_GPU_LAYERS` en >8GB) |
| `granite` | ✓ | ✗ (Ollama ✓) | HELD | mismo fallo híbrido-recurrente que `qwen35` (2026-10-07); vía Ollama GO |
| `granitemoe` | ✓ | ✗ (Ollama ✓) | HELD | ídem `granite` |
| `granitehybrid` | ✓ | ✗ (Ollama ✓) | smoke-OK, bench HELD | P3b 2026-10-07: granite-4.0-h-micro SMOKE-OK 1244s (slot-error GONE); bench tb2 --limit 1 FAIL 0/1 verdict `timeout` (cap 30s vs turno CPU ~1244s); vía Ollama GO (granite4:3b PASS 1/1) |
| `mistral3` | ✓ | ✗ (Ollama ✓) | registrado, smoke pendiente | — |
| `gemma` / `gemma2` | ✓ | ✗ (Ollama ✓) | registrado, smoke pendiente | — |
| `phi2` / `phi3` / `phimoe` | ✓ | ✗ (Ollama ✓) | registrado, smoke pendiente | — |
| `starcoder2` | ✓ | ✗ (Ollama ✓) | registrado, smoke pendiente | — |
| `deepseek2` | ✓ | ✗ (Ollama ✓) | registrado, smoke pendiente | DeepSeekV2/V3/GLM4MoeLite según identidad+tensores |
| `glm4` / `glm4moe` | ✓ | ✗ (Ollama ✓) | registrado, smoke pendiente | mRoPE (`rope.dimension_sections`) rechazado |
| `smollm3` | ✓ | ✗ (Ollama ✓) | registrado, smoke pendiente | — |
| `gpt-oss` | ✓ | ✗ (Ollama ✓) | registrado, smoke pendiente | — |
| `hunyuan-dense` / `hunyuan-moe` | ✓ | ✗ (Ollama ✓) | registrado, smoke pendiente | — |
| `lfm2` / `lfm2moe` | ✓ | ✗ (Ollama ✓) | registrado, smoke pendiente | shortconv |
| otros (fuera del registry) | ✗ | ✗ | no soportado | early-check bails: portar torre o usar Ollama |

Notas: MiniCPM5-2B-Q4_K_M.gguf humo-verificado (turnos OK) con sidecar
`tokenizer.json` obligatorio (sin perfil de conversión standalone); su arch
la declara su propio header (el early-check la valida, no este doc).
`general.architecture` es case-insensitive en el check.

## Conversores a GGUF / Ollama

HF safetensors → GGUF (vía llama.cpp, perfil oficial):

```bash
python llama.cpp/convert_hf_to_gguf.py /ruta/al/repo-hf \
  --outfile modelo.gguf --outtype q4_k_m
```

`--outtype`: `q4_k_m` (uso general 8GB), `q8_0`/`f16` (máxima fidelidad),
`q5_k_m`/`q6_k` intermedios. Verificar con `general.architecture` del
header que cae en la tabla de arriba antes del smoke.

GGUF local → Ollama:

```bash
# Modelfile
# FROM /ruta/absoluta/modelo.gguf
ollama create mi-modelo -f Modelfile
```

`hf2ollama`: importa un repo HF (safetensors) a Ollama sin pasar por GGUF;
útil cuando el arch no está en la tabla (no soportado embebido pero GO
vía Ollama, ej. `qwen35`/`granitehybrid` hoy).

MLX → GGUF: NO hay conversor oficial (MLX es Apple-Silicon-only; ej.
`Ornith-1.5-9B-MLX` no cargable en Linux/NVIDIA ni vía kitty-infer ni
vía Ollama). Los scripts comunitarios MLX→GGUF no son oficiales:
úsalos solo si verificas tensores/hash y el smoke; preferir el GGUF
oficial del publicador cuando exista.
