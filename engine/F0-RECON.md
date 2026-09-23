# F0-RECON — kitty-infer fork reconnaissance (2026-09-20)

Baseline: upstream `d5ae0f1` (v0.9.3). License: **MIT** (Eric Buehler 2024) —
verified `LICENSE`, inherited unchanged, attribution in `NOTICE`.

## Environment (this box)

- GPU RTX 2060 SUPER 8 GB, compute capability **7.5 (Turing)**.
- CUDA toolkit **13.4** installed at `/opt/cuda` (`PATH`+`CUDA_HOME` needed).
- **sm_75 consequences (documented, not broken)**: flash-attn kernels in-tree
  are sm80+ → no FA on this box (plain CUDA attention path applies);
  FP8/NVFP4 tensor cores absent (Hopper+); FA3 absent. The Q4/Q8 + paged
  path — the whole 1–35B mission — applies fully.

## Workspace inventory (14 members, sizes on disk)

| Crate | Size | Verdict |
|---|---|---|
| `mistralrs-core` | 13M | KEEP (engine) — amputate inside (see below) |
| `mistralrs` (SDK/builders) | 600K | KEEP stripped (`text_model`, `gguf`, builders; cut diffusion/embedding/speech/multimodal/lora/xlora/anymoe/agent) |
| `mistralrs-cli` | 3.6M | KEEP stripped (`run`, `bench`, `tune`, `doctor`; cut `serve`, `ui`, server/sandbox/login/manage args) |
| `mistralrs-quant` | 6.8M | KEEP (ISQ/UQFF/GGUF/GPTQ/AWQ/FP8) |
| `mistralrs-paged-attn` | 1.7M | KEEP |
| `mistralrs-server-core` | 972K | **CUT** (serve/UI/Responses API — the harness is the server) |
| `mistralrs-flash-attn` | 564K | KEEP (sm80+; inert on sm_75, needed for portability) |
| `mistralrs-pyo3` | 308K | **CUT** (Python SDK) |
| `mistralrs-code-exec` | 148K | **CUT** (harness executes tools) |
| `mistralrs-mcp` | 124K | **CUT** (Kitty owns MCP) |
| `mistralrs-sandbox` | 84K | **CUT** (Kitty owns Landlock sandbox) |
| `mistralrs-vision` | 36K | **CUT** |
| `mistralrs-audio` | 16K | **CUT** |
| `mistralrs-macros` | 24K | KEEP iff core needs it |
| `mistralrs-metal-compile` | 20K | KEEP (macOS portability, compiles only on Apple) |

Inside `mistralrs-core/src`: CUT `vision_models/` (3.1M), `speech_models/`,
`diffusion_models/`, `embedding_models/`, `xlora_models/`, `lora/`,
`search/`, `distributed.rs`, `video_input.rs`, `engine/agentic_loop.rs`,
`engine/agentic_session.rs`, `engine/file_tools.rs`, `engine/tool_dispatch.rs`
(server-side agent loop — Kitty's harness owns this), `pipeline/` multimodal
loaders, `amoe.rs` (AnyMoE). KEEP `models/` text (888K: llama/mistral/mixtral/
qwen2-3/moE/phi/gemma/deepseek/gpt-oss/...), `gguf/`, `sampler.rs`,
`prefix_cacher.rs`, `kv_cache/`, `paged_attention`, `scheduler/`,
`speculative/`, `topology/`, `tuning.rs`, `reasoning_parsers/`, `tools/`
(text tool-calling + grammar), `attention/`, `mla/`, `moe/`.

## Feature matrix (what compiles where)

- `cuda` (candle-cuda + paged-attn/cuda + cudaforge kernel builds — needs
  nvcc, CUTLASS fetch at build time): **primary target, this box**.
- `flash-attn` (sm80+ kernels): builds, inert on sm_75. Kept for portability.
- `metal`/`accelerate`: Apple-only, kept (no cost on Linux).
- `mkl`: CPU BLAS option, kept.
- `nccl`: CUT with multi-GPU (single-GPU mission).
- `cudnn`/`cutile`: future F5, off in F1.
- `code-execution`: CUT with code-exec.

## Sept-2026 model coverage (verified 2026-09-20 via `ollama show`)

| Model | GGUF `general.architecture` | kitty-infer path | Status |
|---|---|---|---|
| ornith-1.5:9b (9B, 262K ctx, Q4_K_M) | `qwen35` | `Qwen3_5TextModel` + `build_qwen35_text_bindings` (salvaged) | ✅ text path |
| MiniCPM5-2B (2.5B, 128K, Q4_K_M) | `llama` | `LlamaLoader` (standard GQA Llama) | ✅ text path |
| Qwen3.5 dense / MoE (HF) | `Qwen3_5ForCausalLM` / Moe | salvaged text towers + mtp + packed-GDN | ✅ text path |
| Qwen3 / Qwen2.5 / Llama / Mistral / Gemma2 / Phi / DeepSeek / GPT-OSS / GLM4 / Granite / HunYuan / SmolLM3 / LFM2 | various | `models/` (untouched, all text) | ✅ inherited |
| K2-Horizon-3.7B | `k2_horizon` (custom `XllmForCausalLM`) | **none — manual port required (F5)** | ❌ blocked |
| Multimodal-only (Gemma3n/4, Llama4, Muse Glimmer, Qwen3-VL…) | various | honest refusal (`requires_multimodal_projector` / mmproj bail) | refused |
| Pre-2025 (Qwen2-VL and older) | — | purged from projector list; generic unsupported path | refused |

Note: ornith-1.5 ships a 456M clip projector — ignored by design (vision
requests are refused; text requests never touch it).

## Retained dormant (F1 audit, 2026-09-20)

Compiles but unreachable without the cut selection surfaces. Documented so a
future phase can excise without archaeology:

- `amoe/` mixin + `lora/` + `xlora_models/` + dynamic-LoRA plumbing
  (`GGUFLoader.with_dynamic_lora`, `validate_native_dynamic_lora`): no
  `ModelSelected`/`TomlModelSelected` variant constructs them; no CLI/SDK
  builder offers them.
- `matformer.rs` + config fields: CSV-gated, never passed in text paths.
- `ModelCategory::{Multimodal, Diffusion, Audio, Speech, Embedding}` variants:
  never constructed (all pipelines report `Text`).
- `ToolCallbacks` registry: write-only (keys read for routing, never
  dispatched — no server/shell/code-exec).
- `TuneKind::{Multimodal, Embedding}`: `infer_kind` refuses before reaching them.
- `ForwardInputsResult::RawLogits`: logits-debugging path, kept intentionally.
- `VideoFrameSampling`, `Modalities`, `MultimodalPromptPrefixer`: types only.
- `qwen3_5_moe` text tower: compiles, unused (normal registry maps
  `qwen3_5_moe` checkpoints to the `Qwen3Next` loader).

## Amputation rule (binding)

If it does not serve `kitty run` with a 1–35B GGUF on a consumer GPU, it
goes. Target: −60/70% of the tree. New additions must serve the same gate
or they do not enter.

## Nexus-One robo — optimizado para kitty-infer

Gate: todo lo que entra sirve a `kitty run` GGUF 1–35B consumer, o no entra.

### ROBAR (mapeado F1/F2/F5)

- F2 `ModelSpec.auto` + registry single-residente con evict (corrige bug
  `_` sin import + locks `nexus_one/engine/registry.py:55,68,57`).
- F2 superficie API probada: chat stream/non-stream + `tool_calls`,
  `models`, admin `load/status/cancel`, gpu `status/reclaim`,
  `health`+hint (origen `gateway/app.py:223-555,655-887`).
- F2 SSE con `tool_calls` delta + `usage` final
  (`gateway/app.py:347-410` + `gateway/sse.py`).
- F2 `args_builder` por formato (`vllm_engine.py:80-170` adaptado, sin vLLM).
- F1 GGUF loader con VRAM pre-check (`llamacpp_engine.py:91-180`).
- F2 daemon `setsid` + `BASE_DIR`
  (`nexus_daemon.sh:14-132` + `config/settings.py:18-45`).
- F2 snapshot GPU unificado (`observability/gpu.py:26-186`).
- F2 errores friendly con hint (`gateway/app.py:705-726`).

### NO ROBAR

- KV/PD ficticio (`kv_transfer.py`, `router.py` EWMA, `placement.py`) — stubs simbólicos, sin backend real.
- Simulación progreso (`app.py:742-754`) + `nvidia-smi` por poll — sustituir por progreso real + snapshot cacheado.
- Port TS calculadora (`frontend/lib/api.ts`) — consumir `POST /calculator/estimate`, no forkear.
- `install.sh` CUDA-13/sm_120 + `LD_PRELOAD` + `source .env` + `pkill -f` + `os.kill(9)` — inadaptado/peligroso en sm_75; solo detección backend y venv-pin.
- Defaults inseguros (JWT `change-me`, gateway sin auth, CORS `*`, claves en claro sqlite, `.env` sourced) — invertir a fail-closed.

### Fase → qué entra → gate

| Fase | Entra | Gate |
|---|---|---|
| F1-CUDA build | GGUF loader + VRAM pre-check | `kitty run` GGUF 1–35B carga en sm_75 sin OOM |
| F2 serving/API | ModelSpec+registry, API shape, SSE, daemon, snapshot GPU, errores+hint | shape compatible + 1 residente + reclaim real |
| F5 K2 port | `args_builder` adaptado a torre K2 | K2-Horizon-3.7B responde por mismo `POST /chat` |

### Parche candle#3909 (persistido)

- Qué: backport mínimo del guard de `huggingface/candle#3909` (open) sobre rev `35d7ae7` — fichero `docs/patches/candle-compatibility-cuh-3909.patch`.
- Por qué: el shim `__hmax_nan/__hmin_nan` bajo `#if __CUDA_ARCH__ < 800` (post-#3558) colisiona en sm_75 porque CUDA≥12.2 ya define `__hmax_nan` nativo en `cuda_fp16.hpp` (rompe affine/cast/indexing/sort, afq/hqq/mmvq/mmq/dequant).
- Aplicar en clon fresco: tras el primer `cargo fetch` que materialice el checkout, desde `~/.cargo/git/checkouts/candle-*/35d7ae7/` correr `patch -p0 < <repo>/docs/patches/candle-compatibility-cuh-3909.patch`; verificar con `grep -n CANDLE_CUDA_VERSION candle-kernels/src/compatibility.cuh`.
- Retirar cuando #3909 mergee upstream (subir la rev de candle y borrar este parche).

## Apéndice F5 — port K2-Horizon-3.7B (manual)

Checkpoint: `IFM/K2-Horizon-3.7B-GGUF/K2-Horizon-4B-BF16.gguf` (9.5GB BF16,
`general.architecture=k2-horizon`),/tokenizer `IFM/K2-Horizon-3.7B/tokenizer.json`
(`pre=k2-horizon` → sidecar, mismo patrón que MiniCPM5).

### Arquitectura mapeada (ref `modeling_k2_horizon.py`)

- Decoder pre-norm DENSO 36 capas (este checkpoint: `mlp_only_layers`=todas,
  `num_experts=0`): `input_layernorm → GQA → +residual → post_attention_layernorm → SwiGLU → +residual`, RMSNorm final, lm_head sin atar.
- GQA 32Q/8KV con `head_dim=128` EXPLÍCITO (2560/32=80 NO es head dim);
  Q/K/V/O = 4096/1024/1024/4096. RoPE default θ=1e7, full-head, estilo NeoX;
  el flag rota por `rope_pairing` (Adjacent→false, como Llama) — hardcodearlo
  a true destruye atención (lección F5b, revertida).
- `K2HorizonRMSNorm`: varianza POR GRUPO (`n_groups=2`), float32, con peso.
- MLP SwiGLU 10240, sin biases/qk_norm/sliding-window. Vocab 250624.
- Layout tensores 100% estilo llama (`blk.N.*`, `token_embd/output/output_norm`);
  normas F32, resto BF16. Sin projector (327 = 36×9+3 tensores).

### Decisiones

- Torre propia `models/k2_horizon.rs` (no reutilizar Llama: head_dim y norma
  difieren; reutilizar en silencio = numerics erróneos). `GroupedRmsNorm`
  fiel a la referencia (upcast F32, reshape por grupos).
- Registro: `CanonicalGgufArchitecture::K2Horizon` + `NormalLoaderType::K2Horizon`
  (schema Adjacent, single-candidate), `build_k2_horizon` (head_dim desde
  `rope.dimension_count`), `K2HorizonLoader` con `runtime_config` que aplica
  `max_model_len` (nativo 524288 OOMea el batch KV en 8GB).
- Refusals honestos: projector (marcadores tensor) y MoE/MoVA (`expert_count>0`)
  en síntesis; AnyMoE/LoRA sin implementar (mixin con defaults que hacen bail).
- infer_policy: familia `k2-horizon` → sidecar `<dir>/K2-Horizon-3.7B/tokenizer.json`.
- Sin dtype especial conocido: default Auto (F16). F32 tiene bug propio
  (`slice-set` F32/F16 en `kv_cache.append`, pendiente).

### Bugs cazados en el humo (ambos con fix)

1. KV batch preallocado a max_pos nativo 524288 → OOM en prefill.
   Fix: cap `max_position_embeddings` vía `runtime_config` (harness CTX=4096).
2. `GroupedRmsNorm` dividía por `recip(sqrt(var))` (doble inversión):
   salida RMS 0.0006 en vez de ~0.47 → logits ≈ uniformes (-12.34).
   Repro aislado con candle puro (`/tmp/normtest`): normalizado 0.0014=var.
   Fix: `broadcast_mul` por rsqrt. Verificado: distribuciones nítidas CPU≈CUDA.

### Gaps abiertos

- Device-map auto deja 13 capas en GPU + 23 en CPU (9.5GB no cabe en 8GB);
  paged-attention se desactiva en mapping mixto (límite conocido, eager OK).
- F32-CUDA (cierre F5): `dtype mismatch in slice-set` venía de appends con
  dtype mezclado en una misma capa (F32 compute vs F16 CPU-cache); fix mínimo
  en `SingleCache::append` (unifica al dtype del cache, no-op si coinciden;
  sin tocar paged-attn/candle). Verificado: k2-f32 exit 0, peak 7482 MiB,
  mismo argmax que F16 (ruido de precisión ~0.01).
- Humo embebido (KittyInferClient, cierre F5): `info` + `run` llegan a
  `resident ready` en CPU host (MiniCPM5) y contenedor (Qwen3-4B); el turno
  agente completo excede el time-box CPU (no colgado: quema CPU real) —
  generación probada a nivel engine en CUDA para los 4 modelos.
- Paridad vs upstream: sin baseline en el entorno; se entregan top-5 Ferrox.
- MoVA-36B y variantes multimodales: fuera de scope (refusal).

## Apéndice review ora-1 — follow-ups D3-D8 (documentados, NO implementados)

- D3 `Kitty-Code/crates/kitty-runtime/src/error.rs:111`: `error_class` catch-all `_ => Permanent` puede clasificar como permanente un error nuevo transitorio; auditar al añadir variantes.
- D4 `kitty-infer/mistralrs-core/src/serving/errors.rs:51-55,87-96`: `classify` over-matchea (`memory`/`cuda error` sueltos → OOM) y `body` propaga `detail` crudo (posible fuga de paths); endurecer predicados + sanear detalle.
- D5 `kitty-infer/mistralrs-core/src/pipeline/loaders/normal_loaders.rs:3560`: `K2HorizonLoader::is_gptx` devuelve `Ok(true)` incondicional; derivarlo del pairing cuando el layout deje de ser Adjacent-only.
- D6 `kitty-infer/mistralrs-core/src/gguf/normal_config.rs:1142`: `build_k2_horizon` no valida rango de `rope.freq_base` (θ absurdo pasa hasta RoPE); añadir guarda explícita.
- D7 `kitty-infer/mistralrs-core/src/models/k2_horizon.rs:63,102`: `rms_norm_eps: f64` con cómputo F32 y `to_dtype` por etapa; auditar precisión mixta si aparecen NaN en otros rangos de ctx.
- D8 `Kitty-Code/crates/kitty-runtime/src/client.rs:238`: `dynamic_max_output` deriva `num_predict` de la ventana Ollama, no de `KITTY_INFER_CTX`; alinear límites cuando el backend embebido sea default.

## Apéndice Fase 1 (Deepwork) — turno K2 embebido GPU: envelope 8GB

Perfil que corre sin OOM: `KITTY_INFER_CTX=4096` (retry 0.6 → ctx 2457) +
`KITTY_INFER_GPU_LAYERS=12` (12 GPU + 24 CPU), `KITTY_INFER_NO_THINK=1`,
pico VRAM ~6.4GB estable (sin deriva). Knobs nuevos mínimos:
`KITTY_INFER_GPU_LAYERS` (spill explícito vía `DeviceMapSetting::Map`) y
`KITTY_INFER_NO_THINK` (bloque think vacío en templates con reasoning).
Fix requerido por el turno: el mensaje `VRAM pre-check failed` no
clasificaba como OOM (ni en `KittyError::classify` ni en el matcher del
registry) → sin retry; añadido el literal en ambos (verificado: retry 0.9→0.6
con logs `failed with OOM` + `loaded after OOM retry`).
Límites medidos (RTX 2060S 8GB): spill-20 OOMea en prompt-step; spill-12 no
OOMea pero el prefill del prompt agente (~miles de tokens, 24 capas CPU
BF16) no completa el turno en 25 min (proceso vivo quemando CPU, sin error).
Gate Fase 1 K2-turno: NO verde en 8GB (envelope hardware, no defecto de
código: carga, retry, spill, generación sin corrupción y errores friendly
verificados; engine-level F16/F32 verde + determinista).
Riesgo aceptado (re-review Gate 1 attempt 3/3, HOLD levantado): el estimado
spill cubre pesos+KV, no transitorios/grafos de prefill (~1-2GB); spill ≥13/36
en 8GB puede pasar el pre-check y OOMear ruidosamente en prompt-step con hint;
envelope soportado ≤12/36; fallos siempre loud, nunca silenciosos.

## Apéndice CTX-mínimo embebido (medido, RTX 2060S 8GB, Fase 3)

CTX = `KITTY_INFER_CTX` del turno embebido (el retry 0.6 lo escala; el valor
listado es el configurado). Prompt agente = turno `kitty run` con tools.

| Modelo (GGUF) | CTX mín. turno | Capas GPU | Pico VRAM | Tiempo turno | Nota |
|---|---|---|---|---|---|
| MiniCPM5-2B Q4_K_M (1.5GB) | 8192 | auto (todas) | 3124 MiB | 117s | CTX=4096 = panic scheduler (`reserved prompt chunk must fit in the KV cache`, paged_attention/scheduler.rs:313); 8192 → 128 bloques, verde |
| Qwen3-4B Q4_K_M (2.4GB) | 8192 (2048 en humo engine) | auto (todas) | 4779 MiB | 41s | verde F16; tokenizer embebido OK |
| qwen2.5-coder-7B Q4 (4.4GB) | 4096 | auto (todas) | 5980 MiB | n/a engine (turno embebido no medido) | SOLO F32 (NaN F16-CUDA); ver `infer_policy::F32_OVERRIDE_PREFIXES` |
| K2-Horizon 4B BF16 (9.5GB) | 4096 → retry 2457 | 12 (`KITTY_INFER_GPU_LAYERS=12`) | 6473 MiB estable | sin completar en 25 min | spill-20 = OOM prompt-step (`CUDA_ERROR_OUT_OF_MEMORY`); `KITTY_INFER_NO_THINK=1` para no eternizar thinking |
| K2-Horizon 4B Q4_K_M (3.0GB, sha 07773aed) | 16384 (460 bloques) | auto (todas) | 6422-7350 MiB | exit 0 en 408s/578s pero output degenerado (repetición `{""` / role-leak `user/thinking`) | humo engine verde 8/8 determinista; `with_gpu_memory(ContextSize(ctx))` requerido (default 128 bloques panickea); `chat_with_options` honra `num_predict` (sin cota = timeout) |

Regla: el CTX embebido lo fija el prompt agente (miles de tokens con tools),
no el humo engine de prompt corto. Si el scheduler paged panickea por chunk,
subir CTX (más bloques) antes que tocar código.
Soak (turnos largos, deriva VRAM plana): /tmp/kitty-logs/k2-embedded.vram.csv
(28 min, 7493–7656 MiB, sin deriva; OOM solo por picos sobre 7.6GB).

## Apéndice A/B K2-Q4 receta vs actual (cierra Gate A de verdad)
Receta HF = thinking high + temp 1.0/top_p 0.95 + `tool_call_format=json`
(parser k2 ifm implementado: `tools/parsers/k2.rs` json/xml/xml_typed +
`reasoning_parsers::tag_based::{new_k2_think,is_k2_think_template}` +
knob por-request `RequestBuilder::with_tool_call_format`, default xml;
sin parser k2 dedicado previo en el árbol).
Brazo A receta-COMPLETA: parser separa think (content vacío con cap chico →
fail-fast limpio); con ventana amplia genera 18+ min sin cerrar en 25-min box.
Brazo B actual (NO_THINK + greedy + xml): exit 0 pero degenerado (`{""` /
role-leak). Veredicto: A-nítido NO → ambos igual → PESOS (reabrir artefacto
Q6/Q8 o turno BF16 completo antes que cirugía de prompt); engine-plain nítido
y control Qwen3 coherente descartan infra (sin NaN/OOM/panics en ningún brazo).
