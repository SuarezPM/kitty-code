# SDD-PLAN-INFER-MK — MiniCPM5 + K2 vía kitty-infer (Gate 1 incorporado)

Plan de especificación (sin código salvo celdas timeouts). Base: Gate 1 CONDITIONAL + correcciones verificadas directo (sidecars presentes, Q4 oficial, env exacto). Regla: 1 proceso = 1 modelo (runs seriales o contención VRAM); sin celda K2 → infer 150s.

## 1. REQUISITOS Y ALCANCE
- HU-1 MiniCPM5-infer ready: Given `MiniCPM5-2B-Q4_K_M.gguf` + sidecar presente, When `info` + smoke 1 celda con `KITTY_BACKEND=kitty-infer`, Then Connected + load-smoke OK + celda `minicpm5:2b/infer:150` ya existente (sin celda nueva salvo medición contraria).
- HU-2 K2-infer ready: Given `K2-Horizon-4B-Q4_K_M.gguf` (oficial IFM, sha == etag) + sidecar, When `info` + smoke 1 celda, Then Connected + load-smoke OK + celda `k2-horizon/infer` NUEVA solo con número medido (wall×2) o default infer 150s.
- HU-3 manifiesto maratón: Given matriz verde, When se pre-registra, Then run trazable por (modelo,backend).
- Fuera: BF16-10GB en 8GB (no cabe), SGLang (frozen), Ollama para K2 (no existe tag), re-bench de Qwen3 (base hecha).
- DoD por smoke: exit capturado sin pipe + wall time + VRAM pre/post (en lane CUDA; N/A en CPU parcial — ver T2) + `health_check` previo (necesario, no suficiente).

## 2. MODELO DE DATOS Y ENTIDADES
- Envs: `KITTY_BACKEND=kitty-infer`, `KITTY_INFER_MODEL` (global único por proceso), `KITTY_INFER_TOKENIZER_DIR=~/.cache/kitty/tokenizers` (EXACTO, verificado: ambos sidecars con sizes oficiales 9894271/20637260), `KITTY_INFER_CTX` default 4096.
- Paths: `$MODELS/MiniCPM5-2B-Q4_K_M.gguf` (1.56GB) + `$TOK/MiniCPM5-2B/tokenizer.json`; `$MODELS/K2-Horizon-4B-Q4_K_M.gguf` (3.16GB) + `$TOK/K2-Horizon-3.7B/tokenizer.json`. Sibling-junto-al-GGUF como alternativa (nunca layout HF-cache). Definiciones obligatorias al inicio de cada sesión: `export MODELS=~/.cache/kitty/models TOK=~/.cache/kitty/tokenizers` (ambos verificados en disco; paths rotos si vacíos).
- Timeouts: `minicpm5:2b` ollama 60/infer 150 (existen); ornith ollama 60 (solo Ollama, sin GGUF); K2 sin celda → infer-backend 150 hasta medición.
- Labels `--model` libres pero deben casar celdas (`minicpm5:2b`, `k2-horizon`).

## 3. ENFOQUE TÉCNICO Y ARQUITECTURA
- Disciplina VRAM: `ollama stop <residente>` + `nvidia-smi` antes de cada smoke infer (lección OOM: ornith 5.9GB vs 1.75GB libres).
- Orden: sidecars (ya presentes, verificar) → info MiniCPM5 → smoke MiniCPM5 → info K2 → smoke K2 → celdas (medidas) → manifiesto → GO maratón.
- Criterio readiness = smoke de carga real (health solo-fichero no basta); timeout-path OK cuenta como pipeline probado, no como PASS.
- Rollback: unset env + borrar celdas añadidas; cero cambios de código (salvo `timeouts.json`).
- Prohibido: BF16 en 8GB, `--exercises` sobre polyglot esperando filtro, exit bajo pipe, inventar timeouts.

## 4. DESCOMPOSICIÓN DE TAREAS
- T1 verificar sidecars (sizes 9894271/20637260) — HECHO en investigación, re-verificar al ejecutar.
- T2 `info` MiniCPM5-infer (Connected esperado), luego T3 smoke `bench tb2 --limit 1` (time-box 25min) [T3 tras T2]. Invocación canónica (vale para T2-T5): binario `target/debug/kitty-cli` (CPU `--features kitty-infer` ya compilado 24-sep) salvo rebuild necesario (`cargo build -p kitty-cli --features kitty-infer`); GPU solo vía feature `kitty-infer-cuda` en lane/contenedor CUDA (si no compila local por cudarc/CUDA-13.4 → CPU declarado parcial y el criterio VRAM pre/post no aplica; wall/exit sí).
- T4 `info` K2-infer (Connected esperado), luego T5 smoke K2 1 celda (time-box 30min; loader/registry primera vez) [T5 tras T4].
- T6 celdas `timeouts.json`: K2/infer solo con wall medido ×2 con binario debug (perfil declarado en `origin` de la celda; si la maratón corre release, re-medir; si no hay dato → default 150, declarado); MiniCPM5 sin cambios salvo medición contraria.
- T7 manifiesto pre-registrado maratón (norma del repo) + matriz (modelo,backend)→estado.
- T8 GO/NO-GO por modelo con evidencia (logs, walls, exits sin pipe).
- Dependencias: T3 tras T2, T5 tras T4, T6 tras T3+T5, T7 tras todo, T8 tras todo ("tras" = dependencia; "luego" en cada tarea = secuencia; sin flechas). Serie global por VRAM (un resident a la vez); [P] solo T2+T4 si hay VRAM para ambos (no hay: serie).
