# Serving K2-Horizon-3.7B vía SGLang (carril K2)

**Estado 2026-09-20: CONGELADO-documentado.** SGLang no está instalado en este
box y el GGUF oficial exige el fork llama.cpp de MBZUAI (el PR upstream sigue
abierto: stock llama.cpp / Ollama **no** lo cargan). El `OpenAiCompatClient`
(`crates/kitty-runtime/src/openai_compat.rs`) está implementado y testeado
contra mock (11 tests); el levantamiento real queda como criterio de
desbloqueo abajo. K2 vuelve a vigilancia — el resto del plan no se bloquea.

## Receta vendor (IFM/K2-Horizon-3.7B, rev pinneada)

```bash
pip install "sglang[all]"  # + torch 2.13,_flash-attn fa3
sglang serve \
  --model-path IFM/K2-Horizon-3.7B \
  --revision c177771836a4c460743c00002c22483f6f18d1eb \
  --tp 1 \
  --dtype bfloat16 \
  --attention-backend fa3 \
  --reasoning-parser k2_horizon \
  --host 0.0.0.0 \
  --port 30000
```

Parámetros de request (ya cableados en el cliente): `temperature=1.0`,
`top_p=0.95`, `max_tokens=32768`, `extra_body={"reasoning_effort":"high"}`.

## Límite operativo en este box (8 GB VRAM + 48 GB RAM)

NO servir a 128K: el KV-cache f16 a esa ventana hace OOM. Arrancar con
`--max-model-len 32768` (el vendor capa su receta vLLM en 131072; 32K es
nuestro techo operativo). Esto es límite de máquina, no tuning del modelo.

## Conexión con Kitty-Code

```bash
KITTY_OPENAI_BASE_URL=http://127.0.0.1:30000 \
  kitty run --model IFM/K2-Horizon-3.7B --profile small "..."
# (el cableado CLI `--backend sglang` es trabajo futuro; hoy el cliente se
#  usa programáticamente con ModelTuning::k2_family())
```

`ModelTuning::k2_family()` ya resuelve temp 1.0 / top_p 0.95 /
num_predict 32768 / thinking ON para ids `*k2-horizon*` / `*k2_horizon*`;
perfil `small` + turn-cap 12 aplican por `is_small_sized_model`.

## Criterio de desbloqueo

1. SGLang levanta y `/v1/models` responde.
2. `health_check()` del cliente en verde contra `:30000`.
3. Un smoke `kitty run` completa 1 tarea.
4. Entonces: subset-30 oficial mismo protocolo que MiniCPM (Fase 4).

## Por qué no Ollama

Arquitectura custom (`k2_horizon`/`XllmForCausalLM`, `trust_remote_code`).
Ollama solo importa GGUFs de arquitecturas conocidas; el GGUF oficial de IFM
documenta que requiere su fork. Sin upstream merge no hay camino Ollama.
