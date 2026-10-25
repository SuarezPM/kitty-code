# opencode 1.19.0 + Ollama local (smoke F3 Lane C)

Fecha: 2026-10-25. Scope: solo este fichero nuevo (no se tocó resto del repo).

## Entorno

- Binario: `/home/thelinconx/.opencode/bin/opencode` → `opencode --version` = `1.19.0`
- Ollama: `http://127.0.0.1:11434` — `/v1/models` responde OK y lista `minicpm5:2b`, `ornith-1.5:9b`
- `ollama show minicpm5:2b` → `num_ctx 32768`, context length 131072
- `opencode models` hoy NO lista provider `ollama`/`openai-compatible` genérico, pero sí lista
  `local-ornith/ornith:9b` y `local-qwen/qwen38-27b` porque `~/.config/opencode/opencode.jsonc`
  ya declara ese patrón (vía válida conocida).

## Provider local (aislado, sin tocar config del usuario)

Fichero: `/tmp/opencode-ollama-test/opencode.json`

```json
{
  "$schema": "https://opencode.ai/config.json",
  "permission": "allow",
  "provider": {
    "local-ollama": {
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://127.0.0.1:11434/v1",
        "apiKey": "ollama",
        "timeout": 180000,
        "chunkTimeout": 120000
      },
      "models": {
        "minicpm5:2b": { "name": "MiniCPM5 2B local (Ollama)" },
        "ornith-1.5:9b": { "name": "Ornith 1.5 9B local (Ollama)" }
      }
    }
  }
}
```

Notas:

- Es el mismo patrón que ya funciona en `~/.config/opencode/opencode.jsonc`
  (`local-ornith`: `npm @ai-sdk/openai-compatible` + `baseURL http://127.0.0.1:11434/v1`).
  La vía harbor PR #2940 (inyectar `OPENAI_BASE_URL` + remapear `openai→openai-compatible`
  porque `@ai-sdk/openai` habla `/v1/responses` que Ollama rechaza) queda obsoleta en
  opencode moderno: se declara `npm: @ai-sdk/openai-compatible` directamente.
- Credencial dummy: `apiKey: "ollama"` en el JSON basta. Por seguridad se exportó además
  `OPENAI_API_KEY=x` en los intentos (algunas variantes del SDK la exigen aunque sea dummy).
- Workdir de ejecución: `/tmp/opencode-ollama-test` (opencode carga `opencode.json` del cwd).

## Smoke REAL (resultado: NO pasa — límite de contexto de Ollama, no de endpoint)

Objetivo: `crea /tmp/opencode-smoke/ok.txt con el texto ok`, timeout 180 s.

| # | Comando exacto (workdir `/tmp/opencode-ollama-test`) | Exit | Resultado |
|---|------------------------------------------------------|------|-----------|
| 1 | `OPENAI_API_KEY=x timeout 170 /home/thelinconx/.opencode/bin/opencode run -m local-ollama/minicpm5:2b "crea el fichero /tmp/opencode-smoke/ok.txt con el texto exacto: ok"` | 124 (timeout 170 s, bucle de reintentos) | `Error: request (61541 tokens) exceeds the available context size (32768 tokens)` |
| 2 | `OPENAI_API_KEY=x timeout 170 /home/thelinconx/.opencode/bin/opencode run --pure -m local-ollama/minicpm5:2b "crea el fichero /tmp/opencode-smoke/ok.txt con el texto exacto: ok"` | 124 | `Error: request (58588 tokens) exceeds the available context size (32768 tokens)` (ni `--pure` baja de ~58 k) |
| 3 | `OPENAI_API_KEY=x timeout 170 /home/thelinconx/.opencode/bin/opencode run --pure -m local-ollama/ornith-1.5:9b "crea el fichero /tmp/opencode-smoke/ok.txt con el texto exacto: ok"` | 124 | `Error: request (58220 tokens) exceeds the available context size (16384 tokens)` |

Evidencia post-smoke:

```text
$ ls /tmp/opencode-smoke/   → (vacío)
$ cat /tmp/opencode-smoke/ok.txt → No existe el fichero o el directorio (exit 1)
```

## Conclusión

- **Endpoint y provider: CORRECTOS.** Ollama responde en `/v1/models` y el error viene del
  runner de Ollama (tokens vs `num_ctx`), no de conexión ni de `/v1/responses`. La declaración
  `package @ai-sdk/openai-compatible + options.baseURL=http://127.0.0.1:11434/v1` es la vía válida.
- **Bloqueo real:** el system prompt base de opencode 1.19.0 (~58–61 k tokens incluso con
  `--pure`) excede el `num_ctx` por defecto de ambos modelos
  (`minicpm5:2b` = 32768, `ornith-1.5:9b` = 16384 según el error).
- **Comando final recomendado** (cuando se eleve `num_ctx` en Ollama, fuera de los 3 intentos
  de esta lane):

```bash
mkdir -p /tmp/opencode-smoke
cd /tmp/opencode-ollama-test
OPENAI_API_KEY=x timeout 170 /home/thelinconx/.opencode/bin/opencode run --pure \
  -m local-ollama/minicpm5:2b \
  "crea el fichero /tmp/opencode-smoke/ok.txt con el texto exacto: ok"
echo "EXIT:$?"
cat /tmp/opencode-smoke/ok.txt
```

- **Desbloqueo 2026-10-25 (post-apagón, verificado):** variantes 64K creadas y
  smoke OK — `ollama create ornith-1.5-9b-64k` y `minicpm5-2b-64k` con
  `PARAMETER num_ctx 65536`; `opencode run --model
  local-ollama/ornith-1.5-9b-64k "crea /tmp/opencode-smoke2/ok.txt..."` →
  EXIT 0, `ok` creado (con 35% spill a CPU en 9B: 7.8GB > 8GB con escritorio;
  documentar como asimetría). Provider `local-ollama` =
  `@ai-sdk/openai-compatible` + `baseURL http://127.0.0.1:11434/v1`.
- TB2.0 no ejecutado (lo corre F3).
