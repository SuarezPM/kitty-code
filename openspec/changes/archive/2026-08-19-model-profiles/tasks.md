## 1. ModelProfile y presets

- [ ] 1.1 Definir struct `ModelProfile` (context_window, max_thinking_tokens, max_iterations, tool_subset, use_native_tools, caps, keep_count, system prompt)
- [ ] 1.2 Implementar presets `tiny`/`medium`/`large` (4K sin thinking / 16K+2048 / 32K) con selecciones de tools correspondientes
- [ ] 1.3 Migrar defaults dispersos (agent.rs:47-63, context.rs:111, thinking_budget.rs:106, tools.rs:83/413/514) a datos del perfil

## 2. Abstracción ChatClient

- [ ] 2.1 Crear trait `ChatClient` (`chat`, `count_tokens`, `health_check`) en `kitty-runtime` e implementar para `OllamaClient`
- [ ] 2.2 Cambiar `Agent` a `Box<dyn ChatClient>` y añadir flag CLI `--profile tiny|medium|large` (default `medium`)

## 3. Verificación

- [ ] 3.1 Tests de caps por perfil y de `MockClient` para loop sin servidor real (scenarios tiny/medium/large, ChatClient, caps)
