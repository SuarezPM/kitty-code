## 1. Adapter llama.cpp con ChatClient

- [ ] 1.1 Crear adapter llama.cpp que implemente `ChatClient` (`chat`, `count_tokens`, `health_check`)
- [ ] 1.2 Generar gramática GBNF a partir del set de schemas activo (perfil + categoría de routing)

## 2. Integración y paridad

- [ ] 2.1 Inyectar adapter en `Agent` (`Box<dyn ChatClient>`) y verificar loop Read→Edit sin regresión vs OllamaClient
- [ ] 2.2 Sincronizar gramática con 2-stage routing (solo schemas de categoría ganadora)

## 3. Verificación

- [ ] 3.1 A/B de tool-call validity con/sin gramática (target >95% / 88%→98%) y validar escenarios de spec
