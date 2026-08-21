# Plugin System — Kitty Code

Static plugin registry — terceros registran tools sin tocar `crates/kitty-harness/src/tools.rs`.

> No `libloading` / `dlopen`. Todos los plugins se enlazan estáticamente vía `PluginRegistry::register`.

## Quick start

### 1. Implementar `Plugin`

```rust
use kitty_harness::plugin::{Plugin, PluginError, PluginRegistry};
use kitty_harness::tools::{ToolRegistry, ToolResult};
use kitty_runtime::ToolDefinition;

pub struct HelloPlugin;

impl Plugin for HelloPlugin {
    fn name(&self) -> &str { "hello" }

    fn tools(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition::new(
            "HelloTool",
            "Dummy hello tool from plugin system",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "name to greet" }
                },
                "required": ["name"]
            }),
        )]
    }

    fn execute(&self, tool_name: &str, args: &serde_json::Value)
        -> Result<ToolResult, PluginError>
    {
        if tool_name != "HelloTool" {
            return Err(PluginError::NotFound(tool_name.to_string()));
        }
        let who = args.get("name").and_then(|v| v.as_str()).unwrap_or("world");
        Ok(ToolResult { success: true, output: format!("hello {who}"), retry: false })
    }
}
```

### 2. Registrar en `ToolRegistry`

```rust
let mut plugins = PluginRegistry::new();
plugins.register(Box::new(HelloPlugin)).unwrap();

let mut tools = ToolRegistry::new().with_plugin_registry(plugins);
// o incremental:
// let mut tools = ToolRegistry::new();
// tools.register_plugin(Box::new(HelloPlugin)).unwrap();

// Exponer al LLM:
let defs = tools.all_tool_definitions(); // 6 core + N plugin

let tc = ParsedToolCall { name: "HelloTool".into(), arguments: json!({"name":"kitty"}) };
let res = tools.execute(&tc).await.unwrap();
assert!(res.success);
```

### 3. Guard inheritance

Todo plugin que exponga un argumento `command: string` hereda `Guard::check_chain`:

- `eval` → siempre bloqueado (`BashBlocked`), incluso si el plugin permite `eval` como nombre.
- `rm -rf /`, `dd if=/dev/zero of=/dev/sda`, `echo $(rm -rf /)` → bloqueado.
- `cat > file.txt`, `tee out.log` → bloqueado como write-vía-redirect.
- `echo hi > /dev/null` → permitido (non-destructive).

Si `Guard` bloquea, `ToolRegistry::execute` retorna `Err(ToolError::BashBlocked)`;
errores de ejecución del plugin se aíslan como `ToolResult { success:false }` y no abortan el loop (`agent.rs:307` pattern).

### 4. Cargar desde `plugins.toml` (opcional)

```toml
# plugins.toml
[[plugin]]
name = "hello"
enabled = true

[[plugin]]
name = "unknown_plugin_xyz"
enabled = true  # ignorado con warning, no dlopen
```

```rust
let registry = PluginRegistry::load_from_manifest(Path::new("plugins.toml")).unwrap();
// o helper tolerante:
let registry = PluginRegistry::load_from_manifest_or_empty(Path::new("plugins.toml"));
let tools = ToolRegistry::new().with_plugin_registry(registry);
```

Format parse: TOML (`[[plugin]]`) con fallback JSON para tests. Factory estática: `"hello"` → `HelloPlugin`; nombres desconocidos se ignoran con `tracing::warn!`.

### 5. Cargo features (alternativa sin archivo)

En lugar de `plugins.toml`, registra por feature:

```toml
# Cargo.toml
[features]
plugin-hello = ["kitty-harness/plugin-hello"]
```

```rust
#[cfg(feature = "plugin-hello")]
plugins.register(Box::new(HelloPlugin)).unwrap();
```

No hay `libloading`; cada plugin es un crate/feature linkeado y registrado explícitamente.

## Aislamiento y core no modificado

- **Core intacto**: `standard_tool_definitions()` siempre retorna 6 (`Read, Edit, Write, Bash, Glob, Grep`). Registrar plugins no muta ese vector; `all_tool_definitions()` es `6 + N plugin` solo en la instancia `ToolRegistry` que los contiene.
- **Duplicado bloqueado**: dos plugins no pueden exponer el mismo `function.name` (ni siquiera dos tools dentro del mismo plugin). `register` retorna `DuplicateTool`, sin overwrite silencioso.
- **Error aislado**: `Plugin::execute` que retorna `Err(Execution)` se convierte en `ToolResult { success:false, retry:false }`; el loop del agente no aborta y otros plugins siguen funcionando.
- **Guard heredado**: ya documentado arriba; plugins con `command` no bypassan `eval`/`rm -rf /`/`$(…)`/`> file`.

Verificado por tests:
`plugin_two_tool_plugin_registers_two_tools_core_intact`, `plugin_core_not_modified_after_registry_ops`, `plugin_isolation_between_plugins_and_error_isolated`, `plugin_integration_two_tool_plugin_core_not_modified_and_isolated`, `plugin_integration_error_isolation_between_plugins` — modelo `TwoToolPlugin` (ToolA/ToolB) demuestra 2 tools registrados sin tocar core.

## WASM — feature gate obligatorio

No añadir soporte WASM sin feature gate. Si en el futuro se introduce ejecución WASM para plugins, debe estar detrás de `features = ["wasm"]` (ej. `wasmtime` opt-in), nunca por defecto, para no aumentar binario ni superficie de ataque en `tiny`/`medium`.

```toml
[features]
default = []
wasm = ["dep:wasmtime"]
```

## `plugins.toml` — ejemplo completo

```toml
# plugins.toml — manifest estático, sin dlopen
# Cada [[plugin]] requiere `name`; `enabled` por defecto true.

[[plugin]]
name = "hello"
enabled = true          # HelloPlugin -> HelloTool

[[plugin]]
name = "two_tool"
enabled = true          # TwoToolPlugin -> ToolA + ToolB (ejemplo test, registra 2 tools)

[[plugin]]
name = "unknown_plugin_xyz"
enabled = true          # ignorado con tracing::warn!, no dlopen

[[plugin]]
name = "hello"
enabled = false         # deshabilitado -> no se registra, permite toggle sin borrar entrada
```

Carga:

```rust
let registry = PluginRegistry::load_from_manifest(Path::new("plugins.toml")).unwrap();
assert_eq!(registry.len(), 2); // hello + two_tool (unknown ignorado, disabled skip)
assert!(registry.contains_tool("HelloTool"));
assert!(registry.contains_tool("ToolA"));

let missing_ok = PluginRegistry::load_from_manifest_or_empty(Path::new("plugins.toml"));
```

Persistir `plugins.toml` en el repo es opcional; alternativamente registrar por código con `register`/`register_plugin` o por feature `plugin-hello`.

## Errors

- `PluginError::DuplicateTool(name)` — dos plugins exponen el mismo `function.name`; registro falla, no hay overwrite.
- `PluginError::NotFound(name)` — `Plugin::execute` recibe nombre desconocido.
- `PluginError::Execution(msg)` — fallo aislado → `ToolResult success:false`.
- `PluginError::Manifest(msg)` — `plugins.toml` ilegible o inválido.

## Tests

```bash
cargo test -p kitty-harness --lib plugin
cargo test -p kitty-harness --lib plugin_integration
cargo test -p kitty-harness --lib -- plugin
cargo clippy -p kitty-harness -- -D warnings
```

Tests de integración en `crates/kitty-harness/src/tools.rs`:

- `plugin_integration_delegates_to_registry_success`
- `plugin_integration_guard_blocks_bash_plugin`
- `plugin_integration_guard_blocks_eval_via_plugin`
- `plugin_integration_guard_blocks_redirect_via_plugin`
- `plugin_integration_guard_allows_safe_plugin_bash`
- `plugin_integration_error_isolated_as_false`
- `plugin_integration_eval_tool_name_blocked`
- `plugin_integration_all_definitions_merged`
- `plugin_integration_two_tool_plugin_core_not_modified_and_isolated`
- `plugin_integration_error_isolation_between_plugins`

Tests de aislamiento en `crates/kitty-harness/src/plugin.rs`:

- `plugin_two_tool_plugin_registers_two_tools_core_intact` (2 tools, core 6 intacto)
- `plugin_isolation_between_plugins_and_error_isolated`
- `plugin_core_not_modified_after_registry_ops`
