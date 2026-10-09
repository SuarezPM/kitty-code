//! W2 TUI: hilo de agente aislado.
//!
//! `Agent` no es `Send` (contiene `RefCell`/estado no-thread-safe), así que lo
//! confinamos a un hilo propio con un runtime `current_thread` + `LocalSet`.
//! El event loop de la TUI (síncrono) se comunica con él por dos canales:
//!
//!   - UI → worker: [`WorkerCmd`] (prompt, cambio de modelo, reset, shutdown)
//!   - worker → UI: [`TuiEvent`] (tokens, permisos, usage, done, error)
//!
//! El agent se REUSA entre prompts (mismo `Agent`), de modo que la conversación
//! multi-turno funciona de verdad: `agent.messages` acumula. Cambiar de modelo
//! reconstruye el `Agent` (pierde historial en el harness) y emite un aviso.

use std::sync::{Arc, Mutex};

use kitty_harness::agent::{Agent, AgentConfig, RunOverrides};
use kitty_runtime::OllamaClient;
use tokio::sync::mpsc;

use super::client::{
    ModelListing, ModelsOutcome, PermissionBroker, PermissionPolicy, TuiClient, TuiEvent,
};

/// Comandos que la TUI envía al worker.
#[derive(Debug)]
pub enum WorkerCmd {
    /// Ejecutar un turno completo del agent con este prompt.
    Prompt(String),
    /// Cambiar el modelo para turnos futuros (reconstruye el agent).
    SetModel(String),
    /// Listar los modelos instalados en el backend (`/api/tags`).
    FetchModels,
    /// Resetear la conversación (nuevo agent, historial limpio).
    Reset,
    /// Apagar el worker de forma ordenada.
    Shutdown,
}

/// Arranca el hilo worker. Devuelve los sender/receiver para que la TUI
/// se comunique. El `context_window` es la ventana inicial del agent.
pub fn spawn_worker(
    url: String,
    model: String,
    context_window: usize,
    events_tx: mpsc::Sender<TuiEvent>,
    broker: Arc<PermissionBroker>,
    policy: Arc<Mutex<PermissionPolicy>>,
) -> (mpsc::Sender<WorkerCmd>, std::thread::JoinHandle<()>) {
    let (cmd_tx, mut cmd_rx) = mpsc::channel::<WorkerCmd>(16);
    let handle = std::thread::Builder::new()
        .name("kitty-tui-agent".into())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("tui worker runtime");
            let local = tokio::task::LocalSet::new();
            rt.block_on(local.run_until(async move {
                let mut cur_model = model;
                // Cliente dedicado al listado de modelos: mismo base_url y
                // mismo `http` que usa el agent para el streaming, así el
                // selector ve exactamente los modelos de este backend.
                let lister = build_client(&url, &events_tx, &broker, &policy);
                let mut agent = build_agent(
                    &url,
                    &cur_model,
                    context_window,
                    &events_tx,
                    &broker,
                    &policy,
                );
                while let Some(cmd) = cmd_rx.recv().await {
                    match cmd {
                        WorkerCmd::Prompt(p) => {
                            // Anti-stuck: un panic en el agent no debe colgar
                            // la TUI. spawn_local convierte el panic en Err.
                            let join = tokio::task::spawn_local({
                                let mut a = agent;
                                let prompt = p.clone();
                                async move {
                                    let r = a.run_with(&prompt, RunOverrides::default()).await;
                                    (a, r)
                                }
                            });
                            match join.await {
                                Ok((a, Ok(text))) => {
                                    agent = a;
                                    let _ = events_tx.send(TuiEvent::Done(text)).await;
                                }
                                Ok((a, Err(e))) => {
                                    agent = a;
                                    let _ = events_tx.send(TuiEvent::Error(e.to_string())).await;
                                }
                                Err(_) => {
                                    // panic: reconstruimos el agent para no
                                    // dejar estado corrupto.
                                    agent = build_agent(
                                        &url,
                                        &cur_model,
                                        context_window,
                                        &events_tx,
                                        &broker,
                                        &policy,
                                    );
                                    let _ = events_tx
                                        .send(TuiEvent::Error(
                                            "el agente falló; turno abortado".into(),
                                        ))
                                        .await;
                                }
                            }
                        }
                        WorkerCmd::SetModel(m) => {
                            cur_model = m;
                            agent = build_agent(
                                &url,
                                &cur_model,
                                context_window,
                                &events_tx,
                                &broker,
                                &policy,
                            );
                            let _ = events_tx
                                .send(TuiEvent::Done(format!("(modelo activo: {cur_model})")))
                                .await;
                        }
                        WorkerCmd::FetchModels => {
                            // Reusamos el TuiClient ya construido para el
                            // listado: mismo base_url, mismo http cliente.
                            let outcome = match lister.list_models().await {
                                Ok(models) => ModelsOutcome::Ok(models),
                                Err(e) => ModelsOutcome::Err(e),
                            };
                            let ev = match outcome {
                                ModelsOutcome::Ok(models) => TuiEvent::ModelsLoaded(models),
                                ModelsOutcome::Err(e) => TuiEvent::ModelsFailed(e),
                            };
                            let _ = events_tx.send(ev).await;
                        }
                        WorkerCmd::Reset => {
                            agent = build_agent(
                                &url,
                                &cur_model,
                                context_window,
                                &events_tx,
                                &broker,
                                &policy,
                            );
                            let _ = events_tx
                                .send(TuiEvent::Done("(conversación reiniciada)".into()))
                                .await;
                        }
                        WorkerCmd::Shutdown => break,
                    }
                }
            }));
        })
        .expect("spawn tui worker");
    (cmd_tx, handle)
}

fn build_client(
    url: &str,
    events_tx: &mpsc::Sender<TuiEvent>,
    broker: &Arc<PermissionBroker>,
    policy: &Arc<Mutex<PermissionPolicy>>,
) -> TuiClient {
    TuiClient::new(
        OllamaClient::new(url),
        events_tx.clone(),
        broker.clone(),
        policy.clone(),
        0.0, // local: sin coste
    )
}

fn build_agent(
    url: &str,
    model: &str,
    context_window: usize,
    events_tx: &mpsc::Sender<TuiEvent>,
    broker: &Arc<PermissionBroker>,
    policy: &Arc<Mutex<PermissionPolicy>>,
) -> Agent {
    let client = build_client(url, events_tx, broker, policy);
    let config = AgentConfig {
        model: model.to_string(),
        context_window,
        ..Default::default()
    };
    Agent::new(client, config)
}
