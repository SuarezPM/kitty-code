//! Kitty Memory — Episodic memory (blake3 + SQLite).
//!
//! Stores and retrieves session episodes for cross-session recall.

pub mod episodic;
pub mod error;

pub use episodic::{Episode, EpisodeStore};
pub use error::MemoryError;
