//! T6.3 fixed qrels: synthetic recall corpus (SciFact is not fetchable
//! offline, so this file IS the frozen qrel set — edit it only with a new
//! baseline measurement, never to make a red gate green).
//!
//! 30 episodes / 12 queries, one relevant doc each. Queries paraphrase their
//! target sharing rare tokens; distractors share only common words.

use std::collections::HashSet;

use kitty_memory::{
    ndcg_at_k, normalize_embedding, recall_at_k, reciprocal_rank, EpisodeStore, RecallMode,
    RrfParams, ScoringWeights, ENV_HYBRID_RECALL, MODEL_DIM,
};

/// Deterministic PRNG (splitmix64) for synthetic dense vectors — same
/// construction as the embedding unit tests, duplicated here because
/// integration tests cannot see that private helper. Fixed seed ⇒ frozen
/// corpus: identical vectors on every run.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// Random unit vector (deterministic given `rng`).
fn rand_unit_vec(rng: &mut SplitMix64) -> Vec<f32> {
    let mut v: Vec<f32> = (0..MODEL_DIM).map(|_| rng.next_f32()).collect();
    normalize_embedding(&mut v);
    v
}

const TARGETS: &[(&str, &str)] = &[
    (
        "fix flaky websocket reconnect loop in gateway service",
        "gateway/ws.rs",
    ),
    ("deploy kubernetes cluster to production", "infra/k8s.yaml"),
    ("python fibonacci memoization benchmark", "bench/fib.py"),
    (
        "rust borrow checker fight in parser module",
        "src/parser.rs",
    ),
    ("css flexbox centering workaround safari", "web/layout.css"),
    ("bash backup rotation cron script", "ops/backup.sh"),
    ("postgres connection pool exhaustion timeout", "db/pool.rs"),
    (
        "graphql federation gateway schema stitching",
        "api/federation.rs",
    ),
    ("terraform state lock dynamodb backend", "infra/state.tf"),
    (
        "prometheus alertmanager silence webhook receiver",
        "ops/alerts.yaml",
    ),
    (
        "kafka consumer lag rebalance storm",
        "streaming/consumer.rs",
    ),
    ("redis cache stampede thundering herd", "cache/warmer.rs"),
];

const DISTRACTORS: &[&str] = &[
    "fix typo in readme contributing guide",
    "deploy staging environment helm chart",
    "python script to rename files in bulk",
    "rust clippy warnings cleanup",
    "css dark mode toggle variables",
    "bash script to check disk usage",
    "review pull request template checklist",
    "update changelog release notes",
    "fix off-by-one pagination bug",
    "deploy docs site static assets",
    "python virtualenv setup instructions",
    "rust serde json deserialization error",
    "css grid layout dashboard cards",
    "bash history search keybinding",
    "fix login redirect loop",
    "deploy feature flag rollout plan",
    "write unit tests for retry helper",
    "refactor logging middleware config",
];

const QUERIES: &[(&str, usize)] = &[
    ("websocket gateway drops connection intermittently", 0),
    ("kubernetes production rollout stuck pending", 1),
    ("fibonacci memoization too slow python", 2),
    ("borrow checker rejects parser lifetime", 3),
    ("safari flexbox vertical center broken", 4),
    ("cron backup rotation script fails nightly", 5),
    ("postgres pool timeouts under load", 6),
    ("federation gateway schema composition errors", 7),
    ("terraform dynamodb lock contention", 8),
    ("alertmanager webhook silences ignored", 9),
    ("kafka rebalance storm consumer lag", 10),
    ("redis thundering herd cache stampede", 11),
];

fn build_store() -> (
    EpisodeStore,
    tempfile::TempDir,
    Vec<String>,
    i64,
    Vec<Vec<f32>>,
) {
    let tmp = tempfile::TempDir::new().unwrap();
    let store = EpisodeStore::open(tmp.path().join("qrels.db").to_str().unwrap()).unwrap();
    // Spread timestamps over 30 days so recency is live, not tied.
    let now = 1_800_000_000_i64;
    // Synthetic dense channel: query q_i correlates ONLY with target t_i
    // (cos ≈ 0.96); every other episode is quasi-orthogonal (cos ≈ 0).
    // Same stream order on every run ⇒ frozen vectors.
    let mut rng = SplitMix64::new(0x51ab_2026);
    let mut query_embs = Vec::new();
    let mut ids = Vec::new();
    for (i, (goal, file)) in TARGETS.iter().enumerate() {
        let mut ep =
            kitty_memory::Episode::new((*goal).into(), vec![(*file).into()], None, "done".into());
        ep.timestamp = now - (i as i64 + 1) * 2 * 86_400;
        ep.id = kitty_memory::Episode::compute_id(&ep.goal, ep.timestamp);
        ids.push(ep.id.clone());
        let q = rand_unit_vec(&mut rng);
        let r = rand_unit_vec(&mut rng);
        let mut t: Vec<f32> = q.iter().zip(r.iter()).map(|(a, b)| a + 0.3 * b).collect();
        normalize_embedding(&mut t);
        query_embs.push(q);
        store.save_with_embedding(&ep, &t).unwrap();
    }
    for (j, goal) in DISTRACTORS.iter().enumerate() {
        let mut ep =
            kitty_memory::Episode::new((*goal).into(), vec!["x".into()], None, "done".into());
        ep.timestamp = now - (j as i64 + 1) * 86_400;
        ep.id = kitty_memory::Episode::compute_id(&ep.goal, ep.timestamp);
        let d = rand_unit_vec(&mut rng);
        store.save_with_embedding(&ep, &d).unwrap();
    }
    (store, tmp, ids, now, query_embs)
}

fn ranked_ids(
    store: &EpisodeStore,
    query: &str,
    query_emb: &[f32],
    now: i64,
) -> (Vec<String>, Vec<String>) {
    let bm25 = store
        .recall_bm25(query, 10)
        .unwrap()
        .into_iter()
        .map(|e| e.id)
        .collect();
    let hybrid = store
        .recall_hybrid_ranked(
            query,
            Some(query_emb),
            10,
            &RrfParams {
                n: 50,
                ..RrfParams::default()
            },
            &ScoringWeights::default(),
            now,
        )
        .unwrap()
        .into_iter()
        .map(|e| e.id)
        .collect();
    (bm25, hybrid)
}

#[test]
fn qrels_hybrid_meets_baseline() {
    let (store, _tmp, ids, now, query_embs) = build_store();
    let (mut sum_r_b, mut sum_r_h, mut sum_m_b, mut sum_m_h, mut sum_n_b, mut sum_n_h) =
        (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    for (query, target) in QUERIES.iter() {
        let rel: HashSet<String> = [ids[*target].clone()].into_iter().collect();
        let (bm25, hybrid) = ranked_ids(&store, query, &query_embs[*target], now);
        let rb = bm25.iter().position(|id| rel.contains(id));
        let rh = hybrid.iter().position(|id| rel.contains(id));
        println!(
            "q={query:?} target={} bm25_rank={rb:?} hybrid_rank={rh:?}",
            TARGETS[*target].0
        );
        sum_r_b += recall_at_k(&bm25, &rel, 10);
        sum_r_h += recall_at_k(&hybrid, &rel, 10);
        sum_m_b += reciprocal_rank(&bm25, &rel);
        sum_m_h += reciprocal_rank(&hybrid, &rel);
        sum_n_b += ndcg_at_k(&bm25, &rel, 10);
        sum_n_h += ndcg_at_k(&hybrid, &rel, 10);
    }
    let n = QUERIES.len() as f32;
    let (r_b, r_h, m_b, m_h, n_b, n_h) = (
        sum_r_b / n,
        sum_r_h / n,
        sum_m_b / n,
        sum_m_h / n,
        sum_n_b / n,
        sum_n_h / n,
    );
    println!("BM25   recall@10={r_b:.3} mrr={m_b:.3} ndcg@10={n_b:.3}");
    println!("HYBRID recall@10={r_h:.3} mrr={m_h:.3} ndcg@10={n_h:.3}");
    // HU-6 bar on this frozen corpus (measured 2026-09-24, see report).
    assert!(r_h >= 0.78, "hybrid recall@10 {r_h:.3} below HU-6 bar 0.78");
    // HU-6 Δ: hybrid must not regress vs the lexical baseline.
    //
    // Measured 2026-09-24 on this frozen corpus: BM25 == hybrid == 1.000 on
    // all three metrics — the corpus is lexically saturated (every target
    // already rank 0 by BM25), so a strict `>`/`+ε` superiority gate is
    // unmeasurable here and would fail on perfection (1.0 > 1.0 + ε is
    // false). This stays an explicit NO-REGRESSION gate (`>=`); dense
    // decisiveness is pinned independently by
    // `qrels_dense_channel_decisive_without_lexical_overlap` below.
    // Discrimination (Δ > 0) needs a harder corpus — recorded as debt, not
    // faked with a synthetic margin.
    assert!(
        n_h >= n_b,
        "hybrid nDCG@10 {n_h:.3} regressed vs BM25 {n_b:.3}"
    );
    // BM25 itself must clear a floor (guards corpus rot, not the ranker).
    assert!(
        r_b >= 0.78,
        "BM25 recall@10 {r_b:.3} below corpus floor 0.78"
    );
}

/// Pass 2 Lane B — brazo Jaccard@3 sobre holdout disjunto.
///
/// Partición fija y disjunta del corpus congelado: dev = queries[0..6],
/// holdout = queries[6..12] (sin solape). Compara el brazo léxico gated
/// (`RecallMode::Jaccard`, sin `fill_recent`) contra el híbrido gated
/// (`RecallMode::Hybrid` con selector `KITTY_HYBRID_RECALL=1`) a k=3.
///
/// Resultado documentado (2026-10-24, corpus congelado): ambos brazos
/// resuelven el holdout léxicamente saturado; el gate NO exige que el
/// híbrido gane — solo que no colapse (piso) y que el Jaccard gated no
/// rellene con recientes. La discriminación densa sigue pinned por
/// `qrels_dense_channel_decisive_without_lexical_overlap`.
#[test]
fn qrels_jaccard_at3_holdout_disjoint() {
    let (store, _tmp, ids, now, query_embs) = build_store();
    const K: usize = 3;
    // Holdout disjunto: segunda mitad, sin solape con dev [0..6).
    let holdout: &[&usize] = &[&6, &7, &8, &9, &10, &11];
    assert!(
        holdout.iter().all(|&&i| i >= 6),
        "holdout debe ser disjunto del dev [0..6)"
    );
    // Selector default-off: habilitar el híbrido solo dentro del test.
    let saved = std::env::var(ENV_HYBRID_RECALL).ok();
    // SAFETY: test-only; ningún otro test de qrels lee esta env.
    unsafe {
        std::env::set_var(ENV_HYBRID_RECALL, "1");
    }
    let mut sum_j = 0.0_f32;
    let mut sum_h = 0.0_f32;
    for &&qi in holdout {
        let (query, target) = QUERIES[qi];
        let rel: HashSet<String> = [ids[target].clone()].into_iter().collect();
        let jaccard: Vec<String> = store
            .recall_gated(query, None, K, RecallMode::Jaccard, now)
            .unwrap()
            .into_iter()
            .map(|e| e.id)
            .collect();
        let hybrid: Vec<String> = store
            .recall_gated(query, Some(&query_embs[target]), K, RecallMode::Hybrid, now)
            .unwrap()
            .into_iter()
            .map(|e| e.id)
            .collect();
        let rj = recall_at_k(&jaccard, &rel, K);
        let rh = recall_at_k(&hybrid, &rel, K);
        println!("holdout q={query:?} target={} jaccard@3={rj:.3} hybrid@3={rh:.3}", TARGETS[target].0);
        // Sin fill: el brazo que no puntúa devuelve vacío en vez de
        // recientes — aquí ambas queries tienen overlap, así que ambos
        // deben recuperar al relevante dentro del top-3.
        assert!(
            jaccard.contains(&ids[target]),
            "jaccard gated debe hallar el target en holdout q={query:?}"
        );
        sum_j += rj;
        sum_h += rh;
    }
    if let Some(v) = saved {
        // SAFETY: test-only restore.
        unsafe {
            std::env::set_var(ENV_HYBRID_RECALL, v);
        }
    } else {
        // SAFETY: test-only restore.
        unsafe {
            std::env::remove_var(ENV_HYBRID_RECALL);
        }
    }
    let n = holdout.len() as f32;
    println!("HOLDOUT recall@3 jaccard={:.3} hybrid={:.3} (n={n})", sum_j / n, sum_h / n);
    // Piso del holdout (corpus saturado): ambos ≥ 0.5. Sin exigencia de
    // victoria del híbrido — documentar, no ganar.
    assert!(
        sum_j / n >= 0.5,
        "jaccard@3 holdout bajo el piso: {:.3}",
        sum_j / n
    );
    assert!(
        sum_h / n >= 0.5,
        "hybrid@3 holdout bajo el piso: {:.3}",
        sum_h / n
    );
}

/// Dense-channel liveness: with ZERO lexical overlap the BM25 arm must miss
/// while the hybrid arm (fed the target's synthetic query vector) still
/// ranks the target top-1. Fails if dense is skipped, miswired, or unread.
#[test]
fn qrels_dense_channel_decisive_without_lexical_overlap() {
    let (store, _tmp, ids, now, query_embs) = build_store();
    let gibberish = "xyzzy plugh qwerty asdf";
    let bm25 = store.recall_bm25(gibberish, 10).unwrap();
    // No lexical overlap ⇒ BM25 falls back to recency order (a distractor),
    // never the true target at top-1.
    assert!(
        bm25.first().map(|e| &e.id) != Some(&ids[0]),
        "lexical arm must not fluke the target top-1 with zero overlap"
    );
    let hybrid = store
        .recall_hybrid_ranked(
            gibberish,
            Some(&query_embs[0]),
            10,
            &RrfParams {
                n: 50,
                ..RrfParams::default()
            },
            &ScoringWeights::default(),
            now,
        )
        .unwrap();
    assert!(
        !hybrid.is_empty() && hybrid[0].id == ids[0],
        "dense arm must rank target top-1 without lexical overlap"
    );
}
