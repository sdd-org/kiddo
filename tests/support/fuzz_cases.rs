//! Versioned replay keys use the same generators and assertions as the fuzz sweeps.
//! Bump the key version if generator semantics or its RNG algorithm change;
//! preserve an old replay path for any checked-in keys from previous versions.
use super::{FuzzConfig, ReproMeta};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

const TESTS: &[&str] = &[
    "fuzz_v6_mutable_f32",
    "fuzz_v6_mutable_f64",
    "fuzz_v6_immutable_f32",
    "fuzz_v6_immutable_f64",
    "fuzz_v6_metric_pruning_f32",
    "fuzz_v6_metric_pruning_f64",
    "fuzz_v6_metric_pruning_unsigned_fixed",
    "fuzz_v6_adversarial_fast_non_simd",
    "fuzz_v6_adversarial_fast_simd",
    "fuzz_v6_approx_nearest_one_fast_hard",
    "fuzz_v6_approx_nearest_one_quality",
];
const ENV_DEFAULTS: &[(&str, &str)] = &[
    ("KIDDO_FUZZ_V6_RUN_NON_SIMD", "1"),
    ("KIDDO_FUZZ_V6_RUN_SIMD", "1"),
    ("KIDDO_FUZZ_V6_RUN_SIMD_F32", "1"),
    ("KIDDO_FUZZ_V6_SIMD_FAST", "0"),
    ("KIDDO_FUZZ_APPROX_CASES", "4"),
    ("KIDDO_FUZZ_APPROX_QUERY_COUNT", "128"),
    ("KIDDO_FUZZ_APPROX_ENFORCE", "0"),
    ("KIDDO_FUZZ_APPROX_MAX_P99", "8"),
    ("KIDDO_FUZZ_APPROX_MAX_WORST", "32"),
];

pub(super) mod seed_string {
    use serde::{Deserialize, Deserializer, Serializer};
    pub fn serialize<S: Serializer>(value: &u64, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&value.to_string())
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<u64, D::Error> {
        String::deserialize(deserializer)?
            .parse()
            .map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Selection {
    kind: String,
    leaf: String,
    scalar: String,
    strategy: String,
    b: usize,
    k: usize,
}
impl From<ReproMeta> for Selection {
    fn from(m: ReproMeta) -> Self {
        Self {
            kind: m.kind.into(),
            leaf: m.leaf.into(),
            scalar: m.scalar.into(),
            strategy: m.strategy.into(),
            b: m.b,
            k: m.k,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct FailureKey {
    version: u8,
    test: String,
    simd: bool,
    pub config: FuzzConfig,
    selection: Option<Selection>,
    environment: BTreeMap<String, String>,
}
impl FailureKey {
    fn new(test: &str, config: FuzzConfig, selection: Option<Selection>) -> Self {
        Self {
            version: 1,
            test: test.into(),
            simd: cfg!(feature = "simd"),
            config,
            selection,
            environment: ENV_DEFAULTS
                .iter()
                .map(|(name, default)| {
                    (
                        (*name).into(),
                        std::env::var(name).unwrap_or_else(|_| (*default).into()),
                    )
                })
                .collect(),
        }
    }
    fn sampled(
        test: &str,
        mut config: FuzzConfig,
        seed: u64,
        query: Option<usize>,
        selection: Option<Selection>,
    ) -> Self {
        // Capture the effective generator inputs, including the random size draw.
        // Replay advances that same RNG, builds one tree case and runs one query.
        config.seed = seed;
        config.cases = 1;
        config.query_start = query.unwrap_or(0);
        config.query_count = usize::from(query.is_some());
        Self::new(test, config, selection)
    }
    pub fn legacy(cfg: FuzzConfig, meta: ReproMeta, seed: u64, query: Option<usize>) -> Self {
        let kind = if meta.kind == "v6_mutable" {
            "mutable"
        } else {
            "immutable"
        };
        Self::sampled(
            &format!("fuzz_v6_{kind}_{}", meta.scalar),
            cfg,
            seed,
            query,
            Some(meta.into()),
        )
    }
    pub fn generated(test: &str, cfg: FuzzConfig, seed: u64, query: Option<usize>) -> Self {
        Self::sampled(test, cfg, seed, query, None)
    }
    pub fn named(test: &str) -> Self {
        Self::new(test, FuzzConfig::from_env(), None)
    }
    fn validate(&self) {
        assert_eq!(self.version, 1, "unsupported fuzz-key generator version");
        assert!(
            TESTS.contains(&self.test.as_str()),
            "unknown fuzz test: {}",
            self.test
        );
        assert!(
            self.environment
                .keys()
                .all(|k| ENV_DEFAULTS.iter().any(|(name, _)| k == name)),
            "unknown replay environment input"
        );
        assert!(self.config.min_pow < usize::BITS && self.config.max_pow < usize::BITS);
        assert!(self
            .config
            .query_start
            .checked_add(self.config.query_count)
            .is_some());
        let legacy = self.test.starts_with("fuzz_v6_mutable_")
            || self.test.starts_with("fuzz_v6_immutable_");
        assert_eq!(
            legacy,
            self.selection.is_some(),
            "legacy keys need a matrix selector"
        );
        if let Some(selection) = &self.selection {
            assert!([2, 3, 4].contains(&selection.k));
            let mutable = self.test.starts_with("fuzz_v6_mutable_");
            assert_eq!(
                selection.kind,
                if mutable {
                    "v6_mutable"
                } else {
                    "v6_immutable"
                }
            );
            assert!(self.test.ends_with(&selection.scalar));
            assert!(["f32", "f64"].contains(&selection.scalar.as_str()));
            if mutable {
                assert_eq!(selection.leaf, "vec_of_arrays");
            } else {
                assert!(["flat_vec", "vec_of_arenas"].contains(&selection.leaf.as_str()));
            }
            match selection.strategy.as_str() {
                "eytzinger" if mutable => assert!([32, 64, 128].contains(&selection.b)),
                "eytzinger" | "donnelly" => assert!([16, 32, 64].contains(&selection.b)),
                "donnelly_simd_block3" | "donnelly_simd_block4" => {
                    assert!(!mutable && self.simd);
                    assert!([32, 64, 128].contains(&selection.b));
                    assert_eq!(
                        selection.strategy,
                        if selection.scalar == "f32" {
                            "donnelly_simd_block4"
                        } else {
                            "donnelly_simd_block3"
                        }
                    );
                }
                _ => panic!("unsupported fuzz strategy"),
            }
        } else {
            assert!(
                !self.test.starts_with("fuzz_v6_mutable_")
                    && !self.test.starts_with("fuzz_v6_immutable_"),
                "legacy replay needs a matrix selector"
            );
        }
        if self.test.contains("mutable_f") || self.test.contains("metric_pruning") {
            assert_eq!(self.config.cases, 1, "saved fuzz key must select one case");
            assert!(
                self.config.query_count <= 1,
                "saved fuzz key must select at most one query"
            );
        }
    }
}

pub(super) fn replay_key() -> Option<&'static FailureKey> {
    static KEY: OnceLock<Option<FailureKey>> = OnceLock::new();
    KEY.get_or_init(|| {
        std::env::var("KIDDO_FUZZ_REPLAY_KEY").ok().map(|json| {
            let key: FailureKey = serde_json::from_str(&json).expect("invalid fuzz replay key");
            key.validate();
            key
        })
    })
    .as_ref()
}
pub(super) fn selected(meta: ReproMeta) -> bool {
    replay_key()
        .and_then(|key| key.selection.as_ref())
        .is_none_or(|selection| *selection == Selection::from(meta))
}

static REPORT: Mutex<()> = Mutex::new(());
pub(super) struct CaseGuard {
    key: FailureKey,
    report: PathBuf,
}
impl Drop for CaseGuard {
    fn drop(&mut self) {
        // Rayon resumes worker panics on the caller without invoking its panic
        // hook again. Capture at the case boundary so construction panics from
        // those workers retain the caller's reproduction key too.
        if std::thread::panicking() {
            let _lock = REPORT.lock().unwrap_or_else(|e| e.into_inner());
            if let Err(error) = append_report(&self.report, &self.key) {
                eprintln!("could not save fuzz failure key: {error}");
            }
        }
    }
}
pub(super) fn enter(key: FailureKey) -> CaseGuard {
    if replay_key().is_some() {
        println!("KIDDO_FUZZ_REPLAY_ENTERED");
    }
    CaseGuard {
        key,
        report: std::env::var_os("KIDDO_FUZZ_FAILURE_REPORT")
            .map(PathBuf::from)
            .unwrap_or_else(|| "kd_tree_fuzz_failures.jsonl".into()),
    }
}
fn append_report(path: &Path, key: &FailureKey) -> std::io::Result<()> {
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(file, "{}", serde_json::to_string(key)?)
}

pub(super) fn replay_saved() {
    let path = std::env::var("KIDDO_FUZZ_REPLAY_FILE").unwrap_or_else(|_| {
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fuzz_failures.json").into()
    });
    let keys: Vec<FailureKey> =
        serde_json::from_slice(&std::fs::read(&path).expect("read saved fuzz keys"))
            .expect("saved fuzz failures must be a JSON array of versioned keys");
    let mut seen = HashSet::new();
    let mut replayed = 0;
    for key in keys {
        key.validate();
        let json = serde_json::to_string(&key).unwrap();
        assert!(seen.insert(json.clone()), "duplicate fuzz failure key");
        // CI runs both feature modes. A non-SIMD key must not silently replay
        // against a different SIMD leaf/traversal implementation, or vice versa.
        if key.simd != cfg!(feature = "simd") {
            continue;
        }
        replayed += 1;
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", &key.test, "--include-ignored", "--nocapture"])
            .env("KIDDO_FUZZ_REPLAY_KEY", &json)
            .env("RUST_TEST_THREADS", "1");
        for (name, value) in &key.environment {
            command.env(name, value);
        }
        let output = command.output().expect("execute one saved fuzz case");
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stdout.contains("running 1 test"),
            "fuzz key did not select a test: {json}\n{stdout}\n{stderr}"
        );
        assert!(
            stdout.contains("KIDDO_FUZZ_REPLAY_ENTERED"),
            "fuzz key did not enter a case: {json}\n{stdout}\n{stderr}"
        );
        assert!(
            output.status.success(),
            "saved fuzz case failed: {json}\n{stdout}\n{stderr}"
        );
    }
    println!(
        "Replayed {replayed} saved fuzz cases (simd={})",
        cfg!(feature = "simd")
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    fn cfg() -> FuzzConfig {
        FuzzConfig {
            seed: u64::MAX,
            cases: 8,
            query_count: 100,
            query_start: 0,
            min_pow: 8,
            max_pow: 20,
            perturb_min: -5,
            perturb_max: 5,
            max_nearest_n: 64,
        }
    }
    #[test]
    fn seed_round_trip_preserves_all_64_bits() {
        let key = FailureKey::generated("fuzz_v6_metric_pruning_f64", cfg(), u64::MAX, Some(73));
        let encoded = serde_json::to_string(&key).unwrap();
        assert!(encoded.contains("\"18446744073709551615\""));
        let decoded: FailureKey = serde_json::from_str(&encoded).unwrap();
        decoded.validate();
        assert_eq!(key, decoded);
        assert_eq!(decoded.config.cases, 1);
        assert_eq!(decoded.config.query_count, 1);
        assert_eq!(decoded.config.query_start, 73);
        assert_eq!(decoded.config.min_pow, 8);
        assert_eq!(decoded.config.max_pow, 20);
        assert_eq!(decoded.config.max_nearest_n, 64);
    }
    #[test]
    fn construction_key_does_not_run_queries() {
        let key = FailureKey::generated("fuzz_v6_metric_pruning_f32", cfg(), 19, None);
        key.validate();
        assert_eq!(key.config.query_count, 0);
        assert_eq!(key.config.seed, 19);
    }
    #[test]
    fn captures_local_and_propagated_worker_panics() {
        let report =
            std::env::temp_dir().join(format!("kiddo-capture-{}.jsonl", std::process::id()));
        let _ = std::fs::remove_file(&report);
        let key = FailureKey::generated("fuzz_v6_metric_pruning_f32", cfg(), 19, None);
        {
            let _case = CaseGuard {
                key: key.clone(),
                report: report.clone(),
            };
        }
        assert!(!report.exists(), "successful cases must not be recorded");
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        for worker in [false, true] {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _case = CaseGuard {
                    key: key.clone(),
                    report: report.clone(),
                };
                if worker {
                    pool.install(|| panic!("construction worker failure"));
                } else {
                    panic!("query assertion failure");
                }
            }));
            assert!(result.is_err());
        }
        let saved = std::fs::read_to_string(&report).unwrap();
        let keys: Vec<FailureKey> = saved
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(keys, vec![key.clone(), key]);
        std::fs::remove_file(report).unwrap();
    }
    #[test]
    fn unknown_test_cannot_recursively_launch_replay() {
        let mut key = FailureKey::named("fuzz_v6_saved_failures");
        key.version = 1;
        assert!(std::panic::catch_unwind(|| key.validate()).is_err());
    }
}
