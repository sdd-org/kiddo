//! Collect reproducible fuzz keys and publish a data-only regression PR in CI.
use serde::Deserialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const CORPUS: &str = "tests/fuzz_failures.json";
const MARKER: &str = "<!-- kiddo-fuzz-keys: ";
type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn key_id(key: &Value) -> Result<String> {
    // serde_json's default map representation sorts object keys.
    Ok(serde_json::to_string(key)?)
}

fn validate(key: &Value) -> Result<()> {
    const TESTS: &[&str] = &[
        "fuzz_v6_mutable_f32",
        "fuzz_v6_mutable_f64",
        "fuzz_v6_immutable_f32",
        "fuzz_v6_immutable_f64",
        "fuzz_v6_metric_pruning_f32",
        "fuzz_v6_metric_pruning_f64",
        "fuzz_v6_metric_pruning_unsigned_fixed",
        "fuzz_v6_metric_pruning_fixed_i32",
        "fuzz_v6_metric_pruning_unsigned_fixed_u16",
        "fuzz_v6_adversarial_fast_non_simd",
        "fuzz_v6_adversarial_fast_simd",
        "fuzz_v6_approx_nearest_one_fast_hard",
        "fuzz_v6_approx_nearest_one_quality",
    ];
    const ENV: &[&str] = &[
        "KIDDO_FUZZ_V6_RUN_NON_SIMD",
        "KIDDO_FUZZ_V6_RUN_SIMD",
        "KIDDO_FUZZ_V6_RUN_SIMD_F32",
        "KIDDO_FUZZ_V6_SIMD_FAST",
        "KIDDO_FUZZ_APPROX_CASES",
        "KIDDO_FUZZ_APPROX_QUERY_COUNT",
        "KIDDO_FUZZ_APPROX_ENFORCE",
        "KIDDO_FUZZ_APPROX_MAX_P99",
        "KIDDO_FUZZ_APPROX_MAX_WORST",
    ];
    let obj = key.as_object().ok_or("key must be an object")?;
    if obj.len() != 6
        || key["version"].as_u64() != Some(1)
        || !TESTS.contains(&key["test"].as_str().unwrap_or_default())
        || !key["simd"].is_boolean()
    {
        return Err("unsupported fuzz key".into());
    }
    let cfg = key["config"]
        .as_object()
        .ok_or("missing generator configuration")?;
    let fields = [
        "seed",
        "cases",
        "query_count",
        "query_start",
        "min_pow",
        "max_pow",
        "perturb_min",
        "perturb_max",
        "max_nearest_n",
    ];
    if cfg.len() != fields.len()
        || fields.iter().any(|k| !cfg.contains_key(*k))
        || cfg["seed"]
            .as_str()
            .and_then(|s| s.parse::<u64>().ok())
            .is_none()
        || [
            "cases",
            "query_count",
            "query_start",
            "min_pow",
            "max_pow",
            "max_nearest_n",
        ]
        .iter()
        .any(|k| cfg[*k].as_u64().is_none())
        || ["perturb_min", "perturb_max"]
            .iter()
            .any(|k| cfg[*k].as_i64().is_none())
    {
        return Err("invalid generator configuration".into());
    }
    if cfg["min_pow"].as_u64().unwrap() >= usize::BITS.into()
        || cfg["max_pow"].as_u64().unwrap() >= usize::BITS.into()
    {
        return Err("invalid generator size".into());
    }
    let env = key["environment"]
        .as_object()
        .ok_or("invalid replay environment")?;
    if env
        .iter()
        .any(|(k, v)| !ENV.contains(&k.as_str()) || !v.is_string())
    {
        return Err("invalid replay environment".into());
    }
    if !key["selection"].is_null() && !key["selection"].is_object() {
        return Err("invalid matrix selector".into());
    }
    Ok(())
}

fn read_keys(paths: &[PathBuf], lines: bool) -> Result<Vec<Value>> {
    let mut unique = BTreeMap::new();
    for path in paths {
        if !path.exists() {
            continue;
        }
        let text = fs::read_to_string(path)?;
        let values: Vec<Value> = if lines {
            text.lines()
                .filter(|s| !s.trim().is_empty())
                .map(serde_json::from_str)
                .collect::<std::result::Result<_, _>>()?
        } else {
            serde_json::from_str(&text)?
        };
        for key in values {
            validate(&key)?;
            unique.insert(key_id(&key)?, key);
        }
    }
    Ok(unique.into_values().collect())
}
fn write_keys(path: &Path, keys: &[Value]) -> Result<()> {
    fs::write(path, format!("{}\n", serde_json::to_string_pretty(keys)?))?;
    Ok(())
}
fn run(program: &str, args: &[&str]) -> Result<String> {
    let output = Command::new(program).args(args).output()?;
    if !output.status.success() {
        return Err(format!(
            "{program} {args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    Ok(String::from_utf8(output.stdout)?.trim().into())
}
fn verify(report: &Path, output: &Path, command: &[String]) -> Result<()> {
    if command.is_empty() {
        return Err("verify needs a replay command".into());
    }
    let dir = std::env::temp_dir().join(format!("kiddo-fuzz-verify-{}", std::process::id()));
    fs::create_dir(&dir)?;
    let result = (|| {
        let mut verified = Vec::new();
        for key in read_keys(&[report.into()], true)? {
            let case = dir.join("case.json");
            let repeated = dir.join("repeated.jsonl");
            let _ = fs::remove_file(&repeated);
            write_keys(&case, std::slice::from_ref(&key))?;
            let status = Command::new(&command[0])
                .args(&command[1..])
                .env("KIDDO_FUZZ_REPLAY_FILE", &case)
                .env("KIDDO_FUZZ_FAILURE_REPORT", &repeated)
                .status()?;
            let repeated = read_keys(&[repeated], true)?;
            if !status.success() && repeated.contains(&key) {
                verified.push(key);
            } else if !status.success() {
                return Err("replay failed without recapturing its key".into());
            } else {
                println!("Failure did not reproduce: {}", key_id(&key)?);
            }
        }
        write_keys(output, &verified)
    })();
    let _ = fs::remove_dir_all(dir);
    result
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PullRequest {
    head_ref_name: String,
    state: String,
    body: String,
    is_cross_repository: bool,
}
fn marker_keys(body: &str) -> Vec<Value> {
    body.lines()
        .find_map(|line| {
            line.strip_prefix(MARKER)?
                .strip_suffix(" -->")
                .and_then(|s| serde_json::from_str::<Vec<Value>>(s).ok())
        })
        .unwrap_or_default()
}
fn publish(reports: &[PathBuf], repo: &str, sha: &str, run_id: &str, run_url: &str) -> Result<()> {
    if !repo.contains('/')
        || !repo
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"/-_.".contains(&b))
        || sha.len() != 40
        || !sha
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        || run_id.is_empty()
        || !run_id.bytes().all(|b| b.is_ascii_digit())
    {
        return Err("invalid repository/run metadata".into());
    }
    if run("git", &["rev-parse", "HEAD"])? != sha
        || !run("git", &["status", "--porcelain", "--untracked-files=no"])?.is_empty()
    {
        return Err("publication must start from the clean tested commit".into());
    }
    let existing = read_keys(&[CORPUS.into()], false)?;
    let mut known: BTreeSet<String> = existing.iter().map(key_id).collect::<Result<_>>()?;
    let raw = run(
        "gh",
        &[
            "pr",
            "list",
            "--repo",
            repo,
            "--state",
            "all",
            "--limit",
            "1000",
            "--json",
            "headRefName,state,body,isCrossRepository",
        ],
    )?;
    let prs: Vec<PullRequest> = serde_json::from_str(&raw)?;
    for pr in &prs {
        if !pr.is_cross_repository
            && pr.state == "OPEN"
            && pr.head_ref_name.starts_with("fuzz-failure/")
        {
            for key in marker_keys(&pr.body) {
                if validate(&key).is_ok() {
                    known.insert(key_id(&key)?);
                }
            }
        }
    }
    let fresh: Vec<Value> = read_keys(reports, false)?
        .into_iter()
        .filter(|key| !known.contains(&key_id(key).unwrap()))
        .collect();
    if fresh.is_empty() {
        for pr in &prs {
            if !pr.is_cross_repository
                && pr.state == "OPEN"
                && pr
                    .head_ref_name
                    .starts_with(&format!("fuzz-failure/{run_id}-"))
            {
                run(
                    "gh",
                    &[
                        "workflow",
                        "run",
                        "fuzz-regressions.yml",
                        "--repo",
                        repo,
                        "--ref",
                        &pr.head_ref_name,
                    ],
                )?;
            }
        }
        println!("No new reproducible fuzz failures; no PR needed.");
        return Ok(());
    }
    // Used only for the branch name. Keys themselves are compared in full.
    let mut hash = 0xcbf29ce484222325u64;
    for byte in serde_json::to_vec(&fresh)? {
        hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
    }
    let branch = format!("fuzz-failure/{run_id}-{hash:016x}");
    if prs
        .iter()
        .any(|p| !p.is_cross_repository && p.head_ref_name == branch)
    {
        println!("This run already has a PR; leaving its branch intact.");
        return Ok(());
    }
    let reference = format!("refs/heads/{branch}");
    if !run("git", &["ls-remote", "--heads", "origin", &reference])?.is_empty() {
        run("git", &["fetch", "origin", &reference])?;
        let saved: Vec<Value> =
            serde_json::from_str(&run("git", &["show", &format!("FETCH_HEAD:{CORPUS}")])?)?;
        if !fresh.iter().all(|k| saved.contains(k)) {
            return Err("existing branch has unexpected keys; refusing to overwrite it".into());
        }
    } else {
        run("git", &["switch", "-c", &branch])?;
        let mut combined = existing;
        combined.extend(fresh.iter().cloned());
        write_keys(Path::new(CORPUS), &combined)?;
        run("git", &["config", "user.name", "github-actions[bot]"])?;
        run(
            "git",
            &[
                "config",
                "user.email",
                "41898282+github-actions[bot]@users.noreply.github.com",
            ],
        )?;
        run("git", &["add", "--", CORPUS])?;
        if run("git", &["diff", "--cached", "--name-only"])? != CORPUS {
            return Err("regression PR must only change the key array".into());
        }
        run(
            "git",
            &[
                "commit",
                "-m",
                &format!("test(fuzz): retain failures from {}", &sha[..12]),
            ],
        )?;
        run("git", &["push", "origin", &format!("HEAD:{reference}")])?;
    }
    let body = format!("The post-merge fuzz run found {} reproducible failure(s) at `{sha}`.\n\n[Failing run]({run_url}). This PR changes only `{CORPUS}`. The run artifacts preserve `Cargo.lock`.\n\nRun `just fuzz-replay` locally. Push a fix commit to this branch and retain the keys when merging. Automation will not overwrite developer commits.\n\nA dedicated replay workflow runs on this PR head.\n\n{MARKER}{} -->\n", fresh.len(), serde_json::to_string(&fresh)?);
    let mut child = Command::new("gh")
        .args([
            "pr",
            "create",
            "--repo",
            repo,
            "--base",
            "master",
            "--head",
            &branch,
            "--title",
            &format!("test(fuzz): reproduce failures from {}", &sha[..12]),
            "--body-file",
            "-",
        ])
        .stdin(Stdio::piped())
        .spawn()?;
    child
        .stdin
        .take()
        .ok_or("missing gh stdin")?
        .write_all(body.as_bytes())?;
    if !child.wait()?.success() {
        return Err("gh pr create failed".into());
    }
    run(
        "gh",
        &[
            "workflow",
            "run",
            "fuzz-regressions.yml",
            "--repo",
            repo,
            "--ref",
            &branch,
        ],
    )?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("collect") if args.len() == 3 => write_keys(Path::new(&args[2]), &read_keys(&[args[1].clone().into()], true)?),
        Some("verify") if args.len() >= 5 && args[3] == "--" => verify(Path::new(&args[1]), Path::new(&args[2]), &args[4..]),
        Some("publish") if args.len() >= 6 => publish(&args[5..].iter().map(PathBuf::from).collect::<Vec<_>>(), &args[1], &args[2], &args[3], &args[4]),
        _ => Err("usage: fuzz-report collect REPORT OUTPUT | verify REPORT OUTPUT -- COMMAND... | publish REPO SHA RUN_ID RUN_URL REPORT...".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn sample_key() -> Value {
        serde_json::json!({
            "version": 1, "test": "fuzz_v6_mutable_f64", "simd": false,
            "config": {"seed": "18446744073709551615", "cases": 1, "query_count": 1,
                "query_start": 73, "min_pow": 8, "max_pow": 20,
                "perturb_min": -5, "perturb_max": 5, "max_nearest_n": 64},
            "selection": {"kind": "v6_mutable", "leaf": "vec_of_arrays", "scalar": "f64",
                "strategy": "eytzinger", "b": 64, "k": 3}, "environment": {}
        })
    }
    #[test]
    fn rejects_unknown_tests_and_numeric_seeds() {
        let mut key = sample_key();
        validate(&key).unwrap();
        key["test"] = "fuzz_v6_saved_failures".into();
        assert!(validate(&key).is_err());
        key = sample_key();
        key["config"]["seed"] = 7.into();
        assert!(validate(&key).is_err());
    }
    #[cfg(unix)]
    #[test]
    fn replay_failures_without_a_recaptured_key_are_not_promoted() {
        let dir = std::env::temp_dir().join(format!("kiddo-report-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let report = dir.join("reported.jsonl");
        let output = dir.join("verified.json");
        std::fs::write(&report, format!("{}\n", sample_key())).unwrap();
        assert!(verify(&report, &output, &["false".into()]).is_err());
        assert!(!output.exists());
        verify(&report, &output, &["true".into()]).unwrap();
        assert!(read_keys(&[output], false).unwrap().is_empty());
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn marker_round_trips_keys_and_ignores_invalid_markers() {
        let key = serde_json::json!({"seed": "18446744073709551615"});
        let body = format!("{MARKER}{} -->", serde_json::to_string(&[&key]).unwrap());
        assert_eq!(marker_keys(&body), vec![key]);
        assert!(marker_keys("<!-- kiddo-fuzz-keys: [invalid] -->").is_empty());
    }
    #[test]
    fn object_identity_is_independent_of_field_order() {
        let a: Value = serde_json::from_str(r#"{"b":2,"a":1}"#).unwrap();
        let b: Value = serde_json::from_str(r#"{"a":1,"b":2}"#).unwrap();
        assert_eq!(key_id(&a).unwrap(), key_id(&b).unwrap());
    }
}
