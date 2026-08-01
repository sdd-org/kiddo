use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::env;
use std::fs::{self, File};
use std::io::{self, BufReader};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Serialize)]
struct Entry {
    benchmark: String,
    metadata: Value,
    estimates: Value,
}

#[derive(Serialize)]
struct Export<'a> {
    schema_version: u8,
    criterion_root: String,
    collected_at_unix_ms: u128,
    filters: &'a [String],
    results: Vec<Entry>,
}

#[derive(Deserialize)]
struct BenchmarkRecord {
    id: BenchmarkId,
    latest_record: PathBuf,
}

#[derive(Deserialize, Serialize)]
struct BenchmarkId {
    group_id: String,
    function_id: Option<String>,
    value_str: Option<String>,
    throughput: Option<Throughput>,
}

#[derive(Deserialize, Serialize)]
enum Throughput {
    Bytes(u64),
    Elements(u64),
}

#[derive(Deserialize)]
struct SavedStatistics {
    estimates: Value,
}

fn matches_filter(benchmark: &str, filters: &[String]) -> bool {
    filters.iter().any(|filter| {
        let filter = filter.trim_end_matches('/');
        benchmark == filter
            || benchmark
                .strip_prefix(filter)
                .is_some_and(|suffix| suffix.starts_with('/'))
    })
}

fn invalid_data(path: &Path, error: impl std::fmt::Display) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("could not decode {}: {error}", path.display()),
    )
}

fn benchmark_name(id: &BenchmarkId) -> String {
    let mut name = id.group_id.clone();
    for component in [&id.function_id, &id.value_str].into_iter().flatten() {
        if !component.is_empty() {
            name.push('/');
            name.push_str(component);
        }
    }
    name
}

fn collect_cbor_records(
    dir: &Path,
    filters: &[String],
    entries: &mut BTreeMap<String, Entry>,
) -> io::Result<()> {
    if !dir.is_dir() {
        return Ok(());
    }

    for directory_entry in fs::read_dir(dir)? {
        let directory_entry = directory_entry?;
        let path = directory_entry.path();
        let file_type = directory_entry.file_type()?;

        if file_type.is_dir() {
            collect_cbor_records(&path, filters, entries)?;
            continue;
        }
        if path.file_name().and_then(|name| name.to_str()) != Some("benchmark.cbor") {
            continue;
        }

        let record: BenchmarkRecord = serde_cbor::from_reader(BufReader::new(File::open(&path)?))
            .map_err(|error| invalid_data(&path, error))?;
        let benchmark = benchmark_name(&record.id);
        if !matches_filter(&benchmark, filters) {
            continue;
        }

        let measurement_path = path.parent().unwrap_or(dir).join(&record.latest_record);
        let statistics: SavedStatistics =
            serde_cbor::from_reader(BufReader::new(File::open(&measurement_path)?))
                .map_err(|error| invalid_data(&measurement_path, error))?;
        let metadata =
            serde_json::to_value(record.id).map_err(|error| invalid_data(&path, error))?;

        entries.insert(
            benchmark.clone(),
            Entry {
                benchmark,
                metadata,
                estimates: statistics.estimates,
            },
        );
    }

    Ok(())
}

fn collect_legacy_json(
    root: &Path,
    dir: &Path,
    filters: &[String],
    entries: &mut BTreeMap<String, Entry>,
) -> io::Result<()> {
    for directory_entry in fs::read_dir(dir)? {
        let directory_entry = directory_entry?;
        let path = directory_entry.path();
        let file_type = directory_entry.file_type()?;

        if file_type.is_dir() {
            collect_legacy_json(root, &path, filters, entries)?;
            continue;
        }
        if path.file_name().and_then(|name| name.to_str()) != Some("estimates.json") {
            continue;
        }

        let relative = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        if relative.contains("/base/") || relative.contains("/change/") {
            continue;
        }
        let directory_benchmark = relative
            .trim_end_matches("/estimates.json")
            .trim_end_matches("/new")
            .to_string();

        let metadata_path = path.with_file_name("benchmark.json");
        let metadata: Value = serde_json::from_reader(BufReader::new(File::open(&metadata_path)?))
            .map_err(|error| invalid_data(&metadata_path, error))?;
        let benchmark = metadata
            .get("full_id")
            .and_then(Value::as_str)
            .unwrap_or(&directory_benchmark)
            .to_string();
        if !matches_filter(&benchmark, filters) {
            continue;
        }

        let estimates = serde_json::from_reader(BufReader::new(File::open(&path)?))
            .map_err(|error| invalid_data(&path, error))?;
        entries.entry(benchmark.clone()).or_insert(Entry {
            benchmark,
            metadata,
            estimates,
        });
    }

    Ok(())
}

fn main() -> io::Result<()> {
    let mut args = env::args().skip(1);
    let criterion_root = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "target/criterion".to_string()),
    );
    let output_path = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "target/criterion-results.json".to_string()),
    );
    let filters: Vec<String> = args.collect();
    if filters.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "provide at least one Criterion benchmark group filter",
        ));
    }

    let mut entries = BTreeMap::new();
    collect_legacy_json(&criterion_root, &criterion_root, &filters, &mut entries)?;
    collect_cbor_records(&criterion_root.join("data/main"), &filters, &mut entries)?;
    if entries.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "no Criterion results under {} matched: {}",
                criterion_root.display(),
                filters.join(", ")
            ),
        ));
    }

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let export = Export {
        schema_version: 1,
        criterion_root: criterion_root.to_string_lossy().into_owned(),
        collected_at_unix_ms: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis(),
        filters: &filters,
        results: entries.into_values().collect(),
    };
    let output = File::create(output_path)?;
    serde_json::to_writer_pretty(output, &export).map_err(io::Error::other)
}

#[cfg(test)]
mod tests {
    use super::matches_filter;

    #[test]
    fn filter_matches_a_group_and_its_descendants() {
        let filters = vec!["suite/f64/donnelly".to_string()];
        assert!(matches_filter("suite/f64/donnelly", &filters));
        assert!(matches_filter("suite/f64/donnelly/536870912", &filters));
        assert!(!matches_filter("suite/f64/eytzinger/536870912", &filters));
    }

    #[test]
    fn full_benchmark_filter_excludes_other_tree_sizes() {
        let filters = vec!["suite/f64/donnelly/536870912".to_string()];
        assert!(matches_filter(
            "suite/f64/donnelly/536870912",
            &filters
        ));
        assert!(!matches_filter(
            "suite/f64/donnelly/268435456",
            &filters
        ));
    }
}
