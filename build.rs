//! Build script to detect cache line size and set compile-time flags.
//!
//! For cross-compilation to systems with 128-byte cache lines, override with:
//!     RUSTFLAGS="--cfg cache_line_128" cargo build --target <target>

#[cfg(target_arch = "x86_64")]
mod ultimate_f64_avx512_nearest_one_sq_euc_kernel;

#[cfg(feature = "cpp_competitors")]
#[path = "build_cpp_competitors.rs"]
mod cpp_competitors;

fn main() {
    #[cfg(feature = "cpp_competitors")]
    cpp_competitors::build();

    // Register the custom cfg to avoid warnings
    println!("cargo::rustc-check-cfg=cfg(cache_line_128)");
    println!("cargo::rustc-check-cfg=cfg(kiddo_nightly)");
    println!("cargo::rustc-check-cfg=cfg(coverage_nightly)");

    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let rustc_version = std::process::Command::new(rustc)
        .arg("--version")
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .unwrap_or_default();

    if rustc_version.contains("nightly") {
        println!("cargo:rustc-cfg=kiddo_nightly");
    }

    let emit_warnings = std::env::var("KIDDO_CACHELINE_WARN")
        .map(|val| val == "1")
        .unwrap_or(false);

    let target = std::env::var("TARGET").unwrap();
    let host = std::env::var("HOST").unwrap();

    // Only detect on native builds to avoid cross-compilation issues
    if target == host {
        // Try L1 data cache first (most relevant for our use case)
        match yep_cache_line_size::get_cache_line_size(
            yep_cache_line_size::CacheLevel::L1,
            yep_cache_line_size::CacheType::Data,
        ) {
            Ok(cache_line_size) => {
                if emit_warnings {
                    println!(
                        "cargo:warning=Detected L1 data cache line size: {} bytes",
                        cache_line_size
                    );
                }

                if cache_line_size >= 128 {
                    println!("cargo:rustc-cfg=cache_line_128");
                    if emit_warnings {
                        println!(
                            "cargo:warning=Enabling f64 Block4 support (128-byte cache lines detected)"
                        );
                    }
                }
            }
            Err(e) => {
                if emit_warnings {
                    println!(
                        "cargo:warning=Failed to detect cache line size: {}. Assuming 64-byte cache lines.",
                        e
                    );
                }
            }
        }
    } else if emit_warnings {
        println!(
            "cargo:warning=Cross-compiling (host: {}, target: {}). \
             Assuming 64-byte cache lines. Override with RUSTFLAGS='--cfg cache_line_128' if needed.",
            host, target
        );
    }
    println!("cargo:rerun-if-changed=build.rs");
}
