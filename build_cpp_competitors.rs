//! Fetches and compiles the C++ competitor libraries used by
//! `benches/profile_cpp_competitors.rs`.
//!
//! Entirely gated behind the `cpp_competitors` feature (see root
//! `build.rs`), so a normal `cargo build`/`cargo test` never touches the
//! network or requires a C++ toolchain. CI's `cargo hack --each-feature`
//! sweeps do check this feature in isolation, though, and provision the
//! three system packages skd-tree needs (see `require_system_header` below)
//! in a dedicated step before that check runs. This is local, throwaway
//! evaluation tooling: sources are fetched into `OUT_DIR` (never committed)
//! and pinned to exact revisions/URLs for reproducibility.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const NANOFLANN_REPO: &str = "https://github.com/jlblancoc/nanoflann.git";
const NANOFLANN_REV: &str = "936c485fbde748595cda842dec5bea0eaae9a409"; // tag 1.11.0
                                                                        // ALGLIB free C++ edition, GPL 2+. Vendored source-only, never distributed;
                                                                        // local benchmarking only (GPL obligations trigger on distribution).
const ALGLIB_URL: &str = "https://www.alglib.net/translator/re/alglib-4.08.0.cpp.gpl.tgz";
const ALGLIB_DIR_NAME: &str = "alglib-cpp";
const PKDTREE_REPO: &str = "https://github.com/ucrparlay/Pkd-tree.git";
const PKDTREE_REV: &str = "9ea6fb51c3bc6e7e99fdbaec98f35315e59ad307";
// skd-tree (MIT). Artifact repo for an unpublished paper, so the revision pin
// matters more than usual: there are no tags and history may be rewritten.
const SKDTREE_REPO: &str = "https://github.com/achmichalop/skd-tree_2027.git";
const SKDTREE_REV: &str = "1c21f8a3f5b9af9f3e3adb8d51f76dc1c13a4149";

pub fn build() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set by cargo"));
    let vendor_dir = out_dir.join("cpp-competitors-vendor");
    fs::create_dir_all(&vendor_dir).expect("create vendor dir");

    let nanoflann_dir = fetch_git(&vendor_dir, "nanoflann", NANOFLANN_REPO, NANOFLANN_REV);
    let alglib_dir = fetch_tarball(&vendor_dir, "alglib", ALGLIB_URL, ALGLIB_DIR_NAME);
    let pkdtree_dir = fetch_git_with_submodules(&vendor_dir, "pkdtree", PKDTREE_REPO, PKDTREE_REV);
    let skdtree_dir = fetch_git(&vendor_dir, "skdtree", SKDTREE_REPO, SKDTREE_REV);

    build_nanoflann(&nanoflann_dir);
    build_alglib(&alglib_dir);
    build_pkdtree(&pkdtree_dir);
    build_skdtree(&skdtree_dir, &out_dir);

    println!("cargo:rerun-if-changed=build_cpp_competitors.rs");
    println!("cargo:rerun-if-changed=benches/cpp_shims/nanoflann_shim.cpp");
    println!("cargo:rerun-if-changed=benches/cpp_shims/alglib_shim.cpp");
    println!("cargo:rerun-if-changed=benches/cpp_shims/pkdtree_shim.cpp");
    println!("cargo:rerun-if-changed=benches/cpp_shims/skdtree_shim.cpp");
}

/// Unlike the other competitors, skd-tree needs two libraries this build
/// script does not vendor, both system packages. Probing for them here turns a
/// wall of C++ template errors into one actionable line.
///
/// - Boost (`utils/type.hpp` includes `boost/geometry.hpp`), header-only here.
/// - Armadillo (`dimRanking.hpp` uses `arma::mat` and friends to score split
///   dimensions), which must also be linked, and pulls in BLAS/LAPACK.
fn require_system_header(relative: &str, env_var: &str, package_hint: &str) -> Option<String> {
    const PREFIXES: [&str; 3] = [
        "/usr/include",
        "/usr/local/include",
        "/opt/homebrew/include",
    ];
    if let Ok(prefix) = env::var(env_var) {
        if Path::new(&prefix).join(relative).exists() {
            return Some(prefix);
        }
        panic!("cpp_competitors: {env_var}={prefix} does not contain {relative}");
    }
    if PREFIXES
        .iter()
        .any(|prefix| Path::new(prefix).join(relative).exists())
    {
        return None;
    }
    panic!(
        "cpp_competitors: skd-tree needs {relative}, which is not vendored because it is a \
         system package. Install it ({package_hint}) or set {env_var} to its include \
         directory."
    );
}

/// `utils/datautils.hpp` includes `<tpie/tpie.h>`, but nothing in the
/// repository references a single `tpie::` symbol -- it is a dead include, and
/// TPIE is not packaged on most distributions. Satisfying it with an empty
/// header is less invasive than patching vendored source: their code is
/// compiled exactly as written, and an unused include that resolves to nothing
/// is equivalent to the include not being there.
fn write_tpie_stub(out_dir: &Path) -> PathBuf {
    let stub_dir = out_dir.join("skdtree-stubs");
    let tpie_dir = stub_dir.join("tpie");
    fs::create_dir_all(&tpie_dir).expect("create tpie stub dir");
    fs::write(
        tpie_dir.join("tpie.h"),
        "// Deliberately empty. See write_tpie_stub in build_cpp_competitors.rs:\n         // skd-tree includes <tpie/tpie.h> but never uses a tpie symbol.\n         #pragma once\n",
    )
    .expect("write tpie stub");
    stub_dir
}

fn build_skdtree(skdtree_dir: &Path, out_dir: &Path) {
    let boost_prefix = require_system_header(
        "boost/geometry.hpp",
        "BOOST_INCLUDE_DIR",
        "Arch: `pacman -S boost`, Debian/Ubuntu: `apt install libboost-dev`, macOS: `brew install boost`",
    );
    let armadillo_prefix = require_system_header(
        "armadillo",
        "ARMADILLO_INCLUDE_DIR",
        "Arch: `pacman -S armadillo`, Debian/Ubuntu: `apt install libarmadillo-dev`, macOS: `brew install armadillo`",
    );
    let ensmallen_prefix = require_system_header(
        "ensmallen.hpp",
        "ENSMALLEN_INCLUDE_DIR",
        "Arch: `paru -S ensmallen` (AUR), Debian/Ubuntu: `apt install libensmallen-dev`, macOS: `brew install ensmallen`",
    );
    let stub_dir = write_tpie_stub(out_dir);

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++20")
        .flag_if_supported("-march=native")
        // skd-tree searches nodes with AVX-512 intrinsics unconditionally, so
        // there is no scalar fallback to fall back to.
        .flag_if_supported("-mavx512f")
        .flag_if_supported("-mavx512bw")
        // Its own headers are included as "indices/..." and "utils/..."
        // relative to the repository root.
        .include(skdtree_dir)
        .include(&stub_dir)
        .file("benches/cpp_shims/skdtree_shim.cpp")
        .warnings(false);
    for prefix in [boost_prefix, armadillo_prefix, ensmallen_prefix]
        .into_iter()
        .flatten()
    {
        build.include(prefix);
    }
    build.compile("skdtree_shim");
    // Armadillo's matrix routines are not header-only; dimRanking pulls in
    // BLAS/LAPACK through them.
    println!("cargo:rustc-link-lib=armadillo");
}

fn run(cmd: &mut Command, what: &str) {
    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("cpp_competitors: failed to spawn {what}: {e}"));
    if !status.success() {
        panic!("cpp_competitors: {what} failed with {status}");
    }
}

fn fetch_git(vendor_dir: &Path, name: &str, repo: &str, rev: &str) -> PathBuf {
    let dest = vendor_dir.join(name);
    let marker = dest.join(".fetched-rev");
    if let Ok(fetched) = fs::read_to_string(&marker) {
        if fetched.trim() == rev {
            return dest;
        }
    }
    if dest.exists() {
        fs::remove_dir_all(&dest).expect("remove stale vendor checkout");
    }
    println!("cargo:warning=cpp_competitors: cloning {name} from {repo}@{rev}");
    run(
        Command::new("git")
            .args(["clone", "--quiet", repo])
            .arg(&dest),
        &format!("git clone {name}"),
    );
    run(
        Command::new("git")
            .arg("-C")
            .arg(&dest)
            .args(["checkout", "--quiet", rev]),
        &format!("git checkout {name}@{rev}"),
    );
    fs::write(&marker, rev).expect("write fetch marker");
    dest
}

fn fetch_git_with_submodules(vendor_dir: &Path, name: &str, repo: &str, rev: &str) -> PathBuf {
    let dest = vendor_dir.join(name);
    let marker = dest.join(".fetched-rev");
    if let Ok(fetched) = fs::read_to_string(&marker) {
        if fetched.trim() == rev {
            return dest;
        }
    }
    if dest.exists() {
        fs::remove_dir_all(&dest).expect("remove stale vendor checkout");
    }
    println!("cargo:warning=cpp_competitors: cloning {name} from {repo}@{rev} (with submodules)");
    run(
        Command::new("git")
            .args(["clone", "--quiet", repo])
            .arg(&dest),
        &format!("git clone {name}"),
    );
    run(
        Command::new("git")
            .arg("-C")
            .arg(&dest)
            .args(["checkout", "--quiet", rev]),
        &format!("git checkout {name}@{rev}"),
    );
    run(
        Command::new("git").arg("-C").arg(&dest).args([
            "submodule",
            "update",
            "--init",
            "--recursive",
        ]),
        &format!("git submodule update {name}"),
    );
    fs::write(&marker, rev).expect("write fetch marker");
    dest
}

fn fetch_tarball(vendor_dir: &Path, name: &str, url: &str, extracted_name: &str) -> PathBuf {
    let dest = vendor_dir.join(name);
    if dest.exists() {
        return dest;
    }
    let tarball = vendor_dir.join(format!("{name}.tar.gz"));
    println!("cargo:warning=cpp_competitors: downloading {name} from {url}");
    run(
        Command::new("curl")
            .args(["-sSfL", "--retry", "3", "-o"])
            .arg(&tarball)
            .arg(url),
        &format!("downloading {name}"),
    );
    run(
        Command::new("tar")
            .arg("xzf")
            .arg(&tarball)
            .arg("-C")
            .arg(vendor_dir),
        &format!("extracting {name}"),
    );
    let _ = fs::remove_file(&tarball);
    let extracted = vendor_dir.join(extracted_name);
    fs::rename(&extracted, &dest).unwrap_or_else(|e| {
        panic!("cpp_competitors: expected {name} tarball to extract to {extracted_name}: {e}")
    });
    dest
}

fn build_nanoflann(nanoflann_dir: &Path) {
    cc::Build::new()
        .cpp(true)
        .std("c++14")
        .include(nanoflann_dir.join("include"))
        .file("benches/cpp_shims/nanoflann_shim.cpp")
        .warnings(false)
        .compile("nanoflann_shim");
}

fn build_alglib(alglib_dir: &Path) {
    let src = alglib_dir.join("src");
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .include(&src)
        .file("benches/cpp_shims/alglib_shim.cpp")
        .warnings(false);
    for entry in fs::read_dir(&src).expect("read alglib src dir") {
        let path = entry.expect("read alglib src entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("cpp") {
            continue;
        }
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        // ARM/RISC-V specific SIMD kernels; irrelevant and non-portable on x86_64.
        if name.contains("kernels_neon") || name.contains("kernels_rvv10") {
            continue;
        }
        build.file(path);
    }
    build.compile("alglib_shim");
}

fn build_pkdtree(pkdtree_dir: &Path) {
    cc::Build::new()
        .cpp(true)
        .std("c++20")
        .flag_if_supported("-pthread")
        .flag_if_supported("-mcx16")
        .flag_if_supported("-march=native")
        .include(pkdtree_dir.join("include"))
        .include(pkdtree_dir.join("parlaylib/include"))
        .file("benches/cpp_shims/pkdtree_shim.cpp")
        .warnings(false)
        .compile("pkdtree_shim");
    println!("cargo:rustc-link-lib=pthread");
}
