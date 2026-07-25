//! Donnelly-family stem strategies.
//!
//! The Donnelly family shares the same broad layout idea: stems are arranged in
//! fixed-height minor triangles parameterized by a `const BH: usize` block
//! height, so traversal stays cache-friendly while still supporting the normal
//! `StemStrategy` query surface.
//!
//! The variants differ by how much traversal work is specialized:
//!
//! - [`Donnelly`] is the default scalar variant. It traverses one level at a
//!   time, advances dimensions once per level, and includes the current
//!   software-prefetch behavior.
//! - [`DonnellyNoPf`] is the same scalar traversal shape without the “default”
//!   naming. It exists so the public API can distinguish the non-prefetched
//!   scalar baseline from [`Donnelly`].
//! - [`DonnellyUnrolled`] keeps the same Donnelly ordering and per-level
//!   dimension cadence, but unrolls traversal within each minor triangle.
//! - [`DonnellyUnrolledBlockDim`] uses the same unrolled structure, but changes
//!   dimensions once per block rather than once per level. It is primarily the
//!   scalar reference variant for the more specialized block-at-once traversal
//!   strategies.
//! - [`DonnellySimdDescent`] performs block-at-once SIMD child selection during
//!   descent, but still uses scalar backtracking and pruning.
//! - [`DonnellySimdFull`] takes the same block-at-once descent idea and also
//!   uses SIMD-aware backtracking and pruning.
//!
//! Internally, these variants share `core` for scalar Donnelly indexing/state
//! and `simd_full` for the reusable SIMD comparison and backtrack machinery.

mod no_pf;
mod scalar;
mod simd_descent;
mod unrolled;
mod unrolled_block_dim;
#[cfg(feature = "test_utils")]
mod unrolled_leaf_embedded;

#[doc(hidden)]
pub mod core;
#[doc(hidden)]
pub mod simd_full;

#[doc(inline)]
pub use no_pf::DonnellyNoPf;
#[doc(inline)]
pub use scalar::Donnelly;
#[doc(inline)]
pub use simd_descent::DonnellySimdDescent;
#[doc(inline)]
pub use simd_full::DonnellySimdFull;
#[doc(inline)]
pub use unrolled::DonnellyUnrolled;
#[doc(inline)]
pub use unrolled_block_dim::DonnellyUnrolledBlockDim;
#[cfg(feature = "test_utils")]
#[doc(hidden)]
pub use unrolled_leaf_embedded::DonnellyUnrolledLeafEmbedded3;
