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
//! - [`DonnellyCyclicSimdDescent`] performs the same SIMD child selection with
//!   per-depth query lanes, preserving the ordinary cyclic axis cadence.
//! - [`DonnellyCyclicSimdFull`] builds on cyclic SIMD descent with native
//!   block-level SIMD pruning and backtracking.
//! - [`DonnellySimdFull`] takes the same block-at-once descent idea and also
//!   uses SIMD-aware backtracking and pruning.
//!
//! Internally, these variants share `core` for scalar Donnelly indexing/state
//! and `simd_full` for the reusable SIMD comparison and backtrack machinery.

pub(crate) mod cyclic_simd_descent;
mod cyclic_simd_full;
mod no_pf;
mod scalar;
mod simd_descent;
mod unrolled;
mod unrolled_block_dim;

#[doc(hidden)]
pub mod core;
#[doc(hidden)]
pub mod simd_full;

#[doc(inline)]
pub use cyclic_simd_descent::DonnellyCyclicSimdDescent;
#[doc(inline)]
pub use cyclic_simd_full::DonnellyCyclicSimdFull;
#[doc(inline)]
pub use no_pf::DonnellyNoPf;
#[doc(inline)]
pub use scalar::Donnelly;
#[doc(inline)]
pub use simd_descent::DonnellySimdDescent;
#[doc(hidden)]
pub use simd_descent::DonnellySimdInitialDescent;
#[doc(inline)]
pub use simd_full::DonnellySimdFull;
#[doc(inline)]
pub use unrolled::DonnellyUnrolled;
#[doc(inline)]
pub use unrolled_block_dim::DonnellyUnrolledBlockDim;

mod embedded_summary_layout {
    pub trait Sealed {}
}

/// Marker for public stem strategies that share the f64 Donnelly<3> padding
/// layout used by embedded minimum-item summaries.
#[doc(hidden)]
pub trait DonnellyBlock3SummaryLayout:
    crate::StemStrategy + embedded_summary_layout::Sealed
{
}

macro_rules! impl_block3_summary_layout {
    ($($strategy:ty),+ $(,)?) => {
        $(
            impl embedded_summary_layout::Sealed for $strategy {}
            impl DonnellyBlock3SummaryLayout for $strategy {}
        )+
    };
}

impl_block3_summary_layout!(
    Donnelly<3>,
    DonnellyNoPf<3>,
    DonnellyUnrolled<3>,
    DonnellyUnrolledBlockDim<3>,
    DonnellySimdDescent<3>,
    DonnellySimdFull<3>,
    DonnellyCyclicSimdDescent<3>,
    DonnellyCyclicSimdFull<3>,
);
