pub mod donnelly;

pub mod eytzinger;

mod prefetch;

#[doc(hidden)]
pub use donnelly::simd_full::{CompareBlock3, CompareBlock4, SimdPrune, SimdSelectBestChildBlock3};

#[doc(hidden)]
pub use donnelly::Donnelly;
#[cfg(feature = "test_utils")]
#[doc(hidden)]
pub use donnelly::DonnellyUnrolledLeafEmbedded3;
#[doc(hidden)]
pub use donnelly::{
    DonnellyNoPf, DonnellySimdDescent, DonnellySimdFull, DonnellyUnrolled, DonnellyUnrolledBlockDim,
};

#[doc(hidden)]
pub use eytzinger::Eytzinger;
#[doc(hidden)]
pub use eytzinger::EytzingerFlexPf;
#[doc(hidden)]
pub use eytzinger::EytzingerNoPf;

// TODO: modified Donnelly Block stem strategy that pads to the point where
//       the bottom row of the last block is free. Then, use that bottom row
//       to store the offset and leaf size for each left and right child of
//       the level above, eliminating the need for extents to be stored in the
//       leaf strategy, and avoiding the need for one layer of indirection when
//       retrieving the leaf offset
