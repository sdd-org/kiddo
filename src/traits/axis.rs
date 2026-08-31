use num_traits::ConstZero;
use std::fmt::{Debug, Display};
use std::mem::MaybeUninit;
use std::ops::{AddAssign, Sub};

#[cfg(feature = "fixed")]
use fixed::types::extra::{U0, U16, U8};
#[cfg(feature = "fixed")]
use fixed::{FixedI32, FixedU16};

use num_traits::Float;

/// Trait for coordinate/axis types used in the [`KdTree`](crate::kd_tree::KdTree) (the `A` type parameter)
///
/// This trait must be implemented on a type for it to be usable as the coordinates
/// for points stored in a `KdTree`.
///
/// * By default, it is defined for [`f32`], [`f64`], [`u8`], [`u16`], and [`u32`].
/// * Enabling the `f16` feature adds support for the [`f16`](https://docs.rs/half/latest/half/struct.f16.html) type from the [`half`](https://docs.rs/half/latest/half/) crate.
/// * Enabling the `fixed` feature adds support for the following types from the [`fixed`](https://docs.rs/fixed/latest/fixed/) crate:
///     - [`FixedI32<U16>`](https://docs.rs/fixed/latest/fixed/struct.FixedI32.html)
///     - [`FixedI32<U0>`](https://docs.rs/fixed/latest/fixed/struct.FixedI32.html)
///     - [`FixedU16<U8>`](https://docs.rs/fixed/latest/fixed/struct.FixedU16.html)
///     - [`FixedU16<U0>`](https://docs.rs/fixed/latest/fixed/struct.FixedU16.html)
///     - [`FixedU8<U0>`](https://docs.rs/fixed/latest/fixed/struct.FixedU8.html)
///
/// (If you have a requirement to support a `fixed` type that is not listed, please open an issue on GitHub.)
pub trait Axis:
    Copy
    + PartialEq
    + PartialOrd
    + Sub<Output = Self>
    + AddAssign<Self>
    + Debug
    + Display
    + 'static
    + crate::stem_strategy::CompareBlock3
    + crate::stem_strategy::CompareBlock4
{
    /// Whether subtraction can represent a negative result.
    ///
    /// Custom axes default to unsigned semantics so generic algorithms retain
    /// their conservative ordered-subtraction paths unless explicitly opted in.
    const IS_SIGNED: bool = false;

    const VALUE_WIDTH_BYTES: usize = size_of::<Self::Coord>();

    /// Coordinate scalar type stored in the tree and queries.
    type Coord: Copy;

    /// Zero coord.
    fn zero() -> Self::Coord;

    /// Maximum coord value.
    fn max_value() -> Self::Coord;

    /// Minimum coord value.
    fn min_value() -> Self::Coord;

    /// If coord is max value or not.
    fn is_max_value(coord: Self::Coord) -> bool;

    /// Compares two coordinate values.
    fn cmp(a: Self::Coord, b: Self::Coord) -> std::cmp::Ordering;

    /// Absolute/saturating difference along one axis, in coord units.
    fn saturating_dist(a: Self::Coord, b: Self::Coord) -> Self::Coord;

    /// Saturating addition of two coordinate values.
    fn saturating_add(a: Self::Coord, b: Self::Coord) -> Self::Coord;

    /// Returns the maximum of two coordinate values.
    fn max(a: Self::Coord, b: Self::Coord) -> Self::Coord;

    /// Returns the raw packed item-summary word stored in a Donnelly padding
    /// slot, when this axis type supports that representation.
    #[doc(hidden)]
    #[inline(always)]
    fn embedded_item_summary_word(_value: Self::Coord) -> Option<u64> {
        None
    }

    /// Select a path through a Donnelly block whose split dimension advances
    /// at every level. Implemented only by floating-point axes used by the
    /// experimental AVX-512 cyclic-SIMD strategy.
    #[doc(hidden)]
    #[inline(always)]
    fn compare_cyclic_block<const BH: usize, const K: usize>(
        _stems: &[Self],
        _query: &[Self; K],
        _block_base_idx: usize,
        _start_dim: usize,
    ) -> u8 {
        unimplemented!(
            "axis {} does not support cyclic SIMD block comparison",
            std::any::type_name::<Self>()
        )
    }

    /// Select a cyclic path using query lanes prepared once per query.
    #[doc(hidden)]
    #[inline(always)]
    fn compare_prepared_cyclic_block<const BH: usize>(
        _stems: &[Self],
        _query_lanes: &[MaybeUninit<Self>; 16],
        _block_base_idx: usize,
    ) -> u8 {
        unimplemented!(
            "axis {} does not support prepared cyclic SIMD block comparison",
            std::any::type_name::<Self>()
        )
    }
}

/// Macro to implement AxisUnified for floating-point types.
macro_rules! impl_axis_float {
    ($t:ty, SIMD_BLOCK_SUPPORT => ( $( $block_size:literal => ($prune_fn:path, $compare_fn:path) ),* $(,)? )) => {
        impl_axis_float!($t);

        $(
            impl_simd_block_support!($t, $block_size, $prune_fn, $compare_fn);
        )*
    };

    ($t:ty) => {
        impl Axis for $t {
            const IS_SIGNED: bool = true;
            type Coord = $t;

            #[inline(always)]
            fn zero() -> Self::Coord {
                0.0
            }

            #[inline(always)]
            fn max_value() -> Self::Coord {
                <$t>::infinity()
            }

            #[inline(always)]
            fn min_value() -> Self::Coord {
                <$t>::neg_infinity()
            }

            #[inline(always)]
            fn is_max_value(coord: Self::Coord) -> bool {
                coord.is_infinite() && coord.is_sign_positive()
            }

            #[inline(always)]
            fn cmp(a: Self::Coord, b: Self::Coord) -> std::cmp::Ordering {
                // debug_assert!(
                //     a.is_finite() && b.is_finite(),
                //     "NaNs / Infinities should not be present in axis coordinates"
                // );
                if a < b {
                    std::cmp::Ordering::Less
                } else if a > b {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Equal
                }
            }

            #[inline(always)]
            fn saturating_dist(a: Self::Coord, b: Self::Coord) -> Self::Coord {
                (a - b).abs()
            }

            #[inline(always)]
            fn saturating_add(a: Self::Coord, b: Self::Coord) -> Self::Coord {
                a + b
            }

            #[inline(always)]
            fn max(a: Self::Coord, b: Self::Coord) -> Self::Coord {
                a.max(b)
            }

            #[inline(always)]
            fn embedded_item_summary_word(value: Self::Coord) -> Option<u64> {
                if size_of::<$t>() == size_of::<u64>() {
                    // This branch is a compile-time constant for every macro
                    // instantiation. Only f64 reaches the copy.
                    Some(unsafe { std::mem::transmute_copy(&value) })
                } else {
                    None
                }
            }

            #[inline(always)]
            fn compare_cyclic_block<const BH: usize, const K: usize>(
                stems: &[Self],
                query: &[Self; K],
                block_base_idx: usize,
                start_dim: usize,
            ) -> u8 {
                <Self as crate::stem_strategy::donnelly::cyclic_simd_descent::CyclicBlockCompare>::compare_cyclic_block::<BH, K>(
                    stems,
                    query,
                    block_base_idx,
                    start_dim,
                )
            }

            #[inline(always)]
            fn compare_prepared_cyclic_block<const BH: usize>(
                stems: &[Self],
                query_lanes: &[MaybeUninit<Self>; 16],
                block_base_idx: usize,
            ) -> u8 {
                <Self as crate::stem_strategy::donnelly::cyclic_simd_descent::CyclicBlockCompare>::compare_prepared_cyclic_block::<BH>(
                    stems,
                    query_lanes,
                    block_base_idx,
                )
            }
        }
    };
}

impl_axis_float!(f32);
impl_axis_float!(f64);

#[cfg(feature = "fixed")]
/// Macro to implement AxisUnified for fixed-point types.
macro_rules! impl_axis_fixed {
    // Pattern with SIMD block support
    ($t:ty, $is_signed:literal, SIMD_BLOCK_SUPPORT => ( $( $block_size:literal => ($prune_fn:path, $compare_fn:path) ),* $(,)? )) => {
        impl_axis_fixed!($t, $is_signed); // First implement the basic AxisUnified trait

        // Then implement SIMD block support for each specified block size
        $(
            impl_simd_block_support!($t, $block_size, $prune_fn, $compare_fn);
        )*
    };

    // Base pattern without SIMD block support (uses default unimplemented!() from traits)
    ($t:ty, $is_signed:literal) => {
        #[cfg(feature = "fixed")]
        impl Axis for $t {
            const IS_SIGNED: bool = $is_signed;

            type Coord = $t;

            #[inline(always)]
            fn zero() -> Self::Coord {
                <$t>::from_num(0)
            }

            #[inline(always)]
            fn max_value() -> Self::Coord {
                <Self::Coord>::MAX
            }

            #[inline(always)]
            fn min_value() -> Self::Coord {
                unimplemented!("min_value not yet implemented for fixed point types")
            }

            #[inline(always)]
            fn is_max_value(coord: Self::Coord) -> bool {
                coord == <$t>::max_value()
            }

            #[inline(always)]
            fn cmp(a: Self::Coord, b: Self::Coord) -> std::cmp::Ordering {
                a.cmp(&b)
            }

            #[inline(always)]
            fn saturating_dist(a: Self::Coord, b: Self::Coord) -> Self::Coord {
                if a >= b {
                    a - b
                } else {
                    b - a
                }
            }

            #[inline(always)]
            fn saturating_add(a: Self::Coord, b: Self::Coord) -> Self::Coord {
                a.saturating_add(b)
            }

            #[inline(always)]
            fn max(a: Self::Coord, b: Self::Coord) -> Self::Coord {
                a.max(b)
            }
        }
    };
}

#[cfg(feature = "fixed")]
impl_axis_fixed!(FixedI32<U16>, true);
#[cfg(feature = "fixed")]
impl_axis_fixed!(FixedI32<U0>, true);
#[cfg(feature = "fixed")]
impl_axis_fixed!(FixedU16<U8>, false);

#[cfg(feature = "f16")]
impl Axis for half::f16 {
    const IS_SIGNED: bool = true;

    type Coord = half::f16;

    #[inline(always)]
    fn zero() -> Self::Coord {
        half::f16::from_f32(0.0)
    }

    #[inline(always)]
    fn max_value() -> Self::Coord {
        half::f16::from_f32(f32::INFINITY)
    }

    #[inline(always)]
    fn min_value() -> Self::Coord {
        half::f16::from_f32(f32::NEG_INFINITY)
    }

    #[inline(always)]
    fn is_max_value(coord: Self::Coord) -> bool {
        coord.is_infinite() && coord.is_sign_positive()
    }

    #[allow(clippy::ifs_same_cond)]
    #[inline(always)]
    fn cmp(a: Self::Coord, b: Self::Coord) -> std::cmp::Ordering {
        if a < b {
            std::cmp::Ordering::Less
        } else if b > a {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    }

    #[inline(always)]
    fn saturating_dist(a: Self::Coord, b: Self::Coord) -> Self::Coord {
        (a - b).abs()
    }

    #[inline(always)]
    fn saturating_add(a: Self::Coord, b: Self::Coord) -> Self::Coord {
        a + b
    }

    #[inline(always)]
    fn max(a: Self::Coord, b: Self::Coord) -> Self::Coord {
        if a > b {
            a
        } else {
            b
        }
    }
}

// TODO: Remove the current `MAX`-value exclusion for integer/fixed coordinates by
// flipping the tree invariant from `left < pivot, right >= pivot` to
// `left <= pivot, right > pivot`, then updating traversal, immutable pivot
// selection, mutable split-boundary handling, and block-compare semantics to
// match.
/// Macro to implement AxisUnified for unsigned integer types.
macro_rules! impl_axis_uint {
    // Pattern with SIMD block support
    ($t:ty, SIMD_BLOCK_SUPPORT => ( $( $block_size:literal => ($prune_fn:path, $compare_fn:path) ),* $(,)? )) => {
        impl_axis_uint!($t); // First implement the basic AxisUnified trait

        // Then implement SIMD block support for each specified block size
        $(
            impl_simd_block_support!($t, $block_size, $prune_fn, $compare_fn);
        )*
    };

    // Base pattern without SIMD block support (uses default unimplemented!() from traits)
    ($t:ty) => {
        impl Axis for $t {
            type Coord = $t;

            #[inline(always)]
            fn zero() -> Self::Coord {
                <Self::Coord>::ZERO
            }

            #[inline(always)]
            fn max_value() -> Self::Coord {
                <Self::Coord>::MAX
            }

            #[inline(always)]
            fn min_value() -> Self::Coord {
                <$t>::zero()
            }

            #[inline(always)]
            fn is_max_value(coord: Self::Coord) -> bool {
                coord == <$t>::MAX
            }

            #[inline(always)]
            fn cmp(a: Self::Coord, b: Self::Coord) -> std::cmp::Ordering {
                a.cmp(&b)
            }

            #[inline(always)]
            fn saturating_dist(a: Self::Coord, b: Self::Coord) -> Self::Coord {
                if a >= b {
                    a - b
                } else {
                    b - a
                }
            }

            #[inline(always)]
            fn saturating_add(a: Self::Coord, b: Self::Coord) -> Self::Coord {
                a.saturating_add(b)
            }

            #[inline(always)]
            fn max(a: Self::Coord, b: Self::Coord) -> Self::Coord {
                a.max(b)
            }
        }
    };
}

impl_axis_uint!(u8);
impl_axis_uint!(u16);
impl_axis_uint!(u32);

#[cfg(test)]
mod tests {
    use super::Axis;

    #[test]
    fn float_axis_helpers_behave_as_expected() {
        const { assert!(<f32 as Axis>::IS_SIGNED) };
        assert_eq!(<f32 as Axis>::zero(), 0.0);
        assert!(<f32 as Axis>::is_max_value(<f32 as Axis>::max_value()));
        assert_eq!(<f32 as Axis>::cmp(1.0, 2.0), std::cmp::Ordering::Less);
        assert_eq!(<f32 as Axis>::cmp(2.0, 1.0), std::cmp::Ordering::Greater);
        assert_eq!(<f32 as Axis>::cmp(2.0, 2.0), std::cmp::Ordering::Equal);
        assert_eq!(<f32 as Axis>::saturating_dist(5.0, 2.5), 2.5);
        assert_eq!(<f32 as Axis>::saturating_add(1.5, 2.0), 3.5);
        assert_eq!(<f32 as Axis>::max(3.0, 4.0), 4.0);

        assert_eq!(<f64 as Axis>::zero(), 0.0);
        assert!(!<f64 as Axis>::is_max_value(<f64 as Axis>::min_value()));
        assert_eq!(<f64 as Axis>::saturating_dist(2.0, 5.5), 3.5);
    }

    #[test]
    fn unsigned_axis_helpers_behave_as_expected() {
        const { assert!(!<u8 as Axis>::IS_SIGNED) };
        assert_eq!(<u8 as Axis>::zero(), 0);
        assert_eq!(<u8 as Axis>::min_value(), 0);
        assert_eq!(<u8 as Axis>::max_value(), u8::MAX);
        assert!(<u8 as Axis>::is_max_value(u8::MAX));
        assert_eq!(<u8 as Axis>::cmp(1, 2), std::cmp::Ordering::Less);
        assert_eq!(<u8 as Axis>::saturating_dist(2, 5), 3);
        assert_eq!(<u8 as Axis>::saturating_add(250, 10), u8::MAX);
        assert_eq!(<u8 as Axis>::max(3, 4), 4);

        assert_eq!(<u16 as Axis>::saturating_dist(9, 4), 5);
        assert_eq!(<u32 as Axis>::saturating_add(u32::MAX, 1), u32::MAX);
    }

    #[cfg(feature = "fixed")]
    #[test]
    fn fixed_axis_helpers_behave_as_expected() {
        const { assert!(<fixed::FixedI32<fixed::types::extra::U16> as Axis>::IS_SIGNED) };
        const { assert!(!<fixed::FixedU16<fixed::types::extra::U8> as Axis>::IS_SIGNED) };
        let a = fixed::FixedI32::<fixed::types::extra::U16>::from_num(1.5);
        let b = fixed::FixedI32::<fixed::types::extra::U16>::from_num(0.25);
        assert_eq!(
            <fixed::FixedI32<fixed::types::extra::U16> as Axis>::saturating_dist(a, b),
            fixed::FixedI32::<fixed::types::extra::U16>::from_num(1.25)
        );
        assert_eq!(
            <fixed::FixedU16<fixed::types::extra::U8> as Axis>::max(
                fixed::FixedU16::<fixed::types::extra::U8>::from_num(2),
                fixed::FixedU16::<fixed::types::extra::U8>::from_num(3)
            ),
            fixed::FixedU16::<fixed::types::extra::U8>::from_num(3)
        );
    }
}
