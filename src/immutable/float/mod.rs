//! Immutable k-d trees (faster and smaller, but cannot be modified after construction).
//!
//! Intended for use when the smallest possible on-disk serialized
//! size of a tree is of paramount importance, and / or the
//! fastest possible query speed is required.
//!
//! Expect improvements in query time of 10-15%, and a reduction
//! in the size of serialized trees by 33% or so on average vs
//! the standard [`float::kdtree::KdTree`](`crate::float::kdtree::KdTree`)
//!
//! These capabilities come with a few trade-offs:
//! 1) This tree does not provide the capability
//!    to modify its contents after it has been constructed.
//!    The co-ordinates of the points to be stored must
//!    have all been generated in advance.
//! 2) Construction time can be a bit slower - anecdotally
//!    this can be twice as long as the default [`float::kdtree::KdTree`](`crate::float::kdtree::KdTree`).
//!    NB construction time improved massively in Kiddo v5 compared to earlier versions.
//!
//! As per the other Kiddo float-type trees, points being stored
//! in the tree must be floats ([`f64`] or [`f32`],
//! or [`f16`](https://docs.rs/half/latest/half/struct.f16.html) if used with the
//! [`half`](https://docs.rs/half/latest/half) crate.

pub mod kdtree;
#[doc(hidden)]
pub mod query;

#[cfg(feature = "rkyv")]
mod rkyv_07_aligned_vec;

#[cfg(feature = "rkyv_08")]
mod rkyv_aligned_vec;
