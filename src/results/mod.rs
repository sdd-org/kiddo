#[doc(hidden)]
pub mod best_query_result_item;

#[doc(hidden)]
pub mod query_result_item;

pub(crate) mod result_collection;

#[cfg(feature = "result_collection_stats")]
#[doc(hidden)]
pub mod result_collection_stats;

#[cfg(any(feature = "exact_query_stats", feature = "test_utils"))]
#[doc(hidden)]
pub mod exact_query_stats;
