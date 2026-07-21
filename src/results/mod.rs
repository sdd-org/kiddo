#[doc(hidden)]
pub mod best_query_result_item;

#[doc(hidden)]
pub mod query_result_item;

pub(crate) mod result_collection;

pub mod result_buffer;

#[cfg(feature = "result_collection_stats")]
#[doc(hidden)]
pub mod result_collection_stats;
