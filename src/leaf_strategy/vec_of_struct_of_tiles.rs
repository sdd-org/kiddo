use aligned_vec::AVec;

/// Immutable leaf storage using chunk-tiled column-major points with a separate item array.
///
/// This is currently a placeholder for an alternate immutable arena design.
/// The intended shape is similar to [`VecOfArenas`](crate::leaf_strategies::VecOfArenas),
/// but with point coordinates kept in an ordinary `AVec<A>` and items in a
/// separate ordinary `AVec<T>` instead of one mixed-type encoded byte arena.
///
/// The motivation is to isolate the performance impact of separating item storage
/// while keeping the `VecOfArenas` point layout. In other words:
///
/// - preserve the same tiled point stream order seen by the point-distance kernel
/// - but break the "points immediately followed by matching items" property
///
/// That should let us measure how much of `VecOfArenas`' benefit comes from:
///
/// - the tiled point layout itself
/// - versus the fact that the entire leaf, including items, can often be consumed
///   as one continuous prefetch stream
///
/// Intended memory layout:
///
/// ```text
/// leaf_extents = [(tile_start0, len0), (tile_start1, len1), ...]
///
/// point_arena =
///   [ leaf 0 tile32: x[0..32]  y[0..32]  z[0..32] ]
///   [ leaf 0 tile8 : x[32..40] y[32..40] z[32..40] ]
///   [ leaf 0 tile4 : x[40..44] y[40..44] z[40..44] ]
///   [ leaf 1 tile32: ... ]
///
/// items =
///   [ leaf 0 tile32 items[0..32] ]
///   [ leaf 0 tile8  items[32..40] ]
///   [ leaf 0 tile4  items[40..44] ]
///   [ leaf 1 tile32 items[...]   ]
/// ```
///
/// Compared with `VecOfArenas`, points and items would live in separate `AVec`s
/// rather than one shared encoded byte buffer, so item access would become a
/// second phase/stream after point processing.
#[allow(unused)]
pub struct VecOfStructOfTiles<A, T, const K: usize, const B: usize> {
    leaf_extents: Vec<(usize, usize)>,
    point_arena: AVec<A>,
    items: AVec<T>,
    size: usize,
}

impl<A, T, const K: usize, const B: usize> std::fmt::Debug for VecOfStructOfTiles<A, T, K, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VecOfStructOfTiles")
            .field("dimensions", &K)
            .field("bucket_size", &B)
            .field("size", &self.size)
            .field("leaf_count", &self.leaf_extents.len())
            .field("point_arena_len", &self.point_arena.len())
            .field("stored_item_count", &self.items.len())
            .finish()
    }
}
