use super::cursor::JoinTree;
use super::cursor::Tile;
use super::metric::{accepts, JoinFloat, JoinMetric};
use super::traversal::Sink;
use crate::Content;
use std::ops::Range;

#[inline(always)]
fn tile_mask<A: JoinFloat, T: Content, D: JoinMetric<A>, const K: usize, const EX: bool>(
    query: &[D::Output; K],
    tile: &Tile<'_, A, T, K>,
    radius: D::Output,
    distances: &mut [D::Output; 32],
) -> u32 {
    // TypeId checks are constant after monomorphization. Both coordinate and
    // output identity must hold before reinterpreting typed pointers.
    #[cfg(all(
        feature = "simd",
        target_arch = "x86_64",
        any(target_feature = "avx2", target_feature = "avx512f")
    ))]
    if D::SQUARED_EUCLIDEAN {
        #[cfg(not(target_feature = "avx512f"))]
        use super::avx2 as kernel;
        #[cfg(target_feature = "avx512f")]
        use super::avx512 as kernel;
        use std::any::TypeId;
        macro_rules! dispatch {
            ($t:ty, $f:ident) => {
                if TypeId::of::<A>() == TypeId::of::<$t>()
                    && TypeId::of::<D::Output>() == TypeId::of::<$t>()
                {
                    // SAFETY: type identity above establishes every cast; each
                    // column has tile.len initialized coordinates. Kernels use
                    // unaligned loads and never read past that length.
                    return unsafe {
                        kernel::$f::<K, EX>(
                            &*(query as *const _ as *const [$t; K]),
                            &tile.columns.map(|p| p.cast::<$t>()),
                            tile.len,
                            *(&radius as *const _ as *const $t),
                            &mut *(distances as *mut _ as *mut [$t; 32]),
                        )
                    };
                }
            };
        }
        dispatch!(f32, mask_f32);
        dispatch!(f64, mask_f64);
    }
    for (d, q) in query.iter().enumerate() {
        for (j, distance) in distances.iter_mut().enumerate().take(tile.len) {
            D::combine_component(distance, D::dist1(*q, D::widen_coord(tile.coord(d, j))));
        }
    }
    let mut mask = 0;
    for (j, &distance) in distances.iter().enumerate().take(tile.len) {
        if accepts::<_, EX>(distance, radius) {
            mask |= 1 << j;
        }
    }
    mask
}

pub(super) fn leaf_pair<L, R, D, S, const K: usize, const EX: bool>(
    left: &L,
    right: &R,
    li: usize,
    ri: usize,
    rows: &[Range<usize>; 2],
    radius: D::Output,
    sink: &mut S,
) where
    L: JoinTree<K>,
    R: JoinTree<K, A = L::A>,
    D: JoinMetric<L::A>,
    S: Sink<L::Item, R::Item, D::Output>,
{
    #[cfg(feature = "exact_query_stats")]
    super::stats::record(super::stats::Event::Leaf, 1);
    left.tiles(li, rows[0].clone(), |lt| {
        right.tiles(ri, rows[1].clone(), |rt| {
            #[cfg(feature = "exact_query_stats")]
            super::stats::record(super::stats::Event::Distances, lt.len * rt.len);
            for i in 0..lt.len {
                let query: [D::Output; K] = std::array::from_fn(|d| D::widen_coord(lt.coord(d, i)));
                let mut distances = [D::Output::ZERO; 32];
                let mut mask = tile_mask::<_, _, D, K, EX>(&query, &rt, radius, &mut distances);
                #[cfg(feature = "exact_query_stats")]
                super::stats::record(super::stats::Event::Matches, mask.count_ones() as usize);
                if S::COUNT {
                    sink.add_count(mask.count_ones() as usize);
                } else {
                    while mask != 0 {
                        let j = mask.trailing_zeros() as usize;
                        sink.emit(lt.item(i), rt.item(j), distances[j]);
                        mask &= mask - 1;
                    }
                }
            }
        });
    });
}
