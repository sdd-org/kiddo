use crate::leaf_view::LeafView;
use crate::traits::leaf_strategy::{
    BucketLimitType, ConstructibleLeafStrategy, Immutable, LeafProjection,
};
use crate::{Axis, Content, LeafStrategy, StemStrategy};

/// A dummy leaf strategy used for testing.
///
/// This strategy provides no-op implementations and is not meant for production use.
#[allow(dead_code)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Default)]
pub struct DummyLeafStrategy {}

impl std::fmt::Debug for DummyLeafStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DummyLeafStrategy").finish()
    }
}

impl<AX, T, SS, const K: usize, const B: usize> LeafStrategy<AX, T, SS, K, B> for DummyLeafStrategy
where
    AX: Axis<Coord = AX>,
    T: Content,
    SS: StemStrategy,
{
    type Num = ();
    type Mutability = Immutable;

    const BUCKET_LIMIT_TYPE: BucketLimitType = BucketLimitType::Hard;
    const LEAF_PROJECTION: LeafProjection = LeafProjection::LeafView;

    fn size(&self) -> usize {
        unimplemented!()
    }

    fn leaf_count(&self) -> usize {
        unimplemented!()
    }

    fn leaf_len(&self, _leaf_idx: usize) -> usize {
        unimplemented!()
    }

    fn leaf_view(&self, _leaf_idx: usize) -> LeafView<'_, AX, T, K, B> {
        unimplemented!()
    }
}

impl<AX, T, SS, const K: usize, const B: usize> ConstructibleLeafStrategy<AX, T, SS, K, B>
    for DummyLeafStrategy
where
    AX: Axis<Coord = AX>,
    T: Content,
    SS: StemStrategy,
{
    fn new_with_capacity(_capacity: usize) -> Self {
        Self::default()
    }

    fn append_leaf(&mut self, _leaf_points: &[&[AX]; K], _leaf_items: &[T]) {
        // NOOP
    }
}

#[cfg(test)]
mod tests {
    use super::DummyLeafStrategy;
    use crate::{kd_tree, Eytzinger};

    #[test]
    fn default_constructs_dummy_leaf_strategy_kd_tree() {
        let tree: kd_tree::KdTree<f32, u32, Eytzinger, DummyLeafStrategy, 3, 16> =
            kd_tree::KdTree::default();

        assert!(tree.is_empty());
        assert_eq!(tree.size(), 0);
    }
}
