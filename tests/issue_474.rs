#![cfg(feature = "rkyv")]

use kiddo::immutable::float::kdtree::{ImmutableKdTree, ImmutableKdTreeRK};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};

const ALLOCATION_SLOTS: usize = 1024;
static POINTERS: [AtomicUsize; ALLOCATION_SLOTS] =
    [const { AtomicUsize::new(0) }; ALLOCATION_SLOTS];
static SIZES: [AtomicUsize; ALLOCATION_SLOTS] = [const { AtomicUsize::new(0) }; ALLOCATION_SLOTS];
static ALIGNMENTS: [AtomicUsize; ALLOCATION_SLOTS] =
    [const { AtomicUsize::new(0) }; ALLOCATION_SLOTS];
static MISMATCH: [AtomicUsize; 5] = [const { AtomicUsize::new(0) }; 5];

struct TrackingAllocator;

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };

        if !pointer.is_null() && layout.align() >= 64 {
            for index in 0..ALLOCATION_SLOTS {
                if POINTERS[index]
                    .compare_exchange(0, pointer as usize, SeqCst, SeqCst)
                    .is_ok()
                {
                    SIZES[index].store(layout.size(), SeqCst);
                    ALIGNMENTS[index].store(layout.align(), SeqCst);
                    break;
                }
            }
        }

        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        for index in 0..ALLOCATION_SLOTS {
            if POINTERS[index].load(SeqCst) == pointer as usize {
                let allocated_size = SIZES[index].load(SeqCst);
                let allocated_alignment = ALIGNMENTS[index].load(SeqCst);

                if (allocated_size, allocated_alignment) != (layout.size(), layout.align())
                    && MISMATCH[0].load(SeqCst) == 0
                {
                    MISMATCH[1].store(allocated_size, SeqCst);
                    MISMATCH[2].store(allocated_alignment, SeqCst);
                    MISMATCH[3].store(layout.size(), SeqCst);
                    MISMATCH[4].store(layout.align(), SeqCst);
                    MISMATCH[0].store(pointer as usize, SeqCst);
                }

                POINTERS[index].store(usize::MAX, SeqCst);
                break;
            }
        }

        unsafe { System.dealloc(pointer, layout) };
    }
}

#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator;

#[test]
fn converting_to_rkyv_tree_deallocates_stems_with_their_original_layout() {
    let points: Vec<[f64; 3]> = (0..2000)
        .map(|i| [i as f64, (i * 3 % 977) as f64, (i * 7 % 613) as f64])
        .collect();
    let tree: ImmutableKdTree<f64, u32, 3, 32> = ImmutableKdTree::new_from_slice(&points);
    let tree_rk: ImmutableKdTreeRK<f64, u32, 3, 32> = tree.into();
    drop(tree_rk);

    assert_eq!(
        MISMATCH[0].load(SeqCst),
        0,
        "allocation layout mismatch: allocated with size={} align={}, \
         deallocated with size={} align={}",
        MISMATCH[1].load(SeqCst),
        MISMATCH[2].load(SeqCst),
        MISMATCH[3].load(SeqCst),
        MISMATCH[4].load(SeqCst),
    );
}
