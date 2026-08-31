use std::cmp::Ordering;
use std::cmp::Ordering::{Equal, Greater, Less};
use std::mem::MaybeUninit;
use std::{cmp, mem, ptr};

// performs select_nth_unstable_by on target,
// but all the operations performed in the sort are applied to mirror as well
pub fn mirror_select_nth_unstable_by<AA, BB, F>(
    target: &mut [AA],
    mirror: &mut [BB],
    index: usize,
    mut compare: F,
) where
    F: FnMut(&AA, &AA) -> Ordering,
{
    let mut f = |a: &AA, b: &AA| compare(a, b) == Less;

    mirror_partition_at_index_loop(target, mirror, index, &mut f, None)
}

fn mirror_partition_at_index_loop<'a, AA, BB, F>(
    mut target: &'a mut [AA],
    mut mirror: &'a mut [BB],
    mut index: usize,
    is_less: &mut F,
    mut pred: Option<&'a AA>,
) where
    F: FnMut(&AA, &AA) -> bool,
{
    loop {
        // Choose a pivot
        let (pivot, _) = choose_pivot(target, mirror, is_less);

        // If the chosen pivot is equal to the predecessor, then it's the smallest element in the
        // slice. Partition the slice into elements equal to and elements greater than the pivot.
        // This case is usually hit when the slice contains many duplicate elements.
        if let Some(p) = pred {
            if !is_less(p, &target[pivot]) {
                let mid = mirror_partition_equal(target, mirror, pivot, is_less);

                // If we've passed our index, then we're good.
                if mid > index {
                    return;
                }

                // Otherwise, continue sorting elements greater than the pivot.
                target = &mut target[mid..];
                mirror = &mut mirror[mid..];
                index -= mid;
                pred = None;
                continue;
            }
        }

        let (mid, _) = mirror_partition(target, mirror, pivot, is_less);

        // Split the slice into `left`, `pivot`, and `right`.
        let (left, right) = target.split_at_mut(mid);
        let (pivot, right) = right.split_at_mut(1);
        let pivot = &pivot[0];
        let (left_mirror, right_mirror) = mirror.split_at_mut(mid);
        let (_, right_mirror) = right_mirror.split_at_mut(1);

        match mid.cmp(&index) {
            Less => {
                target = right;
                mirror = right_mirror;
                index = index - mid - 1;
                pred = Some(pivot);
            }
            Greater => {
                target = left;
                mirror = left_mirror;
            }
            Equal => {
                // If mid == index, then we're done, since partition() guaranteed that all elements
                // after mid are greater than or equal to mid.
                return;
            }
        }
    }
}

fn mirror_partition_equal<AA, BB, F>(
    target: &mut [AA],
    mirror: &mut [BB],
    pivot: usize,
    is_less: &mut F,
) -> usize
where
    F: FnMut(&AA, &AA) -> bool,
{
    // Place the pivot at the beginning of slice.
    target.swap(0, pivot);
    mirror.swap(0, pivot);

    let (pivot, v) = target.split_at_mut(1);
    let pivot = &mut pivot[0];

    let (mirror_pivot, mirror_v) = mirror.split_at_mut(1);
    let mirror_pivot = &mut mirror_pivot[0];

    // Read the pivot into a stack-allocated variable for efficiency. If a following comparison
    // operation panics, the pivot will be automatically written back into the slice.
    // SAFETY: The pointer here is valid because it is obtained from a reference to a slice.
    let tmp = mem::ManuallyDrop::new(unsafe { ptr::read(pivot) });
    let _pivot_guard = CopyOnDrop {
        src: &*tmp,
        dest: pivot,
    };
    let pivot = &*tmp;

    let mirror_tmp = mem::ManuallyDrop::new(unsafe { ptr::read(mirror_pivot) });
    let _mirror_pivot_guard = CopyOnDrop {
        src: &*mirror_tmp,
        dest: mirror_pivot,
    };
    let _mirror_pivot = &*mirror_tmp;

    // Now partition the slice.
    let mut l = 0;
    let mut r = v.len();
    loop {
        // SAFETY: The unsafety below involves indexing an array.
        // For the first one: We already do the bounds checking here with `l < r`.
        // For the second one: We initially have `l == 0` and `r == v.len()` and we checked that `l < r` at every indexing operation.
        //                     From here we know that `r` must be at least `r == l` which was shown to be valid from the first one.
        unsafe {
            // Find the first element greater than the pivot.
            while l < r && !is_less(pivot, v.get_unchecked(l)) {
                l += 1;
            }

            // Find the last element equal to the pivot.
            while l < r && is_less(pivot, v.get_unchecked(r - 1)) {
                r -= 1;
            }

            // Are we done?
            if l >= r {
                break;
            }

            // Swap the found pair of out-of-order elements.
            r -= 1;
            let ptr = v.as_mut_ptr();
            ptr::swap(ptr.add(l), ptr.add(r));

            let mirror_ptr = mirror_v.as_mut_ptr();
            ptr::swap(mirror_ptr.add(l), mirror_ptr.add(r));
            l += 1;
        }
    }

    // We found `l` elements equal to the pivot. Add 1 to account for the pivot itself.
    l + 1

    // `_pivot_guard` goes out of scope and writes the pivot (which is a stack-allocated variable)
    // back into the slice where it originally was. This step is critical in ensuring safety!
}

fn mirror_partition<AA, BB, F>(
    target: &mut [AA],
    mirror: &mut [BB],
    pivot: usize,
    is_less: &mut F,
) -> (usize, bool)
where
    F: FnMut(&AA, &AA) -> bool,
{
    let (mid, was_partitioned) = {
        // Place the pivot at the beginning of slice.
        target.swap(0, pivot);
        mirror.swap(0, pivot);

        let (pivot, v) = target.split_at_mut(1);
        let (mirror_pivot, mirror_v) = mirror.split_at_mut(1);
        let pivot = &mut pivot[0];
        let mirror_pivot = &mut mirror_pivot[0];

        // Read the pivot into a stack-allocated variable for efficiency. If a following comparison
        // operation panics, the pivot will be automatically written back into the slice.

        // SAFETY: `pivot` is a reference to the first element of `v`, so `ptr::read` is safe.
        let tmp = mem::ManuallyDrop::new(unsafe { ptr::read(pivot) });
        let _pivot_guard = CopyOnDrop {
            src: &*tmp,
            dest: pivot,
        };
        let pivot = &*tmp;

        let mirror_tmp = mem::ManuallyDrop::new(unsafe { ptr::read(mirror_pivot) });
        let _mirror_pivot_guard = CopyOnDrop {
            src: &*mirror_tmp,
            dest: mirror_pivot,
        };
        let _mirror_pivot = &*mirror_tmp;

        // Find the first pair of out-of-order elements.
        let mut l = 0;
        let mut r = v.len();

        // SAFETY: The unsafety below involves indexing an array.
        // For the first one: We already do the bounds checking here with `l < r`.
        // For the second one: We initially have `l == 0` and `r == v.len()` and we checked that `l < r` at every indexing operation.
        //                     From here we know that `r` must be at least `r == l` which was shown to be valid from the first one.
        unsafe {
            // Find the first element greater than or equal to the pivot.
            while l < r && is_less(v.get_unchecked(l), pivot) {
                l += 1;
            }

            // Find the last element smaller that the pivot.
            while l < r && !is_less(v.get_unchecked(r - 1), pivot) {
                r -= 1;
            }
        }

        (
            l + mirror_partition_in_blocks(&mut v[l..r], &mut mirror_v[l..r], pivot, is_less),
            l >= r,
        )

        // `_pivot_guard` goes out of scope and writes the pivot (which is a stack-allocated
        // variable) back into the slice where it originally was. This step is critical in ensuring
        // safety!
    };

    // Place the pivot between the two partitions.
    target.swap(0, mid);
    mirror.swap(0, mid);

    (mid, was_partitioned)
}

// NOTE ON CODE SCANNING ALERTS (GitHub CodeQL `rust/access-invalid-pointer`, alerts #1-#8):
//
// This function is a near-verbatim fork of `partition_in_blocks` from the Rust standard
// library (`library/core/src/slice/sort.rs`, as of 1.80.1, before the ipnsort rewrite):
//
//   https://github.com/rust-lang/rust/blob/1.80.1/library/core/src/slice/sort.rs#L302-L551
//
// The only changes are:
//  * mechanical renames (`v` -> `target`, `T` -> `AA`, `usize::from(..)` -> `.. as usize`); and
//  * every operation on `target` is duplicated on the parallel `mirror` slice.
//
// The duplicated mirror bookkeeping is always identical to the `target` bookkeeping: the
// branchless comparison results are computed once from `target` and drive *both* offsets
// arrays (`*end_l = i` / `*mirror_end_l = i`, etc.), and `mirror_l`/`mirror_r` advance in
// lockstep with `l`/`r`. The upstream SAFETY arguments below therefore apply unchanged to
// the mirrored pointers, and the offset values dereferenced by `mirror_left!()`/
// `mirror_right!()` are bounded exactly like those of `left!()`/`right!()`.
//
// The dereferences CodeQL flags (the `left!/right!/mirror_left!/mirror_right!` macro
// expansions in the cyclic permutation below, and the `ptr::swap` calls in the
// [remaining-elements-safety] tail loops) are false positives on code copied unchanged
// from the standard library, where it was reviewed and Miri-tested for years. Do NOT
// rewrite this with bounds-checked indexing - it would regress this branchless hot path.
fn mirror_partition_in_blocks<AA, BB, F>(
    target: &mut [AA],
    mirror: &mut [BB],
    pivot: &AA,
    is_less: &mut F,
) -> usize
where
    F: FnMut(&AA, &AA) -> bool,
{
    // Number of elements in a typical block.
    const BLOCK: usize = 128;

    // The partitioning algorithm repeats the following steps until completion:
    //
    // 1. Trace a block from the left side to identify elements greater than or equal to the pivot.
    // 2. Trace a block from the right side to identify elements smaller than the pivot.
    // 3. Exchange the identified elements between the left and right side.
    //
    // We keep the following variables for a block of elements:
    //
    // 1. `block` - Number of elements in the block.
    // 2. `start` - Start pointer into the `offsets` array.
    // 3. `end` - End pointer into the `offsets` array.
    // 4. `offsets - Indices of out-of-order elements within the block.

    // The current block on the left side (from `l` to `l.add(block_l)`).
    let mut l = target.as_mut_ptr();
    let mut block_l = BLOCK;
    let mut start_l = ptr::null_mut();
    let mut end_l = ptr::null_mut();
    let mut offsets_l = [MaybeUninit::<u8>::uninit(); BLOCK];

    let mut mirror_l = mirror.as_mut_ptr();
    let mut mirror_block_l = BLOCK;
    let mut mirror_start_l = ptr::null_mut();
    let mut mirror_end_l = ptr::null_mut();
    let mut mirror_offsets_l = [MaybeUninit::<u8>::uninit(); BLOCK];

    // The current block on the right side (from `r.sub(block_r)` to `r`).
    // SAFETY: The documentation for .add() specifically mention that `vec.as_ptr().add(vec.len())` is always safe`
    let mut r = unsafe { l.add(target.len()) };
    let mut block_r = BLOCK;
    let mut start_r = ptr::null_mut();
    let mut end_r = ptr::null_mut();
    let mut offsets_r = [MaybeUninit::<u8>::uninit(); BLOCK];

    let mut mirror_r = unsafe { mirror_l.add(mirror.len()) };
    let mut mirror_block_r = BLOCK;
    let mut mirror_start_r = ptr::null_mut();
    let mut mirror_end_r = ptr::null_mut();
    let mut mirror_offsets_r = [MaybeUninit::<u8>::uninit(); BLOCK];

    // FIXME: When we get VLAs, try creating one array of length `min(v.len(), 2 * BLOCK)` rather
    // than two u16-size arrays of length `BLOCK`. VLAs might be more cache-efficient.

    // Returns the number of elements between pointers `l` (inclusive) and `r` (exclusive).
    fn width<T>(l: *mut T, r: *mut T) -> usize {
        assert!(mem::size_of::<T>() > 0);
        // FIXME: this should *likely* use `offset_from`, but more
        // investigation is needed (including running tests in miri).

        ((r as usize) - (l as usize)) / mem::size_of::<T>()
    }

    loop {
        // We are done with partitioning block-by-block when `l` and `r` get very close. Then we do
        // some patch-up work in order to partition the remaining elements in between.
        let is_done = width(l, r) <= 2 * BLOCK;

        if is_done {
            // Number of remaining elements (still not compared to the pivot).
            let mut rem = width(l, r);
            let mut mirror_rem = width(mirror_l, mirror_r); // TODO: remove, same as rem
            if start_l < end_l || start_r < end_r {
                rem -= BLOCK;
                mirror_rem -= BLOCK;
            }

            // Adjust block sizes so that the left and right block don't overlap, but get perfectly
            // aligned to cover the whole remaining gap.
            if start_l < end_l {
                block_r = rem;
                mirror_block_r = mirror_rem;
            } else if start_r < end_r {
                block_l = rem;
                mirror_block_l = mirror_rem;
            } else {
                // There were the same number of elements to switch on both blocks during the last
                // iteration, so there are no remaining elements on either block. Cover the remaining
                // items with roughly equally-sized blocks.
                block_l = rem / 2;
                block_r = rem - block_l;

                mirror_block_l = mirror_rem / 2;
                mirror_block_r = mirror_rem - mirror_block_l;
            }
            debug_assert!(block_l <= BLOCK && block_r <= BLOCK);
            debug_assert!(width(l, r) == block_l + block_r);

            debug_assert!(mirror_block_l <= BLOCK && mirror_block_r <= BLOCK);
            debug_assert!(width(mirror_l, mirror_r) == mirror_block_l + mirror_block_r);
        }

        if start_l == end_l {
            // Trace `block_l` elements from the left side.

            // TODO: switch back to this once slice_as_mut_ptr is stable
            // start_l = MaybeUninit::slice_as_mut_ptr(&mut offsets_l);
            start_l = &mut offsets_l as *mut _ as *mut u8;

            end_l = start_l;
            let mut elem = l;

            // TODO: switch back to this once slice_as_mut_ptr is stable
            // mirror_start_l = MaybeUninit::slice_as_mut_ptr(&mut mirror_offsets_l);
            mirror_start_l = &mut mirror_offsets_l as *mut _ as *mut u8;

            mirror_end_l = mirror_start_l;
            let mut mirror_elem = mirror_l;

            for i in 0..block_l {
                // SAFETY: The unsafety operations below involve the usage of the `offset`.
                //         According to the conditions required by the function, we satisfy them because:
                //         1. `offsets_l` is stack-allocated, and thus considered separate allocated object.
                //         2. The function `is_less` returns a `bool`.
                //            Casting a `bool` will never overflow `isize`.
                //         3. We have guaranteed that `block_l` will be `<= BLOCK`.
                //            Plus, `end_l` was initially set to the begin pointer of `offsets_` which was declared on the stack.
                //            Thus, we know that even in the worst case (all invocations of `is_less` returns false) we will only be at most 1 byte pass the end.
                //        Another unsafety operation here is dereferencing `elem`.
                //        However, `elem` was initially the begin pointer to the slice which is always valid.
                unsafe {
                    // Branchless comparison.
                    *end_l = i as u8;
                    *mirror_end_l = i as u8;

                    let result = !is_less(&*elem, pivot) as usize;

                    end_l = end_l.add(result);
                    elem = elem.add(1);

                    mirror_end_l = mirror_end_l.add(result);
                    mirror_elem = mirror_elem.add(1);
                }
            }
        }

        if start_r == end_r {
            // Trace `block_r` elements from the right side.

            // TODO: switch back to this once slice_as_mut_ptr is stable
            // start_r = MaybeUninit::slice_as_mut_ptr(&mut offsets_r);
            start_r = &mut offsets_r as *mut _ as *mut u8;

            end_r = start_r;
            let mut elem = r;

            // TODO: switch back to this once slice_as_mut_ptr is stable
            //mirror_start_r = MaybeUninit::slice_as_mut_ptr(&mut mirror_offsets_r);
            mirror_start_r = &mut mirror_offsets_r as *mut _ as *mut u8;

            mirror_end_r = mirror_start_r;
            let mut mirror_elem = mirror_r;

            for i in 0..block_r {
                // SAFETY: The unsafety operations below involve the usage of the `offset`.
                //         According to the conditions required by the function, we satisfy them because:
                //         1. `offsets_r` is stack-allocated, and thus considered separate allocated object.
                //         2. The function `is_less` returns a `bool`.
                //            Casting a `bool` will never overflow `isize`.
                //         3. We have guaranteed that `block_r` will be `<= BLOCK`.
                //            Plus, `end_r` was initially set to the begin pointer of `offsets_` which was declared on the stack.
                //            Thus, we know that even in the worst case (all invocations of `is_less` returns true) we will only be at most 1 byte pass the end.
                //        Another unsafety operation here is dereferencing `elem`.
                //        However, `elem` was initially `1 * sizeof(T)` past the end and we decrement it by `1 * sizeof(T)` before accessing it.
                //        Plus, `block_r` was asserted to be less than `BLOCK` and `elem` will therefore at most be pointing to the beginning of the slice.
                unsafe {
                    // Branchless comparison.
                    elem = elem.sub(1);
                    *end_r = i as u8;
                    mirror_elem = mirror_elem.sub(1);
                    *mirror_end_r = i as u8;

                    let result = is_less(&*elem, pivot) as usize;

                    end_r = end_r.add(result);
                    mirror_end_r = mirror_end_r.add(result);
                }
            }
        }

        // Number of out-of-order elements to swap between the left and right side.
        let count = cmp::min(width(start_l, end_l), width(start_r, end_r));

        if count > 0 {
            macro_rules! left {
                () => {
                    l.add(*start_l as usize)
                };
            }
            macro_rules! right {
                () => {
                    r.sub((*start_r as usize) + 1)
                };
            }

            macro_rules! mirror_left {
                () => {
                    mirror_l.add(*mirror_start_l as usize)
                };
            }
            macro_rules! mirror_right {
                () => {
                    mirror_r.sub((*mirror_start_r as usize) + 1)
                };
            }

            // Instead of swapping one pair at the time, it is more efficient to perform a cyclic
            // permutation. This is not strictly equivalent to swapping, but produces a similar
            // result using fewer memory operations.

            // SAFETY: The use of `ptr::read` is valid because there is at least one element in
            // both `offsets_l` and `offsets_r`, so `left!` is a valid pointer to read from.
            //
            // The uses of `left!` involve calls to `offset` on `l`, which points to the
            // beginning of `v`. All the offsets pointed-to by `start_l` are at most `block_l`, so
            // these `offset` calls are safe as all reads are within the block. The same argument
            // applies for the uses of `right!`.
            //
            // The calls to `start_l.offset` are valid because there are at most `count-1` of them,
            // plus the final one at the end of the unsafe block, where `count` is the minimum number
            // of collected offsets in `offsets_l` and `offsets_r`, so there is no risk of there not
            // being enough elements. The same reasoning applies to the calls to `start_r.offset`.
            //
            // The calls to `copy_nonoverlapping` are safe because `left!` and `right!` are guaranteed
            // not to overlap, and are valid because of the reasoning above.
            //
            // CodeQL flags the dereferences inside `left!/right!/mirror_left!/mirror_right!` as
            // `rust/access-invalid-pointer` (alerts #1-#4). False positive - this block is
            // copied verbatim from `partition_in_blocks` in std 1.80.1 (see the module note
            // above), where the same expressions appear:
            // https://github.com/rust-lang/rust/blob/1.80.1/library/core/src/slice/sort.rs#L449-L479
            unsafe {
                let tmp = ptr::read(left!());
                ptr::copy_nonoverlapping(right!(), left!(), 1);

                let mirror_tmp = ptr::read(mirror_left!());
                ptr::copy_nonoverlapping(mirror_right!(), mirror_left!(), 1);

                for _ in 1..count {
                    start_l = start_l.add(1);
                    ptr::copy_nonoverlapping(left!(), right!(), 1);
                    start_r = start_r.add(1);
                    ptr::copy_nonoverlapping(right!(), left!(), 1);

                    mirror_start_l = mirror_start_l.add(1);
                    ptr::copy_nonoverlapping(mirror_left!(), mirror_right!(), 1);
                    mirror_start_r = mirror_start_r.add(1);
                    ptr::copy_nonoverlapping(mirror_right!(), mirror_left!(), 1);
                }

                ptr::copy_nonoverlapping(&tmp, right!(), 1);
                mem::forget(tmp);
                start_l = start_l.add(1);
                start_r = start_r.add(1);

                ptr::copy_nonoverlapping(&mirror_tmp, mirror_right!(), 1);
                mem::forget(mirror_tmp);
                mirror_start_l = mirror_start_l.add(1);
                mirror_start_r = mirror_start_r.add(1);
            }
        }

        if start_l == end_l {
            // All out-of-order elements in the left block were moved. Move to the next block.

            // block-width-guarantee
            // SAFETY: if `!is_done` then the slice width is guaranteed to be at least `2*BLOCK` wide. There
            // are at most `BLOCK` elements in `offsets_l` because of its size, so the `offset` operation is
            // safe. Otherwise, the debug assertions in the `is_done` case guarantee that
            // `width(l, r) == block_l + block_r`, namely, that the block sizes have been adjusted to account
            // for the smaller number of remaining elements.
            l = unsafe { l.add(block_l) };
            mirror_l = unsafe { mirror_l.add(mirror_block_l) };
        }

        if start_r == end_r {
            // All out-of-order elements in the right block were moved. Move to the previous block.

            // SAFETY: Same argument as [block-width-guarantee]. Either this is a full block `2*BLOCK`-wide,
            // or `block_r` has been adjusted for the last handful of elements.
            r = unsafe { r.sub(block_r) };
            mirror_r = unsafe { mirror_r.sub(mirror_block_r) };
        }

        if is_done {
            break;
        }
    }

    // All that remains now is at most one block (either the left or the right) with out-of-order
    // elements that need to be moved. Such remaining elements can be simply shifted to the end
    // within their block.

    if start_l < end_l {
        // The left block remains.
        // Move its remaining out-of-order elements to the far right.
        debug_assert_eq!(width(l, r), block_l);
        debug_assert_eq!(width(mirror_l, mirror_r), mirror_block_l);
        while start_l < end_l {
            // remaining-elements-safety
            // SAFETY: while the loop condition holds there are still elements in `offsets_l`, so it
            // is safe to point `end_l` to the previous element.
            //
            // The `ptr::swap` is safe if both its arguments are valid for reads and writes:
            //  - Per the debug assert above, the distance between `l` and `r` is `block_l`
            //    elements, so there can be at most `block_l` remaining offsets between `start_l`
            //    and `end_l`. This means `r` will be moved at most `block_l` steps back, which
            //    makes the `r.offset` calls valid (at that point `l == r`).
            //  - `offsets_l` contains valid offsets into `v` collected during the partitioning of
            //    the last block, so the `l.offset` calls are valid.
            //
            // CodeQL flags the `ptr::swap` dereferences below (including the mirrored copies) as
            // `rust/access-invalid-pointer` (alerts #5-#6). False positive - same reasoning as the
            // module note above; cf. std 1.80.1:
            // https://github.com/rust-lang/rust/blob/1.80.1/library/core/src/slice/sort.rs#L516-L531
            unsafe {
                end_l = end_l.sub(1);
                ptr::swap(l.add(*end_l as usize), r.sub(1));
                r = r.sub(1);

                mirror_end_l = mirror_end_l.sub(1);
                ptr::swap(mirror_l.add(*mirror_end_l as usize), mirror_r.sub(1));
                mirror_r = mirror_r.sub(1);
            }
        }
        width(target.as_mut_ptr(), r)
    } else if start_r < end_r {
        // The right block remains.
        // Move its remaining out-of-order elements to the far left.
        debug_assert_eq!(width(l, r), block_r);
        debug_assert_eq!(width(mirror_l, mirror_r), mirror_block_r);
        while start_r < end_r {
            // SAFETY: See the reasoning in [remaining-elements-safety].
            //
            // CodeQL flags the `ptr::swap` dereferences below (including the mirrored copies) as
            // `rust/access-invalid-pointer` (alerts #7-#8). False positive - same reasoning as the
            // module note above; cf. std 1.80.1:
            // https://github.com/rust-lang/rust/blob/1.80.1/library/core/src/slice/sort.rs#L539-L544
            unsafe {
                end_r = end_r.sub(1);
                ptr::swap(l, r.sub((*end_r as usize) + 1));
                l = l.add(1);

                mirror_end_r = mirror_end_r.sub(1);
                ptr::swap(mirror_l, mirror_r.sub((*mirror_end_r as usize) + 1));
                mirror_l = mirror_l.add(1);
            }
        }
        width(target.as_mut_ptr(), l)
    } else {
        // Nothing else to do, we're done.
        width(target.as_mut_ptr(), l)
    }
}

// UNCHANGED FROM sort.rs
fn choose_pivot<AA, BB, F>(v: &mut [AA], w: &mut [BB], is_less: &mut F) -> (usize, bool)
where
    F: FnMut(&AA, &AA) -> bool,
{
    // Minimum length to choose the median-of-medians method.
    // Shorter slices use the simple median-of-three method.
    const SHORTEST_MEDIAN_OF_MEDIANS: usize = 50;
    // Maximum number of swaps that can be performed in this function.
    const MAX_SWAPS: usize = 4 * 3;

    let len = v.len();

    // Three indices near which we are going to choose a pivot.
    let mut a = len / 4;
    let mut b = len / 4 * 2;
    let mut c = len / 4 * 3;

    // Counts the total number of swaps we are about to perform while sorting indices.
    let mut swaps = 0;

    if len >= 8 {
        // Swaps indices so that `v[a] <= v[b]`.
        // SAFETY: `len >= 8` so there are at least two elements in the neighborhoods of
        // `a`, `b` and `c`. This means the three calls to `sort_adjacent` result in
        // corresponding calls to `sort3` with valid 3-item neighborhoods around each
        // pointer, which in turn means the calls to `sort2` are done with valid
        // references. Thus the `v.get_unchecked` calls are safe, as is the `ptr::swap`
        // call.
        let mut sort2 = |a: &mut usize, b: &mut usize| unsafe {
            if is_less(v.get_unchecked(*b), v.get_unchecked(*a)) {
                ptr::swap(a, b);
                swaps += 1;
            }
        };

        // Swaps indices so that `v[a] <= v[b] <= v[c]`.
        let mut sort3 = |a: &mut usize, b: &mut usize, c: &mut usize| {
            sort2(a, b);
            sort2(b, c);
            sort2(a, b);
        };

        if len >= SHORTEST_MEDIAN_OF_MEDIANS {
            // Finds the median of `v[a - 1], v[a], v[a + 1]` and stores the index into `a`.
            let mut sort_adjacent = |a: &mut usize| {
                let tmp = *a;
                sort3(&mut (tmp - 1), a, &mut (tmp + 1));
            };

            // Find medians in the neighborhoods of `a`, `b`, and `c`.
            sort_adjacent(&mut a);
            sort_adjacent(&mut b);
            sort_adjacent(&mut c);
        }

        // Find the median among `a`, `b`, and `c`.
        sort3(&mut a, &mut b, &mut c);
    }

    if swaps < MAX_SWAPS {
        (b, swaps == 0)
    } else {
        // The maximum number of swaps was performed. Chances are the slice is descending or mostly
        // descending, so reversing will probably help sort it faster.
        v.reverse();
        w.reverse();
        (len - 1 - b, true)
    }
}

struct CopyOnDrop<T> {
    src: *const T,
    dest: *mut T,
}

impl<T> Drop for CopyOnDrop<T> {
    fn drop(&mut self) {
        // SAFETY:  This is a helper class.
        //          Please refer to its usage for correctness.
        //          Namely, one must be sure that `src` and `dst` does not overlap as required by `ptr::copy_nonoverlapping`.
        unsafe {
            ptr::copy_nonoverlapping(self.src, self.dest, 1);
        }
    }
}

#[test]
fn test_mirror_select_nth_unstable_by() {
    use core::cmp::Ordering::{Equal, Greater, Less};
    use rand::prelude::IndexedRandom;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    const LEN: usize = 32;

    let mut rng = StdRng::seed_from_u64(0x5eed_fade);

    let mut orig = [0; LEN];
    let mut orig_mirror = [0; LEN];

    for &modulus in &[5, 10, 1000] {
        for _ in 0..10 {
            for i in 0..LEN {
                orig[i] = rng.random::<i32>() % modulus;
                orig_mirror[i] = 1_000_000 - orig[i]
            }

            // Sort in ascending order.
            for pivot in 0..LEN {
                let mut v = orig;
                let mut w = orig_mirror;
                // let (left, pivot, right) = mirror_select_nth_unstable_by(&mut v, &mut w,pivot, |a, b| a.cmp(b));
                let f = |a: &i32, b: &i32| a.cmp(b);
                mirror_select_nth_unstable_by(&mut v, &mut w, pivot, f);

                //assert_eq!(left.len() + right.len(), LEN - 1);

                for l in 0..pivot {
                    assert!(v[l] <= v[pivot]);
                    for r in (pivot + 1)..LEN {
                        assert!(v[pivot] <= v[r]);
                    }
                }

                for i in 0..LEN {
                    assert!(1_000_000 - w[i] == v[i]);
                }
            }

            // Sort in descending order.
            let sort_descending_comparator = |a: &i32, b: &i32| b.cmp(a);
            let v_sorted_descending = {
                let mut v = orig;
                v.sort_by(sort_descending_comparator);
                v
            };

            for pivot in 0..LEN {
                let mut v = orig;
                let mut w = orig_mirror;
                mirror_select_nth_unstable_by(&mut v, &mut w, pivot, sort_descending_comparator);

                assert_eq!(v_sorted_descending[pivot], v[pivot]);
                for i in 0..pivot {
                    for j in pivot..LEN {
                        assert!(v[j] <= v[i]);
                    }
                    assert!(1_000_000 - w[i] == v[i]);
                }
            }
        }
    }

    // Sort at index using a completely random comparison function.
    // This will reorder the elements *somehow*, but won't panic.
    let mut v = [0; 500];
    let mut w = [0; 500];
    for i in 0..v.len() {
        v[i] = i as i32;
        w[i] = 1_000_000 - v[i];
    }

    for pivot in 0..v.len() {
        mirror_select_nth_unstable_by(&mut v, &mut w, pivot, |_, _| {
            *[Less, Equal, Greater].choose(&mut rng).unwrap()
        });
        v.sort();
        w.sort();
        for i in 0..v.len() {
            assert_eq!(v[i], i as i32);
            assert_eq!(w[i], 1_000_000 - 499 + i as i32);
        }
    }
}
