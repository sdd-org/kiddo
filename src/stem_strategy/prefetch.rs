//! Prefetch helper functions

#[cfg(all(target_arch = "aarch64", kiddo_nightly))]
#[inline(always)]
pub(crate) unsafe fn prefetch_t0(ptr: *const u8) {
    use core::arch::aarch64::{_prefetch, _PREFETCH_LOCALITY3, _PREFETCH_READ};

    // prevent LLVM from "helpfully" removing redundant prefetches
    #[allow(clippy::pointers_in_nomem_asm_block)]
    {
        core::arch::asm!("", in("x0") ptr, options(nomem, nostack, preserves_flags));
    }

    _prefetch::<_PREFETCH_READ, _PREFETCH_LOCALITY3>(ptr as *const i8);
}

#[cfg(all(target_arch = "aarch64", not(kiddo_nightly)))]
#[inline(always)]
pub(crate) unsafe fn prefetch_t0(_ptr: *const u8) {}

#[cfg(all(target_arch = "aarch64", kiddo_nightly))]
#[inline(always)]
pub(crate) unsafe fn prefetch_t1(ptr: *const u8) {
    use core::arch::aarch64::{_prefetch, _PREFETCH_LOCALITY2, _PREFETCH_READ};

    // prevent LLVM from "helpfully" removing redundant prefetches
    #[allow(clippy::pointers_in_nomem_asm_block)]
    {
        core::arch::asm!("", in("x0") ptr, options(nomem, nostack, preserves_flags));
    }

    _prefetch::<_PREFETCH_READ, _PREFETCH_LOCALITY2>(ptr as *const i8);
}

#[cfg(all(target_arch = "aarch64", not(kiddo_nightly)))]
#[inline(always)]
pub(crate) unsafe fn prefetch_t1(_ptr: *const u8) {}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) unsafe fn prefetch_t0(ptr: *const u8) {
    use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};

    // prevent LLVM from "helpfully" removing redundant prefetches
    core::arch::asm!("", in("rax") ptr, options(nomem, nostack, preserves_flags));

    _mm_prefetch::<{ _MM_HINT_T0 }>(ptr as *const i8);
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) unsafe fn prefetch_t1(ptr: *const u8) {
    use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T1};

    // prevent LLVM from "helpfully" removing redundant prefetches
    core::arch::asm!("", in("rax") ptr, options(nomem, nostack, preserves_flags));

    _mm_prefetch::<{ _MM_HINT_T1 }>(ptr as *const i8);
}

// Other targets, including wasm32, do not expose a portable hardware-prefetch
// intrinsic. Keep the traversal strategies available there and let these calls
// compile away.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline(always)]
pub(crate) unsafe fn prefetch_t0(_ptr: *const u8) {}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline(always)]
pub(crate) unsafe fn prefetch_t1(_ptr: *const u8) {}
