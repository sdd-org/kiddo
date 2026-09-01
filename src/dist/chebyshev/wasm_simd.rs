use core::arch::wasm32::*;

use crate::dist::distance_metric_wasm_simd::{WasmSimdF32LeafOps, WasmSimdF64LeafOps};

pub struct ChebyshevWasmSimdF64LeafOps;

impl WasmSimdF64LeafOps for ChebyshevWasmSimdF64LeafOps {
    #[inline(always)]
    unsafe fn dist_k0_f64x2(delta: v128) -> v128 {
        f64x2_abs(delta)
    }

    #[inline(always)]
    unsafe fn dist_kn_f64x2(acc: v128, delta: v128) -> v128 {
        f64x2_max(acc, f64x2_abs(delta))
    }

    #[inline(always)]
    fn dist_k0_f64x1(delta: f64) -> f64 {
        delta.abs()
    }

    #[inline(always)]
    fn dist_kn_f64x1(acc: f64, delta: f64) -> f64 {
        acc.max(delta.abs())
    }
}

pub struct ChebyshevWasmSimdF32LeafOps;

impl WasmSimdF32LeafOps for ChebyshevWasmSimdF32LeafOps {
    #[inline(always)]
    unsafe fn dist_k0_f32x4(delta: v128) -> v128 {
        f32x4_abs(delta)
    }

    #[inline(always)]
    unsafe fn dist_kn_f32x4(acc: v128, delta: v128) -> v128 {
        f32x4_max(acc, f32x4_abs(delta))
    }

    #[inline(always)]
    fn dist_k0_f32x1(delta: f32) -> f32 {
        delta.abs()
    }

    #[inline(always)]
    fn dist_kn_f32x1(acc: f32, delta: f32) -> f32 {
        acc.max(delta.abs())
    }
}
