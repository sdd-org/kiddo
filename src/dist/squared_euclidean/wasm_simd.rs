use core::arch::wasm32::*;

use crate::dist::distance_metric_wasm_simd::{WasmSimdF32LeafOps, WasmSimdF64LeafOps};

pub struct SquaredEuclideanWasmSimdF64LeafOps;

impl WasmSimdF64LeafOps for SquaredEuclideanWasmSimdF64LeafOps {
    #[inline(always)]
    unsafe fn dist_k0_f64x2(delta: v128) -> v128 {
        f64x2_mul(delta, delta)
    }

    #[inline(always)]
    unsafe fn dist_kn_f64x2(acc: v128, delta: v128) -> v128 {
        f64x2_add(acc, f64x2_mul(delta, delta))
    }

    #[inline(always)]
    fn dist_k0_f64x1(delta: f64) -> f64 {
        delta * delta
    }

    #[inline(always)]
    fn dist_kn_f64x1(acc: f64, delta: f64) -> f64 {
        acc + delta * delta
    }
}

pub struct SquaredEuclideanWasmSimdF32LeafOps;

impl WasmSimdF32LeafOps for SquaredEuclideanWasmSimdF32LeafOps {
    #[inline(always)]
    unsafe fn dist_k0_f32x4(delta: v128) -> v128 {
        f32x4_mul(delta, delta)
    }

    #[inline(always)]
    unsafe fn dist_kn_f32x4(acc: v128, delta: v128) -> v128 {
        f32x4_add(acc, f32x4_mul(delta, delta))
    }

    #[inline(always)]
    fn dist_k0_f32x1(delta: f32) -> f32 {
        delta * delta
    }

    #[inline(always)]
    fn dist_kn_f32x1(acc: f32, delta: f32) -> f32 {
        acc + delta * delta
    }
}
