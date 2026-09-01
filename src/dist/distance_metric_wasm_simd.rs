use core::arch::wasm32::v128;

/// WebAssembly SIMD128 f64 leaf-kernel operations contract.
pub trait WasmSimdF64LeafOps {
    /// Calculate distance on 2 f64 lanes for the first dimension.
    unsafe fn dist_k0_f64x2(delta: v128) -> v128;

    /// Accumulate distance on 2 f64 lanes for subsequent dimensions.
    unsafe fn dist_kn_f64x2(acc: v128, delta: v128) -> v128;

    /// Calculate scalar f64 distance for the first dimension.
    fn dist_k0_f64x1(delta: f64) -> f64;

    /// Accumulate scalar f64 distance for subsequent dimensions.
    fn dist_kn_f64x1(acc: f64, delta: f64) -> f64;
}

/// Placeholder implementation for metrics without SIMD128 f64 specializations.
pub struct UnsupportedWasmSimdF64LeafOps;

impl WasmSimdF64LeafOps for UnsupportedWasmSimdF64LeafOps {
    #[inline(always)]
    unsafe fn dist_k0_f64x2(_delta: v128) -> v128 {
        panic!("SIMD128 f64 leaf ops are not implemented for this metric")
    }

    #[inline(always)]
    unsafe fn dist_kn_f64x2(_acc: v128, _delta: v128) -> v128 {
        panic!("SIMD128 f64 leaf ops are not implemented for this metric")
    }

    #[inline(always)]
    fn dist_k0_f64x1(_delta: f64) -> f64 {
        panic!("SIMD128 f64 leaf ops are not implemented for this metric")
    }

    #[inline(always)]
    fn dist_kn_f64x1(_acc: f64, _delta: f64) -> f64 {
        panic!("SIMD128 f64 leaf ops are not implemented for this metric")
    }
}

/// WebAssembly SIMD128 f32 leaf-kernel operations contract.
pub trait WasmSimdF32LeafOps {
    /// Calculate distance on 4 f32 lanes for the first dimension.
    unsafe fn dist_k0_f32x4(delta: v128) -> v128;

    /// Accumulate distance on 4 f32 lanes for subsequent dimensions.
    unsafe fn dist_kn_f32x4(acc: v128, delta: v128) -> v128;

    /// Calculate scalar f32 distance for the first dimension.
    fn dist_k0_f32x1(delta: f32) -> f32;

    /// Accumulate scalar f32 distance for subsequent dimensions.
    fn dist_kn_f32x1(acc: f32, delta: f32) -> f32;
}

/// Placeholder implementation for metrics without SIMD128 f32 specializations.
pub struct UnsupportedWasmSimdF32LeafOps;

impl WasmSimdF32LeafOps for UnsupportedWasmSimdF32LeafOps {
    #[inline(always)]
    unsafe fn dist_k0_f32x4(_delta: v128) -> v128 {
        panic!("SIMD128 f32 leaf ops are not implemented for this metric")
    }

    #[inline(always)]
    unsafe fn dist_kn_f32x4(_acc: v128, _delta: v128) -> v128 {
        panic!("SIMD128 f32 leaf ops are not implemented for this metric")
    }

    #[inline(always)]
    fn dist_k0_f32x1(_delta: f32) -> f32 {
        panic!("SIMD128 f32 leaf ops are not implemented for this metric")
    }

    #[inline(always)]
    fn dist_kn_f32x1(_acc: f32, _delta: f32) -> f32 {
        panic!("SIMD128 f32 leaf ops are not implemented for this metric")
    }
}
