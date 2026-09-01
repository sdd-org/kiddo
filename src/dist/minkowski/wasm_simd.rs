use core::arch::wasm32::*;

use crate::dist::distance_metric_wasm_simd::{WasmSimdF32LeafOps, WasmSimdF64LeafOps};

#[inline(always)]
unsafe fn pow_f64x2<const P: u32>(x: v128) -> v128 {
    if P == 0 {
        return f64x2_splat(1.0);
    }
    if P == 1 {
        return x;
    }
    if P == 2 {
        return f64x2_mul(x, x);
    }
    if P == 3 {
        let x2 = f64x2_mul(x, x);
        return f64x2_mul(x2, x);
    }
    if P == 4 {
        let x2 = f64x2_mul(x, x);
        return f64x2_mul(x2, x2);
    }

    let mut acc = f64x2_splat(1.0);
    let mut base = x;
    let mut exp = P;
    while exp != 0 {
        if exp & 1 == 1 {
            acc = f64x2_mul(acc, base);
        }
        exp >>= 1;
        if exp != 0 {
            base = f64x2_mul(base, base);
        }
    }
    acc
}

#[inline(always)]
unsafe fn pow_f32x4<const P: u32>(x: v128) -> v128 {
    if P == 0 {
        return f32x4_splat(1.0);
    }
    if P == 1 {
        return x;
    }
    if P == 2 {
        return f32x4_mul(x, x);
    }
    if P == 3 {
        let x2 = f32x4_mul(x, x);
        return f32x4_mul(x2, x);
    }
    if P == 4 {
        let x2 = f32x4_mul(x, x);
        return f32x4_mul(x2, x2);
    }

    let mut acc = f32x4_splat(1.0);
    let mut base = x;
    let mut exp = P;
    while exp != 0 {
        if exp & 1 == 1 {
            acc = f32x4_mul(acc, base);
        }
        exp >>= 1;
        if exp != 0 {
            base = f32x4_mul(base, base);
        }
    }
    acc
}

#[inline(always)]
fn pow_f64<const P: u32>(x: f64) -> f64 {
    if P == 0 {
        return 1.0;
    }
    if P == 1 {
        return x;
    }
    if P == 2 {
        return x * x;
    }
    if P == 3 {
        return x * x * x;
    }
    if P == 4 {
        let x2 = x * x;
        return x2 * x2;
    }
    x.powi(P as i32)
}

#[inline(always)]
fn pow_f32<const P: u32>(x: f32) -> f32 {
    if P == 0 {
        return 1.0;
    }
    if P == 1 {
        return x;
    }
    if P == 2 {
        return x * x;
    }
    if P == 3 {
        return x * x * x;
    }
    if P == 4 {
        let x2 = x * x;
        return x2 * x2;
    }
    x.powi(P as i32)
}

pub struct MinkowskiWasmSimdF64LeafOps<const P: u32>;

impl<const P: u32> WasmSimdF64LeafOps for MinkowskiWasmSimdF64LeafOps<P> {
    #[inline(always)]
    unsafe fn dist_k0_f64x2(delta: v128) -> v128 {
        pow_f64x2::<P>(f64x2_abs(delta))
    }

    #[inline(always)]
    unsafe fn dist_kn_f64x2(acc: v128, delta: v128) -> v128 {
        f64x2_add(acc, pow_f64x2::<P>(f64x2_abs(delta)))
    }

    #[inline(always)]
    fn dist_k0_f64x1(delta: f64) -> f64 {
        pow_f64::<P>(delta.abs())
    }

    #[inline(always)]
    fn dist_kn_f64x1(acc: f64, delta: f64) -> f64 {
        acc + pow_f64::<P>(delta.abs())
    }
}

pub struct MinkowskiWasmSimdF32LeafOps<const P: u32>;

impl<const P: u32> WasmSimdF32LeafOps for MinkowskiWasmSimdF32LeafOps<P> {
    #[inline(always)]
    unsafe fn dist_k0_f32x4(delta: v128) -> v128 {
        pow_f32x4::<P>(f32x4_abs(delta))
    }

    #[inline(always)]
    unsafe fn dist_kn_f32x4(acc: v128, delta: v128) -> v128 {
        f32x4_add(acc, pow_f32x4::<P>(f32x4_abs(delta)))
    }

    #[inline(always)]
    fn dist_k0_f32x1(delta: f32) -> f32 {
        pow_f32::<P>(delta.abs())
    }

    #[inline(always)]
    fn dist_kn_f32x1(acc: f32, delta: f32) -> f32 {
        acc + pow_f32::<P>(delta.abs())
    }
}
