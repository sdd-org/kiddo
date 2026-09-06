//! Standalone interleaved benchmark. Configuration is through JOIN_* variables.
//! See docs/tree-radius-join.md for reproducible commands and workload selection.
use kiddo::batch::{Executor, ThreadPoolBuilder};
use kiddo::kd_tree::KdTreeAccessor;
use kiddo::leaf_strategy::{FlatVec, VecOfArenas};
use kiddo::stem_strategy::{Donnelly, Eytzinger};
use kiddo::{KdTree, LeafStrategy, SquaredEuclidean, StemStrategy};
use rayon::prelude::*;
use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

fn env(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.into())
}
fn integer(name: &str, default: &str) -> usize {
    env(name, default).parse().unwrap()
}

macro_rules! runner {
    ($name:ident, $a:ty, $store:ident) => {
        fn $name<SS: StemStrategy, const K: usize, const B: usize>() {
            let n = integer("JOIN_N","65536");
            let m = integer("JOIN_M", &n.to_string());
            let threads = integer("JOIN_THREADS","1");
            let rounds = integer("JOIN_ROUNDS","7");
            let kind = env("JOIN_DISTRIBUTION","uniform");
            let sink = env("JOIN_SINK","count");
            let only = env("JOIN_ONLY","both");
            let volume = if K == 2 { std::f64::consts::PI } else { 4.0 * std::f64::consts::PI / 3.0 };
            let default_radius = (16.0 / (m.max(1) as f64 * volume)).powf(1.0 / K as f64);
            let radius: $a = env("JOIN_RADIUS",&default_radius.to_string()).parse().unwrap();
            let r2 = radius * radius;
            let make = |len: usize, mut state: u64, right: bool| {
                (0..len).map(|i| std::array::from_fn(|d| {
                    state ^= state << 13; state ^= state >> 7; state ^= state << 17;
                    let x = (state >> 11) as f64 / (1u64 << 53) as f64;
                    (match kind.as_str() {
                        "uniform" => x,
                        "clustered" => ((i >> d) & 1) as f64 * 0.75 + x * 0.02,
                        "separated" => x + if right { 2.0 } else { 0.0 },
                        "anisotropic" => if d == 0 { x } else { x * 0.001 },
                        "duplicates" => 0.5,
                        _ => panic!("unknown distribution"),
                    }) as $a
                })).collect::<Vec<[$a;K]>>()
            };
            let started = Instant::now();
            let left = KdTree::<$a,u32,SS,$store<$a,u32,K,B>,K,B>::new_from_slice(&make(n,17,false)).unwrap();
            let right = KdTree::<$a,u32,SS,$store<$a,u32,K,B>,K,B>::new_from_slice(&make(m,78,true)).unwrap();
            eprintln!("construction_seconds={:.6}",started.elapsed().as_secs_f64());
            let pool = Arc::new(ThreadPoolBuilder::new().num_threads(threads).build().unwrap());
            let mut executor = if threads == 1 { Executor::serial() } else { Executor::parallel_in_pool(pool.clone()) };
            if let Ok(multiplier) = std::env::var("JOIN_TASK_MULTIPLIER") {
                executor = executor.with_static_chunk_thread_multiplier(std::num::NonZeroUsize::new(multiplier.parse().unwrap()).unwrap());
            }
            let count = left.query_tree(&right).within::<SquaredEuclidean<$a>>(r2).with_executor(&executor).count();
            if sink == "collect" || sink == "collect-distances" {
                assert!(count <= integer("JOIN_MAX_RESULTS","16000000"), "collection exceeds JOIN_MAX_RESULTS; use count/streaming or raise the explicit limit");
            }
            let baseline = || {
                let query_leaf = |leaf| {
                    let leaves = KdTreeAccessor::leaves(&left);
                    let len = <$store<$a,u32,K,B> as LeafStrategy<$a,u32,SS,K,B>>::leaf_len(leaves,leaf);
                    let mut count = 0;
                    let mut out = Vec::new();
                    let mut items_out = Vec::new();
                    let mut scratch = right.create_scratch::<SquaredEuclidean<$a>>();
                    for i in 0..len {
                        let (point,item) = <$store<$a,u32,K,B> as LeafStrategy<$a,u32,SS,K,B>>::leaf_point_item(leaves,leaf,i);
                        match sink.as_str() {
                            "count" => right.query(&point).within::<SquaredEuclidean<$a>>(r2).unsorted().without_items().without_distances()
                                .with_scratch(&mut scratch).visit(|_| count += 1),
                            "items" => right.query(&point).within::<SquaredEuclidean<$a>>(r2).unsorted().without_distances()
                                .with_scratch(&mut scratch).visit(|p| { black_box((item,p.item)); count += 1; }),
                            "distances" => right.query(&point).within::<SquaredEuclidean<$a>>(r2).unsorted()
                                .with_scratch(&mut scratch).visit(|p| { black_box((item,p.item,p.distance)); count += 1; }),
                            "collect" => right.query(&point).within::<SquaredEuclidean<$a>>(r2).unsorted().without_distances()
                                .with_scratch(&mut scratch).visit(|p| { items_out.push((item,p.item)); count += 1; }),
                            "collect-distances" => right.query(&point).within::<SquaredEuclidean<$a>>(r2).unsorted()
                                .with_scratch(&mut scratch).visit(|p| { out.push((item,p.item,p.distance)); count += 1; }),
                            _ => panic!("unknown sink"),
                        }
                    }
                    (count,out,items_out)
                };
                if sink.starts_with("collect") {
                    let chunks: Vec<_> = if threads == 1 { (0..left.leaf_count()).map(query_leaf).collect() }
                        else { pool.install(|| (0..left.leaf_count()).into_par_iter().map(query_leaf).collect()) };
                    let n: usize = chunks.iter().map(|(n,_,_)| n).sum();
                    if sink == "collect" {
                        let mut out = Vec::with_capacity(n);
                        for (_,_,v) in chunks { out.extend(v); }
                        black_box(out);
                    } else {
                        let mut out = Vec::with_capacity(n);
                        for (_,v,_) in chunks { out.extend(v); }
                        black_box(out);
                    }
                    n
                } else if threads == 1 { (0..left.leaf_count()).map(|i| query_leaf(i).0).sum() }
                else { pool.install(|| (0..left.leaf_count()).into_par_iter().map(|i| query_leaf(i).0).sum()) }
            };
            let join = || {
                let q = left.query_tree(&right).within::<SquaredEuclidean<$a>>(r2).with_executor(&executor);
                match sink.as_str() {
                    "count" => assert_eq!(black_box(q.count()),count),
                    "items" => q.without_distances().for_each(|p| { black_box(p); }),
                    "distances" => q.for_each(|p| { black_box(p); }),
                    "collect" => assert_eq!(black_box(q.without_distances().execute()).len(),count),
                    "collect-distances" => assert_eq!(black_box(q.execute()).len(),count),
                    _ => panic!("unknown sink"),
                }
            };
            // Each variant warms once. Repeated runs alternate execution order.
            join();
            if only != "join" {
                let old = baseline();
                if old != count {
                    assert_eq!(env("JOIN_ALLOW_BASELINE_BOUNDARY_DIFFERENCES","0"),"1",
                        "baseline={old} scalar-exact join={count}; investigate with large_radius_boundary_diagnostic before accepting differences");
                    eprintln!("boundary-difference allowed: baseline={old} scalar-exact-join={count}");
                }
            }
            #[cfg(feature = "exact_query_stats")]
            {
                kiddo::tree_join_stats::reset();
                join();
                eprintln!("join_stats={:?}",kiddo::tree_join_stats::snapshot());
            }
            println!("variant,round,float,dimensions,bucket,storage,layout,n,m,threads,distribution,sink,matches,seconds");
            for round in 0..rounds {
                for baseline_first in [round % 2 == 0,round % 2 != 0] {
                    if (only == "join" && baseline_first) || (only == "baseline" && !baseline_first) { continue; }
                    let start = Instant::now();
                    let actual_count = if baseline_first { black_box(baseline()) } else { join(); count };
                    println!("{},{round},{},{K},{B},{},{},{n},{m},{threads},{kind},{sink},{actual_count},{:.9}",
                        if baseline_first { "baseline" } else { "join" },stringify!($a),stringify!($store),std::any::type_name::<SS>(),start.elapsed().as_secs_f64());
                    assert!(actual_count == count || env("JOIN_ALLOW_BASELINE_BOUNDARY_DIFFERENCES","0") == "1");
                }
            }
        }
    };
}
runner!(f32_flat, f32, FlatVec);
runner!(f64_flat, f64, FlatVec);
runner!(f32_arena, f32, VecOfArenas);
runner!(f64_arena, f64, VecOfArenas);

fn main() {
    macro_rules! run {
        ($runner:ident,$k:literal) => {
            match env("JOIN_LAYOUT", "eytzinger").as_str() {
                "eytzinger" => $runner::<Eytzinger, $k, 32>(),
                "donnelly" => $runner::<Donnelly<3>, $k, 32>(),
                _ => panic!("unknown layout"),
            }
        };
    }
    match (
        env("JOIN_FLOAT", "f32").as_str(),
        env("JOIN_STORAGE", "flat").as_str(),
        integer("JOIN_DIMENSIONS", "3"),
    ) {
        ("f32", "flat", 2) => run!(f32_flat, 2),
        ("f32", "flat", 3) => run!(f32_flat, 3),
        ("f64", "flat", 2) => run!(f64_flat, 2),
        ("f64", "flat", 3) => run!(f64_flat, 3),
        ("f32", "arena", 2) => run!(f32_arena, 2),
        ("f32", "arena", 3) => run!(f32_arena, 3),
        ("f64", "arena", 2) => run!(f64_arena, 2),
        ("f64", "arena", 3) => run!(f64_arena, 3),
        _ => panic!("choose f32/f64, flat/arena and 2/3 dimensions"),
    }
}
