#!/usr/bin/env julia
# Standalone benchmark for Julia's NearestNeighbors.jl KDTree, run as its own
# process (there is no practical way to embed a Julia runtime into the Rust
# criterion harness the way the C++ competitors are FFI'd in). Output is a
# JSON file in the same schema tools/criterion-export produces, using the
# same group/function naming convention as profile_cpp_competitors.rs, so
# chart_cpp_competitor_results.py can merge it in as just another library
# series (prefix "nearestneighborsjl_").
#
# Usage: julia bench_nearestneighbors_jl.jl <output.json>
# Env vars (same names/semantics as the Rust benches):
#   KIDDO_PROFILE_QUERIES, KIDDO_PROFILE_MIN_LOG2_POINTS,
#   KIDDO_PROFILE_MAX_LOG2_POINTS, KIDDO_PROFILE_RADIUS

using NearestNeighbors
using Random
using JSON
using Statistics

const K = 3
const MAX_QTYS = [5, 20, 50]
const SAMPLES = 30
const WARMUP = 3

query_count = parse(Int, get(ENV, "KIDDO_PROFILE_QUERIES", "1000"))
min_log2 = parse(Int, get(ENV, "KIDDO_PROFILE_MIN_LOG2_POINTS", "16"))
max_log2 = parse(Int, get(ENV, "KIDDO_PROFILE_MAX_LOG2_POINTS", "19"))
radius = parse(Float64, get(ENV, "KIDDO_PROFILE_RADIUS", "0.05"))
out_path = length(ARGS) >= 1 ? ARGS[1] : "bench_result-nearestneighborsjl.json"

@assert query_count > 0
@assert min_log2 <= max_log2

function build_points(::Type{T}, n::Int, seed::UInt64) where {T}
    rng = MersenneTwister(seed)
    rand(rng, T, K, n)
end

function build_queries(::Type{T}, n::Int, seed::UInt64) where {T}
    rng = MersenneTwister(seed)
    [rand(rng, T, K) for _ in 1:n]
end

function expected_nearest(points::Matrix{T}, q::Vector{T}, max_qty::Int) where {T}
    n = size(points, 2)
    d2 = Vector{T}(undef, n)
    @inbounds for i in 1:n
        s = zero(T)
        for k in 1:K
            diff = points[k, i] - q[k]
            s += diff * diff
        end
        d2[i] = s
    end
    sort(d2)[1:min(max_qty, n)]
end

function expected_within_count(points::Matrix{T}, q::Vector{T}, radius2::T) where {T}
    n = size(points, 2)
    count = 0
    @inbounds for i in 1:n
        s = zero(T)
        for k in 1:K
            diff = points[k, i] - q[k]
            s += diff * diff
        end
        if s <= radius2
            count += 1
        end
    end
    count
end

function validate(points::Matrix{T}, tree, q::Vector{T}, radius::T) where {T}
    max_qty = MAX_QTYS[end]
    expected = expected_nearest(points, q, max_qty)
    _, dists = knn(tree, q, max_qty, true)
    actual2 = sort([d * d for d in dists])
    tol = T === Float32 ? T(1.0e-4) : T(1.0e-10)
    for (a, e) in zip(actual2, expected)
        @assert abs(a - e) <= tol * max(abs(e), one(T)) "distance mismatch: actual=$a expected=$e"
    end
    expected_within = expected_within_count(points, q, radius * radius)
    actual_within = length(inrange(tree, q, radius, false))
    @assert actual_within == expected_within "within-radius count mismatch: actual=$actual_within expected=$expected_within"
    println("Testing nearestneighborsjl ($T) at n=$(size(points, 2)): OK")
end

function timed_trials(f::Function)
    for _ in 1:WARMUP
        f()
    end
    times_ns = Vector{Float64}(undef, SAMPLES)
    for i in 1:SAMPLES
        t0 = time_ns()
        f()
        times_ns[i] = Float64(time_ns() - t0)
    end
    times_ns
end

function estimate(times_ns::Vector{Float64})
    m = mean(times_ns)
    s = std(times_ns)
    n = length(times_ns)
    se = s / sqrt(n)
    # Durations can't be negative; a wide relative spread on fast/noisy
    # samples can otherwise push mean - 1.96*se below zero.
    lower = max(m - 1.96 * se, m * 0.01)
    (mean = m, lower = lower, upper = m + 1.96 * se)
end

function make_result(group_id::String, function_id::String, tree_size::Int, query_count::Int, times_ns::Vector{Float64})
    est = estimate(times_ns)
    Dict(
        "benchmark" => "$group_id/$function_id/$tree_size",
        "metadata" => Dict(
            "group_id" => group_id,
            "function_id" => function_id,
            "value_str" => string(tree_size),
            "throughput" => Dict("Elements" => query_count),
        ),
        "estimates" => Dict(
            "mean" => Dict(
                "point_estimate" => est.mean,
                "confidence_interval" => Dict(
                    "lower_bound" => est.lower,
                    "upper_bound" => est.upper,
                ),
            ),
        ),
    )
end

function bench_scalar(::Type{T}, results::Vector{Any}) where {T}
    scalar = T === Float32 ? "f32" : "f64"
    group_id = "profile_cpp_competitors/$scalar"
    queries = build_queries(T, query_count, UInt64(0x5eed_0000_0000_0002))

    for log2n in min_log2:max_log2
        n = 1 << log2n
        points = build_points(T, n, UInt64(0x5eed_0000_0000_0001))
        tree = KDTree(points)

        if log2n == min_log2
            validate(points, tree, queries[1], T(radius))
        end

        push!(results, make_result(group_id, "nearestneighborsjl_nearest_one", n, query_count, timed_trials(() -> begin
            for q in queries
                knn(tree, q, 1, false)
            end
        end)))

        for max_qty in MAX_QTYS
            push!(results, make_result(group_id, "nearestneighborsjl_nearest_n_k$max_qty", n, query_count, timed_trials(() -> begin
                for q in queries
                    knn(tree, q, max_qty, false)
                end
            end)))
        end

        radius_id = "nearestneighborsjl_within_radius_r$radius"
        push!(results, make_result(group_id, radius_id, n, query_count, timed_trials(() -> begin
            for q in queries
                inrange(tree, q, radius, false)
            end
        end)))

        println(stderr, "nearestneighborsjl $scalar n=$n done")
    end
end

results = Any[]
bench_scalar(Float32, results)
bench_scalar(Float64, results)

open(out_path, "w") do io
    JSON.print(io, Dict(
        "schema_version" => 1,
        "criterion_root" => "julia",
        "collected_at_unix_ms" => round(Int, time() * 1000),
        "filters" => String[],
        "results" => results,
    ))
end

println("wrote $out_path")
