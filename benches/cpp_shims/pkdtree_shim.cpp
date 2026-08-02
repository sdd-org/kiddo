// Thin extern "C" shim over ucrparlay/Pkd-tree (SIGMOD'25), a parallel
// k-d tree built on ParlayLib fork-join parallelism.
//
// Pkd-tree has no clean top-level "kNN(point, k)" API; real usage (mirrored
// from their own tests/testFramework.h) means grabbing the raw root node
// and bounding box, and composing a kBoundedQueue per query yourself. Two
// query modes are exposed here:
//   - *_single_*: one FFI call per query, matching every other competitor's
//     benchmark methodology (sequential, no parallel_for). This is the
//     number that belongs on the same chart as nanoflann/ALGLIB/kiddo.
//   - *_batch_*: one FFI call covering all queries at once via
//     parlay::parallel_for across every core, matching Pkd-tree's actual
//     design intent and its own published benchmark methodology. This is
//     NOT comparable to the sequential per-query numbers above -- it's a
//     batch-throughput metric, charted separately.
//
// Only nearest_one/nearest_n are implemented; Pkd-tree's range query is
// another box-composition API of similar complexity to k_nearest, and
// wasn't worth the added risk for this comparison.
//
// Point indices are recovered by pointer arithmetic against the (build()
// reorders points in place) stored point array, since cpdd::PointType
// carries no id field of its own.
//
// cpdd::PointType and k_nearest's DIM argument are both compile-time-shaped
// around a fixed dimension (Pkd-tree's whole performance case rests on the
// compiler unrolling per-dimension arithmetic), so two dimensions are
// compiled in rather than one: 3, for the f64 comparisons, and 4, for f32 --
// the two block heights the cyclic Donnelly strategies are native to.
// `build` picks the matching specialization at runtime from the `dim`
// argument and tags the returned handle with it; every other entry point
// switches on that tag once per call to recover the concrete type. Any
// dimension other than 3 or 4 aborts, since nothing else is compiled in.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

#include "cpdd/cpdd.h"

namespace {

template <typename Scalar, int Dim>
using Point = cpdd::PointType<Scalar, Dim>;

template <typename Scalar, int Dim>
using Tree = cpdd::ParallelKDtree<Point<Scalar, Dim>>;

template <typename Scalar, int Dim>
using NNPair = std::pair<std::reference_wrapper<Point<Scalar, Dim>>, Scalar>;

template <typename Scalar, int Dim>
struct HandleT {
    Tree<Scalar, Dim> tree;
    typename Tree<Scalar, Dim>::points wp;
    typename Tree<Scalar, Dim>::node* root = nullptr;
    typename Tree<Scalar, Dim>::box root_box;

    explicit HandleT(const Scalar* points, uint64_t n)
        : wp(Tree<Scalar, Dim>::points::uninitialized(n)) {
        for (uint64_t i = 0; i < n; ++i) {
            wp[i] = Point<Scalar, Dim>(const_cast<Scalar*>(points + i * Dim));
        }
        tree.build(parlay::make_slice(wp), static_cast<uint_fast8_t>(Dim));
        root = tree.get_root();
        root_box = tree.get_root_box();
    }

    uint64_t index_of(const Point<Scalar, Dim>& p) const {
        return static_cast<uint64_t>(&p - &wp[0]);
    }
};

// Type-erased, dimension-tagged handle. `inner` points at a HandleT<Scalar,
// 3> or HandleT<Scalar, 4>, decided by `dim` at construction and read back by
// every call below -- the tag, not the pointer's static type, is what makes
// this safe.
template <typename Scalar>
struct Handle {
    uint32_t dim;
    void* inner;
};

// Recovers the concrete HandleT and calls `f` on it, so each entry point
// below is one line instead of a repeated switch.
template <typename Scalar, typename F>
auto dispatch(Handle<Scalar>* handle, F&& f) {
    switch (handle->dim) {
        case 3: return f(*static_cast<HandleT<Scalar, 3>*>(handle->inner));
        case 4: return f(*static_cast<HandleT<Scalar, 4>*>(handle->inner));
        default: std::abort();
    }
}

template <typename Scalar>
void* build(const Scalar* points, uint64_t n, uint32_t dim) {
    void* inner;
    switch (dim) {
        case 3: inner = new HandleT<Scalar, 3>(points, n); break;
        case 4: inner = new HandleT<Scalar, 4>(points, n); break;
        default: std::abort();
    }
    return new Handle<Scalar>{dim, inner};
}

template <typename Scalar>
void free_handle(void* h) {
    auto* handle = static_cast<Handle<Scalar>*>(h);
    switch (handle->dim) {
        case 3: delete static_cast<HandleT<Scalar, 3>*>(handle->inner); break;
        case 4: delete static_cast<HandleT<Scalar, 4>*>(handle->inner); break;
        default: std::abort();
    }
    delete handle;
}

// One query, composed exactly as cpdd's own single-query benchmark does.
template <typename Scalar, int Dim>
uint64_t single_query_impl(
    HandleT<Scalar, Dim>& h, const Scalar* q, uint64_t k, uint64_t* out_idx, Scalar* out_dist2) {
    Point<Scalar, Dim> query(const_cast<Scalar*>(q));
    std::vector<NNPair<Scalar, Dim>> storage(k, NNPair<Scalar, Dim>(std::ref(h.wp[0]), Scalar(0)));
    cpdd::kBoundedQueue<Point<Scalar, Dim>, NNPair<Scalar, Dim>> bq(
        parlay::make_slice(storage.data(), storage.data() + k));
    size_t visited = 0;
    h.tree.k_nearest(h.root, query, Dim, bq, h.root_box, visited);

    const uint64_t found = static_cast<uint64_t>(bq.m_count);
    for (uint64_t i = 0; i < found; ++i) {
        out_idx[i] = h.index_of(storage[i].first.get());
        out_dist2[i] = storage[i].second;
    }
    return found;
}

template <typename Scalar>
uint64_t single_query(Handle<Scalar>* h, const Scalar* q, uint64_t k, uint64_t* out_idx, Scalar* out_dist2) {
    return dispatch(h, [&](auto& handle) { return single_query_impl(handle, q, k, out_idx, out_dist2); });
}

// All queries at once, parallel across every core via parlay::parallel_for.
template <typename Scalar, int Dim>
void batch_query_impl(
    HandleT<Scalar, Dim>& h, const Scalar* queries_flat, uint64_t num_queries, uint64_t k, uint64_t* out_idx,
    Scalar* out_dist2) {
    parlay::parallel_for(0, num_queries, [&](size_t qi) {
        Point<Scalar, Dim> query(const_cast<Scalar*>(queries_flat + qi * Dim));
        std::vector<NNPair<Scalar, Dim>> storage(k, NNPair<Scalar, Dim>(std::ref(h.wp[0]), Scalar(0)));
        cpdd::kBoundedQueue<Point<Scalar, Dim>, NNPair<Scalar, Dim>> bq(
            parlay::make_slice(storage.data(), storage.data() + k));
        size_t visited = 0;
        h.tree.k_nearest(h.root, query, Dim, bq, h.root_box, visited);

        const uint64_t found = static_cast<uint64_t>(bq.m_count);
        for (uint64_t i = 0; i < found; ++i) {
            out_idx[qi * k + i] = h.index_of(storage[i].first.get());
            out_dist2[qi * k + i] = storage[i].second;
        }
        for (uint64_t i = found; i < k; ++i) {
            out_idx[qi * k + i] = 0;
            out_dist2[qi * k + i] = Scalar(0);
        }
    });
}

template <typename Scalar>
void batch_query(
    Handle<Scalar>* h, const Scalar* queries_flat, uint64_t num_queries, uint64_t k, uint64_t* out_idx,
    Scalar* out_dist2) {
    dispatch(h, [&](auto& handle) {
        batch_query_impl(handle, queries_flat, num_queries, k, out_idx, out_dist2);
        return 0;
    });
}

}  // namespace

extern "C" {

void* pkdtree_build_f32(const float* points, uint64_t n, uint32_t dim) { return build<float>(points, n, dim); }
void* pkdtree_build_f64(const double* points, uint64_t n, uint32_t dim) { return build<double>(points, n, dim); }

void pkdtree_free_f32(void* h) { free_handle<float>(h); }
void pkdtree_free_f64(void* h) { free_handle<double>(h); }

uint64_t pkdtree_single_query_f32(void* h, const float* q, uint64_t k, uint64_t* out_idx, float* out_dist2) {
    return single_query<float>(static_cast<Handle<float>*>(h), q, k, out_idx, out_dist2);
}
uint64_t pkdtree_single_query_f64(void* h, const double* q, uint64_t k, uint64_t* out_idx, double* out_dist2) {
    return single_query<double>(static_cast<Handle<double>*>(h), q, k, out_idx, out_dist2);
}

void pkdtree_batch_query_f32(
    void* h, const float* queries_flat, uint64_t num_queries, uint64_t k, uint64_t* out_idx, float* out_dist2) {
    batch_query<float>(static_cast<Handle<float>*>(h), queries_flat, num_queries, k, out_idx, out_dist2);
}
void pkdtree_batch_query_f64(
    void* h, const double* queries_flat, uint64_t num_queries, uint64_t k, uint64_t* out_idx, double* out_dist2) {
    batch_query<double>(static_cast<Handle<double>*>(h), queries_flat, num_queries, k, out_idx, out_dist2);
}

}  // extern "C"
