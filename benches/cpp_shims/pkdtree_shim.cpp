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

#include <cstdint>
#include <cstring>
#include <functional>
#include <vector>

#include "cpdd/cpdd.h"

namespace {

template <typename Scalar>
using Point = cpdd::PointType<Scalar, 3>;

template <typename Scalar>
using Tree = cpdd::ParallelKDtree<Point<Scalar>>;

template <typename Scalar>
using NNPair = std::pair<std::reference_wrapper<Point<Scalar>>, Scalar>;

template <typename Scalar>
struct Handle {
    Tree<Scalar> tree;
    typename Tree<Scalar>::points wp;
    typename Tree<Scalar>::node* root = nullptr;
    typename Tree<Scalar>::box root_box;

    explicit Handle(const Scalar* points, uint64_t n, uint32_t dim)
        : wp(Tree<Scalar>::points::uninitialized(n)) {
        for (uint64_t i = 0; i < n; ++i) {
            wp[i] = Point<Scalar>(const_cast<Scalar*>(points + i * dim));
        }
        tree.build(parlay::make_slice(wp), static_cast<uint_fast8_t>(dim));
        root = tree.get_root();
        root_box = tree.get_root_box();
    }

    uint64_t index_of(const Point<Scalar>& p) const {
        return static_cast<uint64_t>(&p - &wp[0]);
    }
};

template <typename Scalar>
void* build(const Scalar* points, uint64_t n, uint32_t dim) {
    return new Handle<Scalar>(points, n, dim);
}

template <typename Scalar>
void free_handle(void* h) {
    delete static_cast<Handle<Scalar>*>(h);
}

// One query, composed exactly as cpdd's own single-query benchmark does.
template <typename Scalar>
uint64_t single_query(Handle<Scalar>* h, const Scalar* q, uint64_t k, uint64_t* out_idx, Scalar* out_dist2) {
    Point<Scalar> query(const_cast<Scalar*>(q));
    std::vector<NNPair<Scalar>> storage(k, NNPair<Scalar>(std::ref(h->wp[0]), Scalar(0)));
    cpdd::kBoundedQueue<Point<Scalar>, NNPair<Scalar>> bq(parlay::make_slice(storage.data(), storage.data() + k));
    size_t visited = 0;
    h->tree.k_nearest(h->root, query, 3, bq, h->root_box, visited);

    const uint64_t found = static_cast<uint64_t>(bq.m_count);
    for (uint64_t i = 0; i < found; ++i) {
        out_idx[i] = h->index_of(storage[i].first.get());
        out_dist2[i] = storage[i].second;
    }
    return found;
}

// All queries at once, parallel across every core via parlay::parallel_for.
template <typename Scalar>
void batch_query(
    Handle<Scalar>* h, const Scalar* queries_flat, uint64_t num_queries, uint64_t k, uint64_t* out_idx,
    Scalar* out_dist2) {
    parlay::parallel_for(0, num_queries, [&](size_t qi) {
        Point<Scalar> query(const_cast<Scalar*>(queries_flat + qi * 3));
        std::vector<NNPair<Scalar>> storage(k, NNPair<Scalar>(std::ref(h->wp[0]), Scalar(0)));
        cpdd::kBoundedQueue<Point<Scalar>, NNPair<Scalar>> bq(parlay::make_slice(storage.data(), storage.data() + k));
        size_t visited = 0;
        h->tree.k_nearest(h->root, query, 3, bq, h->root_box, visited);

        const uint64_t found = static_cast<uint64_t>(bq.m_count);
        for (uint64_t i = 0; i < found; ++i) {
            out_idx[qi * k + i] = h->index_of(storage[i].first.get());
            out_dist2[qi * k + i] = storage[i].second;
        }
        for (uint64_t i = found; i < k; ++i) {
            out_idx[qi * k + i] = 0;
            out_dist2[qi * k + i] = Scalar(0);
        }
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
