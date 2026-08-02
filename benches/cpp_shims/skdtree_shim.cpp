// skd-tree (achmichalop/skd-tree_2027, MIT) shim.
//
// Needs three system packages that upstream's README lists and this build
// script cannot vendor: Boost, Armadillo and ensmallen. A dead
// `<tpie/tpie.h>` include is satisfied by an empty stub; see
// build_cpp_competitors.rs.
//
// Four properties of the upstream code shape this shim, and each one is a
// place where a naive integration would produce a misleading number.
//
// 1. `SKDTREE::knn_query` wraps the actual search in two
//    `std::chrono::steady_clock::now()` calls and accumulates a counter:
//
//        inline Points knn_query(Point& q, unsigned int k) {
//            auto start = std::chrono::steady_clock::now();
//            results = knnQuery<Dim>[treeType](q, k);
//            auto end = std::chrono::steady_clock::now();
//            knn_count++; knn_time += ...;
//        }
//
//    Two clock reads is roughly 40-50ns on this platform, which is more than
//    kiddo's entire marginal per-query cost. Benchmarking through that
//    wrapper would measure their instrumentation, not their index, so this
//    shim calls the dispatch table `knnQuery<Dim>[treeType]` directly. The
//    constructor still runs, because it is what populates the globals the
//    table reads.
//
// 2. The tree lives in globals -- `inline void *root;` and
//    `inline TreeSelection treeType;` in tree_core.hpp -- not in the object.
//    Only one skd-tree can exist process-wide, and building a second one
//    silently invalidates the first. The handle below is therefore a token
//    for "the global tree", and `skdtree_build_f64` refuses to build while
//    another handle is live rather than corrupting it.
//
// 3. Coordinates are quantised: queries compute
//    `(uint64_t)(query[i] * CONVERSION_FACTOR)` with
//    `CONVERSION_FACTOR = (ULONG_MAX - 1) >> 1`, i.e. ~2^63, and separators
//    are bit-prefix masks of that value. Negative coordinates cast to
//    uint64_t are undefined, and coordinates >= 2 overflow the scale. The
//    benchmark's uniform [0,1) points are fine, but this shim range-checks
//    rather than trusting the caller, because the failure would otherwise be
//    silent wrong answers rather than a crash.
//
// 4. `knn_query` returns `std::vector<point_t<Dim>>` -- the coordinates of
//    the neighbours, not indices or distances. Distances are therefore
//    computed here, which is work the other shims do not do. It is the only
//    way to validate the results at all, but it is a handicap and must be
//    reported as one.
//
// Dim is a compile-time template parameter, so 3D is instantiated
// explicitly; a 4D cell would need a second instantiation here.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "indices/nonlearned/skdtree/skdtree.hpp"
#include "utils/type.hpp"

namespace {

constexpr std::size_t kDim = 3;

// SKDTREE lives in bench::index; the query dispatch table and the tree
// globals it reads are at namespace scope in evaluation.hpp / tree_core.hpp.
using SkdTree = bench::index::SKDTREE<kDim>;
using Point = point_t<kDim>;
using Points = std::vector<Point>;

// Owns the point vector, because SKDTREE holds `Points& points` by reference
// and would dangle if the caller's vector went away.
struct Handle {
    Points points;
    std::vector<Point> results_scratch;
    // Constructed for its side effects on the globals; the queries below go
    // to the dispatch table directly. See note 1.
    SkdTree* index = nullptr;

    ~Handle() { delete index; }
};

// Note 2: the upstream globals permit exactly one live tree.
bool global_tree_in_use = false;

}  // namespace

extern "C" {

// Returns nullptr if a tree already exists (note 2) or if any coordinate is
// outside the quantisable range (note 3).
void* skdtree_build_f64(const double* points, uint64_t n, uint32_t dim) {
    if (dim != kDim || global_tree_in_use) {
        return nullptr;
    }

    auto* handle = new Handle();
    handle->points.resize(static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
        for (std::size_t d = 0; d < kDim; ++d) {
            const double coordinate = points[i * kDim + d];
            // Guard the quantisation rather than let it wrap silently.
            if (!(coordinate >= 0.0) || coordinate >= 2.0) {
                delete handle;
                return nullptr;
            }
            handle->points[i][d] = coordinate;
        }
    }

    handle->index = new SkdTree(handle->points);
    global_tree_in_use = true;
    return handle;
}

void skdtree_free_f64(void* raw_handle) {
    if (raw_handle == nullptr) {
        return;
    }
    delete static_cast<Handle*>(raw_handle);
    global_tree_in_use = false;
}

// Writes squared Euclidean distances for the k nearest neighbours, ascending,
// and returns how many were produced. Distances are computed here because
// upstream returns coordinates (note 4).
uint64_t skdtree_nearest_n_f64(void* raw_handle, const double* q, uint64_t k,
                               double* out_dist2) {
    auto* handle = static_cast<Handle*>(raw_handle);

    Point query;
    for (std::size_t d = 0; d < kDim; ++d) {
        query[d] = q[d];
    }

    // Note 1: bypass SKDTREE::knn_query's clock instrumentation.
    Points results = knnQuery<kDim>[treeType](query, static_cast<uint32_t>(k));

    const std::size_t produced = std::min<std::size_t>(results.size(), static_cast<std::size_t>(k));
    for (std::size_t i = 0; i < produced; ++i) {
        double sum = 0.0;
        for (std::size_t d = 0; d < kDim; ++d) {
            const double delta = results[i][d] - query[d];
            sum += delta * delta;
        }
        out_dist2[i] = sum;
    }
    // Best-first search returns in order, but the contract the other shims
    // present is "ascending", so do not rely on it.
    std::sort(out_dist2, out_dist2 + produced);
    return produced;
}

void skdtree_nearest_one_f64(void* raw_handle, const double* q, double* out_dist2) {
    double best = std::numeric_limits<double>::infinity();
    if (skdtree_nearest_n_f64(raw_handle, q, 1, &best) == 0) {
        best = std::numeric_limits<double>::infinity();
    }
    *out_dist2 = best;
}

}  // extern "C"
