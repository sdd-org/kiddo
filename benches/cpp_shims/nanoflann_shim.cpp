// Thin extern "C" shim over nanoflann's header-only KDTreeSingleIndexAdaptor,
// so it can be driven from the Rust criterion bench via FFI.
//
// Query semantics match kiddo's own conventions: distances are squared
// Euclidean, radius arguments are squared radii, and within_radius results
// are written into a caller-provided buffer up to `cap` entries while the
// true (possibly larger) match count is always returned.
//
// Dimension is a compile-time template parameter on nanoflann's Index (its
// whole performance case rests on the compiler unrolling per-dimension
// arithmetic), so two dimensions are compiled in rather than one: 3, for the
// f64 comparisons, and 4, for f32 -- the two block heights the cyclic
// Donnelly strategies are native to. `build` picks the matching
// specialization at runtime from the `dim` argument and tags the returned
// handle with it; every other entry point switches on that tag once per call
// to recover the concrete type. Any dimension other than 3 or 4 aborts,
// since nothing else is compiled in.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <nanoflann.hpp>

namespace {

template <typename Scalar>
struct FlatCloud {
    const Scalar* data;
    std::size_t n;
    std::size_t dim;

    inline std::size_t kdtree_get_point_count() const { return n; }
    inline Scalar kdtree_get_pt(std::size_t idx, std::size_t d) const {
        return data[idx * dim + d];
    }
    template <class BBox>
    bool kdtree_get_bbox(BBox&) const { return false; }
};

template <typename Scalar, int Dim>
struct HandleT {
    using Cloud = FlatCloud<Scalar>;
    using Adaptor = nanoflann::L2_Simple_Adaptor<Scalar, Cloud, Scalar, uint64_t>;
    using Index = nanoflann::KDTreeSingleIndexAdaptor<Adaptor, Cloud, Dim, uint64_t>;

    Cloud cloud;
    Index index;

    HandleT(const Scalar* points, uint64_t n, uint32_t dim)
        : cloud{points, static_cast<std::size_t>(n), static_cast<std::size_t>(dim)},
          index(dim, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(32)) {
        index.buildIndex();
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

// Recovers the concrete HandleT and calls `f` on it, so each query function
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
        case 3: inner = new HandleT<Scalar, 3>(points, n, dim); break;
        case 4: inner = new HandleT<Scalar, 4>(points, n, dim); break;
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

template <typename Scalar>
void nearest_one(void* h, const Scalar* q, uint64_t* out_idx, Scalar* out_dist2) {
    dispatch(static_cast<Handle<Scalar>*>(h), [&](auto& handle) {
        handle.index.knnSearch(q, 1, out_idx, out_dist2);
    });
}

template <typename Scalar>
uint64_t nearest_n(void* h, const Scalar* q, uint64_t k, uint64_t* out_idx, Scalar* out_dist2) {
    return dispatch(static_cast<Handle<Scalar>*>(h), [&](auto& handle) {
        return static_cast<uint64_t>(handle.index.knnSearch(q, static_cast<std::size_t>(k), out_idx, out_dist2));
    });
}

template <typename Scalar>
uint64_t within_radius(
    void* h, const Scalar* q, Scalar radius2, uint64_t* out_idx, Scalar* out_dist2, uint64_t cap) {
    return dispatch(static_cast<Handle<Scalar>*>(h), [&](auto& handle) {
        std::vector<nanoflann::ResultItem<uint64_t, Scalar>> matches;
        const uint64_t total = static_cast<uint64_t>(handle.index.radiusSearch(q, radius2, matches));
        const uint64_t copy_n = total < cap ? total : cap;
        for (uint64_t i = 0; i < copy_n; ++i) {
            out_idx[i] = matches[i].first;
            out_dist2[i] = matches[i].second;
        }
        return total;
    });
}

}  // namespace

extern "C" {

void* nanoflann_build_f32(const float* points, uint64_t n, uint32_t dim) { return build<float>(points, n, dim); }
void* nanoflann_build_f64(const double* points, uint64_t n, uint32_t dim) { return build<double>(points, n, dim); }

void nanoflann_free_f32(void* h) { free_handle<float>(h); }
void nanoflann_free_f64(void* h) { free_handle<double>(h); }

void nanoflann_nearest_one_f32(void* h, const float* q, uint64_t* out_idx, float* out_dist2) {
    nearest_one<float>(h, q, out_idx, out_dist2);
}
void nanoflann_nearest_one_f64(void* h, const double* q, uint64_t* out_idx, double* out_dist2) {
    nearest_one<double>(h, q, out_idx, out_dist2);
}

uint64_t nanoflann_nearest_n_f32(void* h, const float* q, uint64_t k, uint64_t* out_idx, float* out_dist2) {
    return nearest_n<float>(h, q, k, out_idx, out_dist2);
}
uint64_t nanoflann_nearest_n_f64(void* h, const double* q, uint64_t k, uint64_t* out_idx, double* out_dist2) {
    return nearest_n<double>(h, q, k, out_idx, out_dist2);
}

uint64_t nanoflann_within_radius_f32(
    void* h, const float* q, float radius2, uint64_t* out_idx, float* out_dist2, uint64_t cap) {
    return within_radius<float>(h, q, radius2, out_idx, out_dist2, cap);
}
uint64_t nanoflann_within_radius_f64(
    void* h, const double* q, double radius2, uint64_t* out_idx, double* out_dist2, uint64_t cap) {
    return within_radius<double>(h, q, radius2, out_idx, out_dist2, cap);
}

}  // extern "C"
