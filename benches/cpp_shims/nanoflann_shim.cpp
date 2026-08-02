// Thin extern "C" shim over nanoflann's header-only KDTreeSingleIndexAdaptor,
// so it can be driven from the Rust criterion bench via FFI.
//
// Query semantics match kiddo's own conventions: distances are squared
// Euclidean, radius arguments are squared radii, and within_radius results
// are written into a caller-provided buffer up to `cap` entries while the
// true (possibly larger) match count is always returned.

#include <cstdint>
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

template <typename Scalar>
struct Handle {
    using Cloud = FlatCloud<Scalar>;
    using Adaptor = nanoflann::L2_Simple_Adaptor<Scalar, Cloud, Scalar, uint64_t>;
    using Index = nanoflann::KDTreeSingleIndexAdaptor<Adaptor, Cloud, 3, uint64_t>;

    Cloud cloud;
    Index index;

    Handle(const Scalar* points, uint64_t n, uint32_t dim)
        : cloud{points, static_cast<std::size_t>(n), static_cast<std::size_t>(dim)},
          index(dim, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(32)) {
        index.buildIndex();
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

template <typename Scalar>
void nearest_one(void* h, const Scalar* q, uint64_t* out_idx, Scalar* out_dist2) {
    auto* handle = static_cast<Handle<Scalar>*>(h);
    handle->index.knnSearch(q, 1, out_idx, out_dist2);
}

template <typename Scalar>
uint64_t nearest_n(void* h, const Scalar* q, uint64_t k, uint64_t* out_idx, Scalar* out_dist2) {
    auto* handle = static_cast<Handle<Scalar>*>(h);
    return static_cast<uint64_t>(handle->index.knnSearch(q, static_cast<std::size_t>(k), out_idx, out_dist2));
}

template <typename Scalar>
uint64_t within_radius(
    void* h, const Scalar* q, Scalar radius2, uint64_t* out_idx, Scalar* out_dist2, uint64_t cap) {
    auto* handle = static_cast<Handle<Scalar>*>(h);
    std::vector<nanoflann::ResultItem<uint64_t, Scalar>> matches;
    const uint64_t total = static_cast<uint64_t>(handle->index.radiusSearch(q, radius2, matches));
    const uint64_t copy_n = total < cap ? total : cap;
    for (uint64_t i = 0; i < copy_n; ++i) {
        out_idx[i] = matches[i].first;
        out_dist2[i] = matches[i].second;
    }
    return total;
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
