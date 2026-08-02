// Thin extern "C" shim over ALGLIB's free C++ edition kd-tree
// (alglib::kdtree in alglibmisc). ALGLIB's "real" type is hard-coded to
// double throughout, so this competitor is f64-only, like ANN was.
//
// ALGLIB's own distances are in "corresponding norm" units (norm type 2 =
// Euclidean, i.e. NOT squared), so this shim squares them before returning
// to match every other competitor's squared-distance convention. Point
// indices come back via kdtreebuildtagged's tags (0..n-1), since a plain
// untagged build has no way to recover which original point a result
// corresponds to.

#include <cmath>
#include <cstdint>
#include <vector>

#include "alglibmisc.h"

namespace {

struct Handle {
    alglib::kdtree kdt;
};

}  // namespace

extern "C" {

void* alglib_build_f64(const double* points, uint64_t n, uint32_t dim) {
    auto* h = new Handle();
    alglib::real_2d_array xy;
    xy.setcontent(static_cast<alglib::ae_int_t>(n), static_cast<alglib::ae_int_t>(dim), points);

    std::vector<alglib::ae_int_t> tag_buf(n);
    for (uint64_t i = 0; i < n; ++i) {
        tag_buf[i] = static_cast<alglib::ae_int_t>(i);
    }
    alglib::integer_1d_array tags;
    tags.setcontent(static_cast<alglib::ae_int_t>(n), tag_buf.data());

    alglib::kdtreebuildtagged(xy, tags, static_cast<alglib::ae_int_t>(n), static_cast<alglib::ae_int_t>(dim), 0, 2, h->kdt);
    return h;
}

void alglib_free_f64(void* handle) { delete static_cast<Handle*>(handle); }

void alglib_nearest_one_f64(void* handle, const double* q, uint64_t* out_idx, double* out_dist2) {
    auto* h = static_cast<Handle*>(handle);
    alglib::real_1d_array x;
    x.attach_to_ptr(3, const_cast<double*>(q));
    alglib::kdtreequeryknn(h->kdt, x, 1, false);

    alglib::real_1d_array r;
    alglib::integer_1d_array result_tags;
    alglib::kdtreequeryresultsdistances(h->kdt, r);
    alglib::kdtreequeryresultstags(h->kdt, result_tags);
    *out_idx = static_cast<uint64_t>(result_tags[0]);
    *out_dist2 = r[0] * r[0];
}

uint64_t alglib_nearest_n_f64(void* handle, const double* q, uint64_t k, uint64_t* out_idx, double* out_dist2) {
    auto* h = static_cast<Handle*>(handle);
    alglib::real_1d_array x;
    x.attach_to_ptr(3, const_cast<double*>(q));
    const alglib::ae_int_t found = alglib::kdtreequeryknn(h->kdt, x, static_cast<alglib::ae_int_t>(k), false);

    alglib::real_1d_array r;
    alglib::integer_1d_array result_tags;
    alglib::kdtreequeryresultsdistances(h->kdt, r);
    alglib::kdtreequeryresultstags(h->kdt, result_tags);
    for (alglib::ae_int_t i = 0; i < found; ++i) {
        out_idx[i] = static_cast<uint64_t>(result_tags[i]);
        out_dist2[i] = r[i] * r[i];
    }
    return static_cast<uint64_t>(found);
}

uint64_t alglib_within_radius_f64(
    void* handle, const double* q, double radius2, uint64_t* out_idx, double* out_dist2, uint64_t cap) {
    auto* h = static_cast<Handle*>(handle);
    alglib::real_1d_array x;
    x.attach_to_ptr(3, const_cast<double*>(q));
    const double radius = std::sqrt(radius2);
    const alglib::ae_int_t found = alglib::kdtreequeryrnn(h->kdt, x, radius, false);

    alglib::real_1d_array r;
    alglib::integer_1d_array result_tags;
    alglib::kdtreequeryresultsdistances(h->kdt, r);
    alglib::kdtreequeryresultstags(h->kdt, result_tags);
    const uint64_t copy_n = static_cast<uint64_t>(found) < cap ? static_cast<uint64_t>(found) : cap;
    for (uint64_t i = 0; i < copy_n; ++i) {
        out_idx[i] = static_cast<uint64_t>(result_tags[i]);
        out_dist2[i] = r[i] * r[i];
    }
    return static_cast<uint64_t>(found);
}

}  // extern "C"
