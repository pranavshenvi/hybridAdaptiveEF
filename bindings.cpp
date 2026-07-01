// bindings.cpp — pybind11 bridge for AdaptiveHNSW
//
// Build with:  python setup.py build_ext --inplace
//
// Exposes:
//   adaptive_hnsw_cpp.AdaptiveHNSW  (class)
//     .add_items(data_2d)
//     .search_knn(queries, k, ef)           → (labels, distances)
//     .search_knn_adaptive(query_1d, k, entry_node, start_layer, ef)
//     .profile_query(query_1d)              → list[(layer, node, dist)]
//     .entry_point, .max_level, .num_elements
//     .reset_dist_count(), .get_dist_count()

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "cpp/adaptive_hnsw.h"

namespace py = pybind11;
using Index = ahnsw::AdaptiveHNSW;

PYBIND11_MODULE(adaptive_hnsw_cpp, m) {
    m.doc() = "Adaptive HNSW — HNSW with per-layer entry-point control";

    py::class_<Index>(m, "AdaptiveHNSW")

        // ── constructor ─────────────────────────────────────────────────
        .def(py::init<int, int, int, int>(),
             py::arg("dim"),
             py::arg("max_elements"),
             py::arg("M") = 16,
             py::arg("ef_construction") = 200)

        // ── add_items ───────────────────────────────────────────────────
        .def("add_items",
            [](Index& self, py::array_t<float, py::array::c_style> data) {
                auto buf = data.request();
                if (buf.ndim != 2)
                    throw std::runtime_error("add_items expects a 2-D array");
                int n = static_cast<int>(buf.shape[0]);
                int d = static_cast<int>(buf.shape[1]);
                if (d != self.dim())
                    throw std::runtime_error("Dimension mismatch");
                self.addItems(static_cast<const float*>(buf.ptr), n);
            },
            py::arg("data"),
            "Insert vectors from a (N × dim) float32 array.")

        // ── Serialization ───────────────────────────────────────────────
        .def("save_index", &Index::saveIndex, py::arg("path"),
             "Save the custom AdaptiveHNSW index to a binary file.")
        .def("load_index", &Index::loadIndex, py::arg("path"),
             "Load the custom AdaptiveHNSW index from a binary file.")

        // ── search_knn (standard, batch) ────────────────────────────────
        .def("search_knn",
            [](Index& self,
               py::array_t<float, py::array::c_style> queries,
               int k, int ef) {

                auto buf = queries.request();
                int nq, d;
                if (buf.ndim == 1) { nq = 1; d = static_cast<int>(buf.shape[0]); }
                else if (buf.ndim == 2) {
                    nq = static_cast<int>(buf.shape[0]);
                    d  = static_cast<int>(buf.shape[1]);
                } else throw std::runtime_error("Expected 1-D or 2-D array");
                if (d != self.dim()) throw std::runtime_error("Dim mismatch");

                const float* ptr = static_cast<const float*>(buf.ptr);

                py::array_t<int>   labels({nq, k});
                py::array_t<float> dists ({nq, k});
                auto lb = labels.mutable_unchecked<2>();
                auto db = dists .mutable_unchecked<2>();

                for (int i = 0; i < nq; i++) {
                    auto res = self.searchKnn(
                            ptr + static_cast<size_t>(i) * d, k, ef);
                    for (int j = 0; j < k; j++) {
                        if (j < static_cast<int>(res.size())) {
                            lb(i, j) = res[j].second;
                            db(i, j) = res[j].first;
                        } else {
                            lb(i, j) = -1;
                            db(i, j) = 1e30f;
                        }
                    }
                }
                return py::make_tuple(labels, dists);
            },
            py::arg("queries"), py::arg("k") = 10, py::arg("ef") = 50,
            "Standard KNN search.  Returns (labels, distances) numpy arrays.")

        // ── search_knn_adaptive (single query) ──────────────────────────
        .def("search_knn_adaptive",
            [](Index& self,
               py::array_t<float, py::array::c_style> query,
               int k, int entry_node, int start_layer, int ef) {

                auto buf = query.request();
                if (buf.ndim != 1)
                    throw std::runtime_error("Expected 1-D query vector");
                if (static_cast<int>(buf.shape[0]) != self.dim())
                    throw std::runtime_error("Dim mismatch");

                auto res = self.searchKnnFrom(
                    static_cast<const float*>(buf.ptr),
                    k, entry_node, start_layer, ef);

                py::array_t<int>   labels(k);
                py::array_t<float> dists(k);
                auto lb = labels.mutable_unchecked<1>();
                auto db = dists .mutable_unchecked<1>();
                for (int j = 0; j < k; j++) {
                    if (j < static_cast<int>(res.size())) {
                        lb(j) = res[j].second;
                        db(j) = res[j].first;
                    } else {
                        lb(j) = -1;
                        db(j) = 1e30f;
                    }
                }
                return py::make_tuple(labels, dists);
            },
            py::arg("query"),
            py::arg("k")           = 10,
            py::arg("entry_node")  = 0,
            py::arg("start_layer") = 0,
            py::arg("ef")          = 50,
            "Adaptive KNN: start from the given node/layer with custom ef.")

        // ── search_knn_dynamic (single query) ──────────────────────────
        .def("search_knn_dynamic",
            [](Index& self,
               py::array_t<float, py::array::c_style> query,
               int k,
               std::vector<float> bins,
               std::vector<float> poly_coeffs,
               int min_ef, int max_ef, int probe_count) {

                auto buf = query.request();
                if (buf.ndim != 1)
                    throw std::runtime_error("Expected 1-D query vector");
                if (static_cast<int>(buf.shape[0]) != self.dim())
                    throw std::runtime_error("Dim mismatch");

                auto res = self.searchKnnDynamic(
                    static_cast<const float*>(buf.ptr),
                    k, bins, poly_coeffs, min_ef, max_ef, probe_count);

                py::array_t<int>   labels(k);
                py::array_t<float> dists(k);
                auto lb = labels.mutable_unchecked<1>();
                auto db = dists .mutable_unchecked<1>();
                for (int j = 0; j < k; j++) {
                    if (j < static_cast<int>(res.size())) {
                        lb(j) = res[j].second;
                        db(j) = res[j].first;
                    } else {
                        lb(j) = -1;
                        db(j) = 1e30f;
                    }
                }
                return py::make_tuple(labels, dists);
            },
            py::arg("query"),
            py::arg("k"),
            py::arg("bins"),
            py::arg("poly_coeffs"),
            py::arg("min_ef"),
            py::arg("max_ef"),
            py::arg("probe_count"),
            "Dynamic KNN search with in-flight EF prediction.")

        // ── profile_query ───────────────────────────────────────────────
        .def("profile_query",
            [](Index& self,
               py::array_t<float, py::array::c_style> query) {
                auto buf = query.request();
                if (buf.ndim != 1)
                    throw std::runtime_error("Expected 1-D query vector");
                if (static_cast<int>(buf.shape[0]) != self.dim())
                    throw std::runtime_error("Dim mismatch");
                return self.profileQuery(
                    static_cast<const float*>(buf.ptr));
            },
            py::arg("query"),
            "Trace the greedy path from top → layer 0.  "
            "Returns list of (layer, node_id, distance).")

        // ── readonly properties ─────────────────────────────────────────
        .def_property_readonly("entry_point",  &Index::entryPoint)
        .def_property_readonly("max_level",    &Index::maxLevel)
        .def_property_readonly("num_elements", &Index::size)

        // ── distance counter ────────────────────────────────────────────
        .def("reset_dist_count", &Index::resetDistCount)
        .def("get_dist_count",   &Index::getDistCount);
}
