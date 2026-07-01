#pragma once
//
// adaptive_hnsw.h — HNSW with per-layer entry point control
//
// Key addition over standard HNSW:
//   searchKnnFrom(query, k, entry_node, start_layer, ef)
//     Start the search from ANY node at ANY layer with ANY ef.
//     This lets centroid-based routing skip layers and start closer
//     to the query's true neighborhood.
//
//   profileQuery(query)
//     Trace a query's greedy path from the top layer down.
//     Returns (layer, nearest_node, distance) at each layer.
//     Used offline to cache per-centroid entry points.
//

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <queue>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>
#include <fstream>
#include <string>

namespace ahnsw {

class AdaptiveHNSW {
public:
    using Result = std::pair<float, int>;  // (distance, node_id)

    // ─────────────────────────── Construction ───────────────────────────

    AdaptiveHNSW(int dim, int max_elements, int M = 16, int ef_construction = 200)
        : dim_(dim), max_elements_(max_elements),
          M_(M), M_max_(M), M_max0_(2 * M),
          ef_construction_(ef_construction),
          entry_point_(-1), max_level_(-1), num_elements_(0),
          mult_(1.0 / std::log(static_cast<double>(M))),
          rng_(42), dist_computations_(0)
    {
        data_.resize(static_cast<size_t>(max_elements) * dim, 0.0f);
        levels_.resize(max_elements, -1);
        adj_.resize(max_elements);
        visited_.resize(max_elements, 0);
        stamp_ = 1;
    }

    // ─────────────────────────── Properties ─────────────────────────────

    int dim()        const { return dim_; }
    int size()       const { return num_elements_; }
    int entryPoint() const { return entry_point_; }
    int maxLevel()   const { return max_level_; }
    int nodeLevel(int id) const { return levels_[id]; }

    const float* getDataPtr(int id) const {
        return &data_[static_cast<size_t>(id) * dim_];
    }

    // Distance computation counter (for benchmarking)
    mutable size_t dist_computations_;
    void   resetDistCount() { dist_computations_ = 0; }
    size_t getDistCount()   const { return dist_computations_; }

    // ─────────────────────────── Insertion ───────────────────────────────

    void addItem(const float* vec) {
        if (num_elements_ >= max_elements_)
            throw std::runtime_error("Index is full");

        int id = num_elements_++;
        std::memcpy(&data_[static_cast<size_t>(id) * dim_], vec,
                     dim_ * sizeof(float));

        int level = randomLevel();
        levels_[id] = level;
        adj_[id].resize(level + 1);

        // First node — becomes the global entry point
        if (entry_point_ == -1) {
            entry_point_ = id;
            max_level_   = level;
            return;
        }

        int ep = entry_point_;

        // Phase 1: greedy descent through layers above the new node's level
        for (int lc = max_level_; lc > level; lc--)
            ep = greedyClosest(vec, ep, lc);

        // Phase 2: insert at each layer [min(max_level_, level) … 0]
        for (int lc = std::min(max_level_, level); lc >= 0; lc--) {
            int max_conn = (lc == 0) ? M_max0_ : M_max_;

            auto candidates = searchLayer(vec, ep, ef_construction_, lc);
            auto neighbors  = selectNeighbors(candidates, max_conn);

            adj_[id][lc] = neighbors;

            for (int n : neighbors) {
                adj_[n][lc].push_back(id);
                if (static_cast<int>(adj_[n][lc].size()) > max_conn)
                    shrinkNeighbors(n, lc, max_conn);
            }

            if (!candidates.empty())
                ep = candidates.front().second;   // nearest found
        }

        if (level > max_level_) {
            entry_point_ = id;
            max_level_   = level;
        }
    }

    void addItems(const float* data, int n) {
        for (int i = 0; i < n; i++) {
            addItem(data + static_cast<size_t>(i) * dim_);
            if ((i + 1) % 50000 == 0)
                std::cout << "  [build] " << (i + 1) << " / " << n << std::endl;
        }
        if (n % 50000 != 0)
            std::cout << "  [build] " << n << " / " << n << " (done)" << std::endl;
    }

    // ─────────────────────────── Serialization ─────────────────────────────

    void saveIndex(const std::string& path) const {
        std::ofstream out(path, std::ios::binary);
        if (!out) throw std::runtime_error("Cannot open file for writing: " + path);
        
        out.write((char*)&dim_, sizeof(int));
        out.write((char*)&max_elements_, sizeof(int));
        out.write((char*)&M_, sizeof(int));
        out.write((char*)&M_max_, sizeof(int));
        out.write((char*)&M_max0_, sizeof(int));
        out.write((char*)&ef_construction_, sizeof(int));
        out.write((char*)&entry_point_, sizeof(int));
        out.write((char*)&max_level_, sizeof(int));
        out.write((char*)&num_elements_, sizeof(int));
        out.write((char*)&mult_, sizeof(double));
        
        size_t dsize = data_.size();
        out.write((char*)&dsize, sizeof(size_t));
        out.write((char*)data_.data(), dsize * sizeof(float));
        
        size_t lsize = levels_.size();
        out.write((char*)&lsize, sizeof(size_t));
        out.write((char*)levels_.data(), lsize * sizeof(int));
        
        size_t adj_size = adj_.size();
        out.write((char*)&adj_size, sizeof(size_t));
        for (const auto& node_layers : adj_) {
            size_t layer_count = node_layers.size();
            out.write((char*)&layer_count, sizeof(size_t));
            for (const auto& layer_nbrs : node_layers) {
                size_t nbr_count = layer_nbrs.size();
                out.write((char*)&nbr_count, sizeof(size_t));
                out.write((char*)layer_nbrs.data(), nbr_count * sizeof(int));
            }
        }
    }
    
    void loadIndex(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("Cannot open file for reading: " + path);
        
        in.read((char*)&dim_, sizeof(int));
        in.read((char*)&max_elements_, sizeof(int));
        in.read((char*)&M_, sizeof(int));
        in.read((char*)&M_max_, sizeof(int));
        in.read((char*)&M_max0_, sizeof(int));
        in.read((char*)&ef_construction_, sizeof(int));
        in.read((char*)&entry_point_, sizeof(int));
        in.read((char*)&max_level_, sizeof(int));
        in.read((char*)&num_elements_, sizeof(int));
        in.read((char*)&mult_, sizeof(double));
        
        size_t dsize = 0;
        in.read((char*)&dsize, sizeof(size_t));
        data_.resize(dsize);
        in.read((char*)data_.data(), dsize * sizeof(float));
        
        size_t lsize = 0;
        in.read((char*)&lsize, sizeof(size_t));
        levels_.resize(lsize);
        in.read((char*)levels_.data(), lsize * sizeof(int));
        
        size_t adj_size = 0;
        in.read((char*)&adj_size, sizeof(size_t));
        adj_.resize(adj_size);
        for (size_t i = 0; i < adj_size; i++) {
            size_t layer_count = 0;
            in.read((char*)&layer_count, sizeof(size_t));
            adj_[i].resize(layer_count);
            for (size_t j = 0; j < layer_count; j++) {
                size_t nbr_count = 0;
                in.read((char*)&nbr_count, sizeof(size_t));
                adj_[i][j].resize(nbr_count);
                in.read((char*)adj_[i][j].data(), nbr_count * sizeof(int));
            }
        }
        
        visited_.assign(max_elements_, 0);
        stamp_ = 1;
    }

    // ─────────────────────────── Search ──────────────────────────────────

    /// Standard KNN search (enters at global entry point, top layer).
    std::vector<Result> searchKnn(const float* query, int k, int ef) {
        if (entry_point_ == -1) return {};
        return searchKnnFrom(query, k, entry_point_, max_level_, ef);
    }

    /// Adaptive KNN: start search from `start_node` at `start_layer` with
    /// the given `ef`.  This is the core method that enables centroid routing.
    ///
    /// When the query is highly similar to a profiled centroid, call this
    /// with the centroid's cached layer-0 entry node and a low ef to skip
    /// the entire top-down traversal and start the beam search right in the
    /// query's neighborhood.
    std::vector<Result> searchKnnFrom(const float* query, int k,
                                       int start_node, int start_layer,
                                       int ef) {
        if (entry_point_ == -1 || start_node < 0 ||
            start_node >= num_elements_)
            return {};

        // Clamp to the node's actual level
        start_layer = std::max(0, std::min(start_layer, levels_[start_node]));

        int ep = start_node;

        // Greedy descent from start_layer down to layer 1
        for (int lc = start_layer; lc >= 1; lc--)
            ep = greedyClosest(query, ep, lc);

        // Beam search at layer 0
        auto candidates = searchLayer(query, ep, std::max(ef, k), 0);

        if (static_cast<int>(candidates.size()) > k)
            candidates.resize(k);
        return candidates;
    }

    /// Dynamic KNN search with runtime quantile binning and EF prediction.
    std::vector<Result> searchKnnDynamic(const float* query, int k,
                                         const std::vector<float>& bins,
                                         const std::vector<float>& poly_coeffs,
                                         int min_ef, int max_ef, int probe_count) {
        if (entry_point_ == -1) return {};
        
        int ep = entry_point_;
        
        // Greedy descent down to layer 1
        for (int lc = max_level_; lc >= 1; lc--)
            ep = greedyClosest(query, ep, lc);
            
        // Dynamic Beam search at layer 0
        auto candidates = searchLayerDynamicEf(query, ep, bins, poly_coeffs, min_ef, max_ef, std::max(probe_count, k), 0);
        
        if (static_cast<int>(candidates.size()) > k)
            candidates.resize(k);
        return candidates;
    }

    /// Profile a query through all layers.
    /// Returns [(layer, node_id, distance), …] from top layer → layer 0.
    /// Used offline to cache per-centroid entry points at each layer.
    std::vector<std::tuple<int, int, float>> profileQuery(const float* query) {
        std::vector<std::tuple<int, int, float>> path;
        if (entry_point_ == -1) return path;

        int ep = entry_point_;
        for (int lc = max_level_; lc >= 0; lc--) {
            ep = greedyClosest(query, ep, lc);
            float d = computeDist(query, getDataPtr(ep));
            path.emplace_back(lc, ep, d);
        }
        return path;
    }

    // ─────────────────────────── Internals ───────────────────────────────
private:
    int    dim_, max_elements_;
    int    M_, M_max_, M_max0_;
    int    ef_construction_;
    int    entry_point_, max_level_, num_elements_;
    double mult_;

    std::mt19937 rng_;

    std::vector<float> data_;                           // flat [id * dim]
    std::vector<int>   levels_;                         // node → max level
    std::vector<std::vector<std::vector<int>>> adj_;    // [node][layer] → neighbors

    // Fast visited tracking (stamp-based, avoids clearing per search)
    mutable std::vector<int> visited_;
    mutable int stamp_;

    void newStamp()              const { stamp_++; }
    bool isVisited(int id)       const { return visited_[id] == stamp_; }
    void markVisited(int id)     const { visited_[id] = stamp_; }

    int randomLevel() {
        std::uniform_real_distribution<double> d(0.0, 1.0);
        return static_cast<int>(-std::log(d(rng_)) * mult_);
    }

    /// Cosine distance for L2-normalised vectors: 1 − dot(a, b).
    float computeDist(const float* a, const float* b) const {
        dist_computations_++;
        float ip = 0.0f;
        for (int i = 0; i < dim_; i++)
            ip += a[i] * b[i];
        return 1.0f - ip;
    }

    /// Greedy walk at a single layer.  Returns the nearest node reached.
    int greedyClosest(const float* query, int ep, int layer) const {
        float best = computeDist(query, getDataPtr(ep));
        bool changed = true;
        while (changed) {
            changed = false;
            if (layer >= static_cast<int>(adj_[ep].size())) break;
            int best_n = ep;
            for (int n : adj_[ep][layer]) {
                float d = computeDist(query, getDataPtr(n));
                if (d < best) { best = d; best_n = n; changed = true; }
            }
            ep = best_n;
        }
        return ep;
    }

    /// Beam search at one layer (Algorithm 2 of the HNSW paper).
    /// Returns results sorted ascending by distance (nearest first).
    std::vector<Result> searchLayer(const float* query, int ep, int ef,
                                     int layer) const {
        newStamp();
        float d_ep = computeDist(query, getDataPtr(ep));
        markVisited(ep);

        // candidates: min-heap (extract nearest first)
        auto cmp_min = [](const Result& a, const Result& b) {
            return a.first > b.first;
        };
        std::priority_queue<Result, std::vector<Result>, decltype(cmp_min)>
            candidates(cmp_min);

        // W: max-heap (extract farthest for eviction)
        auto cmp_max = [](const Result& a, const Result& b) {
            return a.first < b.first;
        };
        std::priority_queue<Result, std::vector<Result>, decltype(cmp_max)>
            W(cmp_max);

        candidates.push({d_ep, ep});
        W.push({d_ep, ep});

        while (!candidates.empty()) {
            auto [c_dist, c_id] = candidates.top();
            candidates.pop();

            if (c_dist > W.top().first)
                break;                    // all remaining candidates are worse

            if (layer >= static_cast<int>(adj_[c_id].size())) continue;

            for (int n : adj_[c_id][layer]) {
                if (isVisited(n)) continue;
                markVisited(n);

                float nd = computeDist(query, getDataPtr(n));

                if (nd < W.top().first ||
                    static_cast<int>(W.size()) < ef) {
                    candidates.push({nd, n});
                    W.push({nd, n});
                    if (static_cast<int>(W.size()) > ef)
                        W.pop();
                }
            }
        }

        // Extract and sort
        std::vector<Result> results;
        results.reserve(W.size());
        while (!W.empty()) { results.push_back(W.top()); W.pop(); }
        std::sort(results.begin(), results.end());
        return results;
    }

    /// Beam search with dynamic EF evaluation
    std::vector<Result> searchLayerDynamicEf(const float* query, int ep, 
                                             const std::vector<float>& bins,
                                             const std::vector<float>& poly_coeffs,
                                             int min_ef, int max_ef, int probe_count,
                                             int layer) const {
        newStamp();
        float d_ep = computeDist(query, getDataPtr(ep));
        markVisited(ep);

        auto cmp_min = [](const Result& a, const Result& b) { return a.first > b.first; };
        std::priority_queue<Result, std::vector<Result>, decltype(cmp_min)> candidates(cmp_min);

        auto cmp_max = [](const Result& a, const Result& b) { return a.first < b.first; };
        std::priority_queue<Result, std::vector<Result>, decltype(cmp_max)> W(cmp_max);

        candidates.push({d_ep, ep});
        W.push({d_ep, ep});

        std::vector<float> probe_dists;
        probe_dists.reserve(probe_count);
        probe_dists.push_back(d_ep);
        
        int current_ef = probe_count; // Start with probe size EF
        bool ef_updated = false;

        while (!candidates.empty()) {
            auto [c_dist, c_id] = candidates.top();
            candidates.pop();

            if (c_dist > W.top().first)
                break;

            if (layer >= static_cast<int>(adj_[c_id].size())) continue;

            for (int n : adj_[c_id][layer]) {
                if (isVisited(n)) continue;
                markVisited(n);

                float nd = computeDist(query, getDataPtr(n));
                
                // Collect distance for scoring
                if (!ef_updated) {
                    probe_dists.push_back(nd);
                    if (static_cast<int>(probe_dists.size()) >= probe_count) {
                        float score = 0;
                        for (float d : probe_dists) {
                            for (float b : bins) {
                                if (d > b) score += 1.0f;
                            }
                        }
                        
                        float pred_ef = 0;
                        float cur_x = 1.0f;
                        for (int i = static_cast<int>(poly_coeffs.size()) - 1; i >= 0; i--) {
                            pred_ef += poly_coeffs[i] * cur_x;
                            cur_x *= score;
                        }
                        
                        current_ef = std::max(min_ef, std::min(max_ef, static_cast<int>(pred_ef)));
                        ef_updated = true;
                    }
                }

                if (nd < W.top().first || static_cast<int>(W.size()) < current_ef) {
                    candidates.push({nd, n});
                    W.push({nd, n});
                    if (static_cast<int>(W.size()) > current_ef)
                        W.pop();
                }
            }
        }

        std::vector<Result> results;
        results.reserve(W.size());
        while (!W.empty()) { results.push_back(W.top()); W.pop(); }
        std::sort(results.begin(), results.end());
        return results;
    }

    /// Keep the M nearest neighbours.
    static std::vector<int> selectNeighbors(const std::vector<Result>& cands,
                                             int M) {
        int n = std::min(M, static_cast<int>(cands.size()));
        std::vector<int> out(n);
        for (int i = 0; i < n; i++) out[i] = cands[i].second;
        return out;
    }

    /// Prune a node's adjacency list at one layer to `max_conn` neighbors.
    void shrinkNeighbors(int id, int layer, int max_conn) {
        auto& nbrs = adj_[id][layer];
        if (static_cast<int>(nbrs.size()) <= max_conn) return;

        const float* id_data = getDataPtr(id);
        std::vector<Result> scored;
        scored.reserve(nbrs.size());
        for (int n : nbrs)
            scored.push_back({computeDist(id_data, getDataPtr(n)), n});
        std::sort(scored.begin(), scored.end());

        nbrs.clear();
        nbrs.reserve(max_conn);
        for (int i = 0; i < max_conn; i++)
            nbrs.push_back(scored[i].second);
    }
};

}  // namespace ahnsw
