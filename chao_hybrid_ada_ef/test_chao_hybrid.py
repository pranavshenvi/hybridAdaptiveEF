import chao_hybrid_ada_ef_cpp
import numpy as np

print("Start test")

idx = chao_hybrid_ada_ef_cpp.Index(space='l2', dim=128)
idx.init_index(max_elements=1000, ef_construction=200, M=16)

data = np.random.rand(100, 128).astype(np.float32)
idx.add_items(data)
print("Added items")

query = np.random.rand(128).astype(np.float32)

print("Calling search_knn_adaptive...", flush=True)
print("search_knn_adaptive:", idx.search_knn_adaptive(query, 10, idx.entry_point, idx.max_level, 20), flush=True)

bins = [0.1, 0.2, 0.3, 0.4]
poly = [1.0, 0.5, 0.1]
print("Calling search_knn_dynamic...", flush=True)
print("search_knn_dynamic:", idx.search_knn_dynamic(query, 10, bins, poly, 10, 100, 10), flush=True)

weights = [0.25, 0.25, 0.25, 0.25]
ef_table = [10, 20, 30, 40]
idx.reset_dist_count()
print("Calling search_knn_true_ada...", flush=True)
print("search_knn_true_ada:", idx.search_knn_true_ada(query, 10, bins, weights, ef_table, 10, 100, 10), flush=True)
print("Distance computations:", idx.get_dist_count(), flush=True)

print("Calling profile_query...", flush=True)
print("profile_query:", idx.profile_query(query), flush=True)
print("Successfully called all new hybrid functions!", flush=True)
