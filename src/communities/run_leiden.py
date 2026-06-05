import pandas as pd
import igraph as ig
import leidenalg # pip install igraph leidenalg
import time
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def run_leiden(input_path, output_path):
    print("1. Đang tải dữ liệu và xây dựng đồ thị (Leiden)...")
    start_time = time.time()
    
    df = pd.read_csv(input_path)
    
    # Chuyển data thành dạng TupleList để nạp vào igraph (rất nhanh)
    tuples = [tuple(x) for x in df[['source', 'target', 'weight']].values]
    G = ig.Graph.TupleList(tuples, directed=False, edge_attrs=['weight'])
    
    print(f"2. Đang chạy thuật toán Leiden trên {G.vcount()} đỉnh...")
    # Tối ưu hóa hàm Modularity (giống Louvain nhưng thuật toán chia tốt hơn)
    partition = leidenalg.find_partition(
        G, leidenalg.ModularityVertexPartition, weights=G.es['weight']
    )
    
    print("3. Đang lưu kết quả...")
    results = []
    # igraph lưu tên đỉnh trong thuộc tính 'name'
    for node_idx, comm_id in enumerate(partition.membership):
        user_name = G.vs[node_idx]['name']
        results.append((user_name, comm_id))
        
    df_res = pd.DataFrame(results, columns=['user', 'community_id'])
    df_res.to_csv(output_path, index=False)
    
    print(f"Hoàn thành trong {time.time() - start_time:.2f} giây. Đã lưu tại: {output_path}")

if __name__ == "__main__":
    run_leiden("data/graph/edges.csv", "outputs/communities/leiden_communities.csv")