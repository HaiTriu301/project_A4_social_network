import pandas as pd
import networkx as nx
import community.community_louvain as community_louvain # pip install python-louvain
import time
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def run_louvain(input_path, output_path):
    print("1. Đang tải dữ liệu và xây dựng đồ thị (Louvain)...")
    start_time = time.time()
    
    df = pd.read_csv(input_path)
    # Louvain hoạt động tốt nhất trên đồ thị vô hướng (Undirected Graph)
    G = nx.from_pandas_edgelist(df, 'source', 'target', ['weight'], create_using=nx.Graph())
    
    print(f"2. Đang chạy thuật toán Louvain trên {G.number_of_nodes()} đỉnh...")
    # Chạy thuật toán, có xét đến trọng số (weight)
    partition = community_louvain.best_partition(G, weight='weight')
    
    print("3. Đang lưu kết quả...")
    # Chuyển kết quả dictionary thành DataFrame
    df_res = pd.DataFrame(list(partition.items()), columns=['user', 'community_id'])
    df_res.to_csv(output_path, index=False)
    
    print(f"Hoàn thành trong {time.time() - start_time:.2f} giây. Đã lưu tại: {output_path}")

if __name__ == "__main__":
    run_louvain("data/graph/edges.csv", "outputs/communities/louvain_communities.csv")