import pandas as pd
import networkx as nx
import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def calculate_internal_metrics(edges_path, communities_dir, output_csv):
    print("1. Đang tải đồ thị (Edges)...")
    edges_df = pd.read_csv(edges_path)
    
    # Sử dụng đồ thị vô hướng (Graph) thay vì có hướng (DiGraph)
    # vì thuật toán Modularity tiêu chuẩn hoạt động chính xác nhất trên đồ thị vô hướng
    G = nx.from_pandas_edgelist(edges_df, 'source', 'target', ['weight'], create_using=nx.Graph())
    print(f"   -> Đồ thị gồm {G.number_of_nodes()} đỉnh và {G.number_of_edges()} cạnh.")

    algorithms = ['louvain', 'leiden', 'spectral']
    results = []

    print("\n2. Bắt đầu tính toán Internal Metrics:")
    print("-" * 50)

    for algo in algorithms:
        comm_path = os.path.join(communities_dir, f"{algo}_communities.csv")
        
        if not os.path.exists(comm_path):
            print(f"[Cảnh báo] Không tìm thấy file kết quả của {algo} tại {comm_path}")
            continue
            
        df_pred = pd.read_csv(comm_path)

        # XỬ LÝ ĐẶC BIỆT: Hàm Modularity yêu cầu các cụm phải độc lập (Partition).
        # Vì BigCLAM là Overlapping (1 người nhiều cụm), ta phải lấy cụm đầu tiên của họ để tính xấp xỉ.
        if algo == 'bigclam':
            df_partition = df_pred.drop_duplicates(subset=['user'])
        else:
            df_partition = df_pred

        # Tạo danh sách các tập hợp (set) user cho mỗi cộng đồng
        communities = []
        for comm_id, group in df_partition.groupby('community_id'):
            # Chỉ lấy những node thực sự tồn tại trong đồ thị G
            valid_nodes = set(group['user']).intersection(set(G.nodes()))
            if len(valid_nodes) > 0:
                communities.append(valid_nodes)

        # Đảm bảo toàn bộ đỉnh của đồ thị đều nằm trong một cụm nào đó
        covered_nodes = set().union(*communities)
        missing_nodes = set(G.nodes()) - covered_nodes
        if missing_nodes:
            # Nếu thuật toán bỏ sót node nào (vd bị cô lập), cho node đó thành 1 cụm riêng lẻ
            for node in missing_nodes:
                communities.append({node})

        # ---------------------------------------------------------
        # TÍNH TOÁN 1: MODULARITY (Càng cao càng tốt)
        # ---------------------------------------------------------
        try:
            modularity = nx.community.modularity(G, communities, weight='weight')
        except Exception as e:
            modularity = 0
            print(f"[Lỗi] Không thể tính Modularity cho {algo}: {e}")

        # ---------------------------------------------------------
        # TÍNH TOÁN 2: AVERAGE CONDUCTANCE (Càng thấp càng tốt)
        # ---------------------------------------------------------
        conductance_scores = []
        for comm in communities:
            # Conductance chỉ tính được cho các cụm có từ 2 node trở lên
            if len(comm) > 1:
                try:
                    # Tính tỷ lệ cạnh đâm ra ngoài / tổng cạnh của cụm
                    cond = nx.algorithms.cuts.conductance(G, comm, weight='weight')
                    conductance_scores.append(cond)
                except Exception:
                    pass
        
        avg_conductance = sum(conductance_scores) / len(conductance_scores) if conductance_scores else 0

        # Lưu kết quả
        results.append({
            'Thuật toán': algo.capitalize(),
            'Modularity (Q)': round(modularity, 4),
            'Avg Conductance': round(avg_conductance, 4),
            'Số cụm': len(communities)
        })
        
        print(f"✔️ {algo.capitalize()}: Modularity = {modularity:.4f} | Avg Conductance = {avg_conductance:.4f}")

    # 3. Xuất bảng tổng hợp
    print("-" * 50)
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    
    df_results.to_csv(output_csv, index=False)
    print(f"\n🎉 Đã lưu bảng Internal Metrics tại: {output_csv}")

if __name__ == "__main__":
    # CẬP NHẬT ĐƯỜNG DẪN Ở ĐÂY
    EDGES_PATH = "data/graph/edges.csv"
    COMMUNITIES_DIR = "outputs/communities/"
    OUTPUT_REPORT = "outputs/metrics/internal_metrics.csv"
    
    calculate_internal_metrics(EDGES_PATH, COMMUNITIES_DIR, OUTPUT_REPORT)