# import pandas as pd
# import networkx as nx
# import sys
# import io

# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# def export_to_gephi(edges_path, community_path, raw_data_path, output_gexf):
#     print("1. Đang tải danh sách cạnh (edges)...")
#     edges_df = pd.read_csv(edges_path)
    
#     # Xây dựng đồ thị có hướng (DiGraph)
#     G = nx.from_pandas_edgelist(edges_df, 'source', 'target', ['weight'], create_using=nx.DiGraph())
#     print(f"   -> Đồ thị gồm {G.number_of_nodes()} đỉnh và {G.number_of_edges()} cạnh.")
    
#     # ==========================================
#     # NHÚNG NHÃN COMMUNITY_ID TỪ THUẬT TOÁN
#     # ==========================================
#     print("2. Đang tải dữ liệu cộng đồng (Community ID)...")
#     comm_df = pd.read_csv(community_path)
#     comm_df_unique = comm_df.drop_duplicates(subset=['user'])
#     community_map = dict(zip(comm_df_unique['user'], comm_df_unique['community_id']))
    
#     print("3. Đang nhúng nhãn 'community_id' vào từng đỉnh...")
#     nx.set_node_attributes(G, community_map, 'community_id')
    
#     # ==========================================
#     # NHÚNG NHÃN SUBREDDIT TỪ DỮ LIỆU GỐC
#     # ==========================================
#     print("4. Đang tải dữ liệu gốc để lấy nhãn Subreddit...")
#     raw_df = pd.read_csv(raw_data_path)
    
#     # Loại bỏ các dòng trùng lặp user để lấy 1 subreddit đại diện cho mỗi user
#     # (Dựa trên file test.csv bạn gửi trước đó, cột user là 'author_fullname')
#     raw_df_unique = raw_df.drop_duplicates(subset=['author_fullname'])
    
#     # Tạo từ điển { 'tên_user': 'tên_subreddit' }
#     subreddit_map = dict(zip(raw_df_unique['author_fullname'], raw_df_unique['subreddit']))
    
#     print("5. Đang nhúng nhãn 'subreddit' vào từng đỉnh...")
#     nx.set_node_attributes(G, subreddit_map, 'subreddit')
    
#     # ==========================================
#     # XUẤT FILE GEPHI
#     # ==========================================
#     print("6. Đang xuất ra định dạng Gephi (.gexf)...")
#     nx.write_gexf(G, output_gexf)
#     print(f"🎉 Hoàn thành! File đã được lưu tại: {output_gexf}")

# if __name__ == "__main__":
#     # Lưu ý: Cập nhật lại đường dẫn file raw data cho đúng với máy của bạn
#     export_to_gephi(
#         edges_path="data/graph/edges.csv",
#         community_path="outputs/communities/louvain_communities.csv",
#         raw_data_path="data/processed/RC_2026-01_filtered_final.csv", # <-- THÊM ĐƯỜNG DẪN FILE GỐC VÀO ĐÂY
#         output_gexf="outputs/gephi/reddit_network_louvain.gexf"
#     )


"""
export_gexf.py
--------------
Xuất 4 file GEXF để visualize trong Gephi:

1. graph_raw.gexf          → Đồ thị gốc, không có community label
2. graph_louvain.gexf      → Đồ thị + community_id từ Louvain (dùng để tô màu)
3. graph_leiden.gexf       → Đồ thị + community_id từ Leiden
4. graph_spectral.gexf     → Đồ thị + community_id từ Spectral Clustering

Cách dùng trong Gephi:
- Mở file .gexf
- Chạy layout (ví dụ: ForceAtlas2)
- Tô màu node theo cột "community_id" (Partition → Nodes → community_id)
"""

"""
export_gexf.py
--------------
Xuất 4 file GEXF để visualize trong Gephi:

1. graph_raw.gexf          → Đồ thị gốc, không có community label
2. graph_louvain.gexf      → Đồ thị + community_id (String) + community_label (tên subreddit)
3. graph_leiden.gexf       → Đồ thị + community_id (String) + community_label (tên subreddit)
4. graph_spectral.gexf     → Đồ thị + community_id (String) + community_label (tên subreddit)

Cách dùng trong Gephi:
- Mở file .gexf
- Chạy layout ForceAtlas2 (Scaling=5, Gravity=2, Prevent Overlap=tick)
- Tô màu node: Appearance > Nodes > Partition > community_id
- Hiện tên cụm: bật label (T) > Appearance > Nodes > Labels > Attribute > community_label
"""

"""
export_gexf.py
--------------
Xuất 4 file GEXF để visualize trong Gephi.
Mỗi community chỉ có 1 node hub hiển thị tên subreddit,
các node còn lại để label trống → tránh chồng chéo.
"""

import pandas as pd
import networkx as nx
import os
import sys
import io
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
EDGES_PATH    = "data/graph/edges.csv"
NODES_PATH    = "data/graph/nodes.csv"
RAW_DATA_PATH = "data/processed/RC_2026-01_filtered_final.csv"
LOUVAIN_PATH  = "outputs/communities/louvain_communities.csv"
LEIDEN_PATH   = "outputs/communities/leiden_communities.csv"
SPECTRAL_PATH = "outputs/communities/spectral_communities.csv"
OUTPUT_DIR    = "outputs/gexf"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# HÀM 1: XÂY DỰNG ĐỒ THỊ GỐC
# ==========================================
def build_base_graph(edges_path, nodes_path):
    print("   Đang tải edges và nodes...")
    df_edges = pd.read_csv(edges_path)
    df_nodes = pd.read_csv(nodes_path)

    G = nx.Graph()

    for _, row in df_nodes.iterrows():
        G.add_node(row['user'], label="")

    for _, row in df_edges.iterrows():
        G.add_edge(row['source'], row['target'], weight=float(row['weight']))

    print(f"   -> {G.number_of_nodes()} nodes | {G.number_of_edges()} edges")
    return G


# ==========================================
# HÀM 2: TÍNH TÊN SUBREDDIT CHIẾM ĐA SỐ
# ==========================================
def get_community_labels(community_path, raw_data_path):
    df_comm = pd.read_csv(community_path).drop_duplicates(subset=['user'])
    df_raw  = pd.read_csv(raw_data_path, usecols=['author_fullname', 'subreddit'])
    df_raw  = df_raw.dropna(subset=['author_fullname', 'subreddit'])

    # Majority voting: subreddit chính của mỗi user
    counts = df_raw.groupby(['author_fullname', 'subreddit']).size().reset_index(name='count')
    idx = counts.groupby('author_fullname')['count'].idxmax()
    df_user_sub = counts.loc[idx, ['author_fullname', 'subreddit']].reset_index(drop=True)

    # Merge để biết mỗi community thuộc subreddit nào
    df_merged = df_comm.merge(
        df_user_sub,
        left_on='user', right_on='author_fullname',
        how='left'
    )

    # Subreddit chiếm đa số trong mỗi community
    dominant = (
        df_merged.groupby(['community_id', 'subreddit'])
                 .size()
                 .reset_index(name='count')
                 .sort_values('count', ascending=False)
                 .drop_duplicates(subset='community_id')
    )

    comm_to_label = dict(zip(dominant['community_id'], dominant['subreddit']))
    print(f"   -> {len(comm_to_label)} communities có tên subreddit đại diện")
    return comm_to_label


# ==========================================
# HÀM 3: BẢNG MÀU
# ==========================================
def get_color_palette():
    return [
        "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
        "#42d4f4", "#f032e6", "#bfef45", "#469990", "#dcbeff",
        "#9a6324", "#800000", "#aaffc3", "#808000", "#ffd8b1",
        "#000075", "#a9a9a9", "#ff6347", "#40e0d0", "#ee82ee",
        "#6495ed", "#ff7f50", "#9acd32", "#ba55d3", "#20b2aa",
        "#ff4500", "#da70d6", "#7b68ee"
    ]


# ==========================================
# HÀM 4: GÁN COMMUNITY + MÀU + LABEL HUB
# ==========================================
def attach_community(G, community_path, raw_data_path, algo_name):
    df_comm = pd.read_csv(community_path).drop_duplicates(subset=['user'])
    user_to_comm  = dict(zip(df_comm['user'], df_comm['community_id']))
    comm_to_label = get_community_labels(community_path, raw_data_path)

    # Gán màu cho từng community
    palette = get_color_palette()
    unique_comms  = sorted(df_comm['community_id'].unique())
    comm_to_color = {cid: palette[i % len(palette)] for i, cid in enumerate(unique_comms)}

    # Gán community_id, màu, label trống cho tất cả node
    assigned = 0
    for node in G.nodes():
        comm_id = user_to_comm.get(node, -1)
        color   = comm_to_color.get(comm_id, '#aaaaaa')
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)

        G.nodes[node]['community_id']    = str(comm_id)
        G.nodes[node]['community_label'] = comm_to_label.get(comm_id, 'unknown')
        G.nodes[node]['label']           = ""   # mặc định trống
        G.nodes[node]['viz'] = {
            'color': {'r': r, 'g': g, 'b': b, 'a': 1.0}
        }

        if comm_id != -1:
            assigned += 1

    # ==========================================
    # CHỈ GÁN LABEL CHO NODE HUB (DEGREE CAO NHẤT)
    # TRONG MỖI COMMUNITY
    # ==========================================
    comm_nodes = defaultdict(list)
    for node in G.nodes():
        cid = G.nodes[node].get('community_id', '-1')
        if cid != '-1':
            comm_nodes[cid].append(node)

    for cid, nodes in comm_nodes.items():
        # Node có degree cao nhất = hub của cụm
        hub = max(nodes, key=lambda n: G.degree(n))
        hub_label = comm_to_label.get(int(cid), cid)
        G.nodes[hub]['label'] = hub_label   # chỉ node này có label

    n_communities = df_comm['community_id'].nunique()
    print(f"   -> {assigned}/{G.number_of_nodes()} nodes được gán | {n_communities} communities ({algo_name})")
    print(f"   -> Mỗi community hiển thị đúng 1 label tại node hub")
    return G

def attach_ground_truth(G, raw_data_path):
    """
    Gán màu cho graph_raw dựa trên ground truth subreddit
    (majority voting) thay vì community từ thuật toán.
    """
    print("   Đang gán màu ground truth (subreddit)...")
    df_raw = pd.read_csv(raw_data_path, usecols=['author_fullname', 'subreddit'])
    df_raw = df_raw.dropna(subset=['author_fullname', 'subreddit'])

    # Majority voting
    counts = df_raw.groupby(['author_fullname', 'subreddit']).size().reset_index(name='count')
    idx = counts.groupby('author_fullname')['count'].idxmax()
    df_user_sub = counts.loc[idx, ['author_fullname', 'subreddit']].reset_index(drop=True)
    user_to_sub = dict(zip(df_user_sub['author_fullname'], df_user_sub['subreddit']))

    # Gán màu cho từng subreddit
    palette = get_color_palette()
    unique_subs  = sorted(df_user_sub['subreddit'].unique())
    sub_to_color = {s: palette[i % len(palette)] for i, s in enumerate(unique_subs)}
    sub_to_id    = {s: i for i, s in enumerate(unique_subs)}

    assigned = 0
    for node in G.nodes():
        subreddit = user_to_sub.get(node, 'unknown')
        color     = sub_to_color.get(subreddit, '#aaaaaa')
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)

        G.nodes[node]['community_id']    = str(sub_to_id.get(subreddit, -1))
        G.nodes[node]['community_label'] = subreddit
        G.nodes[node]['label']           = ""
        G.nodes[node]['viz'] = {
            'color': {'r': r, 'g': g, 'b': b, 'a': 1.0}
        }
        if subreddit != 'unknown':
            assigned += 1

    # Gán label tại hub của mỗi subreddit
    sub_nodes = defaultdict(list)
    for node in G.nodes():
        sub = user_to_sub.get(node, 'unknown')
        if sub != 'unknown':
            sub_nodes[sub].append(node)

    for sub, nodes in sub_nodes.items():
        hub = max(nodes, key=lambda n: G.degree(n))
        G.nodes[hub]['label'] = sub

    print(f"   -> {assigned}/{G.number_of_nodes()} nodes được gán màu subreddit")
    print(f"   -> {len(unique_subs)} subreddits (ground truth communities)")
    return G

# ==========================================
# HÀM 5: XUẤT FILE GEXF
# ==========================================
def export_gexf(G, output_path):
    nx.write_gexf(G, output_path)
    print(f"   -> Đã xuất: {output_path}")


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":

    # 1. Đồ thị gốc
    print("\n[1/4] Xuất đồ thị gốc...")
    G = build_base_graph(EDGES_PATH, NODES_PATH)
    G = attach_ground_truth(G, RAW_DATA_PATH)
    export_gexf(G, os.path.join(OUTPUT_DIR, "graph_raw.gexf"))

    # 2. Louvain
    print("\n[2/4] Xuất đồ thị + Louvain communities...")
    G = build_base_graph(EDGES_PATH, NODES_PATH)
    G = attach_community(G, LOUVAIN_PATH, RAW_DATA_PATH, "Louvain")
    export_gexf(G, os.path.join(OUTPUT_DIR, "graph_louvain.gexf"))

    # 3. Leiden
    print("\n[3/4] Xuất đồ thị + Leiden communities...")
    G = build_base_graph(EDGES_PATH, NODES_PATH)
    G = attach_community(G, LEIDEN_PATH, RAW_DATA_PATH, "Leiden")
    export_gexf(G, os.path.join(OUTPUT_DIR, "graph_leiden.gexf"))

    # 4. Spectral
    print("\n[4/4] Xuất đồ thị + Spectral communities...")
    G = build_base_graph(EDGES_PATH, NODES_PATH)
    G = attach_community(G, SPECTRAL_PATH, RAW_DATA_PATH, "Spectral")
    export_gexf(G, os.path.join(OUTPUT_DIR, "graph_spectral.gexf"))

    print("\n✅ Hoàn tất! Các file GEXF được lưu tại:", OUTPUT_DIR)
    print("\nHướng dẫn Gephi Preview:")
    print("  - Show Labels    : ✅")
    print("  - Proportional   : ❌ bỏ tick")
    print("  - Font size      : 14-18")
    print("  - Edge opacity   : 10-20%")
    print("  - Curved         : ❌ bỏ tick")