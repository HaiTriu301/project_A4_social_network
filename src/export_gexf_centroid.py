"""
export_gexf_centroid.py
-----------------------
Đọc file GEXF đã có tọa độ từ Gephi (sau khi chạy ForceAtlas2),
tính centroid của mỗi community, gán label tại node gần centroid nhất.
Chỉ hiện label nếu community có >= min_nodes nodes.

Pipeline:
  Bước 1: chạy export_gexf.py → ra graph_louvain.gexf / graph_leiden.gexf / graph_spectral.gexf
  Bước 2: mở Gephi, chạy ForceAtlas2, export lại → graph_leiden_pos.gexf
  Bước 3: chạy file này → ra graph_leiden_centroid.gexf
  Bước 4: mở file centroid trong Gephi → xuất hình
"""

import pandas as pd
import numpy as np
import networkx as nx
import os
import sys
import io
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
RAW_DATA_PATH = "data/processed/RC_2026-01_filtered_final.csv"

# File GEXF đã export từ Gephi (có tọa độ x, y sau ForceAtlas2)
RAW_POS_PATH      = "outputs/gexf/graph_raw_pos.gexf"
LOUVAIN_POS_PATH  = "outputs/gexf/graph_louvain_pos.gexf"
LEIDEN_POS_PATH   = "outputs/gexf/graph_leiden_pos.gexf"
SPECTRAL_POS_PATH = "outputs/gexf/graph_spectral_pos.gexf"

# File communities
LOUVAIN_PATH  = "outputs/communities/louvain_communities.csv"
LEIDEN_PATH   = "outputs/communities/leiden_communities.csv"
SPECTRAL_PATH = "outputs/communities/spectral_communities.csv"

OUTPUT_DIR = "outputs/gexf"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Ngưỡng số node tối thiểu để hiện label
MIN_NODES = 300


# ==========================================
# HÀM 1: TÍNH TÊN SUBREDDIT CHIẾM ĐA SỐ
# ==========================================
def get_community_labels(community_path, raw_data_path):
    """Majority voting: subreddit chiếm đa số trong mỗi community"""
    df_comm = pd.read_csv(community_path).drop_duplicates(subset=['user'])
    df_raw  = pd.read_csv(raw_data_path, usecols=['author_fullname', 'subreddit'])
    df_raw  = df_raw.dropna(subset=['author_fullname', 'subreddit'])

    counts = df_raw.groupby(['author_fullname', 'subreddit']).size().reset_index(name='count')
    idx = counts.groupby('author_fullname')['count'].idxmax()
    df_user_sub = counts.loc[idx, ['author_fullname', 'subreddit']].reset_index(drop=True)

    df_merged = df_comm.merge(
        df_user_sub,
        left_on='user', right_on='author_fullname',
        how='left'
    )

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
# HÀM 2: GÁN CENTROID LABEL
# ==========================================
def assign_centroid_labels(G, community_path, raw_data_path, min_nodes=300):
    """
    Đọc tọa độ x, y từ G (đã load từ GEXF có position),
    tính centroid mỗi community, gán label tại node gần centroid nhất.
    Chỉ hiện label nếu community có >= min_nodes nodes.
    """
    df_comm = pd.read_csv(community_path).drop_duplicates(subset=['user'])
    user_to_comm = dict(zip(df_comm['user'], df_comm['community_id']))
    comm_size    = df_comm.groupby('community_id').size().to_dict()
    comm_to_label = get_community_labels(community_path, raw_data_path)

    # Đọc tọa độ từ thuộc tính viz.position trong GEXF
    pos_dict = {}
    for node, data in G.nodes(data=True):
        try:
            x = data['viz']['position']['x']
            y = data['viz']['position']['y']
            pos_dict[node] = (float(x), float(y))
        except (KeyError, TypeError):
            pass

    print(f"   -> Đọc được tọa độ của {len(pos_dict)}/{G.number_of_nodes()} nodes")

    # Reset tất cả label về trống
    for node in G.nodes():
        G.nodes[node]['label'] = ""

    # Cập nhật community_id và community_label từ file communities
    # (vì GEXF load lại có thể bị đổi kiểu)
    for node in G.nodes():
        comm_id = user_to_comm.get(node, -1)
        G.nodes[node]['community_id']    = str(comm_id)
        G.nodes[node]['community_label'] = comm_to_label.get(comm_id, 'unknown')

    # Nhóm node theo community (chỉ node có tọa độ)
    comm_nodes = defaultdict(list)
    for node in G.nodes():
        cid = user_to_comm.get(node, -1)
        if cid != -1 and node in pos_dict:
            comm_nodes[cid].append(node)

    # Với mỗi community đủ ngưỡng → tìm node gần centroid nhất
    labeled = 0
    skipped = 0
    for cid, nodes in comm_nodes.items():
        size = comm_size.get(cid, 0)

        if size < min_nodes:
            skipped += 1
            continue

        # Tính centroid
        coords     = np.array([pos_dict[n] for n in nodes])
        centroid_x = coords[:, 0].mean()
        centroid_y = coords[:, 1].mean()

        # Node gần centroid nhất
        distances    = np.sqrt((coords[:, 0] - centroid_x) ** 2 +
                               (coords[:, 1] - centroid_y) ** 2)
        closest_node = nodes[int(np.argmin(distances))]

        # Gán label: tên subreddit + số node
        label = comm_to_label.get(cid, str(cid))
        G.nodes[closest_node]['label'] = f"{label} ({size})"

        labeled += 1
        print(f"      Community {cid}: '{label}' | {size} nodes")

    print(f"   -> {labeled} communities được gán label (>= {min_nodes} nodes)")
    print(f"   -> {skipped} communities bị ẩn label (< {min_nodes} nodes)")
    return G

def assign_raw_centroid_labels(G, raw_data_path, min_nodes=300):
    """
    Dành riêng cho graph_raw — chỉ gán centroid label,
    không đổi màu (màu đã được gán sẵn từ export_gexf.py).
    """
    df_raw = pd.read_csv(raw_data_path, usecols=['author_fullname', 'subreddit'])
    df_raw = df_raw.dropna(subset=['author_fullname', 'subreddit'])

    # Majority voting
    counts = df_raw.groupby(['author_fullname', 'subreddit']).size().reset_index(name='count')
    idx = counts.groupby('author_fullname')['count'].idxmax()
    df_user_sub = counts.loc[idx, ['author_fullname', 'subreddit']].reset_index(drop=True)
    user_to_sub = dict(zip(df_user_sub['author_fullname'], df_user_sub['subreddit']))
    sub_size    = df_user_sub.groupby('subreddit').size().to_dict()

    # Đọc tọa độ từ GEXF
    pos_dict = {}
    for node, data in G.nodes(data=True):
        try:
            x = data['viz']['position']['x']
            y = data['viz']['position']['y']
            pos_dict[node] = (float(x), float(y))
        except (KeyError, TypeError):
            pass

    print(f"   -> Đọc được tọa độ của {len(pos_dict)}/{G.number_of_nodes()} nodes")

    # Reset tất cả label về trống
    for node in G.nodes():
        G.nodes[node]['label'] = ""

    # Nhóm node theo subreddit
    sub_nodes = defaultdict(list)
    for node in G.nodes():
        sub = user_to_sub.get(node, 'unknown')
        if sub != 'unknown' and node in pos_dict:
            sub_nodes[sub].append(node)

    # Tính centroid và gán label
    labeled = 0
    skipped = 0
    for sub, nodes in sub_nodes.items():
        size = sub_size.get(sub, 0)

        if size < min_nodes:
            skipped += 1
            continue

        coords     = np.array([pos_dict[n] for n in nodes])
        centroid_x = coords[:, 0].mean()
        centroid_y = coords[:, 1].mean()
        distances  = np.sqrt((coords[:, 0] - centroid_x) ** 2 +
                             (coords[:, 1] - centroid_y) ** 2)
        closest_node = nodes[int(np.argmin(distances))]
        G.nodes[closest_node]['label'] = f"{sub} ({size})"

        labeled += 1
        print(f"      Subreddit '{sub}' | {size} nodes")

    print(f"   -> {labeled} subreddits được gán label (>= {min_nodes} nodes)")
    print(f"   -> {skipped} subreddits bị ẩn label (< {min_nodes} nodes)")
    return G

# ==========================================
# HÀM 3: XUẤT FILE GEXF
# ==========================================
def export_gexf(G, output_path):
    nx.write_gexf(G, output_path)
    print(f"   -> Đã xuất: {output_path}")


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    # ── Graph Raw ──
    RAW_POS_PATH = "outputs/gexf/graph_raw_pos.gexf"
    print("\n[Raw] Đang xử lý graph_raw...")
    if not os.path.exists(RAW_POS_PATH):
        print(f"   ⚠️  Không tìm thấy {RAW_POS_PATH} — bỏ qua")
    else:
        G = nx.read_gexf(RAW_POS_PATH)
        print(f"   -> {G.number_of_nodes()} nodes | {G.number_of_edges()} edges")
        G = assign_raw_centroid_labels(G, RAW_DATA_PATH, min_nodes=MIN_NODES)
        export_gexf(G, os.path.join(OUTPUT_DIR, "graph_raw_centroid.gexf"))

    configs = [
        ("Louvain",  LOUVAIN_POS_PATH,  LOUVAIN_PATH,  "graph_louvain_centroid.gexf"),
        ("Leiden",   LEIDEN_POS_PATH,   LEIDEN_PATH,   "graph_leiden_centroid.gexf"),
        ("Spectral", SPECTRAL_POS_PATH, SPECTRAL_PATH, "graph_spectral_centroid.gexf"),
    ]

    for algo_name, pos_path, comm_path, out_filename in configs:
        print(f"\n[{algo_name}] Đang xử lý...")

        if not os.path.exists(pos_path):
            print(f"   ⚠️  Không tìm thấy {pos_path} — bỏ qua")
            print(f"       Hãy export từ Gephi sau khi chạy ForceAtlas2")
            continue

        # Đọc GEXF đã có tọa độ từ Gephi
        print(f"   Đang đọc {pos_path}...")
        G = nx.read_gexf(pos_path)
        print(f"   -> {G.number_of_nodes()} nodes | {G.number_of_edges()} edges")

        # Gán centroid label
        G = assign_centroid_labels(G, comm_path, RAW_DATA_PATH, min_nodes=MIN_NODES)

        # Xuất file
        out_path = os.path.join(OUTPUT_DIR, out_filename)
        export_gexf(G, out_path)

    print("\n✅ Hoàn tất!")
    print("\nHướng dẫn Gephi:")
    print("  1. Mở file *_centroid.gexf")
    print("  2. Preview → Show Labels ✅ | Proportional size ❌")
    print("  3. Font size: 14-18 | Edge opacity: 10-20%")
    print(f"\n  Điều chỉnh MIN_NODES (hiện = {MIN_NODES}) nếu muốn ẩn/hiện thêm label")