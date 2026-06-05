import pandas as pd
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def calculate_purity(df_merged):
    """Tính toán chỉ số Purity (Độ tinh khiết) của các cụm"""
    total_samples = len(df_merged)
    majority_sum = 0
    for comm_id, group in df_merged.groupby('community_id'):
        most_common_count = group['subreddit'].value_counts().iloc[0]
        majority_sum += most_common_count
    return majority_sum / total_samples


def build_ground_truth(raw_data_path, nodes_path):
    """
    Chỉ build ground truth cho đúng những user có trong đồ thị (nodes.csv).
    Nhãn = subreddit mà user comment nhiều nhất (majority voting).
    """
    print("1. Đang tải Ground Truth...")

    # Lấy danh sách user hợp lệ từ đồ thị
    df_nodes = pd.read_csv(nodes_path, usecols=['user'])
    graph_users = set(df_nodes['user'])
    print(f"   -> {len(graph_users)} users trong đồ thị (nodes.csv)")

    # Lấy subreddit từ file raw, chỉ giữ user có trong đồ thị
    df_raw = pd.read_csv(raw_data_path, usecols=['author_fullname', 'subreddit'])
    df_raw = df_raw.dropna(subset=['author_fullname', 'subreddit'])
    df_raw = df_raw[df_raw['author_fullname'].isin(graph_users)]  # ← lọc theo đồ thị

    # Majority voting
    counts = df_raw.groupby(['author_fullname', 'subreddit']).size().reset_index(name='count')
    idx = counts.groupby('author_fullname')['count'].idxmax()
    df_true = counts.loc[idx, ['author_fullname', 'subreddit']].reset_index(drop=True)

    print(f"   -> {len(df_true)} users có nhãn subreddit | {df_true['subreddit'].nunique()} subreddits")

    # Kiểm tra user trong đồ thị nhưng không tìm thấy trong raw
    matched = set(df_true['author_fullname'])
    missing = graph_users - matched
    if missing:
        print(f"   -> ⚠️ {len(missing)} users có trong đồ thị nhưng không có trong raw CSV")

    return df_true


def evaluate_all_algorithms(raw_data_path, nodes_path, communities_dir, output_csv):
    # Xây dựng Ground Truth bằng Majority Voting
    df_true = build_ground_truth(raw_data_path, nodes_path)

    algorithms = ['louvain', 'leiden', 'spectral']
    results = []

    print("\n2. Bắt đầu chấm điểm các thuật toán:")
    print("-" * 50)

    for algo in algorithms:
        comm_path = os.path.join(communities_dir, f"{algo}_communities.csv")

        if not os.path.exists(comm_path):
            print(f"[Cảnh báo] Không tìm thấy file kết quả của {algo} tại {comm_path}")
            continue

        df_pred = pd.read_csv(comm_path)

        # Với overlapping community (bigclam), lấy cụm đầu tiên để tính xấp xỉ
        df_pred_unique = df_pred.drop_duplicates(subset=['user'])
        df_pred_unique = df_pred_unique.rename(columns={'user': 'author_fullname'})

        # Merge ground truth với kết quả thuật toán
        df_merged = pd.merge(df_true, df_pred_unique, on='author_fullname', how='inner')

        if len(df_merged) == 0:
            print(f"[Lỗi] Không có user nào khớp nhau cho {algo}.")
            continue

        labels_true = df_merged['subreddit']
        labels_pred = df_merged['community_id']

        nmi     = normalized_mutual_info_score(labels_true, labels_pred)
        ari     = adjusted_rand_score(labels_true, labels_pred)
        purity  = calculate_purity(df_merged)

        results.append({
            'Thuật toán': algo.capitalize(),
            'NMI': round(nmi, 4),
            'ARI': round(ari, 4),
            'Purity': f"{round(purity * 100, 2)}%",
            'Số cụm tìm được': df_merged['community_id'].nunique(),
            'Số user đánh giá': len(df_merged),
        })

        print(f"✔️ {algo.capitalize()}: NMI = {nmi:.4f} | ARI = {ari:.4f} | Purity = {purity:.4f} ({len(df_merged)} users)")

    print("-" * 50)
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))

    df_results.to_csv(output_csv, index=False)
    print(f"\n🎉 Đã lưu bảng External Metrics tại: {output_csv}")


if __name__ == "__main__":
    RAW_DATA          = "data/processed/RC_2026-01_filtered_final.csv"
    NODES_PATH        = "data/graph/nodes.csv"
    COMMUNITIES_FOLDER = "outputs/communities/"
    OUTPUT_REPORT     = "outputs/metrics/external_metrics.csv"

    os.makedirs("outputs/metrics", exist_ok=True)

    evaluate_all_algorithms(RAW_DATA, NODES_PATH, COMMUNITIES_FOLDER, OUTPUT_REPORT)