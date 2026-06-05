# import pandas as pd
# import plotly.graph_objects as go
# import os
# from IPython.display import display
# import sys
# import io

# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# # ==========================================
# # 1. HÀM ĐẾM SỐ TỪ TRÙNG NHAU
# # ==========================================
# def count_overlap(words_a, words_b):
#     if pd.isna(words_a) or pd.isna(words_b):
#         return 0
#     set_a = set([w.strip().lower() for w in str(words_a).split(',')])
#     set_b = set([w.strip().lower() for w in str(words_b).split(',')])
#     return len(set_a.intersection(set_b))


# # ==========================================
# # 2. XÂY DỰNG MA TRẬN VÀ ĐƯỜNG CHÉO
# #    (Dùng lại logic giống heatmap để nhất quán)
# # ==========================================
# def build_diagonal_matrix(subreddits, truth_keywords, communities, algo_keywords):
#     """
#     Trả về:
#       - raw_matrix      : ma trận gốc [subreddit x community], chưa sắp xếp
#       - sorted_matrix   : ma trận sau khi sắp xếp theo đường chéo
#       - diagonal_scores : list điểm tại ô chéo (community 0 → subreddit được assign, ...)
#       - diagonal_pairs  : list (subreddit_name, community_id) theo thứ tự đường chéo
#     """
#     # Sắp xếp cột theo thứ tự số
#     sorted_comm_indices = sorted(range(len(communities)), key=lambda i: int(communities[i]))
#     sorted_communities  = [communities[i] for i in sorted_comm_indices]
#     sorted_algo_kw      = [algo_keywords[i] for i in sorted_comm_indices]

#     # Tạo ma trận gốc (rows = subreddit, cols = community đã sort)
#     matrix = []
#     for sub_kw in truth_keywords:
#         row = [count_overlap(sub_kw, comm_kw) for comm_kw in sorted_algo_kw]
#         matrix.append(row)

#     # Greedy diagonal assignment
#     assigned       = set()
#     diagonal_order = []

#     for col_idx in range(len(sorted_communities)):
#         best_row   = None
#         best_score = -1
#         for row_idx in range(len(subreddits)):
#             if row_idx in assigned:
#                 continue
#             if matrix[row_idx][col_idx] > best_score:
#                 best_score = matrix[row_idx][col_idx]
#                 best_row   = row_idx
#         if best_row is not None:
#             diagonal_order.append(best_row)
#             assigned.add(best_row)

#     # Subreddit chưa được assign → thêm vào cuối
#     for row_idx in range(len(subreddits)):
#         if row_idx not in assigned:
#             diagonal_order.append(row_idx)

#     sorted_matrix = [matrix[i] for i in diagonal_order]
#     sorted_subs   = [subreddits[i] for i in diagonal_order]

#     # Lấy điểm đường chéo: ô [i][i] với i < min(n_subs, n_comms)
#     n_diag         = min(len(sorted_subs), len(sorted_communities))
#     diagonal_scores = [sorted_matrix[i][i] for i in range(n_diag)]
#     diagonal_pairs  = [(sorted_subs[i], sorted_communities[i]) for i in range(n_diag)]

#     return matrix, sorted_matrix, sorted_subs, sorted_communities, diagonal_scores, diagonal_pairs


# # ==========================================
# # 3. TÍNH 3 METRICS
# # ==========================================
# def compute_metrics(sorted_matrix, diagonal_scores, threshold=3):
#     """
#     DDS – Diagonal Dominance Score
#     MCR – Match Coverage Rate
#     DCR – Diagonal Concentration Ratio
#     """
#     n_subs  = len(sorted_matrix)
#     n_comms = len(sorted_matrix[0]) if sorted_matrix else 0
#     n_diag  = len(diagonal_scores)

#     # --- DDS ---
#     total_score    = sum(sum(row) for row in sorted_matrix)
#     diagonal_total = sum(diagonal_scores)
#     dds = (diagonal_total / total_score * 100) if total_score > 0 else 0.0

#     # --- MCR ---
#     matched = sum(1 for s in diagonal_scores if s >= threshold)
#     mcr = (matched / n_subs * 100)   # chia cho tổng subreddit thực tế (29)

#     # --- DCR ---
#     ratios = []
#     for i in range(n_diag):
#         row_max = max(sorted_matrix[i]) if sorted_matrix[i] else 0
#         if row_max > 0:
#             ratios.append(diagonal_scores[i] / row_max * 100)
#         else:
#             ratios.append(0.0)
#     dcr = (sum(ratios) / len(ratios)) if ratios else 0.0

#     return round(dds, 2), round(mcr, 2), round(dcr, 2)


# # ==========================================
# # 4. HÀM CHÍNH
# # ==========================================
# def evaluate_algorithms(truth_path, communities_dir, threshold=3):
#     print("⏳ Đang tải dữ liệu Topic gốc (Subreddit Thực tế)...")
#     if not os.path.exists(truth_path):
#         print(f"❌ LỖI: Không tìm thấy file: {truth_path}")
#         return

#     df_truth       = pd.read_csv(truth_path)
#     subreddits     = df_truth['Subreddit'].tolist()
#     truth_keywords = df_truth['Top 20 Keywords (LDA)'].tolist()

#     algorithms   = ['leiden', 'louvain', 'spectral']
#     summary_rows = []           # Bảng tổng hợp 3 metrics
#     detail_rows  = []           # Bảng chi tiết từng ô chéo

#     for algo in algorithms:
#         comm_path = os.path.join(communities_dir, f"community_topics_{algo}.csv")
#         if not os.path.exists(comm_path):
#             print(f"⚠️ Bỏ qua {algo.upper()}: không tìm thấy file {comm_path}")
#             continue

#         df_algo       = pd.read_csv(comm_path)
#         communities   = df_algo['Community_ID'].tolist()
#         algo_keywords = df_algo['Top 20 Keywords (LDA)'].tolist()

#         print(f"  ✅ Đang tính metrics cho {algo.upper()} "
#               f"({len(communities)} communities × {len(subreddits)} subreddits)...")

#         # Xây dựng ma trận và đường chéo
#         (raw_matrix, sorted_matrix,
#          sorted_subs, sorted_comms,
#          diagonal_scores, diagonal_pairs) = build_diagonal_matrix(
#             subreddits, truth_keywords, communities, algo_keywords
#         )

#         # Tính 3 metrics
#         dds, mcr, dcr = compute_metrics(sorted_matrix, diagonal_scores, threshold)

#         # Xếp hạng tổng thể (trung bình 3 metrics, tính sau)
#         summary_rows.append({
#             'Algorithm'                         : algo.capitalize(),
#             'Num Communities'                   : len(communities),
#             'DDS – Diagonal Dominance Score (%)': dds,
#             'MCR – Match Coverage Rate (%)'     : mcr,
#             f'DCR – Diagonal Concentration Ratio (%)': dcr,
#             'Diagonal Total Score'              : sum(diagonal_scores),
#             'Matrix Total Score'                : sum(sum(r) for r in sorted_matrix),
#             f'Matched (score ≥ {threshold})'   : sum(1 for s in diagonal_scores if s >= threshold),
#         })

#         # Bảng chi tiết từng ô chéo
#         for rank, ((sub, comm_id), score) in enumerate(zip(diagonal_pairs, diagonal_scores), 1):
#             detail_rows.append({
#                 'Algorithm'     : algo.capitalize(),
#                 'Diagonal Rank' : rank,
#                 'Community ID'  : comm_id,
#                 'Subreddit'     : sub,
#                 'Score (/20)'   : score,
#                 'Match?'        : '✅' if score >= threshold else '❌',
#             })

#     if not summary_rows:
#         print("❌ Không có dữ liệu để đánh giá.")
#         return

#     # ==========================================
#     # 5. THÊM CỘT RANK VÀO BẢNG TỔNG HỢP
#     # ==========================================
#     df_summary = pd.DataFrame(summary_rows)

#     # Rank tổng hợp: trung bình thứ hạng của 3 metrics (DDS, MCR, DCR cao hơn = tốt hơn)
#     for col in ['DDS – Diagonal Dominance Score (%)',
#                 'MCR – Match Coverage Rate (%)',
#                 'DCR – Diagonal Concentration Ratio (%)']:
#         df_summary[f'_rank_{col}'] = df_summary[col].rank(ascending=False)

#     rank_cols = [c for c in df_summary.columns if c.startswith('_rank_')]
#     df_summary['Overall Rank'] = (
#         df_summary[rank_cols].mean(axis=1).rank(method='min').astype(int)
#     )
#     df_summary.drop(columns=rank_cols, inplace=True)
#     df_summary.sort_values('Overall Rank', inplace=True)

#     df_detail = pd.DataFrame(detail_rows)

#     # ==========================================
#     # 6. IN KẾT QUẢ
#     # ==========================================
#     print(f"\n{'='*65}")
#     print("📊  BẢNG SO SÁNH HIỆU QUẢ THUẬT TOÁN (3 METRICS)")
#     print(f"{'='*65}")
#     print(f"  Threshold MCR: score ≥ {threshold}/20 từ trùng\n")
#     display(df_summary)

#     print(f"\n{'='*65}")
#     print("📋  BẢNG CHI TIẾT TỪNG Ô CHÉO")
#     print(f"{'='*65}")
#     display(df_detail)

#     # ==========================================
#     # 7. VẼ BAR CHART SO SÁNH 3 METRICS
#     # ==========================================
#     algos = df_summary['Algorithm'].tolist()
#     dds_vals = df_summary['DDS – Diagonal Dominance Score (%)'].tolist()
#     mcr_vals = df_summary['MCR – Match Coverage Rate (%)'].tolist()
#     dcr_vals = df_summary['DCR – Diagonal Concentration Ratio (%)'].tolist()

#     fig = go.Figure()
#     fig.add_trace(go.Bar(name='DDS – Diagonal Dominance Score (%)', x=algos, y=dds_vals,
#                          marker_color='#3B82F6', text=[f"{v:.1f}%" for v in dds_vals],
#                          textposition='outside'))
#     fig.add_trace(go.Bar(name='MCR – Match Coverage Rate (%)',      x=algos, y=mcr_vals,
#                          marker_color='#10B981', text=[f"{v:.1f}%" for v in mcr_vals],
#                          textposition='outside'))
#     fig.add_trace(go.Bar(name='DCR – Diagonal Concentration Ratio (%)', x=algos, y=dcr_vals,
#                          marker_color='#F59E0B', text=[f"{v:.1f}%" for v in dcr_vals],
#                          textposition='outside'))

#     fig.update_layout(
#         title='So sánh hiệu quả thuật toán Community Detection (3 Metrics)',
#         xaxis_title='Algorithm',
#         yaxis_title='Score (%)',
#         yaxis_range=[0, 115],
#         barmode='group',
#         legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
#         width=800, height=500
#     )
#     fig.show()

#     # ==========================================
#     # 8. LƯU FILE
#     # ==========================================
#     os.makedirs("outputs", exist_ok=True)
#     summary_path = "outputs/algorithm_evaluation_summary.csv"
#     detail_path  = "outputs/algorithm_evaluation_detail.csv"
#     df_summary.to_csv(summary_path, index=False)
#     df_detail.to_csv(detail_path,   index=False)

#     print(f"\n💾 Đã lưu bảng tổng hợp : {summary_path}")
#     print(f"💾 Đã lưu bảng chi tiết  : {detail_path}")


# # ==========================================
# # CẤU HÌNH VÀ CHẠY
# # ==========================================
# TRUTH_PATH      = "outputs/LDA/subreddit_topics_ground_truth.csv"
# COMMUNITIES_DIR = "outputs/LDA/"

# # Threshold: số từ trùng tối thiểu để coi là "khớp" (mặc định 3/20)
# THRESHOLD = 3

# evaluate_algorithms(TRUTH_PATH, COMMUNITIES_DIR, threshold=THRESHOLD)

import pandas as pd
import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# ==========================================
# 1. HÀM ĐẾM SỐ TỪ TRÙNG NHAU
# ==========================================
def count_overlap(words_a, words_b):
    if pd.isna(words_a) or pd.isna(words_b):
        return 0
    set_a = set([w.strip().lower() for w in str(words_a).split(',')])
    set_b = set([w.strip().lower() for w in str(words_b).split(',')])
    return len(set_a.intersection(set_b))


# ==========================================
# 2. XÂY DỰNG MA TRẬN VÀ ĐƯỜNG CHÉO
# ==========================================
def build_diagonal_matrix(subreddits, truth_keywords, communities, algo_keywords):
    """
    Trả về:
      - raw_matrix       : ma trận gốc [subreddit x community], chưa sắp xếp
      - sorted_matrix    : ma trận sau khi sắp xếp theo đường chéo
      - sorted_subs      : danh sách subreddit đã sắp xếp
      - sorted_communities: danh sách community đã sắp xếp
      - diagonal_scores  : list điểm tại ô chéo
      - diagonal_pairs   : list (subreddit_name, community_id) theo thứ tự đường chéo
    """
    # Sắp xếp cột theo thứ tự số
    sorted_comm_indices = sorted(range(len(communities)), key=lambda i: int(communities[i]))
    sorted_communities  = [communities[i] for i in sorted_comm_indices]
    sorted_algo_kw      = [algo_keywords[i] for i in sorted_comm_indices]

    # Tạo ma trận gốc (rows = subreddit, cols = community đã sort)
    matrix = []
    for sub_kw in truth_keywords:
        row = [count_overlap(sub_kw, comm_kw) for comm_kw in sorted_algo_kw]
        matrix.append(row)

    # Greedy diagonal assignment
    assigned       = set()
    diagonal_order = []

    for col_idx in range(len(sorted_communities)):
        best_row   = None
        best_score = -1
        for row_idx in range(len(subreddits)):
            if row_idx in assigned:
                continue
            if matrix[row_idx][col_idx] > best_score:
                best_score = matrix[row_idx][col_idx]
                best_row   = row_idx
        if best_row is not None:
            diagonal_order.append(best_row)
            assigned.add(best_row)

    # Subreddit chưa được assign → thêm vào cuối
    for row_idx in range(len(subreddits)):
        if row_idx not in assigned:
            diagonal_order.append(row_idx)

    sorted_matrix = [matrix[i] for i in diagonal_order]
    sorted_subs   = [subreddits[i] for i in diagonal_order]

    # Lấy điểm đường chéo: ô [i][i] với i < min(n_subs, n_comms)
    n_diag          = min(len(sorted_subs), len(sorted_communities))
    diagonal_scores = [sorted_matrix[i][i] for i in range(n_diag)]
    diagonal_pairs  = [(sorted_subs[i], sorted_communities[i]) for i in range(n_diag)]

    return matrix, sorted_matrix, sorted_subs, sorted_communities, diagonal_scores, diagonal_pairs


# ==========================================
# 3. TÍNH 3 METRICS
# ==========================================
def compute_metrics(sorted_matrix, diagonal_scores, threshold=3):
    """
    DDS – Diagonal Dominance Score
    MCR – Match Coverage Rate
    DCR – Diagonal Concentration Ratio
    """
    n_subs  = len(sorted_matrix)
    n_comms = len(sorted_matrix[0]) if sorted_matrix else 0
    n_diag  = len(diagonal_scores)

    # --- DDS ---
    total_score    = sum(sum(row) for row in sorted_matrix)
    diagonal_total = sum(diagonal_scores)
    dds = (diagonal_total / total_score * 100) if total_score > 0 else 0.0

    # --- MCR ---
    matched = sum(1 for s in diagonal_scores if s >= threshold)
    mcr = (matched / n_subs * 100)

    # --- DCR ---
    ratios = []
    for i in range(n_diag):
        row_max = max(sorted_matrix[i]) if sorted_matrix[i] else 0
        if row_max > 0:
            ratios.append(diagonal_scores[i] / row_max * 100)
        else:
            ratios.append(0.0)
    dcr = (sum(ratios) / len(ratios)) if ratios else 0.0

    return round(dds, 2), round(mcr, 2), round(dcr, 2)


# ==========================================
# 4. HÀM CHÍNH
# ==========================================
def evaluate_algorithms(truth_path, communities_dir, threshold=3):
    print("⏳ Đang tải dữ liệu Topic gốc (Subreddit Thực tế)...")
    if not os.path.exists(truth_path):
        print(f"❌ LỖI: Không tìm thấy file: {truth_path}")
        return

    df_truth       = pd.read_csv(truth_path)
    subreddits     = df_truth['Subreddit'].tolist()
    truth_keywords = df_truth['Top 20 Keywords (LDA)'].tolist()

    algorithms   = ['leiden', 'louvain', 'spectral']
    summary_rows = []
    detail_rows  = []

    for algo in algorithms:
        comm_path = os.path.join(communities_dir, f"community_topics_{algo}.csv")
        if not os.path.exists(comm_path):
            print(f"⚠️ Bỏ qua {algo.upper()}: không tìm thấy file {comm_path}")
            continue

        df_algo       = pd.read_csv(comm_path)
        communities   = df_algo['Community_ID'].tolist()
        algo_keywords = df_algo['Top 20 Keywords (LDA)'].tolist()

        print(f"  ✅ Đang tính metrics cho {algo.upper()} "
              f"({len(communities)} communities × {len(subreddits)} subreddits)...")

        (raw_matrix, sorted_matrix,
         sorted_subs, sorted_comms,
         diagonal_scores, diagonal_pairs) = build_diagonal_matrix(
            subreddits, truth_keywords, communities, algo_keywords
        )

        dds, mcr, dcr = compute_metrics(sorted_matrix, diagonal_scores, threshold)

        summary_rows.append({
            'Algorithm'                              : algo.capitalize(),
            'Num Communities'                        : len(communities),
            'DDS – Diagonal Dominance Score (%)'    : dds,
            'MCR – Match Coverage Rate (%)'         : mcr,
            'DCR – Diagonal Concentration Ratio (%)': dcr,
            'Diagonal Total Score'                   : sum(diagonal_scores),
            'Matrix Total Score'                     : sum(sum(r) for r in sorted_matrix),
            f'Matched (score ≥ {threshold})'        : sum(1 for s in diagonal_scores if s >= threshold),
        })

        for rank, ((sub, comm_id), score) in enumerate(zip(diagonal_pairs, diagonal_scores), 1):
            detail_rows.append({
                'Algorithm'     : algo.capitalize(),
                'Diagonal Rank' : rank,
                'Community ID'  : comm_id,
                'Subreddit'     : sub,
                'Score (/20)'   : score,
                'Match?'        : '✅' if score >= threshold else '❌',
            })

    if not summary_rows:
        print("❌ Không có dữ liệu để đánh giá.")
        return

    # ==========================================
    # 5. THÊM CỘT RANK VÀO BẢNG TỔNG HỢP
    # ==========================================
    df_summary = pd.DataFrame(summary_rows)

    for col in ['DDS – Diagonal Dominance Score (%)',
                'MCR – Match Coverage Rate (%)',
                'DCR – Diagonal Concentration Ratio (%)']:
        df_summary[f'_rank_{col}'] = df_summary[col].rank(ascending=False)

    rank_cols = [c for c in df_summary.columns if c.startswith('_rank_')]
    df_summary['Overall Rank'] = (
        df_summary[rank_cols].mean(axis=1).rank(method='min').astype(int)
    )
    df_summary.drop(columns=rank_cols, inplace=True)
    df_summary.sort_values('Overall Rank', inplace=True)

    df_detail = pd.DataFrame(detail_rows)

    # ==========================================
    # 6. IN KẾT QUẢ RA CONSOLE
    # ==========================================
    print(f"\n{'='*65}")
    print("📊  BẢNG SO SÁNH HIỆU QUẢ THUẬT TOÁN (3 METRICS)")
    print(f"{'='*65}")
    print(f"  Threshold MCR: score ≥ {threshold}/20 từ trùng\n")
    print(df_summary.to_string(index=False))

    print(f"\n{'='*65}")
    print("📋  BẢNG CHI TIẾT TỪNG Ô CHÉO")
    print(f"{'='*65}")
    print(df_detail.to_string(index=False))

    # ==========================================
    # 7. LƯU FILE CSV
    # ==========================================
    os.makedirs("outputs", exist_ok=True)
    summary_path = "outputs/algorithm_evaluation_summary.csv"
    detail_path  = "outputs/algorithm_evaluation_detail.csv"
    df_summary.to_csv(summary_path, index=False)
    df_detail.to_csv(detail_path,   index=False)

    print(f"\n💾 Đã lưu bảng tổng hợp : {summary_path}")
    print(f"💾 Đã lưu bảng chi tiết  : {detail_path}")


# ==========================================
# CẤU HÌNH VÀ CHẠY
# ==========================================
if __name__ == "__main__":
    TRUTH_PATH      = "outputs/LDA/subreddit_topics_ground_truth.csv"
    COMMUNITIES_DIR = "outputs/LDA/"
    THRESHOLD       = 3

    evaluate_algorithms(TRUTH_PATH, COMMUNITIES_DIR, threshold=THRESHOLD)