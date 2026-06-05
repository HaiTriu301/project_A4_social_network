import pandas as pd
from collections import Counter
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def build_custom_graph(input_path, output_path):
    print("1. Đang tải dữ liệu...")
    df = pd.read_csv(input_path)    

    # Tập hợp tất cả user hợp lệ = những người có trong file raw (có comment)
    valid_commenters = set(df["author_fullname"].dropna())

    print("2. Tạo từ điển ánh xạ: comment_id (name) -> author_fullname...")
    # Dùng để tra cứu: comment t1_xxx do ai viết
    id_to_author = dict(zip(df["name"], df["author_fullname"]))

    # ==========================================
    # NHÓM 1: B -> A (comment trực tiếp vào post)
    # parent_id bắt đầu bằng t3_
    # ==========================================
    print("3. Lọc tương tác B -> A (reply post)...")
    df_post_replies = df[df["parent_id"].str.startswith("t3_")].copy()

    edges_post = df_post_replies[["author_fullname", "author_id"]].copy()
    edges_post.columns = ["source", "target"]

    # Chỉ giữ cạnh khi A (post author) cũng có trong file raw
    edges_post = edges_post[edges_post["target"].isin(valid_commenters)]

    print(f"   -> {len(edges_post)} cạnh B-A (sau khi lọc post author có trong raw)")

    # ==========================================
    # NHÓM 2: C -> B (reply comment)
    # parent_id bắt đầu bằng t1_
    # ==========================================
    print("4. Lọc tương tác C -> B (reply comment)...")
    df_comment_replies = df[df["parent_id"].str.startswith("t1_")].copy()

    # Tra cứu tác giả của comment cha
    df_comment_replies["target"] = df_comment_replies["parent_id"].map(id_to_author)

    # Bỏ những dòng không tìm được comment cha trong file
    df_comment_replies = df_comment_replies.dropna(subset=["target"])

    edges_comment = df_comment_replies[["author_fullname", "target"]].copy()
    edges_comment.columns = ["source", "target"]

    print(f"   -> {len(edges_comment)} cạnh C-B")

    # ==========================================
    # KẾT HỢP VÀ DỌN DẸP
    # ==========================================
    print("5. Hợp nhất và loại self-loop...")
    df_edges = pd.concat([edges_post, edges_comment], ignore_index=True)
    df_edges = df_edges[df_edges["source"] != df_edges["target"]]

    # ==========================================
    # LỌC NGƯỠNG TƯƠNG TÁC >= 3
    # ==========================================
    print("6. Áp dụng ngưỡng tương tác >= 3...")
    counts = Counter(df_edges["source"]) + Counter(df_edges["target"])
    valid_users = {u for u, c in counts.items() if c >= 3}

    df_edges = df_edges[
        df_edges["source"].isin(valid_users) &
        df_edges["target"].isin(valid_users)
    ]

    # ==========================================
    # TÍNH TRỌNG SỐ
    # ==========================================
    print("7. Tính trọng số...")
    df_edges = df_edges.groupby(["source", "target"]).size().reset_index(name="weight")

    print(f"Tổng số cạnh: {len(df_edges)}")
    print(f"Tổng số node: {len(pd.unique(df_edges[['source', 'target']].values.ravel()))}")

    df_edges.to_csv(output_path, index=False)
    print(f"Đã lưu tại: {output_path}")

    return df_edges

if __name__ == "__main__":
    build_custom_graph(
        "data/processed/RC_2026-01_filtered_final.csv",
        "data/graph/edges.csv"
    )


# def build_graph(input_path, output_path):
#     print("Loading cleaned data...")
#     df = pd.read_csv(input_path)

#     print("Creating comment → author map...")
#     id_to_author = dict(zip(df["name"], df["author_fullname"]))

#     print("Filtering reply comments...")
#     df_reply = df[df["parent_id"].str.startswith("t1_")]

#     print("Mapping parent authors...")
#     df_reply["parent_author"] = df_reply["parent_id"].map(id_to_author)

#     df_edges = df_reply.dropna(subset=["parent_author"])

#     print("Removing self-loops...")
#     df_edges = df_edges[
#         df_edges["author_fullname"] != df_edges["parent_author"]
#     ]

#     print("Applying interaction threshold ≥ 3...")
#     counts = Counter(df_edges["author_fullname"]) + Counter(df_edges["parent_author"])

#     valid_users = {u for u, c in counts.items() if c >= 3} #3

#     df_edges = df_edges[
#         df_edges["author_fullname"].isin(valid_users) &
#         df_edges["parent_author"].isin(valid_users)
#     ]

#     print("Final edges:", df_edges.shape)

#     df_edges[["author_fullname", "parent_author"]].to_csv(output_path, index=False)

#     print("Saved edges!")

# if __name__ == "__main__":
#     build_graph(
#         "data/processed/RC_2026-01_filtered_final.csv",
#         "data/graph/edges.csv"
#     )
