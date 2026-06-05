import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import SpectralClustering
import time
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def run_spectral(input_path, output_path, n_clusters=20):
    print(f"1. Đang tải dữ liệu (Spectral Clustering - {n_clusters} cụm)...")
    start_time = time.time()
    df = pd.read_csv(input_path)
    
    print("2. Xây dựng ma trận thưa (Sparse Adjacency Matrix)...")
    # Lấy danh sách tất cả người dùng duy nhất
    users = pd.unique(df[['source', 'target']].values.ravel())
    user_to_id = {u: i for i, u in enumerate(users)}
    
    # Map sang index số
    src_idx = df['source'].map(user_to_id).values
    dst_idx = df['target'].map(user_to_id).values
    weights = df['weight'].values if 'weight' in df.columns else np.ones(len(df))
    
    # Khởi tạo ma trận thưa kích thước NxN
    N = len(users)
    adj_matrix = sp.csr_matrix((weights, (src_idx, dst_idx)), shape=(N, N))
    
    # Tạo đồ thị vô hướng bằng cách cộng ma trận chuyển vị
    adj_matrix = adj_matrix + adj_matrix.T
    
    print("3. Đang chạy Spectral Clustering (có thể mất vài phút)...")
    sc = SpectralClustering(
        n_clusters=n_clusters, 
        affinity='precomputed', # Báo cho mô hình biết ta truyền vào ma trận kề
        eigen_solver='arpack',  # Giải thuật tối ưu cho ma trận thưa
        random_state=42
    )
    labels = sc.fit_predict(adj_matrix)
    
    print("4. Đang lưu kết quả...")
    df_res = pd.DataFrame({'user': users, 'community_id': labels})
    df_res.to_csv(output_path, index=False)
    
    print(f"Hoàn thành trong {time.time() - start_time:.2f} giây. Đã lưu tại: {output_path}")

if __name__ == "__main__":
    # Bạn có thể thay đổi số lượng cụm (n_clusters) tùy ý
    run_spectral("data/graph/edges.csv", "outputs/communities/spectral_communities.csv", n_clusters=20)