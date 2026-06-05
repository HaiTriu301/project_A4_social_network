import pandas as pd
import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def count_community_members():
    # Danh sách 4 thuật toán bạn đã chạy
    algorithms = ['louvain', 'leiden', 'spectral']
    
    print("Bắt đầu thống kê số lượng thành viên...")
    
    for algo in algorithms:
        input_file = f"outputs/communities/{algo}_communities.csv"
        output_file = f"outputs/count_communities/{algo}_community_counts.csv"

        if os.path.exists(input_file):
            df = pd.read_csv(input_file)
            # Đếm số lượng thành viên (user) cho mỗi nhóm (community_id)
            counts = df['community_id'].value_counts().reset_index()
            counts.columns = ['community_id', 'member_count']
            
            # Lưu ra file CSV
            counts.to_csv(output_file, index=False)
            print(f"✔️ Đã lưu thống kê {algo} tại: {output_file} (Tổng số cộng đồng: {len(counts)})")
        else:
            print(f"❌ Không tìm thấy file {input_file}. Vui lòng chạy thuật toán trước.")

if __name__ == "__main__":
    count_community_members()