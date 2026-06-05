# import networkx as nx

# G_pos = nx.read_gexf("outputs/gexf/graph_leiden_pos.gexf")

# # In thử 5 node đầu tiên để xem có tọa độ không
# for i, (node, data) in enumerate(G_pos.nodes(data=True)):
#     print(node, data)
#     if i >= 4:
#         break

import csv

def count_csv_rows(file_path, has_header=True):
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        
        # Nếu có dòng tiêu đề (header), bỏ qua dòng đầu tiên
        if has_header:
            next(reader, None) 
            
        # Đếm số dòng dữ liệu còn lại
        row_count = sum(1 for row in reader)
        return row_count

# Sử dụng hàm
file_path = "data/processed/RC_2026-01_filtered_final.csv"  # Thay bằng đường dẫn file của bạn
total_rows = count_csv_rows(file_path, has_header=True)
print(f"So dong: 370120 {total_rows}")