import pandas as pd
import re
import os
import gc
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import sys
import io
from sklearn.feature_extraction import text 

# Lấy danh sách stopwords mặc định của sklearn
my_stop_words = list(text.ENGLISH_STOP_WORDS)

# Thêm các từ "vô thưởng vô phạt" mà bạn quan sát thấy
extra_words = ['good', 'great', 'really', 'just', 'think', 'game', 'like', 'time', 'people', 've', 'don', 'games',
               'want', 'play', 'make', 'know', 'playing', 'players', 'need', 'use', 'post', 'subreddit', 'way', 'message']
my_stop_words.extend(extra_words)

# Khi khởi tạo Vectorizer, hãy truyền danh sách này vào
vectorizer = CountVectorizer(stop_words=my_stop_words, max_df=0.8, min_df=5)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ==========================================
# 1. HÀM LÀM SẠCH VĂN BẢN
# ==========================================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ==========================================
# 2. HÀM TỰ ĐỘNG CHẠY 3 THUẬT TOÁN
# ==========================================
def batch_analyze_topics(raw_data_path, communities_dir, output_dir, n_words=10, min_comments=50):
    print("1. Đang tải dữ liệu gốc (Chỉ load 1 lần để tối ưu tốc độ)...")
    df_raw = pd.read_csv(raw_data_path, usecols=['author_fullname', 'body', 'subreddit'])
    df_raw = df_raw.dropna(subset=['author_fullname', 'body'])
    
    # Danh sách 3 thuật toán bạn muốn chạy
    algorithms = ['leiden', 'louvain', 'spectral']
    
    # Tạo thư mục output nếu chưa có
    os.makedirs(output_dir, exist_ok=True)

    for algo in algorithms:
        print(f"\n{'='*60}")
        print(f"🚀 ĐANG XỬ LÝ THUẬT TOÁN: {algo.upper()}")
        print(f"{'='*60}")
        
        comm_path = os.path.join(communities_dir, f"{algo}_communities.csv")
        out_path = os.path.join(output_dir, f"community_topics_{algo}.csv")
        
        if not os.path.exists(comm_path):
            print(f"❌ Không tìm thấy file: {comm_path}. Bỏ qua!")
            continue
            
        print("   - Đang tải và ghép nối dữ liệu...")
        df_comm = pd.read_csv(comm_path)
        if 'user' in df_comm.columns:
            df_comm = df_comm.rename(columns={'user': 'author_fullname'})
            
        df_merged = pd.merge(df_raw, df_comm, on='author_fullname', how='inner')
        print(f"   -> Số bình luận hợp lệ để phân tích: {len(df_merged)}")
        
        print(f"   - Bắt đầu chạy LDA cho các cụm (>= {min_comments} comments)...")
        results = []
        grouped = df_merged.groupby('community_id')
        
        for comm_id, group in grouped:
            if len(group) < min_comments:
                continue
                
            cleaned_docs = group['body'].apply(clean_text)
            cleaned_docs = cleaned_docs[cleaned_docs != ""]
            
            if len(cleaned_docs) < min_comments:
                continue
                
            vectorizer = CountVectorizer(stop_words=my_stop_words, max_df=0.9, min_df=2)
            
            try:
                dtm = vectorizer.fit_transform(cleaned_docs)
                lda = LatentDirichletAllocation(n_components=1, random_state=42)
                lda.fit(dtm)
                
                feature_names = vectorizer.get_feature_names_out()
                top_words = [feature_names[i] for i in lda.components_[0].argsort()[-n_words:][::-1]]
                top_subreddit = group['subreddit'].value_counts().idxmax()
                
                results.append({
                    'Community_ID': comm_id,
                    'Subreddit_Đại_Diện': top_subreddit,
                    'Số lượng Comment': len(cleaned_docs),
                    'Top 20 Keywords (LDA)': ", ".join(top_words)
                })
                
                print(f"     ✔️ Xong Cụm {comm_id} (Đại diện: r/{top_subreddit} | {len(cleaned_docs)} comments)")
                
            except Exception as e:
                print(f"     ❌ Bỏ qua Cụm {comm_id} do lỗi dữ liệu chữ.")

        # Lưu file cho thuật toán hiện tại
        df_results = pd.DataFrame(results)
        if not df_results.empty:
            df_results = df_results.sort_values(by='Số lượng Comment', ascending=False)
            df_results.to_csv(out_path, index=False)
            print(f"\n   🎉 Đã lưu kết quả của {algo.upper()} tại: {out_path}")
        else:
            print(f"\n   ⚠️ {algo.upper()} không có cụm nào đủ điều kiện.")

        # --- TỐI ƯU RAM: Xóa dữ liệu của vòng lặp hiện tại ---
        del df_merged
        del df_comm
        gc.collect()

    print(f"\n{'='*60}")
    print("✅ ĐÃ HOÀN THÀNH CHẠY LDA CHO TOÀN BỘ 3 THUẬT TOÁN!")

if __name__ == "__main__":
    # ĐIỀN ĐƯỜNG DẪN CỦA BẠN VÀO ĐÂY
    RAW_DATA = "data/processed/RC_2026-01_filtered_final.csv" 
    COMMUNITIES_DIR = "outputs/communities/"
    OUTPUT_DIR = "outputs/LDA/"
    
    # Chạy hàm xử lý hàng loạt
    batch_analyze_topics(RAW_DATA, COMMUNITIES_DIR, OUTPUT_DIR, n_words=20, min_comments=50)