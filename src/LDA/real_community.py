import pandas as pd
import re
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
# 1. HÀM LÀM SẠCH VĂN BẢN (Tối ưu tốc độ)
# ==========================================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Xóa link
    text = re.sub(r'\W', ' ', text) # Xóa ký tự đặc biệt
    text = re.sub(r'\s+', ' ', text).strip() # Xóa khoảng trắng thừa
    return text

# ==========================================
# 2. LOGIC CHẠY LDA TỐI ƯU CHO FILE LỚN
# ==========================================
def analyze_real_subreddits(input_csv, output_csv, n_words=10, min_comments=50):
    print("1. Đang tải dữ liệu (Chỉ lấy cột 'subreddit' và 'body' để tiết kiệm RAM)...")
    # usecols giúp tiết kiệm hàng GB RAM nếu file của bạn có nhiều cột thừa
    df = pd.read_csv(input_csv, usecols=['subreddit', 'body'])
    
    # Loại bỏ các dòng bị rỗng
    df = df.dropna(subset=['subreddit', 'body'])
    
    print(f"   -> Đã tải {len(df)} bình luận.")
    print(f"2. Bắt đầu phân tích LDA cho các Subreddit có >= {min_comments} comments...")
    
    results = []
    
    # Gom nhóm theo Subreddit
    grouped = df.groupby('subreddit')
    
    for subreddit, group in grouped:
        # Bỏ qua các subreddit quá nhỏ
        if len(group) < min_comments:
            continue
            
        # Làm sạch văn bản CHỈ cho subreddit hiện tại (Tiết kiệm RAM)
        cleaned_docs = group['body'].apply(clean_text)
        cleaned_docs = cleaned_docs[cleaned_docs != ""] # Bỏ các comment chỉ chứa link/icon
        
        # Kiểm tra lại số lượng sau khi làm sạch
        if len(cleaned_docs) < min_comments:
            continue
            
        # Biến đổi văn bản & Xóa Stopwords tiếng Anh
        vectorizer = CountVectorizer(stop_words=my_stop_words, max_df=0.8, min_df=5)
        
        try:
            dtm = vectorizer.fit_transform(cleaned_docs)
            
            # Chạy LDA lấy 1 Topic tổng quát nhất
            lda = LatentDirichletAllocation(n_components=1, random_state=42)
            lda.fit(dtm)
            
            # Trích xuất từ khóa
            feature_names = vectorizer.get_feature_names_out()
            top_word_indices = lda.components_[0].argsort()[-n_words:][::-1]
            top_words = [feature_names[i] for i in top_word_indices]
            
            results.append({
                'Subreddit': subreddit,
                'Số lượng Comment': len(cleaned_docs),
                'Top 20 Keywords (LDA)': ", ".join(top_words)
            })
            
            print(f"✔️ Đã phân tích xong r/{subreddit} ({len(cleaned_docs)} comments)")
            
        except Exception as e:
            # Bắt lỗi nếu toàn từ khóa rác / stopword không thể phân tích
            print(f"❌ Bỏ qua r/{subreddit} do dữ liệu chữ không đủ tốt.")

    print("\n3. Đang lưu kết quả...")
    # Sắp xếp các Subreddit phổ biến nhất lên đầu
    df_results = pd.DataFrame(results).sort_values(by='Số lượng Comment', ascending=False)
    df_results.to_csv(output_csv, index=False)
    
    print(f"🎉 Hoàn thành! Đã lưu Topic của các Subreddit tại: {output_csv}")

if __name__ == "__main__":
    # Điền đường dẫn file dữ liệu GỐC của bạn vào đây
    # Ví dụ: data/processed/RC_2026-01_filtered_final.csv
    INPUT_DATA = "data/processed/RC_2026-01_filtered_final.csv" 
    OUTPUT_REPORT = "outputs/LDA/subreddit_topics_ground_truth.csv"
    
    # Bạn có thể tăng min_comments = 100 hoặc 500 nếu dữ liệu của bạn quá khổng lồ
    analyze_real_subreddits(INPUT_DATA, OUTPUT_REPORT, n_words=20, min_comments=5)