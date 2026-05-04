# File: scripts/ablation_retrieval_gemini.py
import time
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from dotenv import load_dotenv

print("=" * 60)
print("📊 THỰC NGHIỆM ĐỐI CHỨNG: TF-IDF vs GEMINI DENSE EMBEDDING")
print("=" * 60)

# Cấu hình API Gemini
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# --- DỮ LIỆU GIẢ LẬP (Mô phỏng Chunks giáo trình) ---
corpus = [
    "Cơ sở dữ liệu quan hệ lưu trữ dữ liệu dưới dạng các bảng có liên kết với nhau.",
    "Khóa ngoại (Foreign Key) là một trường trong một bảng tham chiếu đến khóa chính của bảng khác.",
    "SQL Injection là một kỹ thuật tấn công chèn mã SQL độc hại vào input của người dùng.",
    "Tấn công DDoS làm quá tải hệ thống máy chủ bằng hàng ngàn truy vấn giả mạo.",
    "Mô hình Waterfall là phương pháp phát triển phần mềm tuần tự từng bước."
]
corpus = corpus * 20 # Nhân bản lên 100 chunks để đo thời gian TF-IDF

# --- CÂU HỎI THỬ NGHIỆM ---
queries = [
    "Khái niệm khóa ngoại là gì?",
    "Kỹ thuật SQL Injection là gì?"
] * 10 # 20 câu hỏi

print(f"📚 Số lượng Chunks tài liệu: {len(corpus)}")
print(f"❓ Số lượng câu hỏi test (TF-IDF): {len(queries)}\n")

# ==========================================
# 1. PHƯƠNG PHÁP TF-IDF (SPARSE RETRIEVAL - LOCAL)
# ==========================================
print("🚀 ĐANG CHẠY TF-IDF (SPARSE - XỬ LÝ LOCAL)...")
start_time_tfidf = time.time()

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

for query in queries:
    query_vec = vectorizer.transform([query])
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_5_idx = sim_scores.argsort()[-5:][::-1]

end_time_tfidf = time.time()
time_tfidf = (end_time_tfidf - start_time_tfidf) / len(queries)
print(f"✅ TF-IDF Xong! Thời gian trung bình: {time_tfidf:.4f} giây/câu\n")


# ==========================================
# 2. PHƯƠNG PHÁP GEMINI EMBEDDING (DENSE RETRIEVAL - API)
# ==========================================
print("🚀 ĐANG CHẠY GEMINI EMBEDDING (DENSE - GỌI API)...")

try:
    # 2.1 TỰ ĐỘNG DÒ TÌM MÔ HÌNH HỖ TRỢ EMBEDDING
    print("🔍 Đang dò tìm mô hình Embedding khả dụng...")
    valid_embedding_model = None
    for m in genai.list_models():
        if 'embedContent' in m.supported_generation_methods:
            valid_embedding_model = m.name
            break
            
    if not valid_embedding_model:
        raise ValueError("Không tìm thấy mô hình Embedding nào được hỗ trợ cho API Key này.")
        
    print(f"✅ Đã tự động chọn mô hình: {valid_embedding_model}\n")

    start_time_dense = time.time()

    # Encode tập corpus (Chỉ lấy 5 mẫu gốc để giảm API call)
    test_corpus = corpus[:5] 
    corpus_response = genai.embed_content(
        model=valid_embedding_model,
        content=test_corpus,
        task_type="retrieval_document"
    )
    corpus_embeddings = np.array(corpus_response['embedding'])

    # Tiết kiệm API: Chỉ test với 2 câu hỏi đầu tiên
    test_queries = queries[:2]
    
    for query in test_queries:
        query_response = genai.embed_content(
            model=valid_embedding_model,
            content=query,
            task_type="retrieval_query"
        )
        query_embedding = np.array(query_response['embedding']).reshape(1, -1)
        
        # Tính khoảng cách
        sim_scores = cosine_similarity(query_embedding, corpus_embeddings).flatten()
        top_idx = sim_scores.argsort()[::-1]

    end_time_dense = time.time()
    
    # Tính trung bình dựa trên đúng 2 câu đã gọi API
    time_dense = (end_time_dense - start_time_dense) / len(test_queries) 
    print(f"✅ Gemini Dense Xong! Thời gian trung bình: {time_dense:.4f} giây/câu")

except Exception as e:
    print(f"❌ Lỗi khi gọi API Gemini Embedding: {e}")