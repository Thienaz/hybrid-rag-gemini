"""
=============================================================================
CHUẨN BỊ CÁC DỮ LIỆU ĐÁNH GIÁ RAG (DATA PREPARATION)
Mục đích: Tự động trích xuất văn bản từ PDFs trong /data, dùng AI sinh câu hỏi 
          và tóm tắt văn bản tham chiếu từ ngữ cảnh, chuẩn bị cho file evaluate_rag.py.

Tạo ra:
  - data_test.txt (Gồm các delimiter để RAG hiểu cấu trúc tài liệu)
  - qa_dataset.csv (Câu hỏi chuẩn + Đáp án chuẩn)
Tác giả: Võ Huỳnh Thiên - B2203472
=============================================================================
"""

import os
import sys
import re
import time
import pandas as pd
import google.generativeai as genai
from pathlib import Path
from dotenv import load_dotenv

# Thêm thư mục gốc vào sys.path để import được các module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Giả sử bạn đã có 2 module này trong dự án
try:
    from utils.file_handler import FileHandler
    from utils.logger import logger
except ImportError:
    print("Cảnh báo: Không tìm thấy module utils.file_handler hoặc utils.logger.")

# =====================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN TỚI DỮ LIỆU
# =====================================================================
DATA_DIR = Path("data")
OUTPUT_TXT = Path("data_test.txt")
OUTPUT_CSV = Path("qa_dataset.csv")

if not DATA_DIR.exists():
    print(f"❌ Không tìm thấy thư mục '{DATA_DIR}'. Vui lòng tạo thư mục và bỏ PDF vào đó.")
    exit()

pdf_files = list(DATA_DIR.glob("*.pdf"))
if not pdf_files:
    print(f"❌ Không tìm thấy file PDF nào trong '{DATA_DIR}'. Vui lòng kiểm tra lại.")
    exit()

path_str = "\n".join([f"- {f.name}" for f in pdf_files])
print(f"Đã tìm thấy: \n{path_str}")

# =====================================================================
# 2. ĐỌC VÀ BÓC TÁCH VĂN BẢN TỪ PDF, LƯU KÈM DELIMITER
# =====================================================================
print("\n⏳ Đang bóc tách văn bản và chèn delimiter (Để RAG hiểu cấu trúc tài liệu)...")
combined_text = ""
all_clean_sentences = [] # Danh sách tổng hợp tất cả các câu từ mọi file PDF

for pdf_path in pdf_files:
    print(f"   Đang đọc: {pdf_path.name}...")
    try:
        # Đọc file nhị phân
        with open(pdf_path, "rb") as f:
            file_bytes = f.read()
        
        # Bóc tách văn bản thô (loại khoảng trắng thừa)
        text = file_bytes.decode("utf-8", errors="ignore")
        text = re.sub(r'\n{3,}', '\n', text)
        
        # Chia nhỏ câu dựa trên dấu câu cơ bản (tạo các chunk nhỏ)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        clean_sentences = [s.strip() for s in sentences if len(s.split()) > 10]
        
        # Lưu vào mảng tổng để đưa cho Gemini xử lý
        all_clean_sentences.extend(clean_sentences)
        
        # Ghép vào chuỗi lớn, chèn delimiter chuẩn của engine
        combined_text += f"\n--- BẮT ĐẦU TÀI LIỆU: {pdf_path.stem} ---\n"
        combined_text += "\n".join(clean_sentences) + "\n"
        
    except Exception as e:
        print(f"   ❌ Lỗi khi đọc {pdf_path.name}: {e}")

# Lưu data_test.txt
with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    f.write(combined_text)
print(f"\n✅ Đã lưu tài liệu tổng hợp (có delimiter) tại: {OUTPUT_TXT}")


# =====================================================================
# 3. TẠO Q&A BẰNG AI (GEMINI) TỪ CÁC ĐOẠN VĂN BẢN
# =====================================================================
print("\n⏳ Bắt đầu sinh Q&A bằng AI từ các đoạn văn bản (Để đảm bảo RAG trả lời sát nghĩa)...")
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("❌ Không tìm thấy GOOGLE_API_KEY trong file .env. Vui lòng kiểm tra lại.")
    api_key = "YOUR_API_KEY_HERE" 

genai.configure(api_key=api_key)
qa_pairs = []

# Phân chia lại thành các đoạn nhỏ để không vượt quá giới hạn Context
current_chunk = []
CHUNK_SENTENCE_LIMIT = 5 # Gộp 5 câu/1 lô prompt để tránh tràn RAM/Context quá dài

# Khởi tạo model
model = genai.GenerativeModel("gemini-2.5-flash")

for sentence in all_clean_sentences:
    current_chunk.append(sentence)
    
    # Khi đủ câu, tạo Q&A
    if len(current_chunk) >= CHUNK_SENTENCE_LIMIT:
        chunk_text = " ".join(current_chunk)
        
        # 1. Sinh câu hỏi dựa trên đoạn
        q_prompt = f"""Dựa vào đoạn tài liệu sau, hãy tạo MỘT CÂU HỎI học thuật ngắn gọn để kiểm tra kiến thức:
---
{chunk_text}
---
Chỉ xuất ra 1 câu hỏi. KHÔNG thêm giải thích gì thêm."""

        # 2. Sinh câu trả lời tham chiếu
        a_prompt = f"""Dựa vào đoạn tài liệu sau, hãy trả lời ngắn gọn và chính xác CHỈ DỰA VÀO đoạn văn bản:
---
{chunk_text}
---
Trả lời ngắn gọn, không thêm văn bản dư thừa."""

        # Gọi API
        try:
            # Gọi API tạo câu hỏi
            q_res = model.generate_content(q_prompt)
            q_text = q_res.text.strip()
            
            # Gọi API tạo câu trả lời
            a_res = model.generate_content(a_prompt)
            a_text = a_res.text.strip()
            
            # Làm sạch câu hỏi (nếu AI thêm "Câu hỏi:" vào đầu)
            q_clean = re.sub(r'^.*?Câu hỏi:\s*', '', q_text).strip()
            
            qa_pairs.append({
                "Câu hỏi": q_clean,
                "Câu trả lời chuẩn": a_text
            })
            
            current_chunk = [] # Reset đoạn
            time.sleep(4) # Chờ 4s tránh Rate Limit API miễn phí (15 RPM)
            
        except Exception as e:
            print(f"   ❌ Lỗi khi tạo Q&A cho 1 đoạn: {e}")
            current_chunk = []

# Xử lý đoạn text còn dư (nếu số câu không chia hết cho 5)
# (Phần này có thể thêm nếu cần thiết, nhưng với script cơ bản thì bỏ qua cũng không sao)

# Chuyển đổi thành DataFrame và xuất CSV
df_qa = pd.DataFrame(qa_pairs)

# Sửa lỗi string literal ở đây (dùng utf-8-sig để Excel có thể đọc tiếng Việt không bị lỗi font)
df_qa.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

print(f"\n✅ Đã lưu {len(df_qa)} cặp Q&A tại: {OUTPUT_CSV}")
print("\n👉 Bây giờ bạn có thể chạy: python scripts/evaluate_rag.py")