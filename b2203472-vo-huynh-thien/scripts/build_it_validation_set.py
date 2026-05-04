"""
=============================================================================
TẠO TẬP VALIDATION CHUYÊN NGÀNH IT TỪ GIÁO TRÌNH PDF (ORACLE DATASET)
Mục đích: Đọc PDF trong /data, tạo các cặp (Văn bản, Tóm tắt Oracle).
Tóm tắt Oracle = Các câu chứa mật độ từ khóa IT cao nhất.
Từ khóa IT được nạp ĐÚNG từ artifacts/it_dictionary.txt
Dataset đầu ra: it_domain_val_dataset.csv
Tác giả: Võ Huỳnh Thiên - B2203472
=============================================================================
"""

import os
import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.text_processor import TextProcessor

# Cấu hình
DATA_DIR = Path("data")
DICT_FILE = Path("artifacts/it_dictionary.txt") # Đổi sang đọc file từ điển của bạn
OUTPUT_FILE = "it_domain_val_dataset.csv"
TARGET_SAMPLES = 200

print("=" * 60)
print("📚 BẮT ĐẦU XÂY DỰNG TẬP VALIDATION TỪ GIÁO TRÌNH PDF")
print("=" * 60)

# 1. Tải toàn bộ từ khóa IT từ file it_dictionary.txt
all_it_terms = set()
if DICT_FILE.exists():
    with open(DICT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            term = line.strip().lower()
            if term:
                # ⚠️ QUAN TRỌNG: Thay khoảng trắng bằng gạch dưới
                # Vì underthesea tokenize "cơ sở dữ liệu" thành "cơ_sở_dữ_liệu"
                term = term.replace(" ", "_")
                all_it_terms.add(term)
    print(f"✅ Đã tải {len(all_it_terms)} thuật ngữ chuyên ngành từ {DICT_FILE.name}.")
else:
    print(f"❌ Không tìm thấy file {DICT_FILE}. Vui lòng chạy build_dictionary.py trước.")
    sys.exit(1)

# 2. Đọc toàn bộ file PDF trong thư mục /data
pdf_files = list(DATA_DIR.glob("*.pdf"))
if not pdf_files:
    print(f"❌ Không tìm thấy file PDF nào trong thư mục: {DATA_DIR.absolute()}")
    sys.exit(1)

print(f"📄 Tìm thấy {len(pdf_files)} file PDF. Đang bóc tách văn bản...")

import fitz  # PyMuPDF
all_text = ""
for pdf_path in pdf_files:
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text = page.get_text()
            text = re.sub(r'[^\w\s\.!,;:\?\-]', ' ', text)
            all_text += text + "\n"
        print(f"   → Đọc xong: {pdf_path.name}")
    except Exception as e:
        print(f"   ❌ Lỗi đọc {pdf_path.name}: {e}")

# 3. Tách văn bản thành các câu
sentences = re.split(r'(?<=[.!?])\s+', all_text)
clean_sentences = [
    s.strip() for s in sentences 
    if len(s.split()) > 15 and not re.match(r'^[\d\s\.\-]+$', s.strip())
]
print(f"✅ Đã tách được {len(clean_sentences)} câu hợp lệ từ giáo trình.")

# 4. Nhóm câu thành từng chunk (Mỗi chunk ~5 câu để làm 1 văn bản)
CHUNK_SIZE = 5
chunks = [clean_sentences[i:i + CHUNK_SIZE] for i in range(0, len(clean_sentences), CHUNK_SIZE)]

processor = TextProcessor()
rows = []

print(f"🔄 Đang phân tích và tạo cặp Data (Text -> Summary Oracle)...")
for chunk in tqdm(chunks, desc="Xử lý chunks"):
    if len(chunk) < 3: continue
    chunk_text = " ".join(chunk)
    
    # Chấm điểm từng câu trong chunk
    scored_sentences = []
    for sent in chunk:
        tokens = set(processor.tokenize_vietnamese(sent).split())
        overlap = tokens.intersection(all_it_terms)
        scored_sentences.append((len(overlap), sent))
    
    # Sắp xếp giảm dần theo số từ IT
    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    
    # Lấy top 2 câu có nhiều từ IT nhất làm "Bản tóm tắt tham chiếu" (Oracle)
    oracle_summary = " ".join([s[1] for s in scored_sentences[:2]])
    
    # Chỉ lấy mẫu nếu bản tóm tắt thực sự có chứa từ IT
    if scored_sentences[0][0] > 0:
        rows.append({
            "text": chunk_text,
            "summary": oracle_summary
        })

# 5. Trộn ngẫu nhiên và cắt đúng bằng TARGET_SAMPLES
import random
random.seed(42)
random.shuffle(rows)
final_rows = rows[:TARGET_SAMPLES]

df = pd.DataFrame(final_rows)
df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

print("\n" + "=" * 60)
print(f"🎉 HOÀN TẤT!")
print(f"📁 Đã lưu tập dữ liệu chuyên ngành IT tại: {OUTPUT_FILE}")
print(f"📊 Tổng số mẫu: {len(final_rows)}")
print("👉 Bây giờ hãy chạy file: python scripts/tune_parameters_it_domain.py")
print("=" * 60)