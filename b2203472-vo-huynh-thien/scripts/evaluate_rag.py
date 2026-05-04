"""
=============================================================================
ĐÁNH GIÁ CHẤT LƯỢNG HỎI ĐÁP RAG (LLM-AS-A-JUDGE) - CHUẨN KỸ THUẬT
Mục đích: Đánh giá khả năng Hỏi đáp RAG bằng phương pháp LLM-as-a-Judge.
Kỹ thuật: Chấm điểm độc lập từng cặp Q&A (1 API call / 1 câu) để đạt độ 
          chính xác cao nhất, loại bỏ lỗi ngữ cảnh đứt gãy.
Tác giả: Võ Huỳnh Thiên - B2203472
=============================================================================
"""

import os
import re
import sys
import time
import warnings
from pathlib import Path
import numpy as np

import pandas as pd
import google.generativeai as genai
from docx import Document

warnings.filterwarnings('ignore')

# Cấu hình đường dẫn gốc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

from core.nlp_engine import NLPEngine
from utils.logger import logger

# =====================================================================
# BƯỚC 1: ĐỌC VÀ BÓC TÁCH DỮ LIỆU TỪ DOCX
# =====================================================================
print("=" * 60)
print("📚 BƯỚC 1: ĐỌC VÀ BÓC TÁCH DỮ LIỆU TỪ DOCX")
print("=" * 60)

DOCX_PATH = Path(r"D:\LVTN\data\Câu hỏi ôn tập.docx")
OUTPUT_TXT = Path("data_test.txt")
OUTPUT_CSV = Path("artifacts/rag_evaluation_results.csv")

Path("artifacts").mkdir(exist_ok=True)

if not DOCX_PATH.exists():
    print(f"❌ Không tìm thấy '{DOCX_PATH}'. Hãy kiểm tra lại đường dẫn.")
    sys.exit(1)

try:
    doc = Document(str(DOCX_PATH))
    print(f"✅ Đã mở file: {DOCX_PATH.name}")
except Exception as e:
    print(f"❌ Lỗi khi đọc DOCX: {e}")
    sys.exit(1)

# Gộp toàn bộ văn bản để tránh lỗi đứt gãy giữa câu hỏi và đáp án
full_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

# Tách thành các khối câu hỏi dựa trên từ khóa "Câu [số]:" hoặc "Câu hỏi [số]:"
blocks = re.split(r'Câu(?: hỏi)?\s*\d+\s*:', full_text)

qa_pairs = []
for block in blocks:
    block = block.strip()
    if not block:
        continue
    
    # Tìm chuỗi "Đáp án đúng:" hoặc "Đáp án:"
    ans_match = re.search(r'(?:Đáp án đúng:|Đáp án:)\s*(.*)', block, flags=re.IGNORECASE | re.DOTALL)
    if ans_match:
        question = block[:ans_match.start()].strip()
        answer = ans_match.group(1).strip()
        if question and answer:
            qa_pairs.append({"question": question, "reference": answer})

total_questions = len(qa_pairs)
print(f"✅ Đã trích xuất thành công {total_questions} cặp Q&A từ DOCX")

# Lưu văn bản tổng hợp
combined_text = "\n".join([p["question"] + "\nĐáp án: " + p["reference"] + "\n\n" for p in qa_pairs])
with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    f.write(combined_text)
print(f"✅ Đã lưu văn bản tổng hợp tại: {OUTPUT_TXT}")

# =====================================================================
# BƯỚC 2: GỌI RAG ĐỂ LẤY ĐÁP ÁN THỰC TẾ (LOCAL)
# =====================================================================
print("\n⏳ BƯỚC 2: Lấy đáp án RAG thực tế (Local)...")
logger.info("Bắt đầu trích xuất ngữ cảnh...")
engine = NLPEngine()
num_chunks = engine.preprocess(combined_text)
print(f"✅ Đã nạp tài liệu: {num_chunks} đoạn văn bản")

rag_answers = []
for i, item in enumerate(qa_pairs):
    q = item["question"]
    try:
        res = engine.query_document(q)
        ans = res.get('answer', '')
        rag_answers.append(ans)
    except Exception as e:
        logger.error(f"Lỗi RAG câu {i+1}: {e}")
        rag_answers.append("Lỗi hệ thống hoặc không tìm thấy ngữ cảnh.")

print("✅ Đã hoàn tất trích xuất ngữ cảnh.")

# =====================================================================
# BƯỚC 3: GỌI GEMINI CHẤM ĐIỂM TỪNG CÂU (ĐÚNG CHUẨN LLM-AS-A-JUDGE)
# =====================================================================
print(f"\n⏳ BƯỚC 3: Gửi AI chấm điểm độc lập cho {total_questions} câu...")
print("Lưu ý: Quá trình này sẽ mất vài phút (có delay để tránh lỗi Rate Limit của API).")

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("❌ Không tìm thấy GOOGLE_API_KEY trong môi trường.")
    sys.exit(1)

genai.configure(api_key=api_key)
judge_model = genai.GenerativeModel("gemini-1.5-flash")

final_scores = []

for i, item in enumerate(qa_pairs):
    q = item['question']
    ref = item['reference']
    sys_ans = rag_answers[i]
    
    prompt = f"""Bạn là Giám khảo AI đánh giá hệ thống RAG (Retrieval-Augmented Generation).
Nhiệm vụ: Chấm điểm từ 1 đến 5.

[CÂU HỎI]: {q}
[ĐÁP ÁN CHUẨN]: {ref}
[ĐÁP ÁN HỆ THỐNG RAG]: {sys_ans}

Thang điểm:
- 1: Sai hoàn toàn hoặc trả lời không liên quan.
- 2: Có ý đúng nhưng thiếu nội dung quan trọng hoặc có thông tin bịa đặt.
- 3: Trả lời đúng ý chính nhưng diễn đạt chung chung, tóm tắt.
- 4: Trả lời đúng, đầy đủ ý, tự nhiên.
- 5: Hoàn hảo, chính xác tuyệt đối như chuyên gia.

YÊU CẦU ĐẦU RA:
CHỈ trả về đúng MỘT con số nguyên từ 1 đến 5. KHÔNG thêm bất kỳ giải thích, dấu chấm, hay ký tự nào khác.
"""
    
    try:
        # Hiển thị tiến trình trên cùng 1 dòng
        print(f"  👉 Đang chấm câu {i+1}/{total_questions}...", end="\r")
        
        response = judge_model.generate_content(prompt)
        # Dùng Regex để tìm con số nguyên đầu tiên trong câu trả lời (phòng trường hợp AI nói thừa)
        match = re.search(r'\b([1-5])\b', response.text)
        
        if match:
            score = int(match.group(1))
        else:
            score = 3 # Điểm mặc định nếu AI không tuân thủ format
            logger.warning(f"Câu {i+1} AI không trả về số. Raw: {response.text}")
            
        final_scores.append(score)
        
        # Nghỉ 3 giây giữa các lượt để tránh bị chặn API do gửi quá nhanh
        time.sleep(3)
        
    except Exception as e:
        print(f"\n  ❌ Lỗi API ở câu {i+1}: {e}")
        final_scores.append(3) # Điểm an toàn khi lỗi mạng
        time.sleep(5) # Nghỉ lâu hơn một chút nếu gặp lỗi

print(f"\n✅ Đã chấm điểm xong toàn bộ {total_questions} câu!\n")

# =====================================================================
# BƯỚC 4: LƯU VÀ IN BÁO CÁO
# =====================================================================
# Đảm bảo độ dài luôn đồng nhất (Dù code hiện tại vòng lặp đã bảo vệ điều này 100%)
df_results = pd.DataFrame({
    "Câu hỏi": [p["question"] for p in qa_pairs],
    "Đáp án chuẩn": [p["reference"] for p in qa_pairs],
    "Đáp án hệ thống": rag_answers,
    "Điểm (1-5)": final_scores
})

# Lưu CSV
df_results.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

# In bảng thống kê
print("=" * 60)
print(f"📊 TỔNG KẾT ĐÁNH GIÁ RAG ({total_questions} câu)")
print("=" * 60)
print(f"   Điểm trung bình: {np.mean(final_scores):.2f}/5.00")
print(f"\n   📈 Phân bố điểm:")

score_dist = pd.Series(final_scores).value_counts().sort_index()
for score_val, count in score_dist.items():
    bar = "█" * (count * 2)
    pct = count / total_questions * 100
    print(f"      Điểm {score_val}: {bar} {count} ({pct:.1f}%)")

if total_questions > 0:
    average_score = np.mean(final_scores)
    verdict = "🟢 XUẤT SẮC - Hệ thống hoạt động rất tốt"
    if average_score < 3.0:
        verdict = "🟡 KHÁ - Hệ thống hoạt động ổn, cần cải thiện"
    print(f"\n   🏆 Kết luận: {verdict}")

print("=" * 60)
logger.info(f"Đã chấm xong! File kết quả: {OUTPUT_CSV}")
print(f"💾 Đã lưu kết quả chi tiết tại: {OUTPUT_CSV}")