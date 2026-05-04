# scripts/build_dictionary.py
"""
================================================================
Tự động xây dựng từ điển chuyên ngành (Ontology) bằng AI
Tác giả: Võ Huỳnh Thiên - B2203472
================================================================
"""

import os
import sys
import time
import re
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path

# Thêm thư mục gốc vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Nạp API Key từ file .env
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("❌ Thiếu GOOGLE_API_KEY trong file .env. Vui lòng cấu hình trước.")
    sys.exit(1)

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-flash")

# 5 mảng bao quát toàn bộ ngành HTTT và CNTT
categories = [
    "Cơ sở dữ liệu, Dữ liệu lớn và Phân tích dữ liệu",
    "Mạng máy tính, Hạ tầng và An toàn thông tin",
    "Kỹ thuật phần mềm, Lập trình và Thuật toán",
    "Trí tuệ nhân tạo, Máy học và Xử lý ngôn ngữ tự nhiên",
    "Hệ thống thông tin quản lý, ERP và Thương mại điện tử"
]

all_terms = set()  # Dùng set() để tự động loại bỏ các từ bị trùng lặp

print("=" * 60)
print("🚀 BẮT ĐẦU TỰ ĐỘNG XÂY DỰNG TỪ ĐIỂN CHUYÊN NGÀNH (ONTOLOGY)...")
print("=" * 60)

for i, cat in enumerate(categories, 1):
    print(f"\n⏳ [{i}/{len(categories)}] Đang nhờ AI trích xuất 100 thuật ngữ mảng: {cat}...")

    prompt = f"""
Bạn là chuyên gia Công nghệ thông tin. Hãy liệt kê 100 thuật ngữ, từ khóa chuyên ngành tiếng Việt thuộc lĩnh vực: {cat}.

YÊU CẦU NGHIÊM NGẶT:
- Chỉ in ra từ khóa, mỗi từ trên 1 dòng.
- Tuyệt đối không đánh số thứ tự, không gạch đầu dòng, không giải thích ý nghĩa.
- Viết chữ thường toàn bộ.
"""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            terms = response.text.strip().split('\n')

            count_before = len(all_terms)
            # Làm sạch dữ liệu (Xóa số thứ tự, dấu gạch ngang nếu AI lỡ sinh ra)
            for term in terms:
                clean_term = re.sub(r'^[\d\.\-\*\s]+', '', term).strip().lower()
                # Bỏ các từ quá ngắn (< 2 ký tự) hoặc quá dài (> 50 ký tự)
                if 2 < len(clean_term) <= 50:
                    all_terms.add(clean_term)

            count_after = len(all_terms)
            print(f"   ✅ Đã thêm {count_after - count_before} thuật ngữ mới (Tổng: {count_after})")
            break  # Thành công, thoát retry loop

        except Exception as e:
            print(f"   ❌ Lỗi (lần {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = 10 * (attempt + 1)
                print(f"   ⏳ Chờ {wait_time}s rồi thử lại...")
                time.sleep(wait_time)
            else:
                print(f"   ⚠️ Bỏ qua mảng '{cat}' sau {max_retries} lần thất bại.")

    # Nghỉ giữa các lần gọi API để tránh Rate Limit
    if i < len(categories):
        print("   ⏳ Chờ 5s trước lần gọi tiếp...")
        time.sleep(5)

# Đảm bảo thư mục output tồn tại
output_dir = Path("artifacts")
output_dir.mkdir(exist_ok=True)

# Lưu toàn bộ từ khóa ra file txt
output_file = output_dir / "it_dictionary.txt"
with open(output_file, "w", encoding="utf-8") as f:
    for term in sorted(all_terms):
        f.write(term + "\n")

print("\n" + "=" * 60)
print(f"✅ HOÀN TẤT! Đã tạo thành công bộ từ điển '{output_file}' với {len(all_terms)} thuật ngữ chuẩn!")
print("=" * 60)