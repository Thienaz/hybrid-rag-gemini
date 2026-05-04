#scripts/tune_parameters.py

"""
=============================================================================
TINH CHỈNH THAM SỐ (Hyperparameter Tuning) TRÊN TẬP VALIDATION

Mục đích: Tìm giá trị base_bonus tối ưu cho Ontology-weighted TextRank.
Metric đánh giá: ROUGE-L F1
Dataset: augmented_val_dataset.csv (tập Validation)

Quy trình:
  1. Duyệt qua các giá trị base_bonus = [0.5, 1.0, 1.5, 2.0]
  2. Với mỗi giá trị, chạy TextRank trên toàn bộ tập Validation
  3. So sánh ROUGE-L trung bình → chọn giá trị tốt nhất
  4. Lưu kết quả vào artifacts/best_params.json

Tác giả: Võ Huỳnh Thiên - B2203472
=============================================================================
"""

import json
import os
import sys
import warnings
from pathlib import Path

import pandas as pd
from rouge_score import rouge_scorer
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Thêm thư mục gốc vào sys.path để import được các module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.nlp_engine import NLPEngine

print("=" * 60)
print("🔍 BẮT ĐẦU TINH CHỈNH THAM SỐ TRÊN TẬP VALIDATION")
print("=" * 60)

# Đường dẫn lưu kết quả
BEST_PARAM_FILE = Path("artifacts/best_params.json")
BEST_PARAM_FILE.parent.mkdir(exist_ok=True)

# Khởi tạo engine và scorer
engine = NLPEngine()
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)

# Đọc tập Validation
try:
    df_val = pd.read_csv("augmented_val_dataset.csv").head(150)
    print(f"✅ Đã đọc tập Validation: {len(df_val)} mẫu")
except FileNotFoundError:
    print("❌ Không tìm thấy augmented_val_dataset.csv. Chạy lại Bước 1.")
    sys.exit()

# Các giá trị base_bonus cần thử nghiệm
test_weights = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
results = {}

for weight in test_weights:
    total_rouge_l = 0.0
    valid_samples = 0

    for _, row in tqdm(
        df_val.iterrows(), total=len(df_val),
        desc=f"Thử weight={weight}"
    ):
        article = str(row['text'])
        ref_summary = str(row['summary'])

        if len(article.split()) < 50:
            continue

        # Tiền xử lý (tạo chunks + vectorize)
        num_chunks = engine.preprocess(article)

        if num_chunks == 0:
            continue

        # Lấy tên document (mặc định khi không có delimiter)
        doc_name = list(engine.docs_data.keys())[0]

        # Trích xuất bằng TextRank + Ontology với base_bonus hiện tại
        extractive_summary, _ = engine._extract_single_doc(
            doc_name, ratio=0.25, base_bonus=weight
        )

        if not extractive_summary:
            continue

        # Tính ROUGE-L so với reference summary
        rouge_scores = scorer.score(ref_summary, extractive_summary)
        total_rouge_l += rouge_scores['rougeL'].fmeasure * 100
        valid_samples += 1

    avg_rouge_l = total_rouge_l / valid_samples if valid_samples > 0 else 0.0
    results[weight] = avg_rouge_l
    print(f"   → weight={weight}: ROUGE-L = {avg_rouge_l:.2f}% ({valid_samples} mẫu hợp lệ)")

# Chọn tham số tối ưu
best_weight = max(results, key=results.get)

print("\n" + "=" * 60)
print(f"🏆 THAM SỐ TỐI ƯU TỪ TẬP VALIDATION:")
print(f"   base_bonus = {best_weight} (ROUGE-L: {results[best_weight]:.2f}%)")
print("=" * 60)

# In bảng so sánh
print("\n📊 BẢNG KẾT QUẢ CHI TIẾT:")
print(f"{'base_bonus':>12} | {'ROUGE-L (%)':>12} | {'Ghi chú':>15}")
print("-" * 45)
for w, score in sorted(results.items()):
    note = "⭐ TỐT NHẤT" if w == best_weight else ""
    print(f"{w:>12} | {score:>11.2f}% | {note:>15}")

# Lưu kết quả
output_data = {
    "base_bonus": best_weight,
    "rouge_l_score": results[best_weight],
    "all_results": {str(k): v for k, v in results.items()}
}

with open(BEST_PARAM_FILE, "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"\n💾 Đã lưu kết quả vào: {BEST_PARAM_FILE}")