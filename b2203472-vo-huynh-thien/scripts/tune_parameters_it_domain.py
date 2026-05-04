"""
=============================================================================
TINH CHỈNH THAM SỐ TRÊN TẬP DỮ LIỆU CHUYÊN NGÀNH IT
Mục đích: Tìm giá trị base_bonus tối ưu thực sự cho ngành Hệ thống Thông tin.
Dataset: it_domain_val_dataset.csv (Tập Oracle tạo từ giáo trình PDF)
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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.nlp_engine import NLPEngine

print("=" * 60)
print("🔍 TINH CHỈNH THAM SỐ TRÊN TẬP GIÁO TRÌNH CHUYÊN NGÀNH IT")
print("=" * 60)

# Đường dẫn lưu kết quả
BEST_PARAM_FILE = Path("artifacts/best_params_it_domain.json")
BEST_PARAM_FILE.parent.mkdir(exist_ok=True)

# Khởi tạo engine
engine = NLPEngine()

# ĐÃ SỬA LỖI CÚ PHÁP Ở DÒNG DƯỚI ĐÂY
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)

# Đọc tập IT Validation
try:
    df_val = pd.read_csv("it_domain_val_dataset.csv")
    print(f"✅ Đã đọc tập Validation IT: {len(df_val)} mẫu")
except FileNotFoundError:
    print("❌ Không tìm thấy it_domain_val_dataset.csv. Hãy chạy script build_it_validation_set.py trước.")
    sys.exit()

# Mở rộng dải thử nghiệm
test_weights = [0.55, 0.6, 0.65, 0.70, 0.75, 0.8]
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

        if len(article.split()) < 30:
            continue

        # Tiền xử lý
        num_chunks = engine.preprocess(article)
        if num_chunks == 0:
            continue

        doc_name = list(engine.docs_data.keys())[0]

        # Trích xuất (Tăng ratio lên 0.4 vì chunk ngắn)
        extractive_summary, _ = engine._extract_single_doc(
            doc_name, ratio=0.4, base_bonus=weight
        )

        if not extractive_summary:
            continue

        # Tính ROUGE-L
        rouge_scores = scorer.score(ref_summary, extractive_summary)
        total_rouge_l += rouge_scores['rougeL'].fmeasure * 100
        valid_samples += 1

    avg_rouge_l = total_rouge_l / valid_samples if valid_samples > 0 else 0.0
    results[weight] = avg_rouge_l
    print(f"   → weight={weight}: ROUGE-L = {avg_rouge_l:.2f}% ({valid_samples} mẫu)")

# Chọn tham số tối ưu
best_weight = max(results, key=results.get)

print("\n" + "=" * 60)
print(f"🏆 THAM SỐ TỐI ƯU DÀNH CHO NGÀNH HỆ THỐNG THÔNG TIN:")
print(f"   base_bonus = {best_weight} (ROUGE-L: {results[best_weight]:.2f}%)")
print("=" * 60)

# In bảng
print("\n📊 BẢNG KẾT QUẢ CHI TIẾT:")
print(f"{'base_bonus':>12} | {'ROUGE-L (%)':>12} | {'Ghi chú':>15}")
print("-" * 45)
for w, score in sorted(results.items()):
    note = "⭐ TỐI ƯU IT" if w == best_weight else ""
    print(f"{w:>12} | {score:>11.2f}% | {note:>15}")

# Lưu kết quả
output_data = {
    "base_bonus": best_weight,
    "rouge_l_score": results[best_weight],
    "dataset_used": "it_domain_val_dataset2.csv (Oracle from PDFs)",
    "all_results": {str(k): v for k, v in results.items()}
}

with open(BEST_PARAM_FILE, "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"\n💾 Đã lưu kết quả vào: {BEST_PARAM_FILE}")