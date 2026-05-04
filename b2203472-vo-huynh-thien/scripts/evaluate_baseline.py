"""
=============================================================================
THỰC NGHIỆM SO SÁNH TEXTRANK (BASELINE VS ONTOLOGY)

Mục đích: Đánh giá hiệu năng của mô hình TextRank thuần (0.0) 
          so với TextRank tích hợp Ontology chuyên ngành (0.6).
Cấu hình: base_bonus = 0.0 và base_bonus = 0.6
Dataset: XLSum Vietnamese (tập Test, 1000 mẫu đầu tiên)
Metrics: ROUGE-1, ROUGE-2, ROUGE-L (F1)

Tác giả: Võ Huỳnh Thiên - B2203472
=============================================================================
"""

import json
import os
import sys
import warnings
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from rouge_score import rouge_scorer
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Thêm thư mục gốc vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.nlp_engine import NLPEngine

print("=" * 70)
print("⚙️ THỰC NGHIỆM ĐÁNH GIÁ SO SÁNH (Baseline: 0.0 vs Ontology: 0.6)")
print("=" * 70)

# ===== BƯỚC 1: Cấu hình tham số =====
WEIGHT_BASELINE = 0.0
WEIGHT_ONTOLOGY = 1.0

print(f"\n🔎 Cấu hình thực nghiệm:")
print(f"   Phương pháp 1: TextRank Thuần (base_bonus = {WEIGHT_BASELINE})")
print(f"   Phương pháp 2: TextRank + Ontology (base_bonus = {WEIGHT_ONTOLOGY})")

# ===== BƯỚC 2: Khởi tạo Engine và Scorer =====
engine = NLPEngine()
scorer = rouge_scorer.RougeScorer(
    ['rouge1', 'rouge2', 'rougeL'], use_stemmer=False
)

# ===== BƯỚC 3: Nạp Dataset =====
print("\n📥 Đang tải dataset XLSum Vietnamese (test split)...")
dataset = load_dataset("csebuetnlp/xlsum", "vietnamese", split="test[:1000]")
print(f"✅ Đã tải {len(dataset)} mẫu")

# ===== BƯỚC 4: Chạy thực nghiệm =====
results = []
skipped = 0

for i, item in enumerate(tqdm(dataset, desc="Đang chạy thực nghiệm đối chứng")):
    article = str(item['text'])
    ref_summary = str(item['summary'])

    # Bỏ qua bài viết quá ngắn
    if len(article.split()) < 50:
        skipped += 1
        continue

    # Tiền xử lý (Khởi tạo ma trận và đồ thị dùng chung)
    num_chunks = engine.preprocess(article)

    if num_chunks == 0:
        skipped += 1
        continue

    # Lấy tên document
    doc_name = list(engine.docs_data.keys())[0]

    # --- Chạy Phương pháp 1: Baseline (0.0) ---
    summary_baseline, _ = engine._extract_single_doc(
        doc_name, ratio=0.25, base_bonus=WEIGHT_BASELINE
    )

    # --- Chạy Phương pháp 2: Ontology (0.6) ---
    summary_ontology, _ = engine._extract_single_doc(
        doc_name, ratio=0.25, base_bonus=WEIGHT_ONTOLOGY
    )

    # Kiểm tra tính hợp lệ
    if not summary_baseline or not summary_ontology:
        skipped += 1
        continue

    # Tính ROUGE cho cả hai
    scores_base = scorer.score(ref_summary, summary_baseline)
    scores_onto = scorer.score(ref_summary, summary_ontology)

    results.append({
        'Bài_số': i + 1,
        # Điểm Baseline (0.0)
        'R1_0.0': scores_base['rouge1'].fmeasure * 100,
        'R2_0.0': scores_base['rouge2'].fmeasure * 100,
        'RL_0.0': scores_base['rougeL'].fmeasure * 100,
        # Điểm Ontology (0.6)
        'R1_0.6': scores_onto['rouge1'].fmeasure * 100,
        'R2_0.6': scores_onto['rouge2'].fmeasure * 100,
        'RL_0.6': scores_onto['rougeL'].fmeasure * 100,
    })

# ===== BƯỚC 5: Tổng kết và Xuất kết quả =====
df = pd.DataFrame(results)

print("\n" + "=" * 70)
print(f"📊 TỔNG KẾT THỰC NGHIỆM ĐỐI CHỨNG ({len(df)} mẫu hợp lệ, {skipped} mẫu bỏ qua)")
print("=" * 70)

# Tính trung bình cho cả 2 phương pháp
metrics_summary = {
    "ROUGE-1": (df['R1_0.0'].mean(), df['R1_0.6'].mean()),
    "ROUGE-2": (df['R2_0.0'].mean(), df['R2_0.6'].mean()),
    "ROUGE-L": (df['RL_0.0'].mean(), df['RL_0.6'].mean()),
}

# In bảng so sánh trực quan
print(f"\n{'Metric':>10} | {'Baseline (0.0)':>15} | {'Ontology (0.6)':>15} | {'Chênh lệch':>12}")
print("-" * 62)

for metric, (val_base, val_onto) in metrics_summary.items():
    diff = val_onto - val_base
    trend = "🟢 Tăng" if diff > 0 else ("🔴 Giảm" if diff < 0 else "⚪ Hòa")
    print(f"{metric:>10} | {val_base:>14.2f}% | {val_onto:>14.2f}% | {diff:>6.2f}% {trend}")

# Lưu kết quả chi tiết ra CSV
output_csv = "ket_qua_so_sanh.csv"
df.to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"\n💾 Đã lưu kết quả chi tiết: {output_csv}")

# Lưu kết quả tổng hợp ra JSON
output_json = Path("artifacts/comparison_results.json")
output_json.parent.mkdir(exist_ok=True)

summary_data = {
    "config": {
        "method": "Comparison: Baseline vs Ontology",
        "weights": {"baseline": WEIGHT_BASELINE, "ontology": WEIGHT_ONTOLOGY},
        "dataset": "csebuetnlp/xlsum (vietnamese, test[:1000])",
        "total_samples": len(df),
        "skipped_samples": skipped,
    },
    "results": {
        metric: {
            "Baseline_0.0": round(val_base, 4),
            "Ontology_0.6": round(val_onto, 4),
            "Diff": round(val_onto - val_base, 4)
        }
        for metric, (val_base, val_onto) in metrics_summary.items()
    }
}

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(summary_data, f, ensure_ascii=False, indent=2)

print(f"💾 Đã lưu báo cáo tổng hợp: {output_json}")