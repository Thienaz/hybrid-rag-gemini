"""
=============================================================================
THỰC NGHIỆM SO SÁNH 2 PHƯƠNG PHÁP TÓM TẮT TRÊN MIỀN CHUYÊN NGÀNH IT
Mục đích: Đánh giá tác động của Tri thức chuyên ngành (Domain Ontology).

  - Phương pháp A (Base): TextRank Cải tiến (base_bonus=0.0 - Tắt Ontology)
  - Phương pháp B (Core): TextRank Cải tiến (base_bonus=0.6 - Bật Ontology)

Dataset: it_domain_val_dataset.csv (200 mẫu Oracle từ Giáo trình IT)
Tác giả: Võ Huỳnh Thiên - B2203472
=============================================================================
"""

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

print("=" * 70)
print("🧬 THỰC NGHIỆM SO SÁNH: BASE (0.0) vs CORE (0.6)")
print("=" * 70)

OUTPUT_CSV = Path("artifacts/ab_test_base_vs_core_results.csv")
OUTPUT_TXT = Path("artifacts/method_b_extracted_summaries.txt")
OUTPUT_CSV.parent.mkdir(exist_ok=True)

# Cấu hình trọng số
WEIGHT_A = 0.0  # Tắt tri thức
WEIGHT_B = 0.6  # Bật tri thức tối ưu

# Khởi tạo Engine và Scorer
engine = NLPEngine()
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)

# Đọc Dataset
try:
    df_data = pd.read_csv("it_domain_val_dataset.csv")
    print(f"✅ Đã tải {len(df_data)} mẫu chuyên ngành IT.")
except FileNotFoundError:
    print("❌ Không tìm thấy it_domain_val_dataset.csv.")
    sys.exit()

results = []
txt_lines = [] # Dùng để lưu kết quả ra file .txt

print("\n⏳ Bắt đầu chạy đánh giá (100% Local, cực nhanh)...")
for idx, row in tqdm(df_data.iterrows(), total=len(df_data), desc="Đang xử lý"):
    text = str(row['text'])
    ref_summary = str(row['summary'])
    
    if len(text.split()) < 30: continue

    # Tiền xử lý chung (Chỉ cần 1 lần cho mỗi văn bản)
    num_chunks = engine.preprocess(text)
    if num_chunks == 0: continue
    doc_name = list(engine.docs_data.keys())[0]

    # ==========================================
    # PHƯƠNG PHÁP A: BASE (base_bonus=0.0)
    # ==========================================
    summary_a, _ = engine._extract_single_doc(doc_name, ratio=0.4, base_bonus=WEIGHT_A)
    if not summary_a: continue
    scores_a = scorer.score(ref_summary, summary_a)

    # ==========================================
    # PHƯƠNG PHÁP B: CORE (base_bonus=0.6)
    # ==========================================
    summary_b, _ = engine._extract_single_doc(doc_name, ratio=0.4, base_bonus=WEIGHT_B)
    if not summary_b: continue
    scores_b = scorer.score(ref_summary, summary_b)

    # Lưu kết quả vào list để xuất CSV
    results.append({
        'id': idx,
        'A_R1': scores_a['rouge1'].fmeasure, 'B_R1': scores_b['rouge1'].fmeasure,
        'A_R2': scores_a['rouge2'].fmeasure, 'B_R2': scores_b['rouge2'].fmeasure,
        'A_RL': scores_a['rougeL'].fmeasure, 'B_RL': scores_b['rougeL'].fmeasure,
    })

    # Chuẩn bị dòng ghi vào file TXT
    r1 = round(scores_b['rouge1'].fmeasure * 100, 2)
    r2 = round(scores_b['rouge2'].fmeasure * 100, 2)
    rl = round(scores_b['rougeL'].fmeasure * 100, 2)
    
    txt_lines.append(f"=== MẪU {idx} ===")
    txt_lines.append(f"[ĐIỂM SỐ] ROUGE-1: {r1}% | ROUGE-2: {r2}% | ROUGE-L: {rl}%")
    txt_lines.append(f"[TÓM TẮT BỎI RÚT TRÍCH (Phương pháp B)]:\n{summary_b}")
    txt_lines.append("-" * 70 + "\n")

# =====================================================================
# 1. XUẤT FILE TXT TỔNG HỢP KẾT QUẢ CỦA B
# =====================================================================
with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(txt_lines))
print(f"📝 Đã lưu toàn bộ tóm tắt và điểm của B ra file: {OUTPUT_TXT}")

# =====================================================================
# 2. XUẤT FILE CSV VÀ IN BẢNG THỐNG KÊ
# =====================================================================
df = pd.DataFrame(results)

def m(col): return round(col.mean() * 100, 2)

print("\n" + "=" * 70)
print(f"📊 BẢNG TỔNG HỢP SO SÁNH 2 PHƯƠNG PHÁP ({len(df)} mẫu)")
print("=" * 70)
print(f"{'Chỉ số':<15} | {'A: Base (0.0)':<18} | {'B: Core (0.6)':<18} | {'Δ (B - A)':>10}")
print("-" * 65)

metrics = [("ROUGE-1 (F1)", "R1"), ("ROUGE-2 (F1)", "R2"), ("ROUGE-L (F1)", "RL")]

for name, suf in metrics:
    a_val = m(df[f'A_{suf}'])
    b_val = m(df[f'B_{suf}'])
    delta = b_val - a_val
    print(f"{name:<15} | {a_val:<18}% | {b_val:<18}% | {delta:>+9.2f}%")

print("=" * 70)
df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
print(f"💾 Đã lưu bảng số liệu chi tiết tại: {OUTPUT_CSV}")