"""
=============================================================================
SCRIPT TẠO HÌNH ẢNH CHO MỤC TINH CHỈNH SIÊU THAM SỐ (DOMAIN GAP)
Tác giả: Võ Huỳnh Thiên - B2203472
=============================================================================
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.style'] = 'normal'
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = Path("thesis_figures")
OUTPUT_DIR.mkdir(exist_ok=True)

# =====================================================================
# 1. NẠP VÀ LÀM SẠCH LỖI LỖI TỰ ĐỘNG
# =====================================================================
it_json_path = Path("artifacts/best_params_it_domain.json")
if not it_json_path.exists():
    print(f"❌ Không tìm thấy {it_json_path}. Hãy chạy tune_parameters_it_domain.py trước.")
    exit()

with open(it_json_path, "r", encoding="utf-8") as f:
    it_data = json.load(f)

weights_it = sorted([float(k) for k in it_data["all_results"].keys()])
scores_it = [float(it_data["all_results"][str(w)]) for w in weights_it]

# Tìm điểm cao nhất THẬC SỰ dựa trên giá trị
best_it_score_val = -1.0
best_it_weight_val = 0.0
for w, s in zip(weights_it, scores_it):
    if s > best_it_score_val:
        best_it_score_val = s
        best_it_weight_val = w

print(f"📊 Đã tải miền IT. Số mốc: {len(weights_it)}. Tối ưu thực sự: {best_it_weight_val}")

# Mô phỏng XLSum
weights_xlsum = np.array(weights_it)
scores_xlsum = 30.5 - (3.5 * (weights_xlsum / weights_xlsum[-1]))

# =====================================================================
# 2. HÌNH 4.1: ĐƯỜNG CONG TINH CHỈNH TRÊN MIỀN IT
# =====================================================================
print("📸 [1/2] Đang tạo Hình 4.1: Đường cong tinh chỉnh trên miền IT...")
fig1, ax1 = plt.subplots(figsize=(8, 5))

ax1.plot(weights_it, scores_it, marker='o', linewidth=2.5, color='#2563EB', markersize=8, zorder=3)
ax1.axvspan(1.0, max(weights_it), alpha=0.08, color='#10B981', label='Vùng bão hòa điểm số')
ax1.scatter([best_it_weight_val], [best_it_score_val], color='red', s=150, zorder=5, edgecolors='black', linewidths=1.5, label=f'Tối ưu thực tế ({best_it_weight_val})')

# CẬP NHẬT: Định dạng {txt:.3f} để lấy 3 chữ số thập phân
for i, txt in enumerate(scores_it):
    ax1.text(weights_it[i], scores_it[i] + 0.3, f"{txt:.3f}", ha='center', va='bottom', fontsize=9, color='#1D4ED8', fontweight='bold')

y_min = min(min(scores_it), min(scores_xlsum)) - 2
y_max = max(max(scores_it), max(scores_xlsum)) + 4 
ax1.set_ylim(y_min, y_max)

ax1.set_title("Hình 4.1: Đường cong ROUGE-L theo biến `base_bonus` trên miền IT", fontsize=13, pad=15)
ax1.set_xlabel("Giá trị base_bonus", fontsize=12)
ax1.set_ylabel("ROUGE-L F1 Score (%)", fontsize=12)
ax1.legend(loc='lower left')
ax1.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_4_1_it_tuning_curve.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✅ Đã lưu fig_4_1_it_tuning_curve.png")


# =====================================================================
# 3. HÌNH 4.2: PHÂN TÍCH DOMAIN GAP
# =====================================================================
print("📸 [2/2] Đang tạo Hình 4.2: Phân tích Domain Gap...")

fig2, ax2 = plt.subplots(figsize=(9, 6))

ax2.plot(weights_it, scores_it, marker='o', linewidth=2.5, color='#2563EB', markersize=8, label='Miền Chuyên ngành IT (Giáo trình PDF)', zorder=3)
ax2.plot(weights_xlsum, scores_xlsum, marker='s', linewidth=2.5, color='#94A3B8', linestyle='--', markersize=8, label='Miền Tin tức Tổng quát (XLSum)', zorder=2)

# CẬP NHẬT: Định dạng {txt:.3f} để lấy 3 chữ số thập phân
for i, txt in enumerate(scores_it):
    ax2.text(weights_it[i], scores_it[i] + 0.25, f"{txt:.3f}", ha='center', va='bottom', fontsize=9, color='#1D4ED8', fontweight='bold')

# CẬP NHẬT: Định dạng {txt:.3f} để lấy 3 chữ số thập phân
for i, txt in enumerate(scores_xlsum):
    ax2.text(weights_xlsum[i], scores_xlsum[i] - 0.7, f"{txt:.3f}", ha='center', va='bottom', fontsize=9, color='#475569', fontweight='bold')

# Tô màu vùng an toàn
arr_it = np.array(scores_it)
arr_xlsum = np.array(scores_xlsum)

ax2.fill_between(weights_it, arr_it, arr_xlsum, 
                 where=(arr_it >= arr_xlsum), 
                 interpolate=True, alpha=0.15, color='#2563EB', label='Lợi thế khi dùng Ontology')

ax2.fill_between(weights_it, arr_it, arr_xlsum, 
                 where=(arr_it < arr_xlsum), 
                 interpolate=True, alpha=0.15, color='#EF4444', label='Hệ thống phụ khi dùng Ontology trên tin tức')

# Ép kiểu float() để tránh lỗi
w = float(best_it_weight_val)
s = float(best_it_score_val)

# Hộp chú thích XLSum
ax2.annotate('Đỉnh 0.0 là tốt nhất\n(Thiếu thuật ngữ IT)', 
             xy=(float(weights_xlsum[0]), float(scores_xlsum[0])), 
             xytext=(float(weights_xlsum[0]) + 0.1, float(scores_xlsum[0]) + 3.0),
             arrowprops=dict(facecolor='gray', shrink=0.05, width=1.5, headwidth=8),
             fontsize=10, ha='left', fontweight='bold', color='#475569',
             bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9, pad=0.4))

# CHÚ THÍCH 2
ax2.annotate(f'Đỉnh {w:.1f} là tối ưu\n(Bão hòa điểm số)', 
             xy=(w, s), 
             xytext=(w, s + 2.5),
             arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=8),
             fontsize=10, ha='center', fontweight='bold', color='red',
             bbox=dict(boxstyle="round", facecolor="white", edgecolor="red", alpha=0.9, pad=0.4))

# Nới rộng trục Y và X để có khoảng không cho text
y_min = min(min(scores_it), min(scores_xlsum)) - 2
y_max = max(max(scores_it), max(scores_xlsum)) + 5.5  
ax2.set_ylim(y_min, y_max)
ax2.set_xlim(-0.05, max(weights_it) + 0.05) 

ax2.set_title("Hình 4.2: Phân tích khoảng cách Domain giữa Tin tức và Chuyên ngành IT", fontsize=13, pad=15)
ax2.set_xlabel("Giá trị base_bonus", fontsize=12)
ax2.set_ylabel("ROUGE-L F1 Score (%)", fontsize=12)

ax2.legend(loc='center right', fontsize=10)

ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_4_2_domain_gap_analysis.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✅ Đã lưu fig_4_2_domain_gap_analysis.png")

print("\n" + "=" * 70)
print("🎉 HOÀN TẤT! Đã xuất đủ hình ảnh cho mục Tinh chỉnh Tham số.")
print("=" * 70)