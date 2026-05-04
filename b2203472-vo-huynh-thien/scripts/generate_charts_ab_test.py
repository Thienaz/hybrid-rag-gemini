"""
=============================================================================
SCRIPT TỰ ĐỘNG VẼ BIỂU ĐỒ THỊ CHO CHƯƠNG 4 (THỰC NGHIỆM 3 PHƯƠNG PHÁP)
Mục đích: Đọc kết quả từ ab_test_3_methods_it_results.csv và xuất ra 5 hình ảnh
          chuẩn học thuật (300 DPI) để đưa vào Luận văn.

Tác giả: Võ Huỳnh Thiên - B2203472
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Cấu hình toàn cục cho Luận văn (Bắt buộc font và độ phân giải)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font_scale=1.1)

# Thư mục lưu hình
OUTPUT_DIR = Path("thesis_figures")
OUTPUT_DIR.mkdir(exist_ok=True)

# Bảng màu chuyên nghiệp (Mù màu, thân thiện với người mù màu)
COLORS = {
    'A': '#94A3B8', # Xám nhạt (Baseline)
    'B': '#2563EB', # Xanh dương (Core)
    'C': '#10B981'  # Xanh lá (Full Hybrid)
}

def save_fig(fig, filename):
    """Hàm chuẩn hóa lưu hình ảnh"""
    filepath = OUTPUT_DIR / filename
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"   ✅ Đã lưu: {filepath}")

# =====================================================================
# ĐỌC DỮ LIỆU
# =====================================================================
CSV_PATH = Path("artifacts/ab_test_3_methods_it_results.csv")
if not CSV_PATH.exists():
    print(f"❌ Không tìm thấy file {CSV_PATH}. Hãy chạy script thực nghiệm trước.")
    exit()

df = pd.read_csv(CSV_PATH)
print(f"📊 Đã tải dữ liệu: {len(df)} mẫu")

# Tính Mean cho các biểu đồ cột
mean_scores = {
    'ROUGE-1': {'A': df['A_R1'].mean() * 100, 'B': df['B_R1'].mean() * 100, 'C': df['C_R1'].mean() * 100},
    'ROUGE-2': {'A': df['A_R2'].mean() * 100, 'B': df['B_R2'].mean() * 100, 'C': df['C_R2'].mean() * 100},
    'ROUGE-L': {'A': df['A_RL'].mean() * 100, 'B': df['B_RL'].mean() * 100, 'C': df['C_RL'].mean() * 100},
}

print("🚀 Bắt đầu tạo 5 hình ảnh cho Luận văn...\n")

# =====================================================================
# HÌNH 1: SO SÁNH TỔNG QUAN ROUGE-L (Biểu đồ cột đơn giản)
# Dùng cho phần Mở đầu mục 4.X để người đọc nắm bắt ngay
# =====================================================================
print("📸 [1/5] Đang tạo Hình 4.X.1: So sánh tổng quan ROUGE-L...")
fig, ax = plt.subplots(figsize=(7, 5))
methods = ['A: Base\n(0.0)', 'B: Core\n(1.5)', 'C: Hybrid\n(Gemini)']
scores = [mean_scores['ROUGE-L']['A'], mean_scores['ROUGE-L']['B'], mean_scores['ROUGE-L']['C']]
colors = [COLORS['A'], COLORS['B'], COLORS['C']]

bars = ax.bar(methods, scores, color=colors, width=0.5, edgecolor='black', linewidth=0.5)
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

ax.set_ylabel("ROUGE-L F1 Score (%)", fontsize=11)
ax.set_ylim(0, max(scores) * 1.2)
save_fig(fig, "fig_4_x1_overall_rougel.png")

# =====================================================================
# HÌNH 2: SO SÁNH CHI TIẾT TẤT CẢ CÁC CHỈ SỐ (Grouped Bar Chart)
# =====================================================================
print("📸 [2/5] Đang tạo Hình 4.X.2: So sánh chi tiết R1, R2, RL...")
fig, ax = plt.subplots(figsize=(9, 5.5))
metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
x = np.arange(len(metrics))
width = 0.25

r1_scores = [mean_scores['ROUGE-1']['A'], mean_scores['ROUGE-2']['A'], mean_scores['ROUGE-L']['A']]
r2_scores = [mean_scores['ROUGE-1']['B'], mean_scores['ROUGE-2']['B'], mean_scores['ROUGE-L']['B']]
r3_scores = [mean_scores['ROUGE-1']['C'], mean_scores['ROUGE-2']['C'], mean_scores['ROUGE-L']['C']]

bars1 = ax.bar(x - width, r1_scores, width, label='A: Base (0.0)', color=COLORS['A'], edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x, r2_scores, width, label='B: Core (1.5)', color=COLORS['B'], edgecolor='black', linewidth=0.5)
bars3 = ax.bar(x + width, r3_scores, width, label='C: Hybrid (Gemini)', color=COLORS['C'], edgecolor='black', linewidth=0.5)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8)

ax.set_ylabel("Điểm số (%)", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend(loc='upper right')
ax.set_ylim(0, max(r3_scores) * 1.2)
save_fig(fig, "fig_4_x2_detailed_metrics.png")

# =====================================================================
# HÌNH 3: TỶ LỆU TĂNG TRƯỞNG CỦA PHƯƠNG C SO VỚI A (Biểu đồ thanh)
# Dùng để nhấn mạnh "Giá trị gia tăng" mà đề tài mang lại
# =====================================================================
print("📸 [3/5] Đang tạo Hình 4.X.3: Tỷ lệ cải thiện của phương pháp đề xuất...")
delta_data = {
    'B so với A': [mean_scores['ROUGE-1']['B'] - mean_scores['ROUGE-1']['A'],
                 mean_scores['ROUGE-2']['B'] - mean_scores['ROUGE-2']['A'],
                 mean_scores['ROUGE-L']['B'] - mean_scores['ROUGE-L']['A']],
    'C so với A': [mean_scores['ROUGE-1']['C'] - mean_scores['ROUGE-1']['A'],
                 mean_scores['ROUGE-2']['C'] - mean_scores['ROUGE-2']['A'],
                 mean_scores['ROUGE-L']['C'] - mean_scores['ROUGE-L']['A']]
}

df_delta = pd.DataFrame(delta_data, index=['ROUGE-1', 'ROUGE-2', 'ROUGE-L'])

fig, ax = plt.subplots(figsize=(8, 5))
df_delta.plot(kind='bar', ax=ax, color=[COLORS['B'], COLORS['C']], edgecolor='black', linewidth=0.5, width=0.6)

# Thêm đường 0 làm mốc
ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)

# Thêm nhãn phần trăm
for p in ax.patches:
    ax.annotate(f"{p.get_height():+.1f}%", 
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='bold', fontsize=9)

ax.set_ylabel("Chênh lệ điểm (%)", fontsize=11)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.legend(title='So sánh với', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
save_fig(fig, "fig_4_x3_improvement_delta.png")

# =====================================================================
# HÌNH 4: PHÂN PHỐI ĐIỂM SCORE (BOXPLOT) - RẤT QUAN TRỌNG CHO LUẬN VĂN
# Chứng minh rằng phương pháp B/C không chỉ thắng trung bình mà thắng ở đa số mẫu
# =====================================================================
print("📸 [4/5] Đang tạo Hình 4.X.4: Phân phối điểm ROUGE-L (Boxplot)...")
# Chuyển đổi dữ liệu từ wide sang long format để vẽ boxplot
df_melted = pd.melt(df, id_vars=['id'], 
                     value_vars=['A_RL', 'B_RL', 'C_RL'], 
                     var_name='Method', value_name='Score')
df_melted['Score'] = df_melted['Score'] * 100 # Chuyển sang %

# Map tên lại cho đẹp
df_melted['Method'] = df_melted['Method'].map({
    'A_RL': 'A: Base', 'B_RL': 'B: Core', 'C_RL': 'C: Hybrid'
})

fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x='Method', y='Score', hue='Method', data=df_melted, 
            palette=[COLORS['A'], COLORS['B'], COLORS['C']], 
            width=0.4, ax=ax, showfliers=True, legend=False)

ax.set_ylabel("ROUGE-L F1 Score (%)", fontsize=11)
ax.set_xlabel("Phương pháp", fontsize=11)
save_fig(fig, "fig_4_x4_score_distribution_boxplot.png")

# =====================================================================
# HÌNH 5: BIỂU ĐỒ NHỌI (RADAR CHART) - TRỰC QUAN TRỰNG ĐA CHIỀU
# Dùng để khẳng định phương pháp C toàn diện nhất
# =====================================================================
print("📸 [5/5] Đang tạo Hình 4.X.5: Biểu đồ Radar tổng hợp năng lực...")
categories = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # Đóng vòng tròn

# Lấy dữ liệu
values_A = [mean_scores['ROUGE-1']['A'], mean_scores['ROUGE-2']['A'], mean_scores['ROUGE-L']['A']]
values_B = [mean_scores['ROUGE-1']['B'], mean_scores['ROUGE-2']['B'], mean_scores['ROUGE-L']['B']]
values_C = [mean_scores['ROUGE-1']['C'], mean_scores['ROUGE-2']['C'], mean_scores['ROUGE-L']['C']]
values_A += values_A[:1]
values_B += values_B[:1]
values_C += values_C[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.plot(angles, values_A, 'o-', linewidth=2, color=COLORS['A'], label='A: Base (0.0)')
ax.fill(angles, values_A, alpha=0.1, color=COLORS['A'])

ax.plot(angles, values_B, 'o-', linewidth=2, color=COLORS['B'], label='B: Core (1.5)')
ax.fill(angles, values_B, alpha=0.1, color=COLORS['B'])

ax.plot(angles, values_C, 'o-', linewidth=2, color=COLORS['C'], label='C: Hybrid (Gemini)')
ax.fill(angles, values_C, alpha=0.15, color=COLORS['C'])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
save_fig(fig, "fig_4_x5_radar_chart.png")

print("\n" + "=" * 70)
print("🎉 HOÀN TẤT! Đã xuất 5 hình ảnh chất lượng cao vào thư mục thesis_figures/")
print("=" * 70)