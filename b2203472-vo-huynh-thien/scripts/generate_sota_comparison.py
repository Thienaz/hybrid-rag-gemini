"""
=============================================================================
SCRIPT TẠO HÌNH 4.7: SO SÁNH HIỆU NĂNG VỚI SOTA (mT5, ViT5) VÀ BASELINE
Tác giả: Võ Huỳnh Thiên - B2203472
=============================================================================
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# =====================================================================
# 1. CẤU HÌNH FONT CHUẨN HỌC THUẬT & SIZE CHỮ TO
# ==========================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.unicode_minus': False,
    'font.size': 13,           # Tăng cỡ chữ tổng thể
    'axes.labelsize': 14,      # Nhãn trục X to hơn
    'xtick.labelsize': 12,     # Chữ số trên trục X
    'ytick.labelsize': 13,     # Tên các mô hình trục Y
    'legend.fontsize': 12      # Chữ trong chú thích
})

OUTPUT_DIR = Path("thesis_figures")
OUTPUT_DIR.mkdir(exist_ok=True)

# =====================================================================
# 2. DỮ LIỆU SO SÁNH
# =====================================================================
models = [
    'mT5\n(Fine-tuned LLM)', 
    'ViT5-base\n(SOTA Tiếng Việt)', 
    'TextRank\n(Baseline)', 
    'Hybrid Pipeline\n(Đề xuất)'
]

# Số liệu lấy từ bảng
rouge1 = [32.88, 36.31, 32.17, 35.50]
rouge2 = [12.60, 14.54, 14.36, 13.80]
rougeL = [26.08, 28.53, 25.84, 21.91]

# Tính toán vị trí các thanh ngang
y = np.arange(len(models))
height = 0.26 # Tăng nhẹ độ dày của thanh để biểu đồ trông đầy đặn hơn

# Bảng màu học thuật
colors = ['#94A3B8', '#3B82F6', '#10B981'] # Xám, Xanh dương, Xanh lá

# =====================================================================
# 3. KHỞI TẠO FIGURE (TỐI ƯU TỶ LỆ 4:3)
# =====================================================================
# figsize=(8, 6) tạo tỷ lệ 4:3 chuẩn. Nó sẽ cao ráo và không bị dàn ngang
fig, ax = plt.subplots(figsize=(8, 6))

# Vẽ 3 nhóm thanh ngang (R1, R2, RL)
rects1 = ax.barh(y - height, rouge1, height, label='ROUGE-1', color=colors[0], edgecolor='black', linewidth=0.8, alpha=0.9)
rects2 = ax.barh(y, rouge2, height, label='ROUGE-2', color=colors[1], edgecolor='black', linewidth=0.8, alpha=0.9)
rects3 = ax.barh(y + height, rougeL, height, label='ROUGE-L', color=colors[2], edgecolor='black', linewidth=0.8, alpha=0.9)

# =====================================================================
# 4. ĐỊNH DẠNG VÀ TRANG TRÍ TRỤC
# =====================================================================
ax.set_xlabel('Điểm số ROUGE (%)', fontweight='bold', labelpad=12)
ax.set_yticks(y)
ax.set_yticklabels(models, fontweight='bold')

# Đảo ngược trục Y để mô hình đầu tiên (mT5) nằm ở trên cùng
ax.invert_yaxis()

# Đưa Legend lên trên cùng (dàn 3 cột) để biểu đồ hẹp bề ngang, tối ưu không gian hiển thị
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False)

# Nới giới hạn trục X để chữ số không bị cắt mất khi dán nhãn
ax.set_xlim(0, 43)

# Thêm lưới dọc (Grid) mờ giúp hội đồng dễ dóng điểm số
ax.grid(axis='x', linestyle='--', alpha=0.5)

# Bỏ viền trên, phải và trái cho biểu đồ thanh thoát, tập trung vào Data
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

# Ẩn các vạch gạch nhỏ (tick marks) trên trục Y để chữ áp sát vào biểu đồ hơn
ax.tick_params(axis='y', length=0)

# =====================================================================
# 5. DÁN NHÃN SỐ LIỆU TỰ ĐỘNG LÊN ĐUÔI THANH
# =====================================================================
def autolabel(rects):
    for rect in rects:
        width = rect.get_width()
        ax.annotate(f'{width:.2f}',
                    xy=(width, rect.get_y() + rect.get_height() / 2),
                    xytext=(5, 0),  # Dịch sang phải 5 points
                    textcoords="offset points",
                    ha='left', va='center', 
                    fontsize=12, fontweight='bold', color='black') # Chữ in đậm và to hơn (12)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

# =====================================================================
# 6. XUẤT ẢNH CHUẨN IN ẤN
# =====================================================================
plt.tight_layout()

# bbox_inches='tight' cắt gọn viền trắng thừa xung quanh ảnh
output_path = OUTPUT_DIR / "fig_4_7_sota_comparison_horizontal.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Đã tạo thành công biểu đồ thanh ngang tại: {output_path}")