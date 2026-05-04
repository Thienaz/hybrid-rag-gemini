import matplotlib.pyplot as plt
import numpy as np

# 1. Cấu hình font chuẩn học thuật và kích thước lớn dễ đọc
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,           # Size chữ chung
    'axes.labelsize': 13,      # Nhãn trục X, Y
    'xtick.labelsize': 11,     # Chữ ở các tick X
    'ytick.labelsize': 11,     # Chữ ở các tick Y
    'legend.fontsize': 11      # Chữ trong chú thích
})

# Dữ liệu thực tế của bạn (15 người)
questions = [f'Câu {i}' for i in range(1, 11)]

data = np.array([
    [0, 0, 3, 9, 3],  # Q1
    [0, 2, 8, 3, 2],  # Q2
    [0, 0, 4, 8, 3],  # Q3
    [1, 4, 3, 5, 2],  # Q4
    [0, 0, 5, 6, 4],  # Q5
    [0, 3, 6, 3, 3],  # Q6
    [0, 1, 6, 3, 5],  # Q7
    [1, 3, 5, 2, 4],  # Q8
    [0, 2, 5, 6, 2],  # Q9
    [0, 3, 4, 4, 4]   # Q10
])

# Bỏ ký tự \n để chú thích (legend) hiển thị gọn gàng, chia làm 2 dòng đẹp hơn
categories = ['Hoàn toàn không ĐÝ', 'Không đồng ý', 'Bình thường', 'Đồng ý', 'Hoàn toàn đồng ý']

# Bảng màu học thuật
colors = ['#EF4444', '#F97316', '#D1D5DB', '#6EE7B7', '#10B981']

# 2. Thay đổi figsize thành (8, 6) để đạt tỷ lệ 4:3 chuẩn
fig, ax = plt.subplots(figsize=(8, 6))

# Vẽ biểu đồ cột chồng dọc
bottom = np.zeros(len(questions))

for i, col in enumerate(data.T):
    bars = ax.bar(questions, col, bottom=bottom, label=categories[i], 
                  color=colors[i], edgecolor='white', width=0.65)
    
    # Thêm số liệu vào giữa các cột (chữ in đậm để dễ đọc khi thu nhỏ)
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_y() + height/2,
                    int(height), ha='center', va='center', 
                    color='black', fontsize=11, fontweight='bold')
    
    bottom += col

# 3. Căn chỉnh UI
ax.set_ylabel('Số lượng người dùng', fontweight='bold', labelpad=10)
ax.set_ylim(0, 15)

# Thêm lưới ngang mờ để dễ dóng số liệu
ax.grid(axis='y', linestyle='--', alpha=0.4)

# Chú thích đặt phía trên, chia 3 cột (tạo thành 2 dòng) để tiết kiệm chiều ngang
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), 
          ncol=3, frameon=False, columnspacing=1.5)

# Bỏ viền trên, viền phải và viền trái cho thanh thoát
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False) # Bỏ viền trái vì đã có lưới ngang

# Đảm bảo layout không bị cắt xén
plt.tight_layout()
plt.savefig('sus_vertical_chart_4x3.png', dpi=300, bbox_inches='tight')
print("✅ Đã vẽ xong biểu đồ dọc tối ưu tỷ lệ 4:3: sus_vertical_chart_4x3.png")