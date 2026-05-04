"""
Tạo biểu đồ chứng minh Tối ưu hóa thời gian nhờ Hybrid Intent Recognition (Regex vs LLM)
Tác giả: Võ Huỳnh Thiên
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Cấu hình font
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = Path("thesis_figures")
OUTPUT_DIR.mkdir(exist_ok=True)

# Dữ liệu thời gian phản hồi (Giây)
methods = ['Luồng chỉ dùng LLM\n(Zero-shot toàn bộ)', 'Luồng Đề xuất\n(Regex + LLM Fallback)']
time_intent_classification = [1.8, 0.05] # Regex mất 0.05s, LLM mất 1.8s
time_generation = [3.5, 3.5] # Thời gian LLM sinh câu trả lời là như nhau

x = np.arange(len(methods))
width = 0.5

fig, ax = plt.subplots(figsize=(8, 6))

# Vẽ biểu đồ cột chồng (Stacked Bar Chart)
p1 = ax.bar(x, time_intent_classification, width, label='Thời gian Phân loại Ý định (Routing)', color='#EF4444', edgecolor='black', alpha=0.8)
p2 = ax.bar(x, time_generation, width, bottom=time_intent_classification, label='Thời gian Sinh câu trả lời (GenAI)', color='#3B82F6', edgecolor='black', alpha=0.8)

ax.set_ylabel('Thời gian thực thi trung bình (Giây)', fontsize=12, fontweight='bold')
ax.set_title('Hình 4.X: Tối ưu hóa độ trễ nhờ cơ chế Nhận diện Ý định Lai (Regex)', fontsize=13, pad=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=11, fontweight='bold')
ax.legend(loc='upper right')

# Tính tổng để ghi nhãn trên đỉnh cột
for i in range(len(methods)):
    total_time = time_intent_classification[i] + time_generation[i]
    ax.text(x[i], total_time + 0.1, f'{total_time:.2f} s', ha='center', fontweight='bold', fontsize=11)

ax.set_ylim(0, 6)
ax.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()

output_path = OUTPUT_DIR / "fig_4_routing_performance.png"
plt.savefig(output_path, dpi=300)
print(f"✅ Đã tạo biểu đồ tối ưu thời gian tại: {output_path}")