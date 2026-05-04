# File: validate_llm_judge.py
import pandas as pd
from scipy.stats import pearsonr
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import Counter

# ==========================================
# CẤU HÌNH BIỂU ĐỒ (Phông chữ chuẩn học thuật)
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = Path("thesis_figures")
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("📊 KIỂM ĐỊNH ĐỘ ĐỒNG THUẬN GIỮA LLM VÀ CON NGƯỜI (PEARSON)")
print("=" * 60)

# ==========================================
# 1. ĐƯỜNG DẪN VÀ TÊN CỘT CSV
# ==========================================
csv_path = r"D:\LVTN\artifacts\rag_scored_results.csv"

# Hãy đổi tên này cho khớp chính xác với tiêu đề cột trong file CSV của bạn
LLM_SCORE_COLUMN = "Điểm LMM"    
HUMAN_SCORE_COLUMN = "Điểm Human" 

try:
    # Đọc dữ liệu từ file CSV
    df = pd.read_csv(csv_path)
    
    # Kiểm tra xem các cột có tồn tại không
    if LLM_SCORE_COLUMN not in df.columns or HUMAN_SCORE_COLUMN not in df.columns:
        print(f"❌ Lỗi: Không tìm thấy cột '{LLM_SCORE_COLUMN}' hoặc '{HUMAN_SCORE_COLUMN}' trong file CSV.")
        print(f"Các cột hiện có trong file: {list(df.columns)}")
        print("-> Vui lòng mở code và sửa lại tên biến LLM_SCORE_COLUMN và HUMAN_SCORE_COLUMN cho khớp.")
    else:
        # Lọc bỏ các hàng bị trống điểm để tránh lỗi tính toán
        df_clean = df.dropna(subset=[LLM_SCORE_COLUMN, HUMAN_SCORE_COLUMN])
        
        # Ép kiểu dữ liệu sang int (nguyên) và lấy dưới dạng list
        llm_scores = df_clean[LLM_SCORE_COLUMN].astype(int).tolist()
        human_scores = df_clean[HUMAN_SCORE_COLUMN].astype(int).tolist()
        
        # ==========================================
        # 2. TÍNH HỆ SỐ TƯƠNG QUAN PEARSON
        # ==========================================
        correlation, p_value = pearsonr(llm_scores, human_scores)
        
        print(f"📁 Tệp dữ liệu: {os.path.basename(csv_path)}")
        print(f"Số lượng câu hỏi đánh giá: {len(llm_scores)} câu")
        print(f"Hệ số tương quan Pearson (r): {correlation:.3f}")
        print(f"Giá trị p-value: {p_value:.4f}")
        print("-" * 60)
        
        if correlation >= 0.7:
            print("✅ KẾT LUẬN: Mức độ đồng thuận RẤT CAO.")
            print("-> Phương pháp LLM-as-a-Judge hoạt động cực kỳ đáng tin cậy. Đủ tiêu chuẩn hàn lâm để đưa vào luận văn!")
        elif correlation >= 0.5:
            print("🟡 KẾT LUẬN: Mức độ đồng thuận KHÁ (Trung bình).")
            print("-> Có thể sử dụng được, nhưng bạn nên kiểm tra lại các câu có độ lệch điểm lớn giữa bạn và LLM.")
        else:
            print("❌ KẾT LUẬN: Mức độ đồng thuận THẤP.")
            print("-> LLM chấm điểm khác xa con người. Bạn cần tối ưu lại Prompt hướng dẫn chấm điểm cho mô hình!")

        # ==========================================
        # 3. VẼ BIỂU ĐỒ BONG BÓNG (SCATTER PLOT)
        # ==========================================
        print("\n🎨 Đang tạo biểu đồ phân tán (Scatter Plot)...")
        
        # Đếm tần suất các cặp điểm trùng nhau để tăng kích thước bong bóng
        points = list(zip(llm_scores, human_scores))
        point_counts = Counter(points)

        x = [p[0] for p in point_counts.keys()]
        y = [p[1] for p in point_counts.keys()]
        sizes = [point_counts[p] * 150 for p in point_counts.keys()] # Hệ số phóng to

        fig, ax = plt.subplots(figsize=(7, 6))

        # Vẽ biểu đồ bong bóng (Scatter Plot)
        scatter = ax.scatter(x, y, s=sizes, c='#2563EB', alpha=0.7, edgecolors='black', linewidth=1.5)

        # Đường xu hướng (Hoàn hảo r=1.0)
        ax.plot([1, 5], [1, 5], linestyle='--', color='gray', label='Đường đồng thuận lý tưởng')

        ax.set_xlabel('Điểm số do Giám khảo LLM chấm (1-5)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Điểm số do Con người chấm (1-5)', fontsize=11, fontweight='bold')
        
        # Tiêu đề tự động cập nhật hệ số r
        ax.set_title(f'Hình 4.X: Biểu đồ phân tán sự đồng thuận giữa LLM và Con người\n(Hệ số tương quan Pearson r = {correlation:.2f})', fontsize=12, pad=15, fontweight='bold')

        # Cấu hình grid và tick
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(loc='upper left')

        # Ghi chú tần suất
        for px, py, count in zip(x, y, point_counts.values()):
            ax.text(px, py, str(count), ha='center', va='center', color='white', fontweight='bold', fontsize=9)

        plt.tight_layout()
        output_path = OUTPUT_DIR / "fig_4_pearson_correlation.png"
        plt.savefig(output_path, dpi=300)
        
        print(f"✅ Đã tạo biểu đồ thành công tại: {output_path}")

except FileNotFoundError:
    print(f"❌ Không tìm thấy file tại đường dẫn: {csv_path}")
    print("Vui lòng kiểm tra lại đường dẫn tệp tin.")
except Exception as e:
    print(f"❌ Đã xảy ra lỗi không xác định: {e}")