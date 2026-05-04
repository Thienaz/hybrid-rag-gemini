# scripts/prepare_augmented_data.py
"""
================================================================
CHUẨN BỊ DỮ LIỆU TĂNG CƯỜNG CHO TRAINING/TUNING
Nguồn: XL-Sum Vietnamese (tập Train) + Dữ liệu chuyên ngành HTTT
Tác giả: Võ Huỳnh Thiên - B2203472
================================================================
"""

import pandas as pd
from datasets import load_dataset
import os
import sys

print("=" * 60)
print("🌍 BẮT ĐẦU TẢI VÀ TĂNG CƯỜNG DỮ LIỆU (DATA AUGMENTATION)")
print("=" * 60)

# 1. Tải 200 bài viết từ tập TRAIN của XL-Sum (Tin tức chung)
print("\n📥 1. Đang tải tập Train từ XL-Sum...")
try:
    dataset = load_dataset("csebuetnlp/xlsum", "vietnamese", split="train[:200]")
    df_general = pd.DataFrame(dataset)
    print(f"   ✅ Đã tải {len(df_general)} mẫu tin tức chung")
except Exception as e:
    print(f"   ❌ Lỗi tải dataset: {e}")
    sys.exit(1)

# 2. Dữ liệu chuyên ngành (Data Augmentation) do AI cung cấp
print("\n📚 2. Đang nạp dữ liệu tăng cường (Domain: Information Systems)...")
augmented_data = [
    {
        "text": "Hệ thống thông tin quản lý (MIS) là một tập hợp các phần cứng, phần mềm, dữ liệu, quy trình và con người tương tác với nhau để cung cấp thông tin hỗ trợ ra quyết định. MIS đóng vai trò quan trọng trong việc thu thập, lưu trữ và phân tích dữ liệu doanh nghiệp. Các nhà quản lý sử dụng báo cáo từ MIS để tối ưu hóa quy trình làm việc và tăng cường lợi thế cạnh tranh.",
        "summary": "Hệ thống thông tin quản lý (MIS) tích hợp phần cứng, phần mềm và con người để xử lý dữ liệu, hỗ trợ doanh nghiệp ra quyết định và tối ưu hóa quy trình."
    },
    {
        "text": "Cơ sở dữ liệu quan hệ (RDBMS) tổ chức dữ liệu thành các bảng có cấu trúc với hàng và cột. Ngôn ngữ truy vấn có cấu trúc (SQL) được sử dụng rộng rãi để thao tác và quản lý các cơ sở dữ liệu này. Đảm bảo tính toàn vẹn dữ liệu và bảo mật thông tin là ưu tiên hàng đầu khi thiết kế kiến trúc cơ sở dữ liệu cho các ứng dụng quy mô lớn.",
        "summary": "Cơ sở dữ liệu quan hệ sử dụng SQL để quản lý dữ liệu theo dạng bảng, trong đó tính toàn vẹn và bảo mật là yếu tố then chốt khi thiết kế."
    },
    {
        "text": "Điện toán đám mây cung cấp các tài nguyên máy tính như máy chủ, lưu trữ và phần mềm qua internet. Thay vì đầu tư vào phần cứng vật lý, doanh nghiệp có thể thuê hạ tầng ảo hóa (IaaS) hoặc nền tảng (PaaS) để triển khai ứng dụng. Sự linh hoạt và khả năng mở rộng của đám mây giúp giảm thiểu chi phí vận hành cho các dự án phần mềm.",
        "summary": "Điện toán đám mây cho phép doanh nghiệp thuê tài nguyên IT qua internet, mang lại sự linh hoạt, dễ mở rộng và tiết kiệm chi phí vận hành phần cứng."
    },
    {
        "text": "Trí tuệ nhân tạo (AI) và máy học (Machine Learning) đang thay đổi cách các hệ thống thông tin hoạt động. Bằng cách huấn luyện thuật toán trên các tập dữ liệu lớn (Big Data), hệ thống có thể tự động nhận diện mẫu và đưa ra dự đoán chính xác. Khai phá dữ liệu kết hợp với AI giúp thương mại điện tử đề xuất sản phẩm cá nhân hóa cho người dùng.",
        "summary": "Trí tuệ nhân tạo và máy học ứng dụng dữ liệu lớn để huấn luyện thuật toán, giúp hệ thống dự đoán và cá nhân hóa trải nghiệm người dùng."
    },
    {
        "text": "Quy trình phát triển phần mềm Agile tập trung vào tính linh hoạt và sự cộng tác liên tục giữa các nhóm chức năng. Trong Agile, phương pháp Scrum chia dự án thành các vòng lặp ngắn gọi là Sprint. Điều này giúp đội ngũ lập trình nhanh chóng phản hồi các thay đổi từ khách hàng và phát hành các tính năng ứng dụng một cách liên tục và an toàn.",
        "summary": "Quy trình Agile và phương pháp Scrum chia nhỏ dự án thành các Sprint, giúp nhóm phát triển phần mềm linh hoạt đáp ứng thay đổi của khách hàng."
    }
]

df_augmented = pd.DataFrame(augmented_data)
print(f"   ✅ Đã nạp {len(df_augmented)} mẫu chuyên ngành HTTT")

# 3. Trộn dữ liệu (Merge & Shuffle)
print("\n🔀 3. Đang trộn dữ liệu...")
df_final = pd.concat([df_general, df_augmented], ignore_index=True)

# Trộn ngẫu nhiên (Shuffle) các hàng để bài IT xen kẽ bài báo thường
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

# Validate: loại bỏ các mẫu thiếu text hoặc summary
before_count = len(df_final)
df_final = df_final.dropna(subset=['text', 'summary'])
df_final = df_final[df_final['text'].str.strip().astype(bool)]
df_final = df_final[df_final['summary'].str.strip().astype(bool)]
after_count = len(df_final)
if before_count != after_count:
    print(f"   ⚠️ Đã loại bỏ {before_count - after_count} mẫu không hợp lệ")

# 4. Lưu thành file CSV để dùng cho việc Train/Tune tham số
output_file = "augmented_train_dataset.csv"
df_final.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\n" + "=" * 60)
print(f"✅ Hoàn tất! Tập dữ liệu đã được lưu tại: {output_file}")
print(f"📊 Tổng số mẫu: {len(df_final)} bài")
print(f"   - Tin tức chung (XL-Sum): {len(df_general)} bài")
print(f"   - Chuyên ngành HTTT: {len(df_augmented)} bài")
print("=" * 60)