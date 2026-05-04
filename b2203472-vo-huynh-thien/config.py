#config.py
"""
=============================================================================
CẤU HÌNH HỆ THỐNG - AI STUDY ASSISTANT
Luận văn Tốt nghiệp: Ứng dụng NLP trong hỗ trợ học tập
Tác giả: Võ Huỳnh Thiên - B2203472
=============================================================================
"""

import os
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# 1. THÔNG TIN HỆ THỐNG
# =============================================================================
SYSTEM_INFO = {
    "app_name": "AI Study Assistant",
    "version": "2.0 - Thesis Edition",
    "author": "Võ Huỳnh Thiên",
    "student_id": "B2203472",
    "university": "Trường Đại học Cần Thơ",
    "faculty": "Công nghệ Thông tin & Truyền thông",
    "major": "Hệ thống Thông tin",
}

# =============================================================================
# 2. CẤU HÌNH MODEL AI
# =============================================================================
AI_CONFIG = {
    "model_name": "gemini-2.5-flash",

    # Rate limiting (Free Tier: 15 RPM)
    "min_request_interval": 4.5,
    "max_retries": 3,
    "retry_delay": 5,

    # Generation config
    "generation_config": {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "max_output_tokens": 2048,
    }
}

# =============================================================================
# 3. CẤU HÌNH XỬ LÝ VĂN BẢN
# =============================================================================
TEXT_CONFIG = {
    "chunk_size": 150,              # Số TỪ mỗi đoạn (word-based, không phải ký tự)
    "chunk_overlap": 30,            # Số TỪ gối đầu giữa các đoạn
    "default_summary_ratio": 0.25,
    "top_k_retrieval": 5,
}

# =============================================================================
# 4. ONTOLOGY CHUYÊN NGÀNH HỆ THỐNG THÔNG TIN
#    (Mở rộng song ngữ Việt-Anh, dùng underscore cho từ ghép theo chuẩn underthesea)
# =============================================================================
DOMAIN_ONTOLOGY = {
    "AI_ML": {
        # Tiếng Việt (dạng underthesea tokenize)
        "trí_tuệ_nhân_tạo", "máy_học", "học_sâu", "học_máy",
        "khai_phá_dữ_liệu", "xử_lý_ngôn_ngữ_tự_nhiên",
        "mạng_nơ-ron", "thuật_toán", "mô_hình",
        "phân_loại", "hồi_quy", "dự_đoán", "huấn_luyện",
        "tập_huấn_luyện", "tập_kiểm_tra", "overfitting", "underfitting",
        # Tiếng Anh
        "artificial_intelligence", "machine_learning", "deep_learning",
        "nlp", "ai", "ml", "dl", "neural_network",
        "data_mining", "algorithm", "model",
        "classification", "regression", "prediction", "training",
        "transformer", "bert", "gpt", "attention",
        "supervised", "unsupervised", "reinforcement",
    },
    "DATABASE": {
        "cơ_sở_dữ_liệu", "dữ_liệu_lớn", "kho_dữ_liệu",
        "truy_vấn", "bảng", "khóa_chính", "khóa_ngoại",
        "chỉ_mục", "lược_đồ", "quan_hệ", "chuẩn_hóa",
        "database", "sql", "nosql", "query",
        "rdbms", "data_warehouse", "big_data",
        "mongodb", "mysql", "postgresql", "oracle",
        "primary_key", "foreign_key", "index",
        "etl", "olap", "oltp", "schema",
        "table", "view", "trigger", "stored_procedure",
    },
    "SOFTWARE_ENG": {
        "phần_mềm", "mã_nguồn", "hệ_thống",
        "kiểm_thử", "triển_khai", "bảo_trì",
        "yêu_cầu", "thiết_kế", "kiến_trúc",
        "software", "source_code", "system",
        "agile", "scrum", "waterfall", "kanban",
        "api", "testing", "deployment", "devops",
        "ci/cd", "git", "microservice", "monolithic",
        "uml", "use_case", "class_diagram",
        "design_pattern", "mvc", "mvvm",
        "frontend", "backend", "fullstack",
    },
    "INFRASTRUCTURE": {
        "mạng_máy_tính", "đám_mây", "máy_chủ",
        "bảo_mật", "hệ_điều_hành", "băng_thông", "định_tuyến",
        "tường_lửa", "mã_hóa", "xác_thực",
        "computer_network", "network", "cloud", "cloud_computing",
        "server", "security", "operating_system",
        "bandwidth", "routing", "firewall",
        "vpn", "tcp/ip", "http", "https", "dns",
        "docker", "kubernetes", "aws", "azure", "gcp",
        "linux", "windows", "virtualization",
    },
    "BUSINESS_IS": {
        "thương_mại_điện_tử", "chuyển_đổi_số",
        "quản_lý", "doanh_nghiệp", "tích_hợp", "quy_trình",
        "chiến_lược", "phân_tích_nghiệp_vụ",
        "e-commerce", "ecommerce", "digital_transformation",
        "management", "enterprise", "integration", "process",
        "erp", "crm", "bi", "business_intelligence",
        "kpi", "dashboard", "report",
        "stakeholder", "workflow", "automation",
    }
}

# =============================================================================
# 5. PATTERNS NHẬN DIỆN INTENT CỤC BỘ (Không cần API)
# =============================================================================
INTENT_PATTERNS = {
    "DINH_NGHIA": [
        "là gì", "nghĩa là", "khái niệm", "định nghĩa",
        "giải thích", "có nghĩa", "hiểu thế nào", "được hiểu là",
        "what is", "define", "explain", "meaning",
        "hãy cho biết", "em muốn biết", "thế nào là",
    ],
    "SO_SANH": [
        "so sánh", "khác nhau", "giống nhau", "phân biệt",
        "khác gì", "giống gì", "hơn gì", "thua gì",
        "so với", "versus", "vs", "difference",
        "ưu điểm", "nhược điểm", "pros", "cons",
        "nên chọn", "tốt hơn",
    ],
    "TOM_TAT": [
        "tóm tắt", "tổng kết", "tổng hợp", "liệt kê",
        "các ý chính", "điểm chính", "nội dung chính",
        "summarize", "summary", "overview",
        "trình bày ngắn gọn", "nêu các", "kể tên",
    ],
}

# =============================================================================
# 6. VIETNAMESE STOPWORDS (Mở rộng cho NLP tiếng Việt)
# =============================================================================
VIETNAMESE_STOPWORDS = {
    # Đại từ
    "tôi", "bạn", "anh", "chị", "em", "nó", "họ", "chúng_tôi", "chúng_ta",
    "mình", "ta", "tui", "cậu", "ông", "bà",
    # Giới từ
    "của", "và", "với", "trong", "ngoài", "trên", "dưới", "từ", "đến",
    "về", "cho", "bằng", "theo", "qua", "giữa", "tại", "lên", "xuống",
    # Liên từ
    "nhưng", "hoặc", "hay", "mà", "nên", "vì", "để", "nếu", "thì",
    "tuy", "dù", "song", "do", "khi", "lúc",
    # Trợ từ
    "là", "được", "có", "không", "đã", "sẽ", "đang", "rất", "cũng",
    "này", "đó", "kia", "ấy", "những", "các", "một", "hai", "ba",
    "thật", "quá", "lắm", "khá", "hơi",
    # Phụ từ
    "thế", "vậy", "sao", "như", "lại", "còn", "chỉ", "mới", "đều",
    "luôn", "nào", "gì", "đâu", "bao_giờ", "bao_nhiêu",
    # Từ nối câu phổ biến
    "tuy_nhiên", "ngoài_ra", "hơn_nữa", "bên_cạnh", "mặt_khác",
    "vì_vậy", "do_đó", "cho_nên", "bởi_vì", "nhờ_vậy",
    "đồng_thời", "sau_đó", "trước_hết", "cuối_cùng",
    # Từ đệm
    "ừ", "ờ", "à", "ạ", "nhé", "nhỉ", "nha", "hen",
    "okay", "ok", "well",
}

# =============================================================================
# 7. PROMPT TEMPLATES
# =============================================================================
PROMPTS = {
    # 1. Prompt Tóm tắt Hybrid (TextRank → AI Polish)
    "hybrid_summary": """
Bạn là một chuyên gia tóm tắt tài liệu khoa học.
Nhiệm vụ của bạn là tổng hợp nội dung từ các câu quan trọng đã được trích xuất.

DỮ LIỆU ĐẦU VÀO (TextRank):
---
{extracted_text}
---

YÊU CẦU XỬ LÝ:
1. DIỄN ĐẠT LẠI: Không sao chép nguyên văn, hãy dùng văn phong học thuật mạch lạc.
2. CẤU TRÚC: Viết thành một đoạn văn hoàn chỉnh, tập trung vào luận điểm chính.
3. THUẬT NGỮ: Giữ nguyên các thuật ngữ chuyên ngành (Hệ thống thông tin, NLP...).
4. ĐỘ DÀI: Ngắn gọn, súc tích.

BẢN TÓM TẮT:
""",

    # 2. Prompt Hỏi đáp RAG (có gắn Intent + History + Context)
    "rag_qa": """Bạn là Trợ lý AI chuyên sâu hỗ trợ sinh viên ngành Hệ thống Thông tin.
Nhiệm vụ của bạn là giải thích tài liệu dựa trên ngữ cảnh được cung cấp.

YÊU CẦU ĐẶC BIỆT (DỰA TRÊN INTENT: {intent}): {intent_instruction}

LỊCH SỬ TRÒ CHUYỆN GẦN ĐÂY:
---
{history}
---

NGỮ CẢNH TRÍCH XUẤT TỪ TÀI LIỆU (Đã được gắn mã ID):
---
{context}
---

CÂU HỎI HIỆN TẠI CỦA SINH VIÊN: {question}

HƯỚNG DẪN TRẢ LỜI (BẮT BUỘC TUÂN THỦ NGHIÊM NGẶT):
1. ĐỊNH DẠNG: Trả lời dưới dạng danh sách gạch đầu dòng (-). Mỗi dòng diễn đạt đúng 1 ý (Claim).
2. TRÍCH DẪN NGUỒN: Bắt buộc phải gắn thẻ [Nguồn: Cx] ở cuối MỖI DÒNG, trong đó Cx là mã ID của đoạn ngữ cảnh bạn dùng (Ví dụ: "- Cơ sở dữ liệu quan hệ lưu trữ dạng bảng [Nguồn: C1, C3]").
3. CHỐNG ẢO GIÁC: Nếu câu hỏi nằm ngoài NGỮ CẢNH, TUYỆT ĐỐI KHÔNG tự bịa thông tin. Hãy trả về đúng một dòng: "[NO_CONTEXT] Tài liệu hiện tại không đề cập đến vấn đề này."
4. VĂN PHONG: Ngắn gọn, súc tích, mang tính hàn lâm.

TRẢ LỜI:
""",

    # 3. Prompt Giải thích khái niệm
    "explain_concept": """
Bạn hãy đóng vai giảng viên đại học, giải thích khái niệm sau dựa trên tài liệu nghiên cứu.

KHÁI NIỆM: {concept}
NGỮ CẢNH: {context}

CẤU TRÚC TRẢ LỜI:
1. Định nghĩa chuẩn (Ngắn gọn).
2. Cách tài liệu này áp dụng/đề cập đến khái niệm đó.
3. Một ví dụ minh họa dựa trên Hệ thống thông tin.
"""
}

# =============================================================================
# 8. CẤU HÌNH GIAO DIỆN
# =============================================================================
UI_CONFIG = {
    "page_title": "AI Study Assistant - Thesis Demo",
    "page_icon": "🎓",
    "layout": "wide",
    "chat_max_height": 500,
    "summary_max_height": 500,
}