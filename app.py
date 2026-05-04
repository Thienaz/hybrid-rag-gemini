# app.py
"""
================================================================
GIAO DIỆN NGƯỜI DÙNG - AI STUDY ASSISTANT (MULTI-DOC VERSION)
Luận văn Tốt nghiệp: Hệ thống tóm tắt văn bản tài liệu học tập
                      hỗ trợ sinh viên ngành Hệ thống Thông tin
Tác giả: Võ Huỳnh Thiên - B2203472
================================================================
"""

import os
import streamlit as st
import pandas as pd
import base64
import uuid

from core.nlp_engine import NLPEngine
from core.evaluator import SummaryEvaluator
from utils.file_handler import FileHandler
from utils.logger import logger
from utils.session_manager import SessionManager
from config import SYSTEM_INFO, UI_CONFIG

# Thử import pdf_viewer, fallback nếu chưa cài
try:
    from streamlit_pdf_viewer import pdf_viewer
    HAS_PDF_VIEWER = True
except ImportError:
    HAS_PDF_VIEWER = False

# ==========================================
# CẤU HÌNH TRANG
# ==========================================
st.set_page_config(
    page_title=UI_CONFIG["page_title"],
    page_icon=UI_CONFIG["page_icon"],
    layout=UI_CONFIG.get("layout", "wide")
)

st.markdown("""
<style>
    /* Reset mọi thứ về mặc định để tránh xung đột */
    .main {background-color: #f5f7fa;}
    .stButton>button {width: 100%; border-radius: 8px; font-weight: 600;}
    
    /* Khung chat chính - TỐI ƯU ĐỘ RỘNG VÀ KHOẢNG CÁCH */
    .user-msg, .bot-msg {
        padding: 20px 25px !important;    /* Đồng nhất padding */
        max-width: 90% !important;        /* Giảm từ 92% xuống 90% để đẹp hơn */
        min-height: 50px;
        border-radius: 20px 20px 5px 20px; /* Bo góc mềm mại */
        margin: 12px 0 !important;        /* Khoảng cách giữa các khung */
        font-size: 16px;                  /* Giảm 1px để dễ đọc hơn */
        line-height: 1.5;
        clear: both;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08); /* Giảm độ bóng */
    }
    
    .user-msg {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; 
        float: right; 
        border-radius: 20px 20px 5px 20px;
    }
    
    .bot-msg {
        background: white; 
        color: #333; 
        float: left; 
        border-radius: 20px 20px 20px 5px;
        border: 1px solid #e0e0e0;
    }
    
    /* LOẠI BỎ KHUNG NHỎ KHÔNG CẦN THIẾT */
    .stExpander, .stContainer {
        border: none !important;
        box-shadow: none !important;
        background: transparent !important;
    }
    
    /* Khung tóm tắt */
    .summary-box {
        background: white; 
        padding: 18px; 
        border-radius: 10px;
        border: 1px solid #e0e0e0; 
        max-height: 400px;
        overflow-y: auto; 
        line-height: 1.7;
        font-size: 15px;
    }
    
    .thesis-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; 
        padding: 20px; 
        border-radius: 12px; 
        margin-bottom: 20px;
    }
    
    .session-card {
        background-color: #ffffff; 
        border-radius: 8px; 
        padding: 10px;
        margin-bottom: 10px; 
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    
    .overall-score-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white; 
        padding: 15px; 
        border-radius: 12px;
        text-align: center; 
        font-size: 1.2em; 
        font-weight: bold;
        margin: 10px 0;
    }
    
    .metric-card {
        background: #f8f9fa; 
        border-radius: 8px; 
        padding: 12px;
        border: 1px solid #e9ecef; 
        margin: 5px 0;
    }
    
    /* TỐI ƯU EXPANDER - TRÁNH TẠO KHUNG LỒNG NHAU */
    .stExpander > div:first-child {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    /* Đảm bảo phần phân tích không bị lệch */
    .analysis-section {
        width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# CÁC HÀM XỬ LÝ TRẠNG THÁI (SESSIONS)
# ==========================================
def create_new_session():
    """Tạo một phiên làm việc hoàn toàn mới, trống rỗng"""
    st.session_state.current_session_id = str(uuid.uuid4())
    st.session_state.engine = NLPEngine()
    st.session_state.chat_history = []
    st.session_state.doc_processed = False
    st.session_state.file_metadata = {}
    st.session_state.original_text = ""
    st.session_state.uploaded_files_data = {}
    st.session_state.is_processed = False
    if 'current_summary' in st.session_state:
        del st.session_state.current_summary


def load_selected_session(session_id):
    """Nạp một thẻ phiên cũ vào bộ nhớ"""
    data = SessionManager.load_session(session_id)
    if data:
        st.session_state.current_session_id = data["session_id"]
        st.session_state.doc_processed = data.get("doc_processed", False)
        st.session_state.is_processed = data.get("is_processed", False)
        st.session_state.original_text = data.get("original_text", "")
        st.session_state.chat_history = data.get("chat_history", [])
        st.session_state.current_summary = data.get("current_summary", {})
        st.session_state.file_metadata = data.get("file_metadata", {})
        st.session_state.uploaded_files_data = data.get("uploaded_files_data", {})

        st.session_state.engine = NLPEngine()
        if st.session_state.original_text:
            st.session_state.engine.preprocess(st.session_state.original_text)


if 'current_session_id' not in st.session_state:
    create_new_session()


# ==========================================
# CÁC HÀM TIỆN ÍCH GIAO DIỆN - ĐÃ TỐI ƯU
# ==========================================
def display_pdf(file_name):
    """Hiển thị file PDF hoặc nội dung văn bản đã trích xuất"""
    if file_name in st.session_state.uploaded_files_data:
        bytes_data = st.session_state.uploaded_files_data[file_name]
        if file_name.lower().endswith('.pdf') and HAS_PDF_VIEWER:
            pdf_viewer(bytes_data, width=700, height=800)
        else:
            if not file_name.lower().endswith('.pdf'):
                st.info(f"📄 File `{file_name}` không phải định dạng PDF.")
            else:
                st.info("📄 Thư viện xem PDF chưa được cài đặt. Hiển thị văn bản thay thế.")
            with st.expander("👀 Xem trước nội dung văn bản đã trích xuất", expanded=True):
                st.text_area("", st.session_state.original_text, height=600)
    else:
        st.warning("⚠️ Không tìm thấy file gốc.")


def render_chat(placeholder):
    """
    Hiển thị lịch sử trò chuyện - ĐÃ TỐI ƯU ĐỂ TRÁNH KHUNG NHỎ LỒNG NHAU
    """
    chat_height = UI_CONFIG.get("chat_max_height", 500)
    with placeholder.container(height=chat_height):
        for item in st.session_state.chat_history:
            role = item[0]
            content = item[1]
            context = item[2] if len(item) > 2 else ""
            evaluation = item[3] if len(item) > 3 else None

            if role == "user":
                st.markdown(
                    f'<div class="user-msg">🧑‍🎓 {content}</div>'
                    f'<div style="clear:both"></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="bot-msg">🤖 {content}</div>'
                    f'<div style="clear:both"></div>',
                    unsafe_allow_html=True
                )

                # PHẦN PHÂN TÍCH - ĐÃ TÁCH RA KHỎI KHUNG CHAT CHÍNH
                if context or evaluation:
                    with st.expander("🔬 Phân tích Chuyên môn", expanded=False):
                        # Đảm bảo không tạo thêm khung bên trong
                        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                        
                        if context:
                            st.markdown("**1. NGỮ CẢNH TRÍCH XUẤT (Context Retrieved):**")
                            st.info(f"*{context[:500]}{'...' if len(context) > 500 else ''}*")

                        if isinstance(evaluation, dict):
                            st.divider()
                            st.markdown("**2. ĐÁNH GIÁ CHẤT LƯỢNG RAG (Faithfulness Evaluation):**")
                            c1, c2, c3, c4 = st.columns(4)

                            cosine_val = evaluation.get('cosine_similarity', 0) * 100
                            c1.metric(
                                "Semantic Sim",
                                f"{cosine_val:.1f}%",
                                help="Cosine Similarity giữa Context và Answer (TF-IDF)"
                            )

                            precision_val = evaluation.get('word_precision', 0) * 100
                            c2.metric(
                                "Word Precision",
                                f"{precision_val:.1f}%",
                                help="Tỷ lệ từ trong Answer có xuất hiện trong Context"
                            )

                            hal = evaluation.get('hallucination_rate', 0) * 100
                            hal_status = "An toàn" if hal <= 40 else "Cần kiểm tra"
                            c3.metric(
                                "Hallucination",
                                f"{hal:.1f}%",
                                delta=hal_status,
                                delta_color="inverse" if hal <= 40 else "normal",
                                help="Tỷ lệ từ trong Answer KHÔNG có trong Context (1 - Precision)"
                            )

                            rel = evaluation.get('intent_relevance', 0) * 100
                            c4.metric(
                                "Intent Relevance",
                                f"{rel:.1f}%",
                                help="LLM-as-Judge: Mức độ câu trả lời bám sát đúng Ý định (Intent)"
                            )
                        
                        st.markdown('</div>', unsafe_allow_html=True)


def render_metrics_table(metrics):
    """
    Xây dựng bảng metrics toàn diện - ĐÃ TỐI ƯU ĐỂ TRÁNH KHUNG NHỎ
    """
    if not metrics:
        return pd.DataFrame({"Độ đo": ["Chưa có dữ liệu"], "Giá trị": ["-"]})

    rows = []

    # --- ROUGE ---
    rouge_1 = metrics.get('rouge_1', {})
    rouge_2 = metrics.get('rouge_2', {})
    rouge_l = metrics.get('rouge_l', {})

    rows.append(("ROUGE-1 (F1)", f"{rouge_1.get('f1', 0) * 100:.2f}%"))
    rows.append(("ROUGE-1 (Precision)", f"{rouge_1.get('precision', 0) * 100:.2f}%"))
    rows.append(("ROUGE-1 (Recall)", f"{rouge_1.get('recall', 0) * 100:.2f}%"))
    rows.append(("ROUGE-2 (F1)", f"{rouge_2.get('f1', 0) * 100:.2f}%"))
    rows.append(("ROUGE-L (F1)", f"{rouge_l.get('f1', 0) * 100:.2f}%"))

    # --- BLEU ---
    bleu = metrics.get('bleu', {})
    rows.append(("BLEU-1", f"{bleu.get('bleu_1', 0) * 100:.2f}%"))
    rows.append(("BLEU-2", f"{bleu.get('bleu_2', 0) * 100:.2f}%"))
    rows.append(("BLEU-3", f"{bleu.get('bleu_3', 0) * 100:.2f}%"))
    rows.append(("BLEU-4", f"{bleu.get('bleu_4', 0) * 100:.2f}%"))
    rows.append(("BLEU (Avg)", f"{bleu.get('bleu_avg', 0) * 100:.2f}%"))

    # --- Similarity ---
    rows.append(("Cosine Similarity", f"{metrics.get('cosine_similarity', 0) * 100:.2f}%"))
    rows.append(("Jaccard Similarity", f"{metrics.get('jaccard_similarity', 0) * 100:.2f}%"))

    # --- Compression ---
    compression = metrics.get('compression', {})
    rows.append(("Tỷ lệ nén (CR)", f"{compression.get('ratio', 0) * 100:.2f}%"))
    rows.append(("Giảm dung lượng", f"{compression.get('reduction_percent', 0):.1f}%"))
    rows.append(("Từ gốc / Tóm tắt", f"{compression.get('original_words', 0)} → {compression.get('summary_words', 0)}"))

    # --- Keyword Coverage ---
    kw = metrics.get('keyword_coverage', {})
    rows.append(("Keyword Coverage", f"{kw.get('coverage', 0) * 100:.2f}%"))
    rows.append(("Từ khóa giữ lại", f"{kw.get('keywords_found', 0)}/{kw.get('total_keywords', 0)}"))

    # --- Information Density ---
    info_den = metrics.get('information_density', {})
    rows.append(("Information Density", f"{info_den.get('density', 0) * 100:.2f}%"))

    # --- Overall Score ---
    overall = metrics.get('overall_score', 0)
    rows.append(("⭐ OVERALL SCORE", f"{overall * 100:.2f}%"))

    data = {"Độ đo": [r[0] for r in rows], "Giá trị": [r[1] for r in rows]}
    return pd.DataFrame(data)


def render_overall_score(metrics):
    """Hiển thị Overall Score nổi bật - KHÔNG TẠO KHUNG NHỎ"""
    overall = metrics.get('overall_score', 0)
    score_pct = overall * 100

    if score_pct >= 70:
        grade = "🟢 Rất tốt"
        color = "#11998e"
    elif score_pct >= 50:
        grade = "🟡 Khá"
        color = "#f2994a"
    else:
        grade = "🔴 Cần cải thiện"
        color = "#eb5757"

    st.markdown(
        f'<div class="overall-score-box">'
        f'⭐ Overall Score: {score_pct:.2f}% — {grade}</div>',
        unsafe_allow_html=True
    )


def render_metrics_visual(metrics):
    """
    Hiển thị metrics dạng visual - ĐÃ TỐI ƯU ĐỂ TRÁNH KHUNG NHỎ
    """
    rouge_1 = metrics.get('rouge_1', {})
    rouge_2 = metrics.get('rouge_2', {})
    rouge_l = metrics.get('rouge_l', {})
    bleu = metrics.get('bleu', {})
    compression = metrics.get('compression', {})
    kw = metrics.get('keyword_coverage', {})
    info_den = metrics.get('information_density', {})

    # Row 1: ROUGE scores - KHÔNG TẠO CỘT NHỎ BÊN TRONG
    st.markdown("##### 📊 ROUGE Scores")
    c1, c2, c3 = st.columns(3)
    c1.metric("ROUGE-1 (F1)", f"{rouge_1.get('f1', 0) * 100:.2f}%", help="Độ trùng khớp unigram")
    c2.metric("ROUGE-2 (F1)", f"{rouge_2.get('f1', 0) * 100:.2f}%", help="Độ trùng khớp bigram")
    c3.metric("ROUGE-L (F1)", f"{rouge_l.get('f1', 0) * 100:.2f}%", help="Longest Common Subsequence")

    # Row 2: BLEU & Similarity
    st.markdown("##### 📐 BLEU & Similarity")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("BLEU-4", f"{bleu.get('bleu_4', 0) * 100:.2f}%", help="Độ chính xác 4-gram")
    c2.metric("BLEU (Avg)", f"{bleu.get('bleu_avg', 0) * 100:.2f}%", help="BLEU trung bình")
    c3.metric("Cosine Sim", f"{metrics.get('cosine_similarity', 0) * 100:.2f}%", help="Tương đồng vector")
    c4.metric("Jaccard", f"{metrics.get('jaccard_similarity', 0) * 100:.2f}%", help="Tỷ lệ từ chung")

    # Row 3: Compression & Coverage
    st.markdown("##### 📦 Nén & Bao phủ")
    c1, c2, c3 = st.columns(3)
    c1.metric("Tỷ lệ nén", f"{compression.get('ratio', 0) * 100:.1f}%", delta=f"Giảm {compression.get('reduction_percent', 0):.0f}%")
    c2.metric("Keyword Coverage", f"{kw.get('coverage', 0) * 100:.1f}%", help="Tỷ lệ từ khóa giữ lại")
    c3.metric("Info Density", f"{info_den.get('density', 0) * 100:.1f}%", help="Mật độ thông tin")


# ==========================================
# SIDEBAR - QUẢN LÝ THẺ PHIÊN LÀM VIỆC
# ==========================================
with st.sidebar:
    st.markdown(
        f'<div class="thesis-header">'
        f'<h3>🎓 {SYSTEM_INFO["app_name"]}</h3>'
        f'<p style="font-size:0.85em; margin:0;">'
        f'{SYSTEM_INFO.get("version", "")}</p>'
        f'</div>',
        unsafe_allow_html=True
    )

    if st.button("➕ Tạo phiên làm việc mới", type="primary", use_container_width=True):
        create_new_session()
        st.rerun()

    st.divider()
    st.markdown("##### 📚 Lịch sử thao tác")
    saved_sessions = SessionManager.get_all_sessions()

    if not saved_sessions:
        st.caption("Chưa có phiên làm việc nào được lưu.")
    else:
        for s in saved_sessions:
            is_active = (s['id'] == st.session_state.current_session_id)
            c1, c2 = st.columns([5, 1])
            disp_name = s['name'][:25] + "..." if len(s['name']) > 25 else s['name']

            if c1.button(
                f"{'👉' if is_active else '📄'} {disp_name}",
                key=f"load_{s['id']}",
                help=s['name'],
                use_container_width=True
            ):
                if not is_active:
                    load_selected_session(s['id'])
                    st.rerun()

            if c2.button("❌", key=f"del_{s['id']}", help="Xóa thẻ này"):
                SessionManager.delete_session(s['id'])
                if is_active:
                    create_new_session()
                st.rerun()

    st.divider()
    st.markdown("##### 📁 Tài liệu của Phiên này")

    if st.session_state.uploaded_files_data:
        st.markdown(
            "<p style='font-size:14px; color:#555;'>Đang mở:</p>",
            unsafe_allow_html=True
        )
        for fname in st.session_state.uploaded_files_data.keys():
            st.success(f"📎 {fname}")

    uploaded_files = st.file_uploader(
        "Tải tài liệu mới lên phiên này",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("🚀 Xử lý tài liệu", type="primary", use_container_width=True):
            with st.spinner("📚 Đang phân tích tài liệu..."):
                try:
                    text, metadata = FileHandler.read_multiple_files(uploaded_files)
                    st.session_state.uploaded_files_data = {
                        f.name: f.getvalue() for f in uploaded_files
                    }
                    st.session_state.file_metadata = metadata
                    st.session_state.original_text = text

                    num_sentences = st.session_state.engine.preprocess(
                        text,
                        session_id=st.session_state.current_session_id
                    )
                    if num_sentences > 0:
                        st.session_state.doc_processed = True
                        st.session_state.is_processed = True
                        st.session_state.chat_history = []
                        if 'current_summary' in st.session_state:
                            del st.session_state.current_summary

                        img_count = metadata.get("total_images_detected", 0)
                        st.success(
                            f"✅ Đã tải {len(uploaded_files)} file! "
                            f"(Bóc tách được {num_sentences} đoạn văn bản"
                            f" và {img_count} hình ảnh)."
                        )

                        SessionManager.save_session(st.session_state)
                        st.rerun()
                    else:
                        st.error("❌ Không tìm thấy nội dung hợp lệ trong tài liệu.")
                except Exception as e:
                    st.error(f"❌ Lỗi khi xử lý tài liệu: {str(e)}")
                    logger.error(f"Lỗi xử lý tài liệu: {e}")

    if st.session_state.doc_processed:
        st.divider()
        st.markdown("##### ⚙️ Tóm tắt Tài liệu")
        use_ai_polish = st.checkbox("✨ Dùng AI Gemini làm mượt", value=True)

        if st.button("📝 TẠO BẢN TÓM TẮT", type="primary", use_container_width=True):
            with st.spinner("🔄 Hệ thống đang xử lý tóm tắt..."):
                result = st.session_state.engine.generate_summary(
                    use_ai_polish=use_ai_polish
                )
                st.session_state.current_summary = result
                SessionManager.save_session(st.session_state)
                st.toast("✅ Đã hoàn tất tóm tắt!")

        # Thống kê API
        stats = st.session_state.engine.get_statistics()
        if stats.get("api_calls", 0) > 0:
            st.caption(f"📡 API calls: {stats['api_calls']} | 📄 Chunks: {stats['sentences']}")

        # Trong phần with st.sidebar:
        st.divider()
        st.header("⚙️ Cấu hình hiển thị")
        # Tạo biến session_state để lưu trạng thái bật/tắt
        if 'show_document_viewer' not in st.session_state:
            st.session_state.show_document_viewer = True

        st.session_state.show_document_viewer = st.toggle(
            "Hiển thị Trình xem tài liệu", 
            value=st.session_state.show_document_viewer,
            help="Bật/Tắt khung hiển thị nội dung PDF/Word ở bên phải"
        )

        # Nút xem dấu vết (Log Trace)
        st.divider()
        st.markdown("##### 🕵️‍♂️ Debug & Trace Log")
        log_path = os.path.join(
            SessionManager.get_session_dir(st.session_state.current_session_id),
            "trace.log"
        )
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                log_content = f.read()
            st.download_button(
                label="📥 Tải file Trace Log",
                data=log_content,
                file_name=f"trace_{st.session_state.current_session_id[:8]}.log",
                mime="text/plain",
                use_container_width=True
            )
            with st.expander("👀 Xem nhanh Log", expanded=False):
                st.code(log_content[-3000:] if len(log_content) > 3000 else log_content, language="log")
        else:
            st.caption("Chưa có trace log cho phiên này.")

    # Thông tin hệ thống
    st.divider()
    with st.expander("ℹ️ Thông tin hệ thống", expanded=False):
        st.caption(f"**Tác giả:** {SYSTEM_INFO.get('author', 'N/A')}")
        st.caption(f"**MSSV:** {SYSTEM_INFO.get('student_id', 'N/A')}")
        st.caption(f"**Trường:** {SYSTEM_INFO.get('university', 'N/A')}")
        st.caption(f"**Ngành:** {SYSTEM_INFO.get('major', 'N/A')}")
        st.caption(f"**Phiên bản:** {SYSTEM_INFO.get('version', 'N/A')}")


# ==========================================
# MAIN CONTENT (BÊN PHẢI) - ĐÃ TỐI ƯU
# ==========================================
if not st.session_state.doc_processed:
    st.markdown(
        f'<div class="thesis-header" style="text-align:center;">'
        f'<h2>🎓 {SYSTEM_INFO["app_name"]}</h2>'
        f'<p>Hệ thống tóm tắt văn bản tài liệu học tập<br>'
        f'hỗ trợ sinh viên ngành Hệ thống Thông tin</p>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.info(
        "👋 **Chào mừng bạn!** Hãy bắt đầu bằng cách tải tài liệu lên "
        "thanh bên trái hoặc chọn một Phiên làm việc cũ."
    )

    # Hướng dẫn nhanh
    with st.expander("📖 Hướng dẫn sử dụng", expanded=True):
        st.markdown("""
        **Bước 1:** Tải lên tài liệu (PDF, DOCX, TXT) ở thanh bên trái.

        **Bước 2:** Nhấn **"🚀 Xử lý tài liệu"** để hệ thống phân tích.

        **Bước 3:** Sử dụng các tính năng:
        - 💬 **Hỏi đáp AI** — Đặt câu hỏi về nội dung tài liệu (RAG)
        - 📄 **Tóm tắt** — Xem bản tóm tắt tự động (TextRank + AI)
        - ⚖️ **So sánh** — So sánh giữa TextRank gốc và AI Gemini
        - 📊 **Đánh giá** — Xem các độ đo chất lượng (ROUGE, BLEU, Cosine...)

        **Tính năng đặc biệt:**
        - Hỗ trợ **đa tài liệu** (upload nhiều file cùng lúc)
        - **Intent Recognition** tự động nhận diện ý định câu hỏi
        - **Faithfulness Evaluation** đánh giá độ trung thực câu trả lời
        - **Session Management** lưu/khôi phục phiên làm việc
        """)

else:
    # KIỂM TRA TRẠNG THÁI NÚT GẠT (TOGGLE)
    if st.session_state.get('show_document_viewer', True):
        # CHẾ ĐỘ 2 CỘT: Hiển thị cả Trình xem tài liệu và Chat
        col_pdf, col_ai = st.columns([1, 1], gap="large")

        with col_pdf:
            st.subheader("📖 Trình xem tài liệu")
            file_list = list(st.session_state.uploaded_files_data.keys())
            if file_list:
                selected_file = st.selectbox("Chọn file để xem:", file_list)
                display_pdf(selected_file)

                # Hiển thị metadata file
                if st.session_state.file_metadata:
                    with st.expander("📋 Metadata tài liệu", expanded=False):
                        meta = st.session_state.file_metadata
                        st.write(f"**Tổng file:** {meta.get('total_files', 0)}")
                        st.write(f"**Tổng hình ảnh phát hiện:** {meta.get('total_images_detected', 0)}")
                        for fname, details in meta.get('file_details', {}).items():
                            st.caption(
                                f"📄 {fname}: {details.get('text_length', 0):,} ký tự, "
                                f"{details.get('images_extracted', 0)} hình"
                            )
    else:
        col_ai = st.container()

    with col_ai:
        tab_chat, tab_summary, tab_compare, tab_metrics, tab_mindmap, tab_flashcard = st.tabs([
            "💬 Hỏi đáp AI", "📄 Tóm tắt", "⚖️ So sánh", "📊 Đánh giá", "🧠 Mindmap", "🃏 Thẻ ôn tập"
        ])

        # ---------- TAB CHAT - CHUẨN NATIVE STREAMLIT ----------
        with tab_chat:
            st.markdown("### 🤖 Trợ lý Học tập RAG")

            # 1. Khung cuộn cố định chứa toàn bộ lịch sử (Chỉ tạo 1 lần)
            chat_height = UI_CONFIG.get("chat_max_height", 650)
            chat_container = st.container(height=chat_height)

            # Hàm nội bộ để render giao diện chuẩn của Streamlit
            def _draw_native_message(role, content, context, evaluation):
                ui_role = "assistant" if role == "bot" else "user"
                # st.chat_message tự động tạo Avatar và khung chat chuẩn
                with st.chat_message(ui_role):
                    st.markdown(content)
                    
                    # Phân tích chuyên môn (chỉ vẽ khi là AI trả lời)
                    if ui_role == "assistant" and (context or evaluation):
                        with st.expander("🔬 Phân tích Chuyên môn", expanded=False):
                            if context:
                                st.markdown("**1. NGỮ CẢNH TRÍCH XUẤT (Context Retrieved):**")
                                st.info(f"*{context[:500]}{'...' if len(context) > 500 else ''}*")

                            if isinstance(evaluation, dict):
                                st.divider()
                                st.markdown("**2. ĐÁNH GIÁ CHẤT LƯỢNG RAG (Faithfulness Evaluation):**")
                                c1, c2, c3, c4 = st.columns(4)
                                cosine_val = evaluation.get('cosine_similarity', 0) * 100
                                c1.metric("Semantic Sim", f"{cosine_val:.1f}%")

                                precision_val = evaluation.get('word_precision', 0) * 100
                                c2.metric("Word Precision", f"{precision_val:.1f}%")

                                hal = evaluation.get('hallucination_rate', 0) * 100
                                hal_status = "An toàn" if hal <= 40 else "Cần kiểm tra"
                                c3.metric("Hallucination", f"{hal:.1f}%", delta=hal_status, delta_color="inverse" if hal <= 40 else "normal")

                                rel = evaluation.get('intent_relevance', 0) * 100
                                c4.metric("Intent Relevance", f"{rel:.1f}%")

            # 2. Đổ lịch sử CŨ vào khung chat
            with chat_container:
                if not st.session_state.chat_history:
                    st.info("👋 Hãy đặt câu hỏi đầu tiên về tài liệu của bạn!")
                else:
                    for item in st.session_state.chat_history:
                        _draw_native_message(item[0], item[1], item[2] if len(item)>2 else "", item[3] if len(item)>3 else None)

            # 3. Lắng nghe và xử lý câu hỏi MỚI
            if prompt := st.chat_input("Hỏi AI về tài liệu trong Phiên này..."):
                
                # A. Lưu và vẽ ngay câu hỏi của User
                st.session_state.chat_history.append(("user", prompt, "", None))
                with chat_container:
                    _draw_native_message("user", prompt, "", None)

                # B. Vẽ khung chờ xử lý cho AI
                with chat_container:
                    with st.chat_message("assistant"):
                        with st.spinner("🤔 Đang phân tích Ý định (Intent) và trích xuất ngữ cảnh..."):
                            result = st.session_state.engine.query_document(
                                prompt,
                                chat_history=st.session_state.chat_history,
                                session_id=st.session_state.current_session_id
                            )

                            answer = result.get('answer', '')
                            context = result.get('context', '')
                            intent_label = result.get('intent', 'KHAC')
                            source = result.get('source', 'unknown')

                        st.toast(f"🎯 Intent: **{intent_label}** | Source: {source}")

                        evaluation_metrics = None
                        if context and answer and "⚠" not in answer and "❌" not in answer:
                            with st.spinner("🔬 Đang kiểm định Answer Relevance..."):
                                evaluation_metrics = st.session_state.engine.evaluate_rag_faithfulness(
                                    context, answer, prompt, intent_label
                                )

                        # Vẽ nội dung câu trả lời
                        st.markdown(answer)
                        
                        # Vẽ khung phân tích nếu có
                        if context or evaluation_metrics:
                            with st.expander("🔬 Phân tích Chuyên môn", expanded=False):
                                if context:
                                    st.markdown("**1. NGỮ CẢNH TRÍCH XUẤT (Context Retrieved):**")
                                    st.info(f"*{context[:500]}{'...' if len(context) > 500 else ''}*")

                                if isinstance(evaluation_metrics, dict):
                                    st.divider()
                                    st.markdown("**2. ĐÁNH GIÁ CHẤT LƯỢNG RAG (Faithfulness Evaluation):**")
                                    c1, c2, c3, c4 = st.columns(4)
                                    cosine_val = evaluation_metrics.get('cosine_similarity', 0) * 100
                                    c1.metric("Semantic Sim", f"{cosine_val:.1f}%")

                                    precision_val = evaluation_metrics.get('word_precision', 0) * 100
                                    c2.metric("Word Precision", f"{precision_val:.1f}%")

                                    hal = evaluation_metrics.get('hallucination_rate', 0) * 100
                                    hal_status = "An toàn" if hal <= 40 else "Cần kiểm tra"
                                    c3.metric("Hallucination", f"{hal:.1f}%", delta=hal_status, delta_color="inverse" if hal <= 40 else "normal")

                                    rel = evaluation_metrics.get('intent_relevance', 0) * 100
                                    c4.metric("Intent Relevance", f"{rel:.1f}%")

                # C. Lưu câu trả lời của AI vào lịch sử
                st.session_state.chat_history.append(("bot", answer, context, evaluation_metrics))
                SessionManager.save_session(st.session_state)

        # ---------- TAB TÓM TẮT ----------
        with tab_summary:
            if 'current_summary' in st.session_state and st.session_state.current_summary:
                st.markdown("### 📝 Bản tóm tắt tài liệu")
                summaries = st.session_state.current_summary
                doc_names = list(summaries.keys())

                if len(doc_names) > 0:
                    tabs = st.tabs(doc_names)
                    for i, doc_name in enumerate(doc_names):
                        doc_data = summaries[doc_name]
                        with tabs[i]:
                            # Badge nguồn tóm tắt
                            source = doc_data.get('source', 'unknown')
                            source_labels = {
                                "hybrid_gemini_flash": "🤖 Hybrid (TextRank + AI Gemini)",
                                "textrank": "🔷 TextRank thuần (Extractive)",
                                "error_fallback_textrank": "⚠️ Dự phòng TextRank (API lỗi)"
                            }
                            st.info(f"Nguồn: **{source_labels.get(source, source)}**")

                            # Hiển thị bản tóm tắt
                            summary_height = UI_CONFIG.get("summary_max_height", 500)
                            st.markdown(
                                f"<div class='summary-box' style='max-height:{summary_height}px'>"
                                f"{doc_data['final_summary']}"
                                f"</div>",
                                unsafe_allow_html=True
                            )

                            # Overall Score nổi bật
                            doc_metrics = doc_data.get('metrics', {})
                            if doc_metrics and doc_metrics.get('overall_score', 0) > 0:
                                render_overall_score(doc_metrics)

                            # Xem bản trích xuất thô
                            with st.expander("📋 Xem bản trích xuất thô (TextRank)"):
                                st.write(doc_data.get('textrank_summary', 'Không có'))

                            # Thống kê nhanh
                            comp = doc_metrics.get('compression', {})
                            if comp:
                                st.caption(
                                    f"📏 Gốc: {comp.get('original_words', 0)} từ → "
                                    f"Tóm tắt: {comp.get('summary_words', 0)} từ "
                                    f"(giảm {comp.get('reduction_percent', 0):.0f}%)"
                                )
            else:
                st.warning(
                    "📝 Chưa có bản tóm tắt nào cho Phiên này. "
                    "Hãy nhấn **'📝 TẠO BẢN TÓM TẮT'** ở thanh bên."
                )

        # ---------- TAB SO SÁNH ----------
        with tab_compare:
            if 'current_summary' in st.session_state and st.session_state.current_summary:
                st.markdown("### ⚖️ So sánh Thuật toán")
                summaries = st.session_state.current_summary
                doc_names = list(summaries.keys())

                if len(doc_names) > 0:
                    tabs = st.tabs(doc_names)
                    for i, doc_name in enumerate(doc_names):
                        doc_data = summaries[doc_name]
                        with tabs[i]:
                            c1, c2 = st.columns(2)
                            with c1:
                                st.markdown("#### 🔷 TextRank (Extractive)")
                                tr_text = doc_data.get("textrank_summary", "")
                                st.markdown(
                                    f'<div class="summary-box" style="height:350px">'
                                    f'{tr_text}</div>',
                                    unsafe_allow_html=True
                                )
                                tr_words = len(tr_text.split()) if tr_text else 0
                                st.caption(f"📏 {tr_words} từ")

                            with c2:
                                st.markdown("#### 🤖 AI Gemini (Abstractive)")
                                ai_text = doc_data.get("ai_polished_summary", "")
                                if ai_text:
                                    st.markdown(
                                        f'<div class="summary-box" style="height:350px">'
                                        f'{ai_text}</div>',
                                        unsafe_allow_html=True
                                    )
                                    ai_words = len(ai_text.split())
                                    st.caption(f"📏 {ai_words} từ")
                                else:
                                    st.info("Không có bản AI (chưa bật hoặc API lỗi).")

                            # So sánh nhanh số liệu
                            if tr_text and ai_text:
                                st.divider()
                                st.markdown("##### 📊 So sánh nhanh")
                                evaluator = SummaryEvaluator()
                                raw_text_for_eval = st.session_state.engine.docs_data.get(
                                    doc_name, {}
                                ).get("raw_text", st.session_state.original_text)[:15000]

                                cc1, cc2 = st.columns(2)
                                with cc1:
                                    quick_tr = evaluator.quick_evaluate(raw_text_for_eval, tr_text)
                                    st.metric("TextRank ROUGE-1", f"{quick_tr['rouge_1_f1']*100:.1f}%")
                                    st.metric("TextRank Cosine", f"{quick_tr['cosine_sim']*100:.1f}%")

                                with cc2:
                                    quick_ai = evaluator.quick_evaluate(raw_text_for_eval, ai_text)
                                    st.metric("AI ROUGE-1", f"{quick_ai['rouge_1_f1']*100:.1f}%")
                                    st.metric("AI Cosine", f"{quick_ai['cosine_sim']*100:.1f}%")
            else:
                st.warning("Chưa có bản tóm tắt để so sánh.")

        # ---------- TAB ĐÁNH GIÁ ----------
        with tab_metrics:
            if 'current_summary' in st.session_state and st.session_state.current_summary:
                st.markdown("### 📋 Đánh giá Chất lượng Tóm tắt")
                summaries = st.session_state.current_summary

                for doc_name, doc_data in summaries.items():
                    doc_metrics = doc_data.get('metrics', {})

                    with st.expander(f"📊 Kết quả: {doc_name}", expanded=True):
                        if doc_metrics and doc_metrics.get('overall_score', 0) > 0:
                            # Tăng tỷ lệ cột giữa (6) để các số có 4-5 ký tự (VD: 100.0%) không bị cắt
                            spacer_left, center_col, spacer_right = st.columns([1, 6, 1])
                            
                            with center_col:
                                # Overall Score nổi bật (Đã được căn giữa)
                                render_overall_score(doc_metrics)

                                # Visual metrics (Đã được căn giữa)
                                render_metrics_visual(doc_metrics)

                            # Đường phân cách vẫn nằm ngoài để kéo dài toàn bộ bề ngang
                            st.divider()

                            # 3. BẢNG CHI TIẾT (Thu gọn vào Expander để tiết kiệm diện tích)
                            with st.expander("📋 Bảng dữ liệu chi tiết tất cả độ đo", expanded=False):
                                try:
                                    df_metrics = render_metrics_table(doc_metrics)
                                    st.dataframe(
                                        df_metrics,
                                        use_container_width=True,
                                        hide_index=True
                                    )
                                except Exception:
                                    pass

                            # 4. GIẢI THÍCH CÁC ĐỘ ĐO (Tự động chia 2 cột đẹp mắt)
                            with st.expander("📚 Hướng dẫn đọc hiểu các chỉ số đánh giá", expanded=False):
                                evaluator = SummaryEvaluator()
                                explanations = evaluator.get_explanations()
                                
                                info_col1, info_col2 = st.columns(2)
                                items = list(explanations.items())
                                mid = (len(items) + 1) // 2
                                
                                for i, (key, info) in enumerate(items):
                                    col = info_col1 if i < mid else info_col2
                                    box_type = col.info if i < mid else col.success
                                    
                                    box_type(
                                        f"**{info['name']}**\n\n"
                                        f"{info['description']}\n\n"
                                        f"- *Ý nghĩa:* {info['interpretation']}\n"
                                        f"- *Khoảng giá trị:* {info['range']}"
                                    )

                            # 5. HIỂN THỊ TỪ KHÓA (Dàn ngang thành 4 cột cho gọn)
                            kw_data = doc_metrics.get('keyword_coverage', {})
                            if kw_data.get('keywords'):
                                with st.expander("🔑 Top từ khóa quan trọng", expanded=False):
                                    top_kw = kw_data.get('keywords', [])
                                    found_kw = kw_data.get('found_keywords', [])
                                    
                                    # Dàn thành 4 cột ngang
                                    kw_cols = st.columns(4)
                                    for i, kw in enumerate(top_kw):
                                        status = "✅" if kw in found_kw else "❌"
                                        kw_cols[i % 4].write(f"{status} `{kw}`")
                        else:
                            st.warning("Chưa có dữ liệu đánh giá cho tài liệu này.")
            else:
                st.warning(
                    "Chưa có bản tóm tắt để đánh giá. "
                    "Hãy nhấn **'📝 TẠO BẢN TÓM TẮT'** ở thanh bên."
                )
	# ========== TAB MINDMAP ==========
        with tab_mindmap:
            st.markdown("### 🧠 Bản đồ tư duy (TextRank Graph Visualization)")
            st.caption("Trực quan hóa mối liên kết giữa các ý chính trong tài liệu bằng thuật toán Toán học (Chi phí API = 0).")
            
            if 'current_summary' in st.session_state and st.session_state.current_summary:
                doc_names = list(st.session_state.current_summary.keys())
                map_tabs = st.tabs(doc_names)

                for i, doc_name in enumerate(doc_names):
                    with map_tabs[i]:
                        with st.spinner("🎨 Đang xây dựng sơ đồ..."):
                            html_string = st.session_state.engine.generate_mindmap_html(doc_name)
                        
                        # CHẾ ĐỘ HIỂN THỊ GỠ LỖI (DEBUGGER)
                        if html_string.startswith("ERROR:"):
                            st.error("❌ CÓ LỖI XẢY RA TRONG QUÁ TRÌNH XỬ LÝ (DEBUG INFO):")
                            st.code(html_string, language="bash")
                        elif html_string:
                            st.components.v1.html(html_string, height=650, scrolling=True)
                            st.info("💡 **Hướng dẫn**: Kéo thả nút để sắp xếp. Zoom cuộn chuột. Hover vào nút để xem nội dung câu.")
                        else:
                            st.error("❌ Không thể tạo sơ đồ (Hàm trả về rỗng).")
            else:
                st.warning("Hãy tạo bản Tóm tắt trước để thu thập dữ liệu cấu trúc đồ thị.")

        # ========== TAB FLASHCARDS ==========
        with tab_flashcard:
            st.markdown("### 🃏 Bộ thẻ ôn tập Điền khuyết (Cloze Quiz)")
            st.caption("Hệ thống tự động phát hiện Thuật ngữ chuyên ngành từ Ontology và tạo câu hỏi điền khuyết (0 Token).")

            if 'current_summary' in st.session_state and st.session_state.current_summary:
                doc_names = list(st.session_state.current_summary.keys())
                card_tabs = st.tabs(doc_names)

                # ĐÃ FIX LỖI: enumerate(doc_names) thay vì enumerate(card_tabs)
                for i, doc_name in enumerate(doc_names):
                    with card_tabs[i]:
                        cards = st.session_state.engine.generate_flashcards(doc_name, max_cards=15)
                        if not cards:
                            st.info("🚫 Không tìm thấy đủ thuật ngữ chuyên ngành trong tài liệu này.")
                            continue

                        csv_data = "Câu hỏi,Đáp án,Nhóm kiến thức\n"
                        for c in cards:
                            q = c['question'].replace('\n', ' ').replace('"', '""')
                            a = c['answer'].replace('"', '""')
                            cat = c['category']
                            csv_data += f'"{q}","{a}","{cat}"\n'

                        col1, col2 = st.columns([3, 1])
                        col1.write(f"🎯 **Đã tạo {len(cards)} thẻ học tập.**")
                        col2.download_button("📥 Xuất CSV (Cho Anki)", data=csv_data.encode('utf-8-sig'), file_name=f"flashcard_{doc_name[:10]}.csv", mime="text/csv", use_container_width=True)
                        st.divider()

                        for idx, card in enumerate(cards):
                            with st.expander(f"Thẻ #{idx+1} | Chủ đề: {card['category']}"):
                                st.markdown(f"**Câu hỏi:** {card['question']}")
                                st.success(f"**Đáp án:** `{card['answer']}`")
                                st.caption(f"*Ngữ cảnh:* {card['context']}")
            else:
                st.warning("Hãy tạo bản Tóm tắt trước để sinh Thẻ ôn tập.")

        