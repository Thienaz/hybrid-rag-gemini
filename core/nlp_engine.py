# core/nlp_engine.py

import time
import os
import re
import numpy as np
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

from config import (
    AI_CONFIG, TEXT_CONFIG, PROMPTS,
    DOMAIN_ONTOLOGY, INTENT_PATTERNS
)
from core.text_processor import TextProcessor
from core.evaluator import SummaryEvaluator
from utils.logger import logger
from utils.session_manager import SessionManager


class NLPEngine:
    def __init__(self):
        logger.info(f"🚀 Đang khởi tạo Engine cho {AI_CONFIG['model_name']}...")
        self._setup_api()
        self.text_processor = TextProcessor()
        self.evaluator = SummaryEvaluator()

        self.original_sentences = []
        self.tokenized_sentences = []
        self.docs_data = {}
        self.tfidf_matrix = None
        self.vectorizer = None

        self.is_processed = False
        self.document_stats = {}

        self.last_api_call = 0
        self.api_call_count = 0

        # Load ontology từ config (dễ mở rộng mà không cần sửa code)
        self.ontology = self._load_ontology()

	# --- MỚI: Bộ nhớ đệm đồ thị TextRank (Zero-Token Mindmap & Flashcard) ---
        self.ranking_graphs = {}

    def _setup_api(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("❌ Thiếu API KEY")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=AI_CONFIG["model_name"],
            generation_config=AI_CONFIG["generation_config"]
        )

    def _load_ontology(self):
        """
        Load ontology từ config và chuẩn hóa lowercase.
        Tách biệt ra config để mở rộng mà không sửa code engine.
        """
        return {
            category: {term.lower() for term in keywords}
            for category, keywords in DOMAIN_ONTOLOGY.items()
        }

    # ==================================================================
    # TIỀN XỬ LÝ (100% Local, không tốn API)
    # ==================================================================

    def preprocess(self, raw_text, session_id=None):
        """Tiền xử lý văn bản: Chunking → Tokenization → Vectorization"""

        # Bước 1: Ghi log bắt đầu
        SessionManager.log_trace(
            session_id, "DOCUMENT_INGESTION",
            "Bắt đầu đọc luồng dữ liệu thô từ các file được tải lên."
        )

        result = self.text_processor.process_document(raw_text)
        if not result:
            return 0

        self.original_sentences = result["original_sentences"]
        self.tokenized_sentences = result["tokenized_sentences"]
        self.docs_data = result.get("docs_data", {})
        self.document_stats = result["statistics"]

        # Bước 2: Ghi log chunking
        SessionManager.log_trace(
            session_id, "TEXT_CHUNKING",
            f"Hoàn tất làm sạch và phân mảnh văn bản. "
            f"Tổng số phân đoạn (chunks): {len(self.original_sentences)} đoạn. "
            f"Có gắn thẻ Metadata nguồn."
        )

        # Bước 3: Xây dựng không gian Vector TF-IDF (Global)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.tokenized_sentences)
        self.is_processed = True

        SessionManager.log_trace(
            session_id, "VECTORIZATION",
            f"Đã xây dựng Không gian Vector TF-IDF. "
            f"Kích thước ma trận: {self.tfidf_matrix.shape}."
        )

        return len(self.original_sentences)

    # ==================================================================
    # SCORING & EXTRACTION (TextRank + Domain Heuristics)
    # ==================================================================

    def _score_sentence_heuristics(self, sentence_tokens, idx, total_sentences, base_bonus):
        """
        Chấm điểm câu dựa trên 3 yếu tố:
        1. Ontology match (thuật ngữ chuyên ngành)
        2. Vị trí câu trong văn bản
        3. Độ dài câu
        """
        score = 1.0
        detected_classes = set()

        # 1. Ontology scoring
        for category, keywords in self.ontology.items():
            intersection = sentence_tokens.intersection(keywords)
            if intersection:
                detected_classes.add(category)
                score += len(intersection) * base_bonus

        # Bonus liên ngành (câu chứa thuật ngữ từ nhiều lĩnh vực)
        if len(detected_classes) > 1:
            score *= 1.5

        # 2. Position weighting
        position_weight = 1.0
        if total_sentences > 0:
            if idx == 0 or idx == total_sentences - 1:
                position_weight = 1.2      # Câu đầu/cuối thường quan trọng
            elif idx < total_sentences * 0.1:
                position_weight = 1.1      # 10% đầu tiên

        # 3. Length weighting
        length_weight = 1.0
        sentence_length = len(sentence_tokens)
        if sentence_length < 8:
            length_weight = 0.8            # Câu quá ngắn → ít thông tin
        elif 15 <= sentence_length <= 35:
            length_weight = 1.1            # Câu vừa phải → chứa nhiều thông tin

        return score * position_weight * length_weight

    def _extract_single_doc(self, doc_name, ratio=0.25, base_bonus=1.5):
        """
        Trích xuất câu quan trọng từ MỘT tài liệu bằng Personalized TextRank.
        Dùng vectorizer riêng cho từng doc (intra-document similarity).
        """
        if doc_name not in self.docs_data:
            return "", {}

        doc = self.docs_data[doc_name]
        sentences = doc["chunks"]
        tokenized = doc["tokenized"]
        total_sentences = len(sentences)

        if total_sentences == 0:
            return "", {}

        # Tạo TF-IDF riêng cho document này (intra-doc similarity)
        local_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=2000)
        try:
            tfidf_matrix = local_vectorizer.fit_transform(tokenized)
        except Exception as e:
            logger.warning(f"Lỗi Vectorizer cục bộ cho {doc_name}: {e}")
            return " ".join(sentences[:3]), {}

        # Tính trọng số heuristic cho từng câu
        weights = {}
        for idx, sentence in enumerate(sentences):
            tokens = set(tokenized[idx].split())
            weights[idx] = self._score_sentence_heuristics(
                tokens, idx, total_sentences, base_bonus
            )

        # Personalization vector cho PageRank
        total_weight = sum(weights.values())
        pers_dict = (
            {k: v / total_weight for k, v in weights.items()}
            if total_weight > 0 else None
        )

        # Xây đồ thị tương đồng → PageRank
        sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        nx_graph = nx.from_numpy_array(sim_matrix)
        scores = nx.pagerank(nx_graph, alpha=0.85, personalization=pers_dict)

        # LƯU ĐỒ THỊ ĐỂ VẼ MINDMAP & TẠO FLASHCARD SAU NÀY
        if not hasattr(self, 'ranking_graphs'):
            self.ranking_graphs = {}
        self.ranking_graphs[doc_name] = {
            "sentences": sentences,
            "scores": scores,
            "sim_matrix": sim_matrix
        }

        # Chọn top câu, giữ thứ tự xuất hiện gốc
        ranked = sorted(
            ((scores[i], s) for i, s in enumerate(sentences)),
            reverse=True
        )
        num_sentences = max(1, int(total_sentences * ratio))
        top_sentences = sorted(
            ranked[:num_sentences],
            key=lambda x: sentences.index(x[1])
        )

        return " ".join([s[1] for s in top_sentences]), scores

    # ==================================================================
    # TÓM TẮT (Hybrid: TextRank Extractive + AI Abstractive)
    # ==================================================================

    def generate_summary(self, use_ai_polish=True):
        """Tạo bản tóm tắt cho từng tài liệu riêng biệt"""
        all_summaries = {}

        if not self.docs_data:
            self.docs_data = {"Tài_liệu_chính": {
                "chunks": self.original_sentences,
                "tokenized": self.tokenized_sentences,
                "raw_text": " ".join(self.original_sentences)
            }}

        for doc_name, doc_info in self.docs_data.items():
            logger.info(f"📝 Đang tóm tắt tài liệu: {doc_name}")

            # Bước 1: Extractive (TextRank + Ontology)
            textrank_summary, scores = self._extract_single_doc(
                doc_name, ratio=0.25, base_bonus=0.5
            )

            # Bước 2: Đánh giá bản trích xuất
            eval_original = doc_info["raw_text"][:15000]
            eval_summary_tr = textrank_summary[:15000]
            textrank_metrics = self.evaluator.evaluate_summary(
                eval_original, eval_summary_tr
            )

            final_summary = textrank_summary
            final_metrics = textrank_metrics
            source = "textrank"
            ai_polished_summary = ""

            # Bước 3: Abstractive Polish (nếu bật)
            if use_ai_polish:
                prompt = PROMPTS["hybrid_summary"].format(
                    extracted_text=textrank_summary
                )
                success, ai_data = self._call_gemini(prompt)

                if success:
                    ai_polished_summary = ai_data
                    final_summary = ai_polished_summary
                    source = "hybrid_gemini_flash"
                    eval_summary_ai = ai_polished_summary[:15000]
                    final_metrics = self.evaluator.evaluate_summary(
                        eval_original, eval_summary_ai
                    )
                else:
                    final_summary = (
                        f"⚠ Lỗi AI ({ai_data}). "
                        f"Dự phòng TextRank:\n\n{textrank_summary}"
                    )
                    source = "error_fallback_textrank"

            all_summaries[doc_name] = {
                "final_summary": final_summary,
                "textrank_summary": textrank_summary,
                "ai_polished_summary": ai_polished_summary,
                "metrics": final_metrics,
                "source": source,
                "sentence_scores": scores
            }

        return all_summaries

    # ==================================================================
    # INTENT RECOGNITION (Hybrid: Local Pattern → API Fallback)
    # ==================================================================

    def _analyze_intent_local(self, question):
        """
        Nhận diện Intent bằng pattern matching CỤC BỘ.
        Ưu điểm: Miễn phí, nhanh, không tốn API quota.
        """
        q_lower = question.lower()

        for intent, patterns in INTENT_PATTERNS.items():
            for pattern in patterns:
                if pattern in q_lower:
                    return intent

        return None  # Không xác định → cần API fallback

    def analyze_intent(self, question):
        """
        Hybrid Intent Recognition:
        - Ưu tiên 1: Pattern matching cục bộ (miễn phí, nhanh)
        - Ưu tiên 2: Zero-shot LLM (chỉ khi cục bộ không xác định được)
        Tiết kiệm ~60-70% API calls cho intent detection.
        """
        # Bước 1: Local detection
        local_intent = self._analyze_intent_local(question)
        if local_intent:
            logger.info(f"🎯 Intent nhận diện CỤC BỘ: {local_intent}")
            return local_intent

        # Bước 2: API fallback
        prompt = f"""Phân tích ý định (Intent) của câu hỏi sau trong ngữ cảnh học tập Đại học:

Câu hỏi: "{question}"

Nhiệm vụ: Phân loại câu hỏi trên vào ĐÚNG 1 TRONG 4 loại Intent sau:
1. DINH_NGHIA (Hỏi về khái niệm, giải thích thuật ngữ, "là gì?")
2. SO_SANH (Phân biệt, so sánh 2 hay nhiều thực thể, "khác nhau thế nào?")
3. TOM_TAT (Yêu cầu tóm lược, liệt kê các ý chính)
4. KHAC (Hỏi cách làm, câu hỏi suy luận mở, hoặc không thuộc 3 loại trên)

Chỉ trả về đúng 1 từ khóa Intent (ví dụ: DINH_NGHIA). Không giải thích gì thêm."""

        success, intent_result = self._call_gemini(prompt)
        intent = intent_result.strip().upper() if success else "KHAC"

        valid_intents = ["DINH_NGHIA", "SO_SANH", "TOM_TAT", "KHAC"]
        for vi in valid_intents:
            if vi in intent:
                logger.info(f"🎯 Intent nhận diện qua API: {vi}")
                return vi
        return "KHAC"

    def _get_intent_instruction(self, intent):
        """Trả về hướng dẫn hành vi cụ thể cho LLM theo từng loại Intent"""
        instructions = {
            "DINH_NGHIA": "Hãy trả lời tập trung vào việc định nghĩa rõ ràng, mạch lạc.",
            "SO_SANH": "Hãy kẻ bảng hoặc dùng gạch đầu dòng để làm rõ sự khác biệt/giống nhau.",
            "TOM_TAT": "Hãy tóm tắt ngắn gọn thành các luận điểm chính.",
            "KHAC": "Hãy trả lời trực tiếp, rõ ràng dựa trên ngữ cảnh được cung cấp."
        }
        return instructions.get(intent, instructions["KHAC"])

    # ==================================================================
    # HỎI ĐÁP RAG (Retrieval-Augmented Generation)
    # ==================================================================

    def query_document(self, question, chat_history=None, session_id=None):
        """
        Pipeline RAG hoàn chỉnh:
        User Input → Intent Detection → Vector Search → Prompt Assembly → LLM → Response
        """
        if not self.is_processed:
            return {"answer": "⚠ Vui lòng tải tài liệu lên trước.", "context": ""}

        # === BƯỚC 1: Nhận câu hỏi ===
        SessionManager.log_trace(
            session_id, "USER_INPUT",
            f"Tiếp nhận truy vấn: '{question}'"
        )

        # === BƯỚC 2: Intent Recognition (Hybrid) ===
        SessionManager.log_trace(
            session_id, "INTENT_ANALYSIS",
            "Phân tích Ý định (Intent) bằng Hybrid (Local Pattern → API Fallback)..."
        )

        detected_intent = self.analyze_intent(question)
        logger.info(f"🎯 Đã nhận diện Intent: {detected_intent}")

        SessionManager.log_trace(
            session_id, "INTENT_MATCHED",
            f"Kết quả Intent: [{detected_intent}]. "
            f"Hệ thống sẽ chuyển hướng trích xuất theo định dạng này."
        )

        # === BƯỚC 3: Vector Search (Retrieval) có gắn ID ===
        q_tokens = self.text_processor.tokenize_vietnamese(question)
        q_vec = self.vectorizer.transform([q_tokens])
        sims = cosine_similarity(q_vec, self.tfidf_matrix).flatten()

        top_k = TEXT_CONFIG.get("top_k_retrieval", 5)
        top_idx = sims.argsort()[-top_k:][::-1]

        # Khởi tạo ngữ cảnh dưới dạng Dict để map ID (Theo chuẩn Nguồn 1)
        context_parts = []
        for rank, i in enumerate(top_idx):
            if sims[i] > 0.01:
                context_parts.append({
                    "id": f"C{rank+1}",
                    "text": self.original_sentences[i]
                })
        
        context_text = "\n".join([f"[{c['id']}] {c['text']}" for c in context_parts]) if context_parts else "\n".join(self.original_sentences[:3])

        SessionManager.log_trace(
            session_id, "VECTOR_SEARCH",
            f"Lọc được {len(context_parts)} đoạn ngữ cảnh. Đã gắn mã định danh C1 -> C{len(context_parts)}."
        )

        # === BƯỚC 4: Xử lý Lịch sử (Hybrid Memory: Sliding + Vector) ===
        history_text = "Không có lịch sử trò chuyện trước đó."

        if chat_history and len(chat_history) > 0:
            # ------------------------------------------------------------
            # 1. KÝ ỨC GẦN (Immediate History): Giữ 2 tin nhắn cuối
            # ------------------------------------------------------------
            immediate_history = chat_history[-2:] 
            
            # ------------------------------------------------------------
            # 2. KÝ ỨC XA (Vector Memory): Truy xuất thông minh bằng TF-IDF
            # ------------------------------------------------------------
            relevant_past_msgs = []
            older_history = chat_history[:-2]
            
            if len(older_history) > 0:
                formatted_old_msgs = []
                for item in older_history:
                    # Trích xuất an toàn dù dữ liệu là Dict hay Tuple
                    role = item.get("role", "") if isinstance(item, dict) else item[0]
                    msg = item.get("content", "") if isinstance(item, dict) else item[1]
                    
                    prefix = "Sinh viên" if role in ["Bạn", "user"] else "Trợ lý AI"
                    formatted_old_msgs.append(f"{prefix}: {msg}")
                
                try:
                    # Tokenize lịch sử. TƯƠNG LAI: Lấy trực tiếp từ session['tokenized_history'] ở đây
                    tokenized_history = [self.text_processor.tokenize_vietnamese(m) for m in formatted_old_msgs]
                    
                    # Khởi tạo Vectorizer thông minh (Lấy từ Cách 1: token_pattern=None)
                    hist_vectorizer = TfidfVectorizer(
                        ngram_range=(1, 2), 
                        max_features=2000,
                        lowercase=True,
                        token_pattern=None # Bắt buộc để giữ nguyên định dạng của underthesea
                    )
                    
                    # Fit & Transform
                    hist_matrix = hist_vectorizer.fit_transform(tokenized_history)
                    q_vec_hist = hist_vectorizer.transform([q_tokens]) # q_tokens đã có ở Bước 3
                    
                    # Tính Cosine Similarity
                    sims_hist = cosine_similarity(q_vec_hist, hist_matrix).flatten()
                    
                    # Lấy Top 3 tin nhắn (Ngưỡng 0.08 từ Cách 1, Sắp xếp Index từ Cách 2)
                    SIMILARITY_THRESHOLD = 0.08
                    top_hist_idx = sims_hist.argsort()[-3:][::-1]
                    
                    relevant_indices = [idx for idx in top_hist_idx if sims_hist[idx] > SIMILARITY_THRESHOLD]
                    
                    # Sắp xếp lại theo thời gian gốc (Lấy từ Cách 2)
                    relevant_indices.sort()
                    relevant_past_msgs = [formatted_old_msgs[i] for i in relevant_indices]
                    
                    SessionManager.log_trace(
                        session_id, "VECTOR_MEMORY_RETRIEVAL",
                        f"Đã quét {len(older_history)} tin cũ. Truy xuất thành công {len(relevant_past_msgs)} tin."
                    )
                except Exception as e:
                    logger.warning(f"Vector hóa lịch sử thất bại: {e}")

            # ------------------------------------------------------------
            # 3. Lắp ráp History Text cuối cùng
            # ------------------------------------------------------------
            history_blocks = []
            
            if relevant_past_msgs:
                history_blocks.append("--- ĐOẠN HỘI THOẠI CŨ CÓ LIÊN QUAN (Truy xuất tự động) ---")
                history_blocks.extend(relevant_past_msgs)
                history_blocks.append("--- HỘI THOẠI HIỆN TẠI ---")
            
            for item in immediate_history:
                role = item.get("role", "") if isinstance(item, dict) else item[0]
                msg = item.get("content", "") if isinstance(item, dict) else item[1]
                
                prefix = "Sinh viên" if role in ["Bạn", "user"] else "Trợ lý AI"
                history_blocks.append(f"{prefix}: {msg}")
            
            history_text = "\n".join(history_blocks)

        # === BƯỚC 5: Lắp ráp Prompt từ Template (config.py) ===
        intent_instruction = self._get_intent_instruction(detected_intent)

        advanced_prompt = PROMPTS["rag_qa"].format(
            intent=detected_intent,
            intent_instruction=intent_instruction,
            history=history_text,
            context=context_text,
            question=question
        )

        SessionManager.log_trace(
            session_id, "PROMPT_ASSEMBLY",
            f"Lắp ráp thành công Context + History + Query từ Template. "
            f"Đính kèm lệnh ép buộc hành vi [{detected_intent}]. "
            f"Tiến hành gọi API LLM sinh văn bản."
        )

        # === BƯỚC 6: Gọi API sinh câu trả lời ===
        success, result_data = self._call_gemini(advanced_prompt)

        status = "Thành công" if success else "Thất bại"
        SessionManager.log_trace(
            session_id, "LLM_RESPONSE",
            f"Trạng thái API: {status}. Bắt đầu đưa qua cổng kiểm duyệt..."
        )

        # === CƠ CHẾ DỰ PHÒNG KHI API LỖI ===
        if not success:
            fallback_answer = (
                f"⚠ **Máy chủ AI hiện đang bận hoặc hết hạn ngạch API.**\n\n"
                f"Dưới đây là nguyên văn tài liệu trích xuất để bạn tham khảo:\n\n> *{context_text}*"
            )
            return {
                "answer": fallback_answer,
                "context": context_text,
                "intent": detected_intent,
                "source": "fallback_raw_retrieval"
            }

        # === BƯỚC 7: CỔNG KIỂM DUYỆT (TRUST & SAFETY GATE) ===
        # Gọi hàm kiểm tra độ tin cậy vừa viết
        trust_report = self._verify_trustworthiness(question, result_data, context_parts, detected_intent, session_id)
        
        # Nếu FAIL, dán cảnh báo màu vàng lên đầu câu trả lời
        if trust_report["verdict"] == "FAIL":
            result_data = trust_report["warning_msg"] + result_data
            source_tag = "hallucinated_warning"
        else:
            source_tag = "verified_rag_with_intent"

        # Ghi log kết quả đánh giá để show ra UI hoặc gỡ lỗi
        details = trust_report.get("details", {})
        SessionManager.log_trace(
            session_id, "TRUST_CHECK_COMPLETED",
            f"Phán quyết: {trust_report.get('verdict', 'PASS')}. "
            f"Tổng số ý: {details.get('total_claims', 0)} | "
            f"Thiếu nguồn: {details.get('missing_cite', 0)} | "
            f"Sai nguồn: {details.get('bad_cite', 0)} | "
            f"Độ tin cậy thấp: {details.get('low_support', 0)}"
        )

        # === TRƯỜNG HỢP THÀNH CÔNG (ĐÃ QUA KIỂM DUYỆT) ===
        return {
            "answer": result_data,
            "context": context_text,
            "intent": detected_intent,
            "source": source_tag,
            "trust_metrics": trust_report # Gửi kèm điểm số để render UI (Đồng hồ ảo giác)
        }

    # ==================================================================
    # ĐÁNH GIÁ ĐỘ TRUNG THỰC RAG (Faithfulness Evaluation)
    # ==================================================================

    def evaluate_rag_faithfulness(self, context, answer, question, intent):
        """
        Đánh giá 4 chiều:
        1. Cosine Similarity (context ↔ answer)
        2. Word Precision (tỷ lệ từ answer có trong context)
        3. Hallucination Rate (1 - precision)
        4. Intent Relevance (LLM-as-Judge)
        """
        if not context or not answer or "⚠" in answer or "❌" in answer:
            return None

        try:
            # 1. Cosine Similarity
            vecs = self.vectorizer.transform([
                self.text_processor.tokenize_vietnamese(context),
                self.text_processor.tokenize_vietnamese(answer)
            ])
            cosine_sim = float(cosine_similarity(vecs[0:1], vecs[1:2])[0][0])

            # 2. Word Precision & Hallucination
            ans_tokens = set(self.text_processor.tokenize_vietnamese(answer).split())
            ctx_tokens = set(self.text_processor.tokenize_vietnamese(context).split())
            overlap = ans_tokens.intersection(ctx_tokens)
            precision = len(overlap) / len(ans_tokens) if ans_tokens else 0.0
            hallucination = 1.0 - precision

            # 3. Intent Relevance (LLM-as-Judge)
            judge_prompt = f"""Đánh giá xem câu trả lời có đáp ứng đúng Ý định (Intent) của câu hỏi không.

Câu hỏi: "{question}" (Intent dự đoán: {intent})
Câu trả lời: "{answer[:500]}"

Hãy chấm điểm từ 0 đến 10 (10 là cực kỳ đúng trọng tâm, 0 là lạc đề hoàn toàn).
CHỈ TRẢ VỀ ĐÚNG 1 CON SỐ TỪ 0 ĐẾN 10, KHÔNG GIẢI THÍCH."""

            suc, score_str = self._call_gemini(judge_prompt)
            relevance_score = 0.85  # Default nếu API lỗi

            if suc:
                try:
                    numbers = re.findall(r'\d+', score_str)
                    if numbers:
                        raw_score = int(numbers[0])
                        relevance_score = min(raw_score, 10) / 10.0
                except Exception:
                    pass

            return {
                "cosine_similarity": cosine_sim,
                "word_precision": float(precision),
                "hallucination_rate": float(hallucination),
                "intent_relevance": float(relevance_score)
            }

        except Exception as e:
            logger.error(f"Lỗi Faithfulness: {e}")
            return None

    # ==================================================================
    # PHÂN HỆ KIỂM DUYỆT ĐỘ TIN CẬY (ANTI-HALLUCINATION GATE)
    # ==================================================================

    def _get_dynamic_thresholds(self, intent):
        """Ngưỡng động linh hoạt theo Ý định người dùng (Nguồn 2)"""
        return {
            "DINH_NGHIA": {"cosine": 0.25}, # Cần độ chính xác từ vựng cao
            "SO_SANH":    {"cosine": 0.20},
            "TOM_TAT":    {"cosine": 0.15}, # Tóm tắt được phép AI viết lại nhiều
            "KHAC":       {"cosine": 0.20},
        }.get(intent, {"cosine": 0.20})

    def _verify_trustworthiness(self, question, answer, context_parts, intent, session_id=None):
        import json
        
        # 1. Bắt trường hợp hệ thống từ chối trả lời (Nguồn 1)
        if "[NO_CONTEXT]" in answer:
            return {"verdict": "PASS", "warning_msg": "", "details": {"no_context": True}}

        # 2. Xây dựng bản đồ ID -> Text
        context_map = {c["id"]: c["text"] for c in context_parts}
        thresholds = self._get_dynamic_thresholds(intent)

        # 3. Trích xuất các ý (Claim) từ câu trả lời
        lines = [ln.strip("-• \t") for ln in answer.splitlines() if ln.strip()]
        lines = lines[:15] # Chống spam độ dài
        
        # Sử dụng Char N-gram để chống lỗi từ vựng/paraphrase (Nguồn 1)
        vec_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=5000)
        
        missing_cite = 0
        bad_cite = 0
        low_support = 0
        hallucinated_lines = []

        cite_regex = re.compile(r"\[Nguồn:\s*([A-Za-z0-9,\s]+)\]", re.IGNORECASE)

        # --- LỚP 1: KIỂM TRA THỐNG KÊ LOCAL (0 Token) ---
        for ln in lines:
            # Tìm trích dẫn [Nguồn: Cx]
            m = cite_regex.search(ln)
            if not m:
                missing_cite += 1
                hallucinated_lines.append(ln)
                continue
                
            cite_ids = [x.strip() for x in m.group(1).split(",")]
            cited_texts = []
            
            for cid in cite_ids:
                if cid not in context_map:
                    bad_cite += 1
                else:
                    cited_texts.append(context_map[cid])
            
            if not cited_texts:
                low_support += 1
                hallucinated_lines.append(ln)
                continue
                
            # Tính Cosine Char N-gram giữa Claim và Dữ liệu được trích dẫn
            support_text = " ".join(cited_texts)
            try:
                X = vec_char.fit_transform([ln, support_text])
                sim = cosine_similarity(X[0:1], X[1:2])[0][0]
                if sim < thresholds["cosine"]:
                    low_support += 1
                    hallucinated_lines.append(ln)
            except Exception:
                pass # Bỏ qua nếu dòng quá ngắn không đủ tạo vector

        total_claims = max(1, len(lines))
        is_local_safe = (missing_cite/total_claims < 0.3) and (bad_cite == 0) and (low_support/total_claims < 0.4)

        # --- LỚP 2: LLM-AS-A-JUDGE (Chain-of-Thought từ Nguồn 2) ---
        judge_verdict = "PASS"
        if not is_local_safe:
            SessionManager.log_trace(session_id, "JUDGE_TRIGGERED", f"Lỗi Local: Thiếu nguồn={missing_cite}, Sai nguồn={bad_cite}, Hỗ trợ thấp={low_support}. Kích hoạt AI Giám khảo...")
            
            context_raw = "\n".join([f"[{c['id']}] {c['text']}" for c in context_parts])
            suspect_claims = "\n".join(hallucinated_lines[:3])
            
            judge_prompt = f"""Bạn là chuyên gia kiểm duyệt. PHÂN TÍCH TỪNG BƯỚC:
NGỮ CẢNH GỐC:
---
{context_raw[:2500]}
---
CÁC MỆNH ĐỀ ĐÁNG NGỜ CỦA HỆ THỐNG:
---
{suspect_claims}
---
Hãy suy luận: Các mệnh đề trên có xuất hiện (trực tiếp hoặc gián tiếp) trong NGỮ CẢNH không? Hay là do AI tự bịa đặt?
Trả về ĐÚNG MỘT KHỐI JSON:
{{"verdict": "PASS" hoặc "FAIL", "reason": "Lý do ngắn gọn", "fake_info": "Thông tin bịa đặt (nếu có)"}}"""

            suc, res = self._call_gemini(judge_prompt)
            if suc and "FAIL" in res.upper():
                judge_verdict = "FAIL"

        # --- XUẤT KẾT QUẢ ---
        final_verdict = "FAIL" if (not is_local_safe and judge_verdict == "FAIL") else "PASS"
        warning_msg = ""
        
        if final_verdict == "FAIL":
            warning_msg = (
                f"⚠️ **Cảnh báo từ Hệ thống Kiểm duyệt:** Một số nội dung dưới đây "
                f"thiếu minh chứng hoặc mâu thuẫn với tài liệu gốc. Vui lòng đối chiếu kỹ.\n\n---\n"
            )

        return {
            "verdict": final_verdict,
            "warning_msg": warning_msg,
            "details": {
                "total_claims": total_claims,
                "missing_cite": missing_cite,
                "bad_cite": bad_cite,
                "low_support": low_support
            }
        }

    # ==================================================================
    # API CALL (Rate Limiting + Retry)
    # ==================================================================

    def _call_gemini(self, prompt):
        """Gọi API Gemini với rate limiting và exponential backoff"""
        max_retries = AI_CONFIG.get("max_retries", 3)
        last_error = "Unknown"

        for attempt in range(max_retries):
            # Rate limiting
            elapsed = time.time() - self.last_api_call
            if elapsed < AI_CONFIG["min_request_interval"]:
                time.sleep(AI_CONFIG["min_request_interval"] - elapsed)

            try:
                response = self.model.generate_content(prompt)
                self.last_api_call = time.time()

                if response and response.text:
                    self.api_call_count += 1
                    return True, response.text.strip()

            except Exception as e:
                last_error = str(e)
                logger.error(
                    f"❌ API Error (lần {attempt + 1}/{max_retries}): {last_error}"
                )
                time.sleep(AI_CONFIG.get("retry_delay", 5))

        return False, last_error

    # ==================================================================
    # TIỆN ÍCH
    # ==================================================================

    def get_detected_language(self):
        return "vi"

    def get_statistics(self):
        return {
            "api_calls": self.api_call_count,
            "sentences": len(self.original_sentences)
        }

    def generate_mindmap_html(self, doc_name, max_nodes=40):
        """Tạo Mindmap sạch, không bị che bởi nhãn nguồn và chống chồng lấp"""
        if not hasattr(self, 'ranking_graphs'): self.ranking_graphs = {}
        
        if doc_name not in self.ranking_graphs:
            if doc_name in self.docs_data:
                self._extract_single_doc(doc_name)
            else:
                return f"ERROR: Không tìm thấy tài liệu '{doc_name}'."

        try:
            from pyvis.network import Network
            import uuid
            import re # Thêm re để lọc text
            
            data = self.ranking_graphs[doc_name]
            sentences = data["sentences"]
            scores = data["scores"]
            sim_matrix = data["sim_matrix"]

            ranked_indices = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            selected_indices = [idx for idx, _ in ranked_indices]

            # Khởi tạo Network với màu nền sáng và font chữ rõ ràng
            net = Network(height="650px", width="100%", bgcolor="#ffffff", font_color="#333333", directed=False)
            
            # CẤU HÌNH VẬT LÝ NÂNG CAO: Chống nốt đè lên nhau (avoidOverlap) và đẩy xa nhau (nodeDistance)
            net.set_options("""
            var options = {
              "nodes": {
                "font": {"size": 14, "face": "Tahoma"},
                "shape": "dot",
                "margin": 10
              },
              "physics": {
                "forceAtlas2Based": {
                  "gravitationalConstant": -150,
                  "centralGravity": 0.01,
                  "springLength": 150,
                  "springConstant": 0.08,
                  "avoidOverlap": 1
                },
                "maxVelocity": 50,
                "solver": "forceAtlas2Based",
                "timestep": 0.35,
                "stabilization": {"iterations": 150}
              },
              "interaction": {
                "hover": true,
                "tooltipDelay": 200
              }
            }""")

            for idx in selected_indices:
                # 1. LÀM SẠCH VĂN BẢN: Loại bỏ gạch đầu dòng và các thẻ [Nguồn...], [C1...]
                raw_text = sentences[idx]
                clean_text = re.sub(r'\[.*?\]', '', raw_text) # Xóa nội dung trong ngoặc vuông
                clean_text = clean_text.strip("-• \t\n")       # Xóa ký tự thừa đầu dòng
                
                # Tính toán kích thước nốt
                s_vals = list(scores.values())
                min_s, max_s = min(s_vals), max(s_vals)
                score_norm = int((scores[idx] - min_s) / (max_s - min_s + 1e-5) * 25) + 12
                
                node_color = "#2E7D32" if score_norm > 25 else "#1976D2"
                
                # Nhãn hiển thị ngắn gọn trên nốt
                display_label = clean_text[:45] + "..." if len(clean_text) > 45 else clean_text
                
                # Chú thích đầy đủ khi di chuột vào (Tooltip) - Đã được làm sạch
                net.add_node(idx, label=display_label, size=score_norm, title=clean_text, color=node_color)

            # Thêm các đường nối
            if len(selected_indices) > 1:
                sub_matrix = sim_matrix[np.ix_(selected_indices, selected_indices)]
                positive_edges = sub_matrix[sub_matrix > 0]
                threshold = np.percentile(positive_edges, 70) if len(positive_edges) > 0 else 0
                
                for i, idx1 in enumerate(selected_indices):
                    for j, idx2 in enumerate(selected_indices):
                        if i >= j: continue
                        weight = sim_matrix[idx1][idx2]
                        if weight > threshold:
                            net.add_edge(idx1, idx2, value=float(weight), color="#E0E0E0")

            path = f"temp_mindmap_{uuid.uuid4().hex[:6]}.html"
            net.save_graph(path)
            with open(path, "r", encoding="utf-8") as f:
                html_data = f.read()
            os.remove(path)
            return html_data
        except Exception as e:
            return f"ERROR: Lỗi hiển thị đồ thị: {str(e)}"

    def generate_flashcards(self, doc_name, max_cards=15):
        """Sinh Flashcard với cơ chế Lazy Loading"""
        if not hasattr(self, 'ranking_graphs'): self.ranking_graphs = {}
        
        if doc_name not in self.ranking_graphs and doc_name in self.docs_data:
            self._extract_single_doc(doc_name)
            
        if doc_name not in self.ranking_graphs: return []

        data = self.ranking_graphs[doc_name]
        sentences = data["sentences"]
        scores = data["scores"]
        
        ranked = sorted(((scores[i], s, i) for i, s in enumerate(sentences)), reverse=True)
        flashcards = []
        used_keywords = set()

        for score, sentence, idx in ranked:
            matched_terms = []
            for category, keywords in self.ontology.items():
                for kw in keywords:
                    pattern = r'(?<!\w)' + re.escape(kw) + r'(?!\w)'
                    if re.search(pattern, sentence, re.IGNORECASE):
                        matched_terms.append((category, kw))
            
            new_terms = [item for item in matched_terms if item[1].lower() not in used_keywords]
            if new_terms:
                category, keyword = new_terms[0]
                used_keywords.add(keyword.lower())
                pattern = r'(?<!\w)' + re.escape(keyword) + r'(?!\w)'
                blanked = re.sub(pattern, '[___]', sentence, flags=re.IGNORECASE)

                flashcards.append({
                    "question": blanked,
                    "answer": keyword.upper(),
                    "context": sentence,
                    "category": category,
                    "importance": score
                })
            if len(flashcards) >= max_cards: break
        return flashcards