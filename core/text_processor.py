#core/text_processor.py

import re
from underthesea import word_tokenize
from config import TEXT_CONFIG, VIETNAMESE_STOPWORDS
from utils.logger import logger


class TextProcessor:
    def __init__(self):
        # Lấy cấu hình cắt văn bản từ config.py (đơn vị: SỐ TỪ)
        self.chunk_size = TEXT_CONFIG.get("chunk_size", 150)
        self.chunk_overlap = TEXT_CONFIG.get("chunk_overlap", 30)

    def process_document(self, raw_text):
        """Xử lý chia nhỏ văn bản và phân nhóm theo Tài liệu"""
        if not raw_text or not raw_text.strip():
            return None

        # Tìm các vách ngăn tài liệu
        doc_pattern = r"--- BẮT ĐẦU TÀI LIỆU: (.*?) ---\n(.*?)(?=--- KẾT THÚC TÀI LIỆU|\Z)"
        documents = re.findall(doc_pattern, raw_text, re.DOTALL)

        if not documents:
            documents = [("Tài_liệu_chính", raw_text)]

        all_valid_chunks = []
        docs_data = {}

        for doc_name, doc_content in documents:
            doc_name = doc_name.strip()
            cleaned_text = self._clean_text(doc_content)
            chunks = self._split_with_overlap(
                cleaned_text, self.chunk_size, self.chunk_overlap
            )

            doc_chunks = []
            for chunk in chunks:
                # Lọc chunk quá ngắn (ít hơn 10 từ → không có giá trị)
                if len(chunk.split()) > 10:
                    tagged_chunk = f"[Nguồn: {doc_name}]\n{chunk}"
                    doc_chunks.append(tagged_chunk)
                    all_valid_chunks.append(tagged_chunk)

            # Lưu riêng rẽ từng file để chạy TextRank độc lập
            docs_data[doc_name] = {
                "chunks": doc_chunks,
                "tokenized": [self.tokenize_vietnamese(c) for c in doc_chunks],
                "raw_text": cleaned_text
            }

        if not all_valid_chunks:
            logger.warning("Không có chunk hợp lệ nào sau khi xử lý.")
            return None

        return {
            "original_sentences": all_valid_chunks,
            "tokenized_sentences": [
                self.tokenize_vietnamese(c) for c in all_valid_chunks
            ],
            "docs_data": docs_data,
            "statistics": {
                "num_chunks": len(all_valid_chunks),
                "num_docs": len(docs_data)
            }
        }

    def _split_with_overlap(self, text, chunk_size, overlap):
        """
        Thuật toán cắt văn bản thành các đoạn (chunk) có phần gối đầu (overlap).
        ĐƠN VỊ: SỐ TỪ (word-based), không phải ký tự.
        Giúp RAG không bị mất ngữ cảnh giữa các đoạn.
        """
        # Tách văn bản thành các câu dựa trên dấu chấm, chấm hỏi, chấm than
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = []
        current_word_count = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_word_count = len(sentence.split())

            # Nếu tổng số từ vượt chunk_size và đã có câu trong chunk hiện tại
            if current_word_count + sentence_word_count > chunk_size and current_chunk:
                # Đóng gói chunk hiện tại
                chunks.append(" ".join(current_chunk))

                # Tạo phần gối đầu (Overlap) bằng cách lấy ngược từ cuối
                overlap_word_count = 0
                overlap_chunk = []
                for s in reversed(current_chunk):
                    s_words = len(s.split())
                    if overlap_word_count + s_words <= overlap:
                        overlap_chunk.insert(0, s)
                        overlap_word_count += s_words
                    else:
                        break

                # Khởi tạo chunk tiếp theo với phần gối đầu
                current_chunk = overlap_chunk
                current_word_count = overlap_word_count

            # Thêm câu mới vào chunk
            current_chunk.append(sentence)
            current_word_count += sentence_word_count

        # Thêm chunk cuối cùng
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _clean_text(self, text):
        """Làm sạch văn bản: loại bỏ khoảng trắng thừa, ký tự đặc biệt"""
        # Loại bỏ nhiều khoảng trắng liên tiếp
        text = re.sub(r'\s+', ' ', text)
        # Loại bỏ các ký tự điều khiển (giữ lại Unicode tiếng Việt)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        return text.strip()

    def tokenize_vietnamese(self, text):
        """
        Tiền xử lý văn bản tiếng Việt cho TF-IDF.

        Pipeline:
        1. Lowercase
        2. Tách từ tiếng Việt (underthesea word_tokenize)
        3. Loại bỏ dấu câu
        4. Loại bỏ stopwords
        5. Loại bỏ từ quá ngắn (1 ký tự)
        """
        if not text or not text.strip():
            return ""

        # Bước 1: Lowercase
        text_lower = text.lower()

        # Bước 2: Tách từ tiếng Việt bằng underthesea
        # format="text" trả về chuỗi với dấu _ nối các từ ghép (ví dụ: "máy_học")
        try:
            tokenized = word_tokenize(text_lower, format="text")
        except Exception as e:
            logger.warning(f"Lỗi word_tokenize, fallback về split(): {e}")
            tokenized = text_lower

        # Bước 3: Loại bỏ dấu câu (giữ lại chữ, số, underscore, khoảng trắng)
        clean_text = re.sub(r'[^\w\s]', ' ', tokenized)

        # Bước 4: Tách thành danh sách token
        tokens = clean_text.split()

        # Bước 5: Loại bỏ stopwords và từ quá ngắn
        filtered_tokens = [
            token for token in tokens
            if token not in VIETNAMESE_STOPWORDS and len(token) > 1
        ]

        return " ".join(filtered_tokens)