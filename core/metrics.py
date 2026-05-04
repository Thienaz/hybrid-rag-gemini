#core/metrics.py

"""
=============================================================================
MODULE ĐÁNH GIÁ CHẤT LƯỢNG TÓM TẮT
Các độ đo chuẩn trong nghiên cứu NLP
=============================================================================

Tài liệu tham khảo:
- ROUGE: Lin, C.Y. (2004). "ROUGE: A Package for Automatic Evaluation of Summaries"
- BLEU: Papineni et al. (2002). "BLEU: a Method for Automatic Evaluation of Machine Translation"
"""

import math
from collections import Counter
from underthesea import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
import numpy as np


class SummaryMetrics:
    """
    Lớp tính toán các độ đo đánh giá chất lượng tóm tắt.
    
    Các độ đo bao gồm:
    1. ROUGE-1, ROUGE-2, ROUGE-L
    2. BLEU-1, BLEU-2, BLEU-3, BLEU-4
    3. Cosine Similarity (TF-IDF)
    4. Jaccard Similarity
    5. Compression Ratio
    6. Keyword Coverage
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
    
    # =========================================================================
    # TIỀN XỬ LÝ
    # =========================================================================
    
    def _tokenize(self, text):
        """Tokenize văn bản tiếng Việt"""
        tokenized = word_tokenize(text.lower(), format="text")
        return tokenized.split()
    
    def _get_ngrams(self, tokens, n):
        """Tạo n-gram từ danh sách token"""
        if len(tokens) < n:
            return []
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    # =========================================================================
    # ROUGE METRICS
    # =========================================================================
    
    def rouge_n(self, reference, candidate, n=1):
        """
        Tính ROUGE-N score
        
        ROUGE-N = (Số n-gram trùng khớp) / (Tổng số n-gram trong reference)
        
        Args:
            reference: Văn bản gốc (ground truth)
            candidate: Văn bản tóm tắt (cần đánh giá)
            n: Độ dài n-gram (1 cho unigram, 2 cho bigram)
            
        Returns:
            dict: {precision, recall, f1}
        """
        ref_tokens = self._tokenize(reference)
        cand_tokens = self._tokenize(candidate)
        
        ref_ngrams = self._get_ngrams(ref_tokens, n)
        cand_ngrams = self._get_ngrams(cand_tokens, n)
        
        if not ref_ngrams or not cand_ngrams:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        # Đếm số lượng n-gram
        ref_counts = Counter(ref_ngrams)
        cand_counts = Counter(cand_ngrams)
        
        # Tính số n-gram trùng khớp
        overlap = 0
        for ngram, count in cand_counts.items():
            overlap += min(count, ref_counts.get(ngram, 0))
        
        # Precision = overlap / |candidate|
        precision = overlap / len(cand_ngrams) if cand_ngrams else 0
        
        # Recall = overlap / |reference|
        recall = overlap / len(ref_ngrams) if ref_ngrams else 0
        
        # F1 = 2 * P * R / (P + R)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4)
        }
    
    def rouge_1(self, reference, candidate):
        """ROUGE-1: Đánh giá dựa trên unigram (từ đơn)"""
        return self.rouge_n(reference, candidate, n=1)
    
    def rouge_2(self, reference, candidate):
        """ROUGE-2: Đánh giá dựa trên bigram (cặp từ liên tiếp)"""
        return self.rouge_n(reference, candidate, n=2)
    
    def rouge_l(self, reference, candidate):
        """
        ROUGE-L: Đánh giá dựa trên Longest Common Subsequence (LCS)
        
        LCS là chuỗi con chung dài nhất giữa 2 chuỗi (không cần liên tiếp)
        
        Returns:
            dict: {precision, recall, f1, lcs_length}
        """
        ref_tokens = self._tokenize(reference)
        cand_tokens = self._tokenize(candidate)
        
        if not ref_tokens or not cand_tokens:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "lcs_length": 0}
        
        # Tính LCS bằng Dynamic Programming
        m, n = len(ref_tokens), len(cand_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_tokens[i-1] == cand_tokens[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[m][n]
        
        # Precision = LCS / |candidate|
        precision = lcs_length / n if n > 0 else 0
        
        # Recall = LCS / |reference|
        recall = lcs_length / m if m > 0 else 0
        
        # F1
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "lcs_length": lcs_length
        }
    
    # =========================================================================
    # BLEU METRIC
    # =========================================================================
    
    def bleu_score(self, reference, candidate, max_n=4):
        """
        Tính BLEU score (Bilingual Evaluation Understudy)
        
        BLEU đo độ chính xác của các n-gram trong candidate so với reference.
        Thường dùng cho Machine Translation nhưng cũng áp dụng cho Summarization.
        
        Args:
            reference: Văn bản gốc
            candidate: Văn bản tóm tắt
            max_n: N-gram tối đa (thường là 4)
            
        Returns:
            dict: {bleu_1, bleu_2, bleu_3, bleu_4, bleu_avg}
        """
        ref_tokens = self._tokenize(reference)
        cand_tokens = self._tokenize(candidate)
        
        if not cand_tokens:
            return {f"bleu_{i}": 0.0 for i in range(1, max_n+1)}
        
        # Brevity Penalty - phạt nếu tóm tắt quá ngắn
        bp = 1.0
        if len(cand_tokens) < len(ref_tokens):
            bp = math.exp(1 - len(ref_tokens) / len(cand_tokens))
        
        # Tính precision cho từng n-gram
        precisions = {}
        for n in range(1, max_n + 1):
            ref_ngrams = self._get_ngrams(ref_tokens, n)
            cand_ngrams = self._get_ngrams(cand_tokens, n)
            
            if not cand_ngrams:
                precisions[n] = 0.0
                continue
            
            ref_counts = Counter(ref_ngrams)
            cand_counts = Counter(cand_ngrams)
            
            # Clipped count
            clipped = 0
            for ngram, count in cand_counts.items():
                clipped += min(count, ref_counts.get(ngram, 0))
            
            precisions[n] = clipped / len(cand_ngrams)
        
        # BLEU-N = BP * precision_n
        result = {}
        for n in range(1, max_n + 1):
            result[f"bleu_{n}"] = round(bp * precisions[n], 4)
        
        # BLEU trung bình (geometric mean)
        valid_precisions = [p for p in precisions.values() if p > 0]
        if valid_precisions:
            log_avg = sum(math.log(p) for p in valid_precisions) / len(valid_precisions)
            result["bleu_avg"] = round(bp * math.exp(log_avg), 4)
        else:
            result["bleu_avg"] = 0.0
        
        return result
    
    # =========================================================================
    # SIMILARITY METRICS
    # =========================================================================
    
    def cosine_similarity(self, reference, candidate):
        """
        Tính Cosine Similarity dựa trên TF-IDF vectors
        
        Cosine Similarity = (A · B) / (||A|| * ||B||)
        
        Đo góc giữa 2 vector trong không gian TF-IDF.
        Giá trị từ 0 (hoàn toàn khác) đến 1 (giống hệt).
        
        Returns:
            float: Cosine similarity score (0-1)
        """
        try:
            # Tokenize
            ref_tokenized = " ".join(self._tokenize(reference))
            cand_tokenized = " ".join(self._tokenize(candidate))
            
            # Fit và transform
            tfidf_matrix = self.vectorizer.fit_transform([ref_tokenized, cand_tokenized])
            
            # Tính cosine similarity
            similarity = sklearn_cosine(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return round(float(similarity), 4)
        except Exception:
            return 0.0
    
    def jaccard_similarity(self, reference, candidate):
        """
        Tính Jaccard Similarity Index
        
        Jaccard = |A ∩ B| / |A ∪ B|
        
        Đo tỷ lệ từ chung so với tổng số từ (loại trùng).
        Giá trị từ 0 đến 1.
        
        Returns:
            float: Jaccard index (0-1)
        """
        ref_tokens = set(self._tokenize(reference))
        cand_tokens = set(self._tokenize(candidate))
        
        if not ref_tokens and not cand_tokens:
            return 0.0
        
        intersection = ref_tokens & cand_tokens
        union = ref_tokens | cand_tokens
        
        jaccard = len(intersection) / len(union) if union else 0
        
        return round(jaccard, 4)
    
    # =========================================================================
    # OTHER METRICS
    # =========================================================================
    
    def compression_ratio(self, reference, candidate):
        """
        Tính Compression Ratio (Tỷ lệ nén)
        
        CR = len(summary) / len(original)
        
        Giá trị nhỏ hơn 1 = nén tốt
        Thường mong muốn 0.2-0.4 (nén còn 20-40%)
        
        Returns:
            dict: {ratio, original_words, summary_words}
        """
        ref_words = len(self._tokenize(reference))
        cand_words = len(self._tokenize(candidate))
        
        ratio = cand_words / ref_words if ref_words > 0 else 0
        
        return {
            "ratio": round(ratio, 4),
            "original_words": ref_words,
            "summary_words": cand_words,
            "reduction_percent": round((1 - ratio) * 100, 2)
        }
    
    def keyword_coverage(self, reference, candidate, top_n=20):
        """
        Tính Keyword Coverage (Độ bao phủ từ khóa)
        
        Đo xem bản tóm tắt có giữ lại các từ khóa quan trọng không.
        Từ khóa quan trọng = từ xuất hiện nhiều nhất trong văn bản gốc.
        
        Args:
            reference: Văn bản gốc
            candidate: Văn bản tóm tắt  
            top_n: Số từ khóa quan trọng cần xét
            
        Returns:
            dict: {coverage, keywords_found, total_keywords, keywords}
        """
        ref_tokens = self._tokenize(reference)
        cand_tokens = set(self._tokenize(candidate))
        
        # Lọc stopwords đơn giản (từ quá ngắn)
        word_freq = Counter([w for w in ref_tokens if len(w) > 2])
        
        # Lấy top N từ khóa
        top_keywords = [word for word, _ in word_freq.most_common(top_n)]
        
        # Đếm số từ khóa xuất hiện trong tóm tắt
        found = [kw for kw in top_keywords if kw in cand_tokens]
        
        coverage = len(found) / len(top_keywords) if top_keywords else 0
        
        return {
            "coverage": round(coverage, 4),
            "keywords_found": len(found),
            "total_keywords": len(top_keywords),
            "keywords": top_keywords[:10],  # Trả về 10 từ khóa đầu
            "found_keywords": found[:10]
        }
    
    def information_density(self, reference, candidate):
        """
        Tính Information Density (Mật độ thông tin)
        
        Đo lượng thông tin mới trong mỗi từ của bản tóm tắt.
        ID = Số từ unique trong summary có trong reference / Tổng từ summary
        
        Returns:
            dict: {density, unique_relevant, total_summary_words}
        """
        ref_tokens = set(self._tokenize(reference))
        cand_tokens = self._tokenize(candidate)
        
        if not cand_tokens:
            return {"density": 0.0, "unique_relevant": 0, "total_summary_words": 0}
        
        # Từ trong summary mà cũng có trong reference
        relevant_words = [w for w in cand_tokens if w in ref_tokens]
        unique_relevant = len(set(relevant_words))
        
        density = unique_relevant / len(cand_tokens)
        
        return {
            "density": round(density, 4),
            "unique_relevant": unique_relevant,
            "total_summary_words": len(cand_tokens)
        }
    
    # =========================================================================
    # COMPREHENSIVE EVALUATION
    # =========================================================================
    
    def evaluate_all(self, reference, candidate):
        """
        Đánh giá toàn diện với tất cả các độ đo
        
        Args:
            reference: Văn bản gốc
            candidate: Văn bản tóm tắt
            
        Returns:
            dict: Tất cả metrics
        """
        results = {
            # ROUGE scores
            "rouge_1": self.rouge_1(reference, candidate),
            "rouge_2": self.rouge_2(reference, candidate),
            "rouge_l": self.rouge_l(reference, candidate),
            
            # BLEU scores
            "bleu": self.bleu_score(reference, candidate),
            
            # Similarity scores
            "cosine_similarity": self.cosine_similarity(reference, candidate),
            "jaccard_similarity": self.jaccard_similarity(reference, candidate),
            
            # Other metrics
            "compression": self.compression_ratio(reference, candidate),
            "keyword_coverage": self.keyword_coverage(reference, candidate),
            "information_density": self.information_density(reference, candidate),
        }
        
        # Tính điểm tổng hợp (Overall Score)
        # Công thức: weighted average của các metrics quan trọng
        overall = (
            0.25 * results["rouge_1"]["f1"] +
            0.20 * results["rouge_2"]["f1"] +
            0.15 * results["rouge_l"]["f1"] +
            0.15 * results["cosine_similarity"] +
            0.15 * results["keyword_coverage"]["coverage"] +
            0.10 * results["information_density"]["density"]
        )
        
        results["overall_score"] = round(overall, 4)
        
        return results
    
    def get_metrics_explanation(self):
        """
        Trả về giải thích cho từng độ đo (dùng cho UI)
        """
        return {
            "rouge_1": {
                "name": "ROUGE-1",
                "description": "Đo độ trùng khớp từ đơn (unigram) giữa tóm tắt và văn bản gốc",
                "interpretation": "Giá trị cao = nhiều từ trong tóm tắt xuất hiện trong văn bản gốc",
                "range": "0-1 (càng cao càng tốt)"
            },
            "rouge_2": {
                "name": "ROUGE-2", 
                "description": "Đo độ trùng khớp cặp từ liên tiếp (bigram)",
                "interpretation": "Giá trị cao = cấu trúc câu/cụm từ được giữ nguyên",
                "range": "0-1 (càng cao càng tốt)"
            },
            "rouge_l": {
                "name": "ROUGE-L",
                "description": "Dựa trên chuỗi con chung dài nhất (Longest Common Subsequence)",
                "interpretation": "Đánh giá mức độ giữ nguyên trật tự từ",
                "range": "0-1 (càng cao càng tốt)"
            },
            "bleu": {
                "name": "BLEU Score",
                "description": "Đo độ chính xác n-gram (thường dùng trong dịch máy)",
                "interpretation": "Giá trị cao = tóm tắt chính xác về mặt từ ngữ",
                "range": "0-1 (càng cao càng tốt)"
            },
            "cosine_similarity": {
                "name": "Cosine Similarity",
                "description": "Độ tương đồng vector TF-IDF giữa 2 văn bản",
                "interpretation": "Đo mức độ tương đồng về chủ đề/nội dung",
                "range": "0-1 (càng cao càng tốt)"
            },
            "jaccard_similarity": {
                "name": "Jaccard Index",
                "description": "Tỷ lệ từ chung / tổng số từ (loại trùng)",
                "interpretation": "Đo overlap về từ vựng",
                "range": "0-1 (càng cao càng tốt)"
            },
            "compression": {
                "name": "Compression Ratio",
                "description": "Tỷ lệ độ dài tóm tắt / độ dài gốc",
                "interpretation": "Đo mức độ nén thông tin",
                "range": "0-1 (0.2-0.4 là tốt, nén còn 20-40%)"
            },
            "keyword_coverage": {
                "name": "Keyword Coverage",
                "description": "Tỷ lệ từ khóa quan trọng được giữ lại",
                "interpretation": "Đo việc bảo toàn thông tin quan trọng",
                "range": "0-1 (càng cao càng tốt)"
            },
            "information_density": {
                "name": "Information Density",
                "description": "Mật độ thông tin có ích trong tóm tắt",
                "interpretation": "Đo hiệu quả của việc tóm tắt",
                "range": "0-1 (càng cao càng tốt)"
            },
            "overall_score": {
                "name": "Overall Score",
                "description": "Điểm tổng hợp (weighted average)",
                "interpretation": "Đánh giá chung chất lượng tóm tắt",
                "range": "0-1 (>0.5 là tốt, >0.7 là rất tốt)"
            }
        }