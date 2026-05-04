#core/evaluator.py

"""
Module Đánh Giá - Wrapper cho SummaryMetrics
"""

from core.metrics import SummaryMetrics
from utils.logger import logger


class SummaryEvaluator:
    """
    Wrapper class cho việc đánh giá tóm tắt
    Sử dụng SummaryMetrics để tính toán các độ đo
    """
    
    def __init__(self):
        self.metrics = SummaryMetrics()
    
    def evaluate_summary(self, original_text, summary_text):
        """
        Đánh giá toàn diện bản tóm tắt
        
        Args:
            original_text: Văn bản gốc
            summary_text: Văn bản tóm tắt
            
        Returns:
            dict: Tất cả các metrics
        """
        logger.info("Bắt đầu đánh giá chất lượng tóm tắt...")
        
        try:
            results = self.metrics.evaluate_all(original_text, summary_text)
            logger.info(f"Đánh giá hoàn tất. Overall Score: {results['overall_score']}")
            return results
        except Exception as e:
            logger.error(f"Lỗi đánh giá: {str(e)}")
            return self._empty_results()
    
    def _empty_results(self):
        """Trả về kết quả rỗng khi có lỗi"""
        return {
            "rouge_1": {"precision": 0, "recall": 0, "f1": 0},
            "rouge_2": {"precision": 0, "recall": 0, "f1": 0},
            "rouge_l": {"precision": 0, "recall": 0, "f1": 0},
            "bleu": {"bleu_1": 0, "bleu_2": 0, "bleu_3": 0, "bleu_4": 0, "bleu_avg": 0},
            "cosine_similarity": 0,
            "jaccard_similarity": 0,
            "compression": {"ratio": 0, "original_words": 0, "summary_words": 0},
            "keyword_coverage": {"coverage": 0, "keywords_found": 0, "total_keywords": 0},
            "information_density": {"density": 0},
            "overall_score": 0
        }
    
    def get_explanations(self):
        """Lấy giải thích cho các độ đo"""
        return self.metrics.get_metrics_explanation()
    
    def quick_evaluate(self, original_text, summary_text):
        """
        Đánh giá nhanh với các metrics chính
        (Dùng khi cần hiển thị nhanh)
        """
        return {
            "rouge_1_f1": self.metrics.rouge_1(original_text, summary_text)["f1"],
            "rouge_2_f1": self.metrics.rouge_2(original_text, summary_text)["f1"],
            "cosine_sim": self.metrics.cosine_similarity(original_text, summary_text),
            "compression": self.metrics.compression_ratio(original_text, summary_text)["ratio"]
        }