# utils/logger.py
"""
Module Logging - Ghi nhận hoạt động hệ thống
Nâng cấp: Thread-safe, chống duplicate handler, ghi log ra file
"""

import logging
import threading
from datetime import datetime
import os


class ThesisLogger:
    """Logger tùy chỉnh cho luận văn - ghi log để demo và debug"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """Singleton pattern đảm bảo chỉ có 1 instance duy nhất"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, name="ThesisAI"):
        # Tránh khởi tạo lại khi Streamlit rerun
        if self._initialized:
            return
        self._initialized = True

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # Xóa tất cả handler cũ để tránh duplicate khi Streamlit rerun
        self.logger.handlers.clear()

        # Format log
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler (ghi log ra file để debug)
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_filename = os.path.join(
            log_dir,
            f"app_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler = logging.FileHandler(
            log_filename, encoding="utf-8", mode="a"
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Lưu log cho phân tích (thread-safe)
        self.activity_log = []
        self._log_lock = threading.Lock()

    def info(self, message):
        self.logger.info(message)
        self._save_activity("INFO", message)

    def warning(self, message):
        self.logger.warning(message)
        self._save_activity("WARNING", message)

    def error(self, message):
        self.logger.error(message)
        self._save_activity("ERROR", message)

    def debug(self, message):
        self.logger.debug(message)

    def _save_activity(self, level, message):
        """Lưu activity để hiển thị trong UI (thread-safe)"""
        with self._log_lock:
            self.activity_log.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "level": level,
                "message": message
            })
            # Giữ tối đa 100 log gần nhất
            if len(self.activity_log) > 100:
                self.activity_log = self.activity_log[-100:]

    def get_recent_logs(self, n=10):
        """Lấy n log gần nhất (thread-safe)"""
        with self._log_lock:
            return list(self.activity_log[-n:])


# Singleton instance
logger = ThesisLogger()