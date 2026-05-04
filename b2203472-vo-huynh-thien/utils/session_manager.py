# utils/session_manager.py

import os
import json
import base64
import time
import shutil
from datetime import datetime
from utils.logger import logger

WORKSPACE_DIR = "workspace/sessions"
MAX_SESSIONS = 5

class SessionManager:
    """Quản lý ĐA PHIÊN dưới dạng THƯ MỤC CÁC NHÂN (Chứa data và log)"""
    
    @staticmethod
    def init_workspace():
        if not os.path.exists(WORKSPACE_DIR):
            os.makedirs(WORKSPACE_DIR)

    @staticmethod
    def get_session_dir(session_id):
        return os.path.join(WORKSPACE_DIR, str(session_id))

    @staticmethod
    def log_trace(session_id, action, details):
        """Ghi lại dấu vết (Trace) của AI vào file trace.log của từng Session"""
        if not session_id: return
        session_dir = SessionManager.get_session_dir(session_id)
        if not os.path.exists(session_dir):
            os.makedirs(session_dir)
            
        log_file = os.path.join(session_dir, "trace.log")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] [STEP: {action}]\n")
            f.write(f"{details}\n")
            f.write("-" * 50 + "\n")

    @staticmethod
    def get_all_sessions():
        SessionManager.init_workspace()
        sessions = []
        for item in os.listdir(WORKSPACE_DIR):
            session_dir = os.path.join(WORKSPACE_DIR, item)
            if os.path.isdir(session_dir):
                state_file = os.path.join(session_dir, "state.json")
                if os.path.exists(state_file):
                    try:
                        with open(state_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            sessions.append({
                                "id": data.get("session_id", item),
                                "name": data.get("session_name", "Phiên chưa đặt tên"),
                                "updated_at": data.get("updated_at", 0)
                            })
                    except: continue
        return sorted(sessions, key=lambda x: x["updated_at"], reverse=True)

    @staticmethod
    def save_session(state):
        SessionManager.init_workspace()
        session_id = state.get("current_session_id")
        if not session_id: return

        # Xóa bớt phiên cũ nếu vượt quá MAX_SESSIONS
        sessions = SessionManager.get_all_sessions()
        if len(sessions) >= MAX_SESSIONS:
            to_delete = sessions[MAX_SESSIONS - 1:]
            for s in to_delete:
                if s["id"] != session_id:
                    SessionManager.delete_session(s["id"])

        files_data = state.get("uploaded_files_data", {})
        file_names = list(files_data.keys())
        if len(file_names) == 0: session_name = "Phiên trò chuyện trống"
        elif len(file_names) == 1: session_name = file_names[0]
        else: session_name = f"{file_names[0]} và {len(file_names)-1} file khác"

        encoded_files = {}
        for filename, bytes_data in files_data.items():
            encoded_files[filename] = base64.b64encode(bytes_data).decode('utf-8')

        data = {
            "session_id": session_id,
            "session_name": session_name,
            "updated_at": time.time(),
            "doc_processed": state.get("doc_processed", False),
            "is_processed": state.get("is_processed", False),
            "original_text": state.get("original_text", ""),
            "chat_history": state.get("chat_history", []),
            "current_summary": state.get("current_summary", {}),
            "file_metadata": state.get("file_metadata", {}),
            "uploaded_files_data": encoded_files
        }

        session_dir = SessionManager.get_session_dir(session_id)
        if not os.path.exists(session_dir):
            os.makedirs(session_dir)
            
        filepath = os.path.join(session_dir, "state.json")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Lỗi khi sao lưu session: {e}")

    @staticmethod
    def load_session(session_id):
        filepath = os.path.join(SessionManager.get_session_dir(session_id), "state.json")
        if not os.path.exists(filepath): return None
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            decoded_files = {}
            if "uploaded_files_data" in data:
                for filename, b64_str in data["uploaded_files_data"].items():
                    decoded_files[filename] = base64.b64decode(b64_str)
            data["uploaded_files_data"] = decoded_files
            return data
        except Exception as e: return None

    @staticmethod
    def delete_session(session_id):
        session_dir = SessionManager.get_session_dir(session_id)
        if os.path.exists(session_dir):
            shutil.rmtree(session_dir) # Xóa toàn bộ thư mục chứa log và json
            logger.info(f"🗑️ Đã xóa thư mục session {session_id}")