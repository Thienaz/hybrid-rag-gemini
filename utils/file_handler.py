# utils/file_handler.py
import io
import fitz  # PyMuPDF
import docx
from utils.logger import logger

class FileHandler:
    @staticmethod
    def read_multiple_files(uploaded_files):
        """Đọc nội dung và bóc tách đa phương tiện (Text + Images) từ nhiều file"""
        combined_text = ""
        metadata = {
            "total_files": len(uploaded_files), 
            "total_images_detected": 0, 
            "file_details": {}
        }
        
        for file in uploaded_files:
            filename = file.name
            bytes_data = file.getvalue()
            text = ""
            images_in_file = 0
            
            try:
                if filename.lower().endswith('.pdf'):
                    # Dùng PyMuPDF để bóc tách sâu
                    doc = fitz.open(stream=bytes_data, filetype="pdf")
                    for page in doc:
                        text += page.get_text() + "\n"
                        # Phát hiện và trích xuất danh sách hình ảnh trên trang
                        image_list = page.get_images(full=True)
                        images_in_file += len(image_list)
                        
                elif filename.lower().endswith('.docx'):
                    # Đọc file Word
                    doc = docx.Document(io.BytesIO(bytes_data))
                    for para in doc.paragraphs:
                        text += para.text + "\n"
                elif filename.lower().endswith('.txt'):
                    text = bytes_data.decode('utf-8')
                
                # Cập nhật Metadata (Yêu cầu của Thầy)
                metadata["total_images_detected"] += images_in_file
                metadata["file_details"][filename] = {
                    "text_length": len(text), 
                    "images_extracted": images_in_file
                }
                
                # Gắn nhãn để NLP Engine phân biệt đa tài liệu
                combined_text += f"\n--- BẮT ĐẦU TÀI LIỆU: {filename} ---\n"
                combined_text += text
                combined_text += f"\n--- KẾT THÚC TÀI LIỆU ---\n"
                
                logger.info(f"📄 Đã đọc {filename}: {len(text)} ký tự, {images_in_file} hình ảnh.")
            except Exception as e:
                logger.error(f"Lỗi đọc file {filename}: {e}")
                
        return combined_text, metadata