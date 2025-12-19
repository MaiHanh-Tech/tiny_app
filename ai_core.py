import google.generativeai as genai
import streamlit as st
import time
from google.api_core.exceptions import ResourceExhausted

class AI_Core:
    def __init__(self):
        try:
            api_key = st.secrets["api_keys"]["gemini_api_key"]
            genai.configure(api_key=api_key)
            
            # Khởi tạo model mặc định (Flash cho nhanh)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            # Model mạnh hơn nếu cần (Pro)
            self.model_pro = genai.GenerativeModel('gemini-1.5-pro')
            
        except Exception as e:
            st.error(f"Lỗi khởi tạo AI: {e}")
            self.model = None

    def generate(self, prompt, model_type="flash", system_instruction=None):
        """Hàm gọi AI duy nhất"""
        if not self.model: return "Lỗi: Chưa cấu hình API Key."

        # 1. Chọn Model
        target_model = self.model_pro if model_type == "pro" else self.model
        
        # 2. Cấu hình an toàn (Tắt kiểm duyệt để dịch thoải mái)
        safety = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        # 3. Ghép System Instruction vào Prompt (Cách an toàn nhất cho mọi version)
        full_prompt = prompt
        if system_instruction:
            full_prompt = f"SYSTEM INSTRUCTION: {system_instruction}\n\nUSER REQUEST: {prompt}"

        # 4. Cơ chế Retry (Lì đòn)
        wait_times = [2, 5, 10, 20] # Giây chờ
        
        for i, wait in enumerate(wait_times):
            try:
                response = target_model.generate_content(full_prompt, safety_settings=safety)
                return response.text
            except ResourceExhausted:
                if i == len(wait_times) - 1:
                    return "⚠️ Server quá tải. Vui lòng thử lại sau 1 phút."
                time.sleep(wait) # Chờ rồi thử lại
            except Exception as e:
                return f"Lỗi AI: {str(e)}"
        
        return None

    # Cache data 1 tiếng cho các tác vụ phân tích tĩnh (Sách/Luật)
    @st.cache_data(ttl=3600)
    def analyze_static(_self, text, task_description):
        """Hàm dùng riêng cho RAG (Phân tích tài liệu) để tiết kiệm quota"""
        prompt = f"{task_description}:\n\n{text[:50000]}" # Giới hạn 50k ký tự
        # Gọi lại hàm generate nội bộ (cần workaround vì cache không nhận self)
        # Cách đơn giản nhất cho cache: Khởi tạo lại model nhẹ trong này
        try:
            m = genai.GenerativeModel('gemini-1.5-flash')
            return m.generate_content(prompt).text
        except:
            return "Lỗi phân tích static."
