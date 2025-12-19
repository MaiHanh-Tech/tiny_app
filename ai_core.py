import google.generativeai as genai
import streamlit as st
import time
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable, InternalServerError

class AI_Core:
    def __init__(self):
        try:
            api_key = st.secrets["api_keys"]["gemini_api_key"]
            genai.configure(api_key=api_key)
            
            # Khởi tạo các model
            self.flash = genai.GenerativeModel('gemini-2.5-flash')
            self.pro = genai.GenerativeModel('gemini-2.5-pro')
            # Thử thêm bản experimental nếu có
            self.exp = genai.GenerativeModel('gemini-2.5-flash-latest')
        except Exception as e:
            st.error(f"Lỗi API Key: {e}")

    def _get_safety(self):
        return [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

    def generate(self, prompt, model_type="flash", system_instruction=None):
        """
        Chiến thuật 'Không lùi bước': 
        Thử Pro -> Lỗi -> Thử Flash -> Lỗi -> Thử Exp -> Lỗi -> Chờ 10s -> Thử lại Flash
        """
        
        # 1. Xây dựng Prompt
        full_prompt = prompt
        if system_instruction:
            full_prompt = f"SYSTEM INSTRUCTION:\n{system_instruction}\n\nUSER REQUEST:\n{prompt}"

        # 2. Lên kế hoạch tác chiến (Danh sách model sẽ thử theo thứ tự)
        # Cấu trúc: (Model_Object, Tên_Model, Thời_gian_chờ_nếu_lỗi)
        if model_type == "pro":
            plan = [
                (self.pro, "Pro", 2),       # Thử Pro trước
                (self.flash, "Flash", 5),   # Lỗi thì sang Flash ngay
                (self.exp, "Exp", 5),       # Lỗi nữa thì sang bản 2.0 Exp
                (self.flash, "Flash Retry", 10) # Cùng đường thì quay lại Flash chờ 10s
            ]
        else:
            plan = [
                (self.flash, "Flash", 2),
                (self.exp, "Exp", 5),
                (self.pro, "Pro", 5),       # Flash lỗi thì thử sang Pro cầu may
                (self.flash, "Flash Retry", 10)
            ]

        # 3. Thực thi kế hoạch
        last_error = ""
        for model, name, wait_time in plan:
            try:
                # Nếu model chưa khởi tạo được thì bỏ qua
                if not model: continue
                
                response = model.generate_content(
                    full_prompt, 
                    safety_settings=self._get_safety()
                )
                
                if response.text:
                    return response.text
                    
            except ResourceExhausted:
                # st.warning(f"⚠️ Model {name} bận, đang chuyển kênh...") # (Có thể bỏ comment để debug)
                time.sleep(wait_time)
                last_error = "Hết Quota (ResourceExhausted)"
            except Exception as e:
                time.sleep(1)
                last_error = str(e)
                
        # Nếu thử hết cách mà vẫn chết
        return f"⚠️ Hệ thống đang quá tải thực sự (Google chặn tạm thời). Lỗi cuối cùng: {last_error}"

    @st.cache_data(ttl=3600)
    def analyze_static(_self, text, instruction):
        """Hàm dùng cho RAG - Có Cache"""
        # Với hàm này ta tạo instance mới để tránh conflict
        try:
            m = genai.GenerativeModel('gemini-2.5-flash')
            res = m.generate_content(f"{instruction}\n\n{text[:50000]}")
            return res.text
        except Exception as e:
            return f"Lỗi phân tích: {e}"
