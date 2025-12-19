# FILE: ai_core.py
import google.generativeai as genai
import streamlit as st
import time
import json
from google.api_core.exceptions import ResourceExhausted

class AI_Engine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if not self.initialized:
            try:
                # Lấy API Key
                api_key = st.secrets["api_keys"]["gemini_api_key"]
                genai.configure(api_key=api_key)
                
                # Cấu hình Model (Ưu tiên Flash cho nhanh và rẻ)
                self.model = genai.GenerativeModel('gemini-2.5-flash')
                self.pro_model = genai.GenerativeModel('gemini-2.5-pro') # Dùng khi cần suy luận sâu
                
            except Exception as e:
                self.model = None
            self.initialized = True

    def generate_content(self, prompt, system_instruction=None, use_pro=False):
        """Hàm gọi AI tổng quát cho mọi tác vụ"""
        target_model = self.pro_model if use_pro else self.model
        if not target_model: return "Lỗi: Chưa cấu hình API Key."

        # Nếu có System Instruction (Dành cho Thúc thúc)
        # Lưu ý: Gemini Flash hỗ trợ system instruction qua cấu hình hoặc gộp vào prompt
        final_prompt = prompt
        if system_instruction:
            final_prompt = f"{system_instruction}\n\nUSER INPUT:\n{prompt}"

        for i in range(3): # Retry 3 lần
            try:
                response = target_model.generate_content(final_prompt)
                return response.text
            except ResourceExhausted:
                time.sleep(5) # Nghỉ 5s nếu hết quota
            except Exception as e:
                time.sleep(1)
        return "⚠️ Hệ thống đang bận hoặc hết Quota. Vui lòng thử lại sau."

    # --- CÁC HÀM DỊCH THUẬT CŨ (GIỮ LẠI ĐỂ KHÔNG HỎNG APP) ---
    def translate_text(self, text, source, target, include_english):
        # ... (Chị có thể giữ nguyên logic dịch cũ ở đây hoặc gọi generate_content) ...
        # Để code gọn, em dùng luôn hàm generate_content
        prompt = f"Translate from {source} to {target}. Text: {text}. Output only translation."
        if include_english and target != 'en':
            prompt += " Also provide English translation on a new line."
        return self.generate_content(prompt)

    def process_chinese_text(self, word, target):
        # Logic phân tích từ vựng (rút gọn)
        prompt = f"Analyze Chinese word '{word}' to {target}. JSON Output: [{{'word': '{word}', 'pinyin': '', 'translations': ['Meaning']}}]"
        res = self.generate_content(prompt)
        try:
            # Xử lý JSON sơ sài để demo, chị dùng logic cũ nếu cần kỹ hơn
            import re
            if "```" in res: res = re.sub(r'```json|```', '', res).strip()
            return json.loads(res)
        except:
            return [{'word': word, 'pinyin': '', 'translations': ['...']}]
