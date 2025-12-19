import google.generativeai as genai
import streamlit as st
import time
from google.api_core.exceptions import ResourceExhausted

class AI_Core:
    def __init__(self):
        try:
            api_key = st.secrets["api_keys"]["gemini_api_key"]
            genai.configure(api_key=api_key)
            self.flash = genai.GenerativeModel('gemini-2.5-flash')
            self.pro = genai.GenerativeModel('gemini-2.5-pro')
        except:
            st.error("Lỗi API Key AI.")

    def generate(self, prompt, model_type="flash", system_instruction=None):
        model = self.pro if model_type == "pro" else self.flash
        
        # Ghép "Tư duy" (System Instruction) vào Prompt
        final_prompt = prompt
        if system_instruction:
            final_prompt = f"HƯỚNG DẪN HỆ THỐNG (SYSTEM INSTRUCTION):\n{system_instruction}\n\nYÊU CẦU NGƯỜI DÙNG:\n{prompt}"

        # Cơ chế Retry (Lì đòn)
        for wait in [2, 5, 10, 20]:
            try:
                res = model.generate_content(final_prompt)
                return res.text
            except ResourceExhausted:
                time.sleep(wait)
            except Exception as e:
                return f"Lỗi AI: {e}"
        return "Server quá tải. Thử lại sau."

    @st.cache_data(ttl=3600)
    def analyze_static(_self, text, instruction):
        """Dùng cho RAG (Sách/Luật)"""
        try:
            m = genai.GenerativeModel('gemini-2.5-flash')
            return m.generate_content(f"{instruction}\n\n{text[:50000]}").text
        except: return "Lỗi phân tích."
