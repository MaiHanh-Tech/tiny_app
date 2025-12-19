import edge_tts
import asyncio
import tempfile
import streamlit as st

class Voice_Engine:
    async def _gen(self, text, voice, rate):
        communicate = edge_tts.Communicate(text, voice, rate=rate)
        # Tạo file tạm thời để tránh lỗi quyền ghi file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            await communicate.save(fp.name)
            return fp.name

    def speak(self, text, lang="vi", speed=0):
        """Chuyển văn bản thành Audio Path"""
        if not text: return None
        
        # Map ngôn ngữ -> Giọng đọc
        voices = {
            "vi": "vi-VN-HoaiMyNeural",
            "en": "en-US-ChristopherNeural",
            "zh": "zh-CN-XiaoyiNeural"
        }
        voice_code = voices.get(lang, "vi-VN-HoaiMyNeural")
        rate_str = f"{'+' if speed >= 0 else ''}{speed}%"

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            path = loop.run_until_complete(self._gen(text, voice_code, rate_str))
            return path
        except Exception as e:
            st.error(f"Lỗi tạo giọng nói: {e}")
            return None
