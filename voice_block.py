import edge_tts
import asyncio
import tempfile
import streamlit as st

class Voice_Engine:
    def __init__(self):
        # Danh sách 6 giọng chuẩn (Nam/Nữ cho 3 ngôn ngữ)
        self.VOICE_OPTIONS = {
            "🇻🇳 VN - Nữ (Hoài My)": "vi-VN-HoaiMyNeural",
            "🇻🇳 VN - Nam (Nam Minh)": "vi-VN-NamMinhNeural",
            "🇺🇸 US - Nữ (Emma)": "en-US-EmmaNeural",
            "🇺🇸 US - Nam (Andrew)": "en-US-AndrewMultilingualNeural",
            "🇨🇳 CN - Nữ (Xiaoyi)": "zh-CN-XiaoyiNeural",
            "🇨🇳 CN - Nam (Yunjian)": "zh-CN-YunjianNeural"
        }

    async def _gen(self, text, voice, rate):
        communicate = edge_tts.Communicate(text, voice, rate=rate)
        # Tạo file tạm thời để tránh lỗi quyền ghi file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            await communicate.save(fp.name)
            return fp.name

    def speak(self, text, voice_key=None, speed=0):
        """
        Chuyển văn bản thành Audio Path
        text: Văn bản cần đọc
        voice_key: Key trong VOICE_OPTIONS (VD: "🇻🇳 VN - Nam (Nam Minh)")
        speed: Tốc độ (-50 đến 50)
        """
        if not text: return None
        
        # Lấy code giọng đọc từ key, nếu không có thì mặc định Hoài My
        voice_code = self.VOICE_OPTIONS.get(voice_key, "vi-VN-HoaiMyNeural")
        
        # Định dạng tốc độ chuẩn cho Edge TTS (VD: "+10%")
        rate_str = f"{'+' if speed >= 0 else ''}{speed}%"

        try:
            # Chạy Async trong môi trường Sync của Streamlit
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            path = loop.run_until_complete(self._gen(text, voice_code, rate_str))
            return path
        except Exception as e:
            st.error(f"Lỗi tạo giọng nói: {e}")
            return None
