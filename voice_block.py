import edge_tts
import asyncio
import tempfile
import streamlit as st
import unicodedata
import re

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
        """Generate audio file asynchronously"""
        try:
            communicate = edge_tts.Communicate(text, voice, rate=rate)
            # Tạo file tạm thời để tránh lỗi quyền ghi file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                await communicate.save(fp.name)
                return fp.name
        except Exception as e:
            st.error(f"Lỗi tạo audio: {e}")
            return None

    def _clean_text_for_speech(self, text, voice_code):
        """
        ✅ LỌC VÀ CHUẨN HÓA VĂN BẢN CHO TỪ GIỌNG NÓI
        
        Lý do cần thiết:
        - Edge TTS không đọc được các ký tự đặc biệt (emoji, ký hiệu toán học)
        - Unicode diacritics (dấu thanh tiếng Việt) đôi khi bị lỗi
        - Cần xử lý khác nhau cho từng ngôn ngữ
        """
        if not text or not text.strip():
            return None
        
        # 1. ✅ XÓA EMOJI VÀ KÝ TỰ ĐẶC BIỆT
        # Regex loại bỏ emoji ranges trong Unicode
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+", 
            flags=re.UNICODE
        )
        text = emoji_pattern.sub('', text)
        
        # 2. ✅ XỬ LÝ THEO NGÔN NGỮ
        if "vi-VN" in voice_code:
            # Tiếng Việt: GIỮ NGUYÊN dấu thanh (không normalize)
            # Chỉ xóa ký tự điều khiển
            text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C')
            
        elif "zh-CN" in voice_code:
            # Tiếng Trung: GIỮ NGUYÊN chữ Hán
            # Chỉ xóa ký tự không in được
            text = ''.join(char for char in text 
                          if unicodedata.category(char)[0] != 'C')
            
        elif "en-US" in voice_code:
            # Tiếng Anh: Normalize về ASCII nếu có thể
            # Nhưng GIỮ các ký tự Unicode nếu không convert được
            try:
                # Thử decompose rồi loại bỏ dấu
                text = unicodedata.normalize('NFKD', text)
                # Chỉ giữ ASCII + một số ký tự Latin mở rộng
                text = text.encode('ascii', 'ignore').decode('ascii')
            except:
                # Nếu lỗi, giữ nguyên
                pass
        
        # 3. ✅ DỌN DẸP CUỐI CÙNG
        # Xóa khoảng trắng thừa
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Xóa các ký tự điều khiển còn sót
        text = ''.join(char for char in text 
                      if char.isprintable() or char.isspace())
        
        # 4. ✅ GIỚI HẠN ĐỘ DÀI (Edge TTS có limit ~5000 chars)
        MAX_LENGTH = 4500
        if len(text) > MAX_LENGTH:
            text = text[:MAX_LENGTH]
            st.warning(f"⚠️ Văn bản quá dài. Chỉ đọc {MAX_LENGTH} ký tự đầu.")
        
        return text if text.strip() else None

    def speak(self, text, voice_key=None, speed=0):
        """
        Chuyển văn bản thành Audio Path
        
        Args:
            text: Văn bản cần đọc
            voice_key: Key trong VOICE_OPTIONS (VD: "🇻🇳 VN - Nam (Nam Minh)")
            speed: Tốc độ (-50 đến 50)
        
        Returns:
            str: Đường dẫn file audio, hoặc None nếu lỗi
        """
        if not text: 
            return None
        
        # Lấy code giọng đọc từ key, nếu không có thì mặc định Hoài My
        voice_code = self.VOICE_OPTIONS.get(voice_key, "vi-VN-HoaiMyNeural")
        
        # ✅ LỌC VÀ CHUẨN HÓA VĂN BẢN
        cleaned_text = self._clean_text_for_speech(text, voice_code)
        
        if not cleaned_text:
            st.warning("⚠️ Văn bản không hợp lệ hoặc chỉ chứa ký tự đặc biệt")
            return None
        
        # Định dạng tốc độ chuẩn cho Edge TTS (VD: "+10%")
        rate_str = f"{'+' if speed >= 0 else ''}{speed}%"

        try:
            # Chạy Async trong môi trường Sync của Streamlit
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            path = loop.run_until_complete(
                self._gen(cleaned_text, voice_code, rate_str)
            )
            loop.close()
            
            return path
            
        except Exception as e:
            st.error(f"❌ Lỗi tạo giọng nói: {e}")
            return None
        finally:
            # ✅ CLEANUP: Đảm bảo đóng event loop
            try:
                loop.close()
            except:
                pass
