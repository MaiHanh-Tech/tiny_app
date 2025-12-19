import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
from docx import Document
from bs4 import BeautifulSoup
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import plotly.express as px
import markdown
import json
import re
from streamlit_agraph import agraph, Node, Edge, Config
import sys
import time

# --- IMPORT CÁC META-BLOCKS DÙNG CHUNG ---
from auth_block import AuthBlock
from ai_core import AI_Core
from voice_block import Voice_Engine
from prompts import DEBATE_PERSONAS, BOOK_ANALYSIS_PROMPT

# --- KHỞI TẠO CÔNG CỤ ĐẶC THÙ CỦA WEAVER ---
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def doc_file(uploaded_file):
    if not uploaded_file: return ""
    ext = uploaded_file.name.split('.')[-1].lower()
    try:
        if ext == "pdf":
            reader = PdfReader(uploaded_file)
            return "\n".join([page.extract_text() for page in reader.pages])
        elif ext == "docx":
            doc = Document(uploaded_file)
            return "\n".join([p.text for p in doc.paragraphs])
        elif ext in ["txt", "md", "html"]:
            return str(uploaded_file.read(), "utf-8")
    except: return ""
    return ""

# --- LOGIC GSHEET (NHẬT KÝ VĨNH VIỄN) ---
def connect_gsheet():
    try:
        if "gcp_service_account" not in st.secrets: return None
        creds_dict = dict(st.secrets["gcp_service_account"])
        if "private_key" in creds_dict:
            creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        return client.open("AI_History_Logs").sheet1
    except: return None

def tai_lich_su_tu_sheet():
    try:
        sheet = connect_gsheet()
        if sheet:
            data = sheet.get_all_records()
            my_user = st.session_state.get("current_user", "")
            if st.session_state.get("is_admin", False): return data
            return [item for item in data if item.get("User") == my_user]
    except: return []
    return []

# --- HÀM CHẠY CHÍNH CỦA MODULE ---
def run():
    # Khởi tạo Trưởng phòng
    ai = AI_Core()
    voice = Voice_Engine()
    auth = AuthBlock()
    
    st.header("🧠 The Cognitive Weaver (Người Dệt Nhận Thức)")

    # Tabs (Giữ nguyên cấu trúc 5 Tab của chị)
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📚 Phân Tích Sách", 
        "✍️ Dịch Giả", 
        "🗣️ Tranh Biện (Uncle Mode)", 
        "🎙️ Phòng Thu AI", 
        "⏳ Nhật Ký"
    ])

    # === TAB 1: RAG & KNOWLEDGE GRAPH ===
    with tab1:
        st.subheader("Trợ lý Nghiên cứu & Knowledge Graph")
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1: file_excel = st.file_uploader("1. Kho Sách (Excel)", type="xlsx", key="w_excel")
        with c2: uploaded_files = st.file_uploader("2. Tài liệu mới", accept_multiple_files=True, key="w_docs")
        with c3: st.write(""); btn_run = st.button("🚀 PHÂN TÍCH NGAY", type="primary")

        if btn_run and uploaded_files:
            vec = load_embedding_model()
            has_db = False
            if file_excel:
                df_db = pd.read_excel(file_excel).dropna(subset=["Tên sách"])
                db_embs = vec.encode([f"{r['Tên sách']} {str(r.get('CẢM NHẬN',''))}" for _, r in df_db.iterrows()])
                has_db = True
                st.success(f"✅ Kết nối {len(df_db)} cuốn sách.")

            for f in uploaded_files:
                text = doc_file(f)
                link = ""
                if has_db:
                    q = vec.encode([text[:2000]])
                    sc = cosine_similarity(q, db_embs)[0]
                    idx = np.argsort(sc)[::-1][:3]
                    for i in idx:
                        if sc[i] > 0.35: link += f"- {df_db.iloc[i]['Tên sách']} ({sc[i]*100:.0f}%)\n"

                with st.spinner(f"Đang dệt nhận thức cho {f.name}..."):
                    prompt = f"Phân tích tài liệu: {f.name}. Liên quan: {link}. Nội dung: {text[:20000]}"
                    # Dùng AI Core có Cache để tiết kiệm quota
                    res = ai.analyze_static(prompt, BOOK_ANALYSIS_PROMPT)
                    st.markdown(f"### 📄 {f.name}")
                    st.markdown(res)
                    # Lưu log
                    if connect_gsheet():
                         connect_gsheet().append_row([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Phân Tích", f.name, res[:5000], st.session_state.current_user, 0, "Neutral"])

    # === TAB 2: DỊCH THUẬT ĐA CHIỀU ===
    with tab2:
        st.subheader("Dịch Thuật Chuyên Sâu")
        txt = st.text_area("Nhập văn bản cần dịch:", height=150)
        c_l, c_s, c_b = st.columns([1,1,1])
        with c_l: target_lang = st.selectbox("Dịch sang:", ["Tiếng Việt", "English", "Chinese", "French", "Japanese"])
        with c_s: style = st.selectbox("Phong cách:", ["Mặc định", "Hàn lâm", "Văn học", "Kinh tế", "Kiếm hiệp"])
        if st.button("✍️ Dịch Ngay") and txt:
            with st.spinner("AI đang chuyển ngữ..."):
                p = f"Dịch văn bản sau sang {target_lang} với phong cách {style}. Nếu sang Trung phải có Pinyin. Văn bản: {txt}"
                res = ai.generate(p, model_type="pro")
                st.markdown(res)

    # === TAB 3: ĐẤU TRƯỜNG TƯ DUY (UNCLE MODE) ===
    with tab3:
        st.subheader("Đấu Trường Tư Duy & Cố Vấn Hệ Thống")
        mode = st.radio("Chế độ:", ["👤 Solo (User vs AI)", "⚔️ Debate (AI vs AI)"], horizontal=True)
        
        persona_name = st.selectbox("Chọn Đối Thủ/Cố Vấn:", list(DEBATE_PERSONAS.keys()))
        
        if "weaver_history" not in st.session_state: st.session_state.weaver_history = []
        
        if st.button("🗑️ Xóa Chat"): 
            st.session_state.weaver_history = []
            st.rerun()

        for msg in st.session_state.weaver_history:
            st.chat_message(msg["role"]).write(msg["content"])

        if prompt := st.chat_input("Nhập luận điểm..."):
            st.chat_message("user").write(prompt)
            st.session_state.weaver_history.append({"role": "user", "content": prompt})
            
            with st.chat_message("assistant"):
                with st.spinner(f"{persona_name} đang suy ngẫm..."):
                    # Ghép lịch sử chat
                    history_context = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.weaver_history[-5:]])
                    # Gọi AI Core với cơ chế Lì đòn (Retry)
                    reply = ai.generate(prompt=f"Lịch sử: {history_context}\nCâu hỏi: {prompt}", 
                                      model_type="pro" if "Thúc Thúc" in persona_name else "flash", 
                                      system_instruction=DEBATE_PERSONAS[persona_name])
                    st.write(reply)
                    st.session_state.weaver_history.append({"role": "assistant", "content": reply})

    # === TAB 4: PHÒNG THU AI (FULL 6 GIỌNG) ===
    with tab4:
        st.subheader("🎙️ Phòng Thu AI Đa Ngôn Ngữ")
        c_in, c_ctrl = st.columns([3, 1])
        with c_in: inp_v = st.text_area("Văn bản cần đọc:", height=200, key="v_input")
        with c_ctrl:
            v_choice = st.selectbox("Chọn Giọng:", list(voice.VOICE_OPTIONS.keys()))
            speed_v = st.slider("Tốc độ:", -50, 50, 0)
        
        if st.button("🔊 TẠO AUDIO") and inp_v:
            with st.spinner("Đang tải giọng đọc..."):
                path = voice.speak(inp_v, voice_key=v_choice, speed=speed_v)
                if path:
                    st.audio(path)
                    with open(path, "rb") as f:
                        st.download_button("⬇️ Tải xuống MP3", f, "audio.mp3")

    # === TAB 5: NHẬT KÝ (Lấy từ GSheet) ===
    with tab5:
        st.subheader("⏳ Lịch Sử Hoạt Động")
        if st.button("🔄 Tải lại Nhật ký"):
            st.session_state.history_cloud = tai_lich_su_tu_sheet()
            st.rerun()
        
        data = st.session_state.get("history_cloud", [])
        if data:
            df_h = pd.DataFrame(data)
            # Vẽ biểu đồ cảm xúc nếu có data
            if "SentimentScore" in df_h.columns:
                fig = px.line(df_h, x="Time", y="SentimentScore", title="📈 Biểu đồ trạng thái tư duy")
                st.plotly_chart(fig, use_container_width=True)
            
            for item in reversed(data):
                with st.expander(f"⏰ {item.get('Time')} | {item.get('Type')} | {item.get('Title')}"):
                    st.markdown(item.get("Content"))
        else:
            st.info("Chưa có dữ liệu lịch sử.")

if __name__ == "__main__":
    run()
