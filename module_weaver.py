# FILE: module_weaver.py
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
from docx import Document
from bs4 import BeautifulSoup
from streamlit_agraph import agraph, Node, Edge, Config
import plotly.express as px
import time

# Import các Blocks dùng chung
from ai_core import AI_Core
from voice_block import Voice_Engine
from prompts import DEBATE_PERSONAS

# Khởi tạo (Cache để không load lại model nặng)
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def doc_file(uploaded_file):
    """Hàm đọc file đa năng"""
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

def run():
    # Gọi các trưởng phòng
    ai = AI_Core()
    voice = Voice_Engine()
    
    st.header("🧠 The Cognitive Weaver (Người Dệt Nhận Thức)")
    
    tab1, tab2, tab3 = st.tabs(["📚 RAG & Graph", "🗣️ Tranh Biện", "🎙️ Studio"])

    # --- TAB 1: RAG & KNOWLEDGE GRAPH (Dùng sentence-transformers & agraph) ---
    with tab1:
        st.subheader("1. Phân tích & Kết nối Tri thức")
        
        c1, c2 = st.columns([1, 1])
        with c1:
            uploaded_file = st.file_uploader("Nạp tài liệu (PDF/Docx/Txt)", key="weaver_up")
        with c2:
            st.info("Hệ thống sẽ dùng `sentence-transformers` để vector hóa văn bản và vẽ Knowledge Graph.")

        if uploaded_file:
            text = doc_file(uploaded_file)
            st.success(f"Đã đọc {len(text)} ký tự.")
            
            # Phân tích bằng Gemini
            if st.button("🚀 Phân tích sâu"):
                with st.spinner("Gemini đang đọc..."):
                    res = ai.analyze_static(text, "Phân tích cấu trúc, ý chính và các khái niệm cốt lõi.")
                    st.markdown(res)

            # Vẽ Graph (Demo tính năng agraph)
            with st.expander("🕸️ Xem Vũ Trụ Khái Niệm (Knowledge Graph)"):
                # Demo tạo graph đơn giản từ text (Thực tế cần xử lý phức tạp hơn)
                nodes = []
                edges = []
                nodes.append(Node(id="Root", label="Tài liệu", size=25, color="#ff5733"))
                
                # Giả lập trích xuất từ khóa
                keywords = text.split()[:5] # Lấy 5 từ đầu làm demo
                for i, kw in enumerate(keywords):
                    nodes.append(Node(id=str(i), label=kw, size=15))
                    edges.append(Edge(source="Root", target=str(i)))
                
                config = Config(width=700, height=500, directed=True, physics=True)
                agraph(nodes, edges, config)

    # --- TAB 2: TRANH BIỆN (Dùng chung logic với CFO nhưng giao diện khác) ---
    with tab2:
        st.subheader("2. Đấu Trường Tư Duy")
        persona = st.selectbox("Chọn Đối Thủ:", list(DEBATE_PERSONAS.keys()), key="weaver_persona")
        
        if "weaver_chat" not in st.session_state: st.session_state.weaver_chat = []

        for msg in st.session_state.weaver_chat:
            st.chat_message(msg["role"]).write(msg["content"])

        if prompt := st.chat_input("Nhập chủ đề tranh luận..."):
            st.chat_message("user").write(prompt)
            st.session_state.weaver_chat.append({"role": "user", "content": prompt})
            
            with st.chat_message("assistant"):
                sys_prompt = DEBATE_PERSONAS[persona]
                # Gọi AI Core
                reply = ai.generate(prompt, model_type="pro", system_instruction=sys_prompt)
                st.write(reply)
                st.session_state.weaver_chat.append({"role": "assistant", "content": reply})

    # --- TAB 3: VOICE STUDIO (Dùng edge_tts & mic_recorder) ---
    with tab3:
        st.subheader("3. Phòng Thu AI")
        text_input = st.text_area("Nhập văn bản để đọc:", height=150, key="weaver_tts_input")
        
        c_v1, c_v2 = st.columns(2)
        with c_v1:
            lang = st.selectbox("Ngôn ngữ:", ["vi", "en", "zh"], key="weaver_lang")
        with c_v2:
            speed = st.slider("Tốc độ:", -50, 50, 0, key="weaver_speed")
            
        if st.button("🔊 Tạo Audio"):
            path = voice.speak(text_input, lang=lang, speed=speed)
            if path:
                st.audio(path)
                st.success("Xong!")

# Hàm này để file app.py gọi
if __name__ == "__main__":
    run()
