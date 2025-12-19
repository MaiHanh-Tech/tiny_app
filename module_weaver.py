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
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import json
import re

# --- IMPORT CÁC META-BLOCKS ---
from ai_core import AI_Core
from voice_block import Voice_Engine
from prompts import DEBATE_PERSONAS, BOOK_ANALYSIS_PROMPT

# ==========================================
# 🌍 BỘ TỪ ĐIỂN ĐA NGÔN NGỮ (TÍCH HỢP VÀO MODULE)
# ==========================================
TRANS = {
    "vi": {
        "lang_select": "Ngôn ngữ / Language / 语言",
        "tab1": "📚 Phân Tích Sách",
        "tab2": "✍️ Dịch Giả",
        "tab3": "🗣️ Tranh Biện",
        "tab4": "🎙️ Phòng Thu AI",
        "tab5": "⏳ Nhật Ký",
        "t1_header": "Trợ lý Nghiên cứu & Knowledge Graph",
        "t1_up_excel": "1. Kết nối Kho Sách (Excel)",
        "t1_up_doc": "2. Tài liệu mới (PDF/Docx)",
        "t1_btn": "🚀 PHÂN TÍCH NGAY",
        "t1_analyzing": "Đang phân tích {name}...",
        "t2_header": "Dịch Thuật Đa Chiều",
        "t2_input": "Nhập văn bản cần dịch:",
        "t2_target": "Dịch sang:",
        "t2_style": "Phong cách:",
        "t2_btn": "✍️ Dịch Ngay",
        "t3_header": "Đấu Trường Tư Duy",
        "t3_persona_label": "Chọn Đối Thủ:",
        "t3_input": "Nhập chủ đề tranh luận...",
        "t3_clear": "🗑️ Xóa Chat",
        "t4_header": "🎙️ Phòng Thu AI Đa Ngôn Ngữ",
        "t4_voice": "Chọn Giọng:",
        "t4_speed": "Tốc độ:",
        "t4_btn": "🔊 TẠO AUDIO",
        "t5_header": "Nhật Ký & Lịch Sử",
        "t5_refresh": "🔄 Tải lại Lịch sử",
        "t5_empty": "Chưa có dữ liệu lịch sử.",
    },
    "en": {
        "lang_select": "Language",
        "tab1": "📚 Book Analysis",
        "tab2": "✍️ Translator",
        "tab3": "🗣️ Debater",
        "tab4": "🎙️ AI Studio",
        "tab5": "⏳ History",
        "t1_header": "Research Assistant & Knowledge Graph",
        "t1_up_excel": "1. Connect Book Database (Excel)",
        "t1_up_doc": "2. New Documents (PDF/Docx)",
        "t1_btn": "🚀 ANALYZE NOW",
        "t1_analyzing": "Analyzing {name}...",
        "t2_header": "Multidimensional Translator",
        "t2_input": "Enter text to translate:",
        "t2_target": "Translate to:",
        "t2_style": "Style:",
        "t2_btn": "✍️ Translate",
        "t3_header": "Thinking Arena",
        "t3_persona_label": "Choose Opponent:",
        "t3_input": "Enter debate topic...",
        "t3_clear": "🗑️ Clear Chat",
        "t4_header": "🎙️ Multilingual AI Studio",
        "t4_voice": "Select Voice:",
        "t4_speed": "Speed:",
        "t4_btn": "🔊 GENERATE AUDIO",
        "t5_header": "Logs & History",
        "t5_refresh": "🔄 Refresh History",
        "t5_empty": "No history data found.",
    },
    "zh": {
        "lang_select": "语言",
        "tab1": "📚 书籍分析",
        "tab2": "✍️ 翻译专家",
        "tab3": "🗣️ 辩论场",
        "tab4": "🎙️ AI 录音室",
        "tab5": "⏳ 历史记录",
        "t1_header": "研究助手 & 知识图谱",
        "t1_up_excel": "1. 连接书库 (Excel)",
        "t1_up_doc": "2. 上传新文档 (PDF/Docx)",
        "t1_btn": "🚀 立即分析",
        "t1_analyzing": "正在分析 {name}...",
        "t2_header": "多维翻译",
        "t2_input": "输入文本:",
        "t2_target": "翻译成:",
        "t2_style": "风格:",
        "t2_btn": "✍️ 翻译",
        "t3_header": "思维竞技场",
        "t3_persona_label": "选择对手:",
        "t3_input": "输入辩论主题...",
        "t3_clear": "🗑️ 清除聊天",
        "t4_header": "🎙️ AI 多语言录音室",
        "t4_voice": "选择声音:",
        "t4_speed": "语速:",
        "t4_btn": "🔊 生成音频",
        "t5_header": "日志 & 历史",
        "t5_refresh": "🔄 刷新历史",
        "t5_empty": "暂无历史数据。",
    }
}

# Hàm lấy text theo ngôn ngữ
def T(key):
    lang = st.session_state.get('weaver_lang', 'vi')
    return TRANS.get(lang, TRANS['vi']).get(key, key)

# --- CÁC HÀM PHỤ TRỢ ---
@st.cache_resource
def load_models():
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

def luu_lich_su(loai, tieu_de, noi_dung):
    thoi_gian = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user = st.session_state.get("current_user", "Unknown")
    try:
        sheet = connect_gsheet()
        if sheet: sheet.append_row([thoi_gian, loai, tieu_de, noi_dung, user, 0.0, "Neutral"])
    except: pass

def tai_lich_su():
    try:
        sheet = connect_gsheet()
        if sheet: return sheet.get_all_records()
    except: return []
    return []

# --- HÀM CHÍNH: RUN() ---
def run():
    # 1. Khởi tạo các Block
    ai = AI_Core()
    voice = Voice_Engine()
    
    # 2. Sidebar chọn ngôn ngữ cho Module này
    with st.sidebar:
        st.markdown("---")
        lang_choice = st.selectbox(
            "🌐 " + TRANS['vi']['lang_select'],
            ["Tiếng Việt", "English", "中文"],
            index=0,
            key="weaver_lang_selector"
        )
        # Lưu ngôn ngữ vào session state
        if lang_choice == "Tiếng Việt": st.session_state.weaver_lang = 'vi'
        elif lang_choice == "English": st.session_state.weaver_lang = 'en'
        elif lang_choice == "中文": st.session_state.weaver_lang = 'zh'
    
    st.header(f"🧠 {T('The Cognitive Weaver')}")
    
    # 5 TABS ĐẦY ĐỦ (Dùng hàm T để dịch tên Tab)
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        T("tab1"), T("tab2"), T("tab3"), T("tab4"), T("tab5")
    ])

    # === TAB 1: RAG & GRAPH ===
    with tab1:
        st.subheader(T("t1_header"))
        
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1: file_excel = st.file_uploader(T("t1_up_excel"), type="xlsx", key="w_t1_ex")
        with c2: uploaded_files = st.file_uploader(T("t1_up_doc"), type=["pdf", "docx", "txt"], accept_multiple_files=True, key="w_t1_doc")
        with c3: 
            st.write("")
            st.write("")
            btn_run = st.button(T("t1_btn"), type="primary", use_container_width=True)

        if btn_run and uploaded_files:
            vec = load_models()
            db, df = None, None
            has_db = False
            
            if file_excel:
                try:
                    df = pd.read_excel(file_excel).dropna(subset=["Tên sách"])
                    db = vec.encode([f"{r['Tên sách']} {str(r.get('CẢM NHẬN',''))}" for _, r in df.iterrows()])
                    has_db = True
                    st.success(T("t1_connect_ok").format(n=len(df)))
                except: st.error("Lỗi đọc Excel.")

            for f in uploaded_files:
                text = doc_file(f)
                link = ""
                if has_db:
                    q = vec.encode([text[:2000]])
                    sc = cosine_similarity(q, db)[0]
                    idx = np.argsort(sc)[::-1][:3]
                    for i in idx:
                        if sc[i] > 0.35: link += f"- {df.iloc[i]['Tên sách']} ({sc[i]*100:.0f}%)\n"

                with st.spinner(T("t1_analyzing").format(name=f.name)):
                    prompt = f"Phân tích tài liệu '{f.name}'. Liên quan: {link}\nNội dung: {text[:30000]}"
                    res = ai.analyze_static(prompt, BOOK_ANALYSIS_PROMPT)
                    
                    st.markdown(f"### 📄 {f.name}")
                    st.markdown(res)
                    st.markdown("---")
                    luu_lich_su("Phân Tích Sách", f.name, res[:200])

        # VẼ GRAPH (AGRAPH)
        if file_excel:
            try:
                with st.expander(T("t1_graph_title"), expanded=False):
                    vec = load_models()
                    if "book_embs" not in st.session_state:
                         st.session_state.book_embs = vec.encode(df["Tên sách"].tolist())
                    
                    embs = st.session_state.book_embs
                    sim = cosine_similarity(embs)
                    nodes, edges = [], []
                    
                    max_nodes = st.slider("Max Nodes:", 5, len(df), min(50, len(df)))
                    threshold = st.slider("Threshold:", 0.0, 1.0, 0.45)

                    for i in range(max_nodes):
                        nodes.append(Node(id=str(i), label=df.iloc[i]["Tên sách"], size=20, color="#FFD166"))
                        for j in range(i+1, max_nodes):
                            if sim[i,j]>threshold: edges.append(Edge(source=str(i), target=str(j), color="#118AB2"))
                    
                    config = Config(width=900, height=600, directed=False, physics=True, collapsible=False)
                    agraph(nodes, edges, config)
            except: pass

    # === TAB 2: DỊCH GIẢ ===
    with tab2:
        st.subheader(T("t2_header"))
        txt = st.text_area(T("t2_input"), height=150, key="w_t2_inp")
        c_l, c_s, c_b = st.columns([1,1,1])
        with c_l: target_lang = st.selectbox(T("t2_target"), ["Tiếng Việt", "English", "Chinese", "French", "Japanese"], key="w_t2_lang")
        with c_s: style = st.selectbox(T("t2_style"), ["Default", "Academic", "Literary", "Business"], key="w_t2_style")
        
        if st.button(T("t2_btn"), key="w_t2_btn") and txt:
            with st.spinner("AI Translating..."):
                p = f"Translate to {target_lang}. Style: {style}. Text: {txt}"
                res = ai.generate(p, model_type="pro")
                st.markdown(res)
                luu_lich_su("Dịch Thuật", f"{target_lang}", txt[:50])

    # === TAB 3: ĐẤU TRƯỜNG TƯ DUY (MULTI-AGENT ARENA) ===
    with tab3:
        st.subheader(T("t3_header"))
        mode = st.radio("Mode:", ["👤 Solo", "⚔️ Multi-Agent"], horizontal=True, key="w_t3_mode")
        
        if "weaver_chat" not in st.session_state: st.session_state.weaver_chat = []

        if mode == "👤 Solo":
            c1, c2 = st.columns([3, 1])
            with c1: persona = st.selectbox(T("t3_persona_label"), list(DEBATE_PERSONAS.keys()), key="w_t3_solo_p")
            with c2: 
                if st.button(T("t3_clear"), key="w_t3_clr"): 
                    st.session_state.weaver_chat = []
                    st.rerun()

            for msg in st.session_state.weaver_chat:
                st.chat_message(msg["role"]).write(msg["content"])

            if prompt := st.chat_input(T("t3_input")):
                st.chat_message("user").write(prompt)
                st.session_state.weaver_chat.append({"role": "user", "content": prompt})
                
                with st.chat_message("assistant"):
                    sys = DEBATE_PERSONAS[persona]
                    with st.spinner("..."):
                        res = ai.generate(prompt, model_type="flash", system_instruction=sys)
                        st.write(res)
                        st.session_state.weaver_chat.append({"role": "assistant", "content": res})
                        luu_lich_su("Tranh Biện Solo", persona, prompt)
        # --- PHẦN HỘI ĐỒNG TRANH BIỆN (ĐÃ SỬA) ---
        else:
            st.info("💡 Chọn tối đa 3 nhân vật để họ tự cãi nhau.")
            participants = st.multiselect(
                "Chọn Hội Đồng Tranh Biện:", 
                list(DEBATE_PERSONAS.keys()), 
                default=[list(DEBATE_PERSONAS.keys())[0], list(DEBATE_PERSONAS.keys())[1]],
                key="w_t3_multi_p"
            )
            topic = st.text_input("Chủ đề tranh luận:", key="w_t3_topic")
            
            c_start, c_del = st.columns([1, 5])
            with c_start:
                start_btn = st.button("🔥 KHAI CHIẾN", key="w_t3_start", disabled=(len(participants)<2))
            with c_del:
                if st.button("🗑️ Xóa Bàn", key="w_t3_multi_clr"):
                    st.session_state.weaver_chat = []
                    st.rerun()

            # Hiện lịch sử cũ để không bị mất khi rerun
            for msg in st.session_state.weaver_chat:
                if msg["role"] == "system":
                    st.chat_message("system").write(msg["content"])
                elif msg["role"] == "assistant":
                    st.chat_message("assistant").write(msg["content"])

            # Logic chạy vòng lặp
            if start_btn and topic:
                st.session_state.weaver_chat = [] # Reset
                
                start_msg = f"📢 **CHỦ TỌA:** Bắt đầu tranh luận về: *{topic}*"
                st.session_state.weaver_chat.append({"role": "system", "content": start_msg})
                st.chat_message("system").write(start_msg)
                
                # Biến tạm để lưu log cho hàm luu_lich_su
                full_transcript = [start_msg]

                with st.status("Cuộc chiến đang diễn ra (3 vòng)...") as status:
                    for round_num in range(1, 4):
                        status.update(label=f"🔄 Vòng {round_num}/3...")
                        for p_name in participants:
                            # Xây dựng ngữ cảnh từ tin nhắn cuối cùng
                            last_content = st.session_state.weaver_chat[-1]['content'] if st.session_state.weaver_chat else topic
                            
                            p_prompt = f"""
                            VAI TRÒ: {p_name}. 
                            CHỦ ĐỀ GỐC: {topic}.
                            TÌNH HUỐNG: Người trước vừa nói: '{last_content}'.
                            NHIỆM VỤ: Hãy phản biện hoặc bổ sung ý kiến ngắn gọn (dưới 100 từ).
                            """
                            
                            # Gọi AI
                            res = ai.generate(p_prompt, model_type="flash", system_instruction=DEBATE_PERSONAS[p_name])
                            
                            # Lưu và Hiện
                            content_fmt = f"**{p_name}:** {res}"
                            st.session_state.weaver_chat.append({"role": "assistant", "content": content_fmt})
                            full_transcript.append(content_fmt)
                            
                            with st.chat_message("assistant"): 
                                st.write(content_fmt)
                            
                            time.sleep(5) # Nghỉ để tránh Quota
                
                # --- ĐÃ BỔ SUNG: LƯU LỊCH SỬ ---
                luu_lich_su("Hội Đồng Tranh Biện", topic, "\n\n".join(full_transcript))
                st.success("Kết thúc tranh luận! Đã lưu vào Nhật ký.")

    # === TAB 4: PHÒNG THU AI ===
    with tab4:
        st.subheader(T("t4_header"))
        c_in, c_ctrl = st.columns([3, 1])
        with c_in: inp_v = st.text_area("Text:", height=200, key="w_t4_input")
        with c_ctrl:
            try:
                v_choice = st.selectbox(T("t4_voice"), list(voice.VOICE_OPTIONS.keys()), key="w_t4_sel")
            except:
                v_choice = st.selectbox(T("t4_voice"), ["vi", "en"], key="w_t4_sel")
            speed_v = st.slider(T("t4_speed"), -50, 50, 0, key="w_t4_spd")
        
        if st.button(T("t4_btn"), key="w_t4_btn") and inp_v:
            with st.spinner("..."):
                path = voice.speak(inp_v, voice_key=v_choice, speed=speed_v)
                if path:
                    st.audio(path)
                    st.success("OK")

    # === TAB 5: NHẬT KÝ ===
    with tab5:
        st.subheader(T("t5_header"))
        if st.button(T("t5_refresh"), key="w_t5_btn"):
            data = tai_lich_su()
            if data:
                st.dataframe(pd.DataFrame(data))
            else:
                st.info(T("t5_empty"))
