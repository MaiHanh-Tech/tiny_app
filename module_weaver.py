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
from datetime import datetime
import json
import re

# ✅ THAY ĐỔI 1: IMPORT SUPABASE (Bỏ gspread, oauth2client)
try:
    from supabase import create_client, Client
except ImportError:
    st.error("⚠️ Thiếu thư viện supabase. Hãy thêm 'supabase' vào requirements.txt")

# --- IMPORT CÁC META-BLOCKS ---
from ai_core import AI_Core
from voice_block import Voice_Engine
from prompts import DEBATE_PERSONAS, BOOK_ANALYSIS_PROMPT

# ==========================================
# 🌍 KẾT NỐI SUPABASE (Thay thế Google Sheet)
# ==========================================
has_db = False
supabase = None

try:
    # Lấy thông tin từ secrets.toml
    SUPA_URL = st.secrets["supabase"]["url"]
    SUPA_KEY = st.secrets["supabase"]["key"]
    supabase: Client = create_client(SUPA_URL, SUPA_KEY)
    has_db = True
except Exception:
    # Nếu chưa cấu hình thì thôi, không báo lỗi đỏ
    has_db = False

# ==========================================
# 🌍 BỘ TỪ ĐIỂN ĐA NGÔN NGỮ (GIỮ NGUYÊN)
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
        "t1_connect_ok": "✅ Đã kết nối {n} cuốn sách.",
        "t1_graph_title": "🪐 Vũ Trụ Sách",
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
        "t1_connect_ok": "✅ Connected {n} books.",
        "t1_graph_title": "🪐 Book Universe",
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
        "t1_connect_ok": "✅ 已连接 {n} 本书。",
        "t1_graph_title": "🪐 书籍宇宙",
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

# --- CÁC HÀM PHỤ TRỢ (GIỮ NGUYÊN) ---
@st.cache_resource
def load_models():
    """Chỉ load khi thực sự cần, và giới hạn 1 instance"""
    try:
        model = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2",
            device='cpu'  # ← BẮT BUỘC dùng CPU trên Streamlit Cloud
        )
        model.max_seq_length = 128
        return model
    except Exception as e:
        return None

def check_model_available():
    model = load_models()
    if model is None:
        st.warning("⚠️ Chức năng Knowledge Graph tạm thời không khả dụng (thiếu RAM)")
        return False
    return True

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

# ✅ THAY ĐỔI 2: HÀM LƯU/TẢI LOG DÙNG SUPABASE
# Hàm này tự động map dữ liệu Supabase về format cũ (Time, Title...) để giao diện không bị lỗi

def luu_lich_su(loai, tieu_de, noi_dung):
    """Lưu log vào Supabase (Bảng history_logs)"""
    if not has_db: return
    
    user = st.session_state.get("current_user", "Unknown")
    
    # Data chuẩn theo cột trong Supabase (chữ thường)
    data = {
        "type": loai,
        "title": tieu_de,
        "content": noi_dung,
        "user_name": user,
        "sentiment_score": 0.0,
        "sentiment_label": "Neutral"
    }
    
    try:
        supabase.table("history_logs").insert(data).execute()
    except Exception as e:
        print(f"Lỗi lưu log: {e}")

def tai_lich_su():
    """Tải log từ Supabase và đổi tên cột cho khớp code cũ"""
    if not has_db: return []
    
    try:
        # Lấy 50 dòng mới nhất
        response = supabase.table("history_logs").select("*").order("created_at", desc=True).limit(50).execute()
        raw_data = response.data
        
        # ✅ CHUYỂN ĐỔI FORMAT (Mapping)
        formatted_data = []
        for item in raw_data:
            # Xử lý thời gian cho đẹp (bỏ chữ T và phần mili giây)
            t = item.get("created_at", "").replace("T", " ")[:19]
            
            formatted_data.append({
                "Time": t,                          # Map created_at -> Time
                "Type": item.get("type"),           # Map type -> Type
                "Title": item.get("title"),         # Map title -> Title
                "Content": item.get("content"),     # Map content -> Content
                "User": item.get("user_name"),      # Map user_name -> User
                "SentimentScore": item.get("sentiment_score", 0.0),
                "SentimentLabel": item.get("sentiment_label", "Neutral")
            })
            
        return formatted_data
    except Exception as e:
        return []

# --- HÀM CHÍNH: RUN() (GIỮ NGUYÊN) ---
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
        if lang_choice == "Tiếng Việt": st.session_state.weaver_lang = 'vi'
        elif lang_choice == "English": st.session_state.weaver_lang = 'en'
        elif lang_choice == "中文": st.session_state.weaver_lang = 'zh'
    
    st.header(f"🧠 The Cognitive Weaver")
    
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
            total_files = len(uploaded_files)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            vec = load_models()
            db, df = None, None
            has_db_rag = False
            
            if file_excel:
                try:
                    df = pd.read_excel(file_excel).dropna(subset=["Tên sách"])
                    db = vec.encode([f"{r['Tên sách']} {str(r.get('CẢM NHẬN',''))}" for _, r in df.iterrows()])
                    has_db_rag = True
                    st.success(T("t1_connect_ok").format(n=len(df)))
                except: st.error("Lỗi đọc Excel.")

            for file_idx, f in enumerate(uploaded_files):
                status_text.text(f"Đang xử lý file {file_idx+1}/{total_files}: {f.name}")
                progress_bar.progress((file_idx) / total_files)
                
                text = doc_file(f)
                link = ""
                if has_db_rag:
                    q = vec.encode([text[:2000]])
                    sc = cosine_similarity(q, db)[0]
                    idx_sim = np.argsort(sc)[::-1][:3]
                    for i in idx_sim:
                        if sc[i] > 0.35: link += f"- {df.iloc[i]['Tên sách']} ({sc[i]*100:.0f}%)\n"

                with st.spinner(T("t1_analyzing").format(name=f.name)):
                    prompt = f"Phân tích tài liệu '{f.name}'. Liên quan: {link}\nNội dung: {text[:30000]}"
                    res = ai.analyze_static(prompt, BOOK_ANALYSIS_PROMPT)
                    
                    st.markdown(f"### 📄 {f.name}")
                    st.markdown(res)
                    st.markdown("---")
                    luu_lich_su("Phân Tích Sách", f.name, res[:200])
                
                progress_bar.progress((file_idx+1) / total_files)
            
            status_text.text("✅ Hoàn thành!")

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

    # === TAB 3: ĐẤU TRƯỜNG TƯ DUY ===
    with tab3:
        st.subheader(T("t3_header"))
        mode = st.radio("Mode:", ["👤 Solo", "⚔️ Multi-Agent"], horizontal=True, key="w_t3_mode")
        
        if "weaver_chat" not in st.session_state: 
            st.session_state.weaver_chat = []

        if mode == "👤 Solo":
            c1, c2 = st.columns([3, 1])
            with c1: 
                persona = st.selectbox(T("t3_persona_label"), list(DEBATE_PERSONAS.keys()), key="w_t3_solo_p")
            with c2: 
                if st.button(T("t3_clear"), key="w_t3_clr"): st.session_state.weaver_chat = []; st.rerun()

            for msg in st.session_state.weaver_chat: st.chat_message(msg["role"]).write(msg["content"])

            if prompt := st.chat_input(T("t3_input")):
                st.chat_message("user").write(prompt)
                st.session_state.weaver_chat.append({"role": "user", "content": prompt})
                recent_history = st.session_state.weaver_chat[-10:]
                context_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in recent_history])
                full_prompt = f"LỊCH SỬ:\n{context_text}\n\nNHIỆM VỤ: Trả lời câu hỏi mới nhất của USER."
                
                with st.chat_message("assistant"):
                    with st.spinner("🤔..."):
                        res = ai.generate(full_prompt, model_type="flash", system_instruction=DEBATE_PERSONAS[persona])
                        if res:
                            st.write(res)
                            st.session_state.weaver_chat.append({"role": "assistant", "content": res})
                            luu_lich_su("Tranh Biện Solo", f"{persona} - {prompt[:50]}...", f"Q: {prompt}\nA: {res}")
                        else: st.error("⚠️ AI Error.")
        else:
            # Multi-Agent
            st.info("💡 Chọn 2-3 nhân vật.")
            participants = st.multiselect("Chọn Hội Đồng:", list(DEBATE_PERSONAS.keys()), default=[list(DEBATE_PERSONAS.keys())[0], list(DEBATE_PERSONAS.keys())[1]], max_selections=3)
            topic = st.text_input("Chủ đề:", key="w_t3_topic")
            
            if st.button("🔥 KHAI CHIẾN", disabled=(len(participants)<2 or not topic)):
                st.session_state.weaver_chat = []
                start_msg = f"📢 **CHỦ TỌA:** Khai mạc tranh luận về: *'{topic}'*"
                st.session_state.weaver_chat.append({"role": "system", "content": start_msg})
                st.info(start_msg)
                full_transcript = [start_msg]
                MAX_DEBATE_TIME = 90; start_time = time.time()
                
                with st.status("🔥 Đang diễn ra...") as status:
                    for round_num in range(1, 4):
                        if time.time() - start_time > MAX_DEBATE_TIME: break
                        status.update(label=f"🔄 Vòng {round_num}...")
                        for p_name in participants:
                            if time.time() - start_time > MAX_DEBATE_TIME: break
                            context_str = topic if len(st.session_state.weaver_chat) <= 1 else "\n".join([f"- {m['content']}" for m in st.session_state.weaver_chat[-3:] if m['role'] != 'system'])
                            p_prompt = f"CHỦ ĐỀ: {topic}\nBỐI CẢNH:\n{context_str}\n\nNHIỆM VỤ (Vòng {round_num}): Phản biện."
                            try:
                                res = ai.generate(p_prompt, model_type="flash", system_instruction=DEBATE_PERSONAS[p_name])
                                if res:
                                    fmt = f"**{p_name}:** {res}"
                                    st.session_state.weaver_chat.append({"role": "assistant", "content": fmt})
                                    full_transcript.append(fmt)
                                    st.chat_message("assistant").write(fmt)
                                    time.sleep(2)
                            except: continue
                    status.update(label="✅ Kết thúc!", state="complete")
                luu_lich_su("Hội Đồng Tranh Biện", topic, "\n".join(full_transcript))

    # === TAB 4: PHÒNG THU AI ===
    with tab4:
        st.subheader(T("t4_header"))
        inp_v = st.text_area("Text:", height=200, key="w_t4_input")
        btn_v = st.button(T("t4_btn"), key="w_t4_btn")
        if btn_v and inp_v:
            path = voice.speak(inp_v)
            if path: st.audio(path)

    # === TAB 5: NHẬT KÝ (DATA TỪ SUPABASE) ===
    with tab5:
        st.subheader("⏳ Nhật Ký & Phản Chiếu Tư Duy")
        if st.button("🔄 Tải lại", key="w_t5_refresh"):
            st.session_state.history_cloud = tai_lich_su()
            st.rerun()
        
        # Lấy dữ liệu (đã được hàm tai_lich_su chuyển đổi về format cũ)
        data = st.session_state.get("history_cloud", tai_lich_su())
        
        if data:
            df_h = pd.DataFrame(data)
            
            # --- BIỂU ĐỒ (Dùng tên cột cũ: Time, SentimentScore...) ---
            if "SentimentScore" in df_h.columns:
                try:
                    df_h["score"] = pd.to_numeric(df_h["SentimentScore"], errors='coerce').fillna(0)
                    fig = px.line(df_h, x="Time", y="score", markers=True, color_discrete_sequence=["#76FF03"])
                    st.plotly_chart(fig, use_container_width=True)
                except: pass

            st.divider()
            for index, item in df_h.iloc[::-1].iterrows(): # Đảo ngược để xem mới nhất
                # Dùng tên cột cũ để hiển thị
                t = str(item.get('Time', ''))
                tp = str(item.get('Type', ''))
                ti = str(item.get('Title', ''))
                ct = str(item.get('Content', ''))
                
                icon = "📝"
                if "Tranh Biện" in tp: icon = "🗣️"
                elif "Dịch" in tp: icon = "✍️"
                
                with st.expander(f"{icon} {t} | {tp} | {ti}"):
                    st.markdown(ct)
        else:
            st.info(T("t5_empty"))
