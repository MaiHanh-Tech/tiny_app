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

# ✅ [SỬA] THAY THẾ GSPREAD BẰNG SUPABASE
try:
    from supabase import create_client, Client
except ImportError:
    st.error("⚠️ Thiếu thư viện supabase. Hãy thêm 'supabase' vào requirements.txt")

# --- IMPORT CÁC META-BLOCKS ---
from ai_core import AI_Core
from voice_block import Voice_Engine
from prompts import DEBATE_PERSONAS, BOOK_ANALYSIS_PROMPT

# ==========================================
# ✅ [SỬA] CẤU HÌNH KẾT NỐI SUPABASE
# ==========================================
has_db = False
supabase = None

try:
    # Lấy key từ secrets.toml
    SUPA_URL = st.secrets["supabase"]["url"]
    SUPA_KEY = st.secrets["supabase"]["key"]
    supabase: Client = create_client(SUPA_URL, SUPA_KEY)
    has_db = True
except Exception as e:
    # Nếu chưa cấu hình thì thôi, chỉ tắt tính năng log
    pass

# ==========================================
# 🌍 BỘ TỪ ĐIỂN ĐA NGÔN NGỮ
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

# --- CÁC HÀM PHỤ TRỢ (ĐÃ SỬA THEO YÊU CẦU) ---
@st.cache_resource
def load_models():
    """Chỉ load khi thực sự cần, và giới hạn 1 instance"""
    try:
        model = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2",
            device='cpu'  # ← BẮT BUỘC dùng CPU trên Streamlit Cloud
        )
        # Giảm kích thước cache
        model.max_seq_length = 128  # Giảm từ 256 (default)
        return model
    except Exception as e:
        # st.error(f"Không load được model: {e}")
        return None

# THÊM HÀM KIỂM TRA
def check_model_available():
    """Kiểm tra model có sẵn không trước khi dùng"""
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

# ==========================================
# ✅ [SỬA] CÁC HÀM TƯƠNG TÁC DB (THAY GSPREAD)
# ==========================================

def luu_lich_su(loai, tieu_de, noi_dung):
    """Lưu log vào Supabase"""
    if not has_db: return
    
    user = st.session_state.get("current_user", "Unknown")
    
    # Map dữ liệu vào đúng tên cột trong Supabase (chữ thường)
    data = {
        "type": loai,
        "title": tieu_de,
        "content": noi_dung,
        "user_name": user,
        "sentiment_score": 0.0,
        "sentiment_label": "Neutral"
    }
    
    try:
        # insert vào bảng history_logs
        supabase.table("History_Logs").insert(data).execute()
    except Exception as e:
        print(f"Lỗi lưu log: {e}")

def tai_lich_su():
    """Tải log từ Supabase và chuyển về format cũ cho Frontend"""
    if not has_db: return []
    
    try:
        # Lấy 50 dòng mới nhất
        response = supabase.table("History_Logs").select("*").order("created_at", desc=True).limit(50).execute()
        raw_data = response.data
        
        # ✅ QUAN TRỌNG: Map lại tên cột để khớp với code Frontend cũ của chị
        # Supabase trả về: created_at, type, title...
        # Chị cần: Time, Type, Title...
        formatted_data = []
        for item in raw_data:
            # Xử lý ngày tháng: "2023-10-10T10:00:00" -> "2023-10-10 10:00:00"
            raw_time = item.get("created_at", "")
            clean_time = raw_time.replace("T", " ")[:19]

            formatted_data.append({
                "Time": clean_time,            # Map created_at -> Time
                "Type": item.get("type"),      # Map type -> Type
                "Title": item.get("title"),    # Map title -> Title
                "Content": item.get("content"),# Map content -> Content
                "User": item.get("user_name"), # Map user_name -> User
                "SentimentScore": item.get("sentiment_score", 0.0),
                "SentimentLabel": item.get("sentiment_label", "Neutral")
            })
            
        return formatted_data
    except Exception as e:
        # st.error(f"Lỗi tải lịch sử từ DB: {e}")
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
    
    st.header(f"🧠 The Cognitive Weaver")
    
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
            # ✅ THÊM: Progress Bar & Status
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

            # ✅ THÊM: Dùng enumerate để theo dõi tiến độ
            for file_idx, f in enumerate(uploaded_files):
                # Update status
                status_text.text(f"Đang xử lý file {file_idx+1}/{total_files}: {f.name}")
                progress_bar.progress((file_idx) / total_files)
                
                # Logic xử lý file cũ
                text = doc_file(f)
                link = ""
                if has_db_rag and vec:
                    q = vec.encode([text[:2000]])
                    sc = cosine_similarity(q, db)[0]
                    # Lưu ý: Đổi tên biến idx thành idx_sim để tránh trùng
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
                
                # Update progress sau khi xong 1 file
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
        
        # Khởi tạo history nếu chưa có
        if "weaver_chat" not in st.session_state: 
            st.session_state.weaver_chat = []

        # ========================================
        # MODE 1: SOLO (USER vs AI với MEMORY)
        # ========================================
        if mode == "👤 Solo":
            c1, c2 = st.columns([3, 1])
            
            with c1: 
                persona = st.selectbox(
                    T("t3_persona_label"), 
                    list(DEBATE_PERSONAS.keys()), 
                    key="w_t3_solo_p"
                )
            
            with c2: 
                if st.button(T("t3_clear"), key="w_t3_clr"): 
                    st.session_state.weaver_chat = []
                    st.rerun()

            # Hiển thị lịch sử chat
            for msg in st.session_state.weaver_chat:
                st.chat_message(msg["role"]).write(msg["content"])

            # Input mới
            if prompt := st.chat_input(T("t3_input")):
                # Thêm user message
                st.chat_message("user").write(prompt)
                st.session_state.weaver_chat.append({
                    "role": "user", 
                    "content": prompt
                })
                
                # XÂY DỰNG CONTEXT TỪ LỊCH SỬ
                recent_history = st.session_state.weaver_chat[-10:]
                
                context_text = "\n".join([
                    f"{m['role'].upper()}: {m['content']}" 
                    for m in recent_history
                ])
                
                # Prompt có ngữ cảnh đầy đủ
                full_prompt = f"""
                LỊCH SỬ HỘI THOẠI:
                {context_text}

                NHIỆM VỤ: Dựa vào lịch sử trên, hãy trả lời câu hỏi mới nhất của USER.
                Nếu USER hỏi "câu hỏi cũ" hoặc "vừa rồi", hãy tham chiếu đến lịch sử để trả lời.
                """
                
                with st.chat_message("assistant"):
                    sys_instruction = DEBATE_PERSONAS[persona]
                    
                    with st.spinner("🤔 Đang suy nghĩ..."):
                        # Gọi AI với context đầy đủ
                        res = ai.generate(
                            full_prompt, 
                            model_type="flash", 
                            system_instruction=sys_instruction
                        )
                        
                        if res:
                            st.write(res)
                            
                            # Lưu assistant response
                            st.session_state.weaver_chat.append({
                                "role": "assistant", 
                                "content": res
                            })
                            
                            # LƯU CẢ CÂU HỎI VÀ TRẢ LỜI
                            full_content = f"""
                            👤 USER: {prompt}

                            🤖 {persona}: {res}
                            """
                            
                            luu_lich_su(
                                loai="Tranh Biện Solo",
                                tieu_de=f"{persona} - {prompt[:50]}...",
                                noi_dung=full_content.strip()
                            )
                        else:
                            st.error("⚠️ AI không phản hồi. Vui lòng thử lại.")
        
        # ========================================
        # MODE 2: MULTI-AGENT (AI vs AI) - ĐÃ SỬA THEO YÊU CẦU
        # ========================================
        else:
            st.info("💡 Chọn 2-3 nhân vật để họ tự tranh luận.")
            
            participants = st.multiselect(
                "Chọn Hội Đồng Tranh Biện:", 
                list(DEBATE_PERSONAS.keys()), 
                default=[list(DEBATE_PERSONAS.keys())[0], list(DEBATE_PERSONAS.keys())[1]],
                max_selections=3,
                key="w_t3_multi_p"
            )
            
            topic = st.text_input(
                "Chủ đề tranh luận:", 
                placeholder="VD: Tiền có mua được hạnh phúc không?",
                key="w_t3_topic"
            )
            
            c_start, c_del = st.columns([1, 5])
            
            with c_start:
                start_btn = st.button(
                    "🔥 KHAI CHIẾN", 
                    key="w_t3_start", 
                    disabled=(len(participants) < 2 or not topic),
                    type="primary"
                )
            
            with c_del:
                if st.button("🗑️ Xóa Bàn", key="w_t3_multi_clr"):
                    st.session_state.weaver_chat = []
                    st.rerun()

            # Hiển thị lịch sử cũ
            for msg in st.session_state.weaver_chat:
                role = msg["role"]
                content = msg["content"]
                
                if role == "system":
                    st.info(content)
                else:
                    st.chat_message("assistant").write(content)
            
            # === PHẦN LOGIC ĐƯỢC SỬA ===
            if start_btn and topic and len(participants) >= 2:
                st.session_state.weaver_chat = []
                
                start_msg = f"📢 **CHỦ TỌA:** Khai mạc tranh luận về: *'{topic}'*"
                st.session_state.weaver_chat.append({"role": "system", "content": start_msg})
                st.info(start_msg)
                
                full_transcript = [start_msg]
                
                # ✅ THÊM: Timeout toàn bộ cuộc tranh luận
                MAX_DEBATE_TIME = 90  # 90 giây
                start_time = time.time()
                
                with st.status("🔥 Cuộc chiến đang diễn ra (tối đa 3 vòng)...") as status:
                    try:
                        for round_num in range(1, 4):
                            # ✅ KIỂM TRA TIMEOUT
                            if time.time() - start_time > MAX_DEBATE_TIME:
                                st.warning("⏰ Đã hết thời gian tranh luận (90s). Kết thúc sớm.")
                                break
                            
                            status.update(label=f"🔄 Vòng {round_num}/3 đang diễn ra...")
                            
                            for i, p_name in enumerate(participants):
                                # ✅ KIỂM TRA TIMEOUT CHO TỪNG NGƯỜI
                                if time.time() - start_time > MAX_DEBATE_TIME:
                                    break
                                
                                # Lấy ngữ cảnh (giữ nguyên logic cũ)
                                if len(st.session_state.weaver_chat) > 1:
                                    recent_context = st.session_state.weaver_chat[-3:]
                                    context_str = "\n".join([
                                        f"- {m['content']}" 
                                        for m in recent_context 
                                        if m['role'] != 'system'
                                    ])
                                else:
                                    context_str = topic
                                
                                # Xây dựng prompt (giữ nguyên)
                                if round_num == 1:
                                    p_prompt = f"""
                                    CHỦ ĐỀ TRANH LUẬN: {topic}

                                    NHIỆM VỤ (Vòng 1 - Khai mạc): 
                                    Bạn là {p_name}. Hãy đưa ra quan điểm mở đầu của mình về chủ đề này.
                                    Nêu rõ lập trường và 2-3 lý lẽ chính (dưới 200 từ).
                                    """
                                else:
                                    p_prompt = f"""
                                    CHỦ ĐỀ: {topic}

                                    TÌNH HUỐNG HIỆN TẠI:
                                    {context_str}

                                    NHIỆM VỤ (Vòng {round_num} - Phản biện):
                                    Bạn là {p_name}. Hãy:
                                    1. Chỉ ra điểm yếu trong lập luận của đối thủ
                                    2. Củng cố quan điểm của mình
                                    3. Đưa ra thêm 1 ví dụ minh họa
                                    (Dưới 200 từ, súc tích)
                                    """
                                
                                try:
                                    # ✅ GIẢM THỜI GIAN CHỜ VÀ DÙNG FLASH
                                    res = ai.generate(
                                        p_prompt, 
                                        model_type="flash",  # ← BẮT BUỘC dùng Flash (Pro quá chậm)
                                        system_instruction=DEBATE_PERSONAS[p_name]
                                    )
                                    
                                    if res:
                                        content_fmt = f"**{p_name}:** {res}"
                                        st.session_state.weaver_chat.append({
                                            "role": "assistant", 
                                            "content": content_fmt
                                        })
                                        full_transcript.append(content_fmt)
                                        
                                        with st.chat_message("assistant"):
                                            st.write(content_fmt)
                                        
                                        # ✅ GIẢM SLEEP: 6s → 2s
                                        time.sleep(2)
                                    
                                except Exception as e:
                                    st.error(f"⚠️ Lỗi khi gọi AI cho {p_name}: {str(e)}")
                                    continue  # ← Bỏ qua người này, tiếp tục với người khác
                        
                        status.update(label="✅ Tranh luận kết thúc!", state="complete")
                        
                    except Exception as e:
                        st.error(f"❌ Lỗi nghiêm trọng: {e}")
                        status.update(label="❌ Tranh luận gặp lỗi", state="error")
                
                # Lưu lịch sử
                full_log = "\n\n".join(full_transcript)
                
                luu_lich_su(
                    loai="Hội Đồng Tranh Biện",
                    tieu_de=f"Chủ đề: {topic}",
                    noi_dung=full_log
                )
                
                st.toast("💾 Đã lưu biên bản cuộc họp vào Nhật Ký!", icon="✅")
                
                with st.expander("📄 Xem Toàn Bộ Biên Bản", expanded=False):
                    st.markdown(full_log)

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

    # === TAB 5: NHẬT KÝ & TƯ DUY BAYES ===
    with tab5:
        st.subheader("⏳ Nhật Ký & Phản Chiếu Tư Duy")
        
        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            if st.button("🔄 Tải lại", key="w_t5_refresh"):
                st.session_state.history_cloud = tai_lich_su()
                st.rerun()
        
        # Lấy dữ liệu
        data = st.session_state.get("history_cloud", tai_lich_su())
        
        if data:
            df_h = pd.DataFrame(data)
            
            # --- BIỂU ĐỒ CẢM XÚC (Dùng tên cột cũ: Time, SentimentScore) ---
            if "SentimentScore" in df_h.columns:
                try:
                    df_h["score"] = pd.to_numeric(df_h["SentimentScore"], errors='coerce').fillna(0)
                    
                    st.caption("📉 Biểu đồ dao động trạng thái cảm xúc/tư duy qua thời gian:")
                    fig = px.line(
                        df_h, 
                        x="Time", 
                        y="score", 
                        markers=True, 
                        color_discrete_sequence=["#76FF03"],
                        labels={"score": "Chỉ số Tích cực (Positivity)", "Time": "Thời gian"}
                    )
                    fig.update_layout(height=250, margin=dict(l=20, r=20, t=10, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    # st.warning(f"Không vẽ được biểu đồ: {e}")
                    pass

            # --- PHẦN 2: TƯ DUY BAYES ---
            with st.expander("🔮 Phân tích Tư duy theo xác suất Bayes (E.T. Jaynes)", expanded=False):
                st.info("AI sẽ coi Lịch sử hoạt động của chị là 'Dữ liệu quan sát' (Evidence) để suy luận ra 'Hàm mục tiêu' (Objective Function) và sự dịch chuyển niềm tin của chị.")
                
                if st.button("🧠 Chạy Mô hình Bayes ngay"):
                    with st.spinner("Đang tính toán xác suất hậu nghiệm (Posterior)..."):
                        # Lấy 10 hoạt động gần nhất làm dữ liệu mẫu
                        recent_logs = df_h.tail(10).to_dict(orient="records")
                        logs_text = json.dumps(recent_logs, ensure_ascii=False)
                        
                        bayes_prompt = f"""
                        Đóng vai một nhà khoa học tư duy theo trường phái E.T. Jaynes (sách 'Probability Theory: The Logic of Science').
                        
                        DỮ LIỆU QUAN SÁT (EVIDENCE):
                        Đây là nhật ký hoạt động của tôi:
                        {logs_text}
                        
                        NHIỆM VỤ:
                        Hãy phân tích chuỗi hành động này như một bài toán suy luận Bayes.
                        1. **Xác định Priors (Niềm tin tiên nghiệm):** Dựa trên các hành động đầu, tôi đang quan tâm/tin tưởng điều gì?
                        2. **Cập nhật Likelihood (Khả năng):** Các hành động tiếp theo củng cố hay làm yếu đi niềm tin đó?
                        3. **Kết luận Posterior (Hậu nghiệm):** Trạng thái tư duy hiện tại của tôi đang hội tụ về đâu? Có mâu thuẫn (Inconsistency) nào trong logic hành động không?
                        
                        Trả lời ngắn gọn, sâu sắc, dùng thuật ngữ xác suất nhưng dễ hiểu.
                        """
                        
                        # Gọi AI Core (Dùng Pro để suy luận sâu)
                        analysis = ai.generate(bayes_prompt, model_type="pro")
                        st.markdown(analysis)

            # --- PHẦN 3: DANH SÁCH CHI TIẾT ---
            st.divider()
            st.write("📜 **Chi tiết Nhật ký:**")
            
            # Đảo ngược để xem mới nhất trước
            for index, item in df_h.iloc[::-1].iterrows():
                # Lấy dữ liệu theo tên cột cũ
                time_str = str(item.get('Time', ''))
                type_str = str(item.get('Type', ''))
                title_str = str(item.get('Title', ''))
                content_str = str(item.get('Content', ''))
                
                icon = "📝"
                if "Tranh Biện" in type_str: icon = "🗣️"
                elif "Dịch" in type_str: icon = "✍️"
                elif "Audio" in type_str: icon = "🎙️"
                
                with st.expander(f"{icon} {time_str} | {type_str} | {title_str}"):
                    st.markdown(content_str)
                    st.caption(f"Sentiment: {item.get('SentimentLabel', 'Neutral')} ({item.get('SentimentScore', 0)})")
        else:
            st.info(T("t5_empty"))

 # --- 👇 DÁN ĐÈ ĐOẠN NÀY VÀO CUỐI CÙNG TAB 5 (THAY CHO ĐOẠN CŨ) ---
        st.divider()
        with st.expander("🛠️ CÔNG CỤ CHUYỂN NHÀ (V3 - Fix lỗi Dấu phẩy & Tên bảng)", expanded=True):
            st.info("Phiên bản V3: Đã xử lý số liệu Việt Nam (0,95 -> 0.95) và Tên bảng chữ Hoa.")
            
            uploaded_csv = st.file_uploader("1. Tải file CSV từ Google Sheet lên đây:", type=["csv"])
            
            if uploaded_csv:
                # Đọc file CSV
                df_old = pd.read_csv(uploaded_csv)
                # Xóa khoảng trắng thừa trong tên cột
                df_old.columns = df_old.columns.str.strip()
                
                st.write(f"Đã tìm thấy {len(df_old)} dòng nhật ký cũ.")
                
                if st.button("🚀 BẮT ĐẦU CHUYỂN DỮ LIỆU"):
                    progress_bar = st.progress(0)
                    success_count = 0
                    error_count = 0
                    errors_log = [] 
                    
                    for idx, row in df_old.iterrows():
                        try:
                            # 1. XỬ LÝ NGÀY THÁNG
                            raw_time = str(row.get('Time', '')).strip()
                            clean_time = datetime.now().isoformat()
                            if raw_time and raw_time.lower() != 'nan':
                                try:
                                    clean_time = pd.to_datetime(raw_time).strftime('%Y-%m-%d %H:%M:%S')
                                except: pass

                            # 2. XỬ LÝ SỐ LIỆU (FIX LỖI 0,95)
                            raw_score = str(row.get('SentimentScore', '0'))
                            # 👉 Thay dấu phẩy thành dấu chấm ngay lập tức
                            clean_score = raw_score.replace(',', '.')
                            try:
                                final_score = float(clean_score)
                            except:
                                final_score = 0.0

                            data = {
                                "created_at": clean_time,
                                "type": str(row.get('Type', 'General')),
                                "title": str(row.get('Title', 'No Title')),
                                "content": str(row.get('Content', '')),
                                "user_name": str(row.get('User', 'Imported')),
                                "sentiment_score": final_score, # ✅ Đã sạch
                                "sentiment_label": str(row.get('SentimentLabel', 'Neutral'))
                            }
                            
                            # 3. GỬI LÊN SUPABASE (Dùng History_Logs chữ Hoa như lỗi gợi ý)
                            try:
                                supabase.table("History_Logs").insert(data).execute()
                            except:
                                # Nếu History_Logs lỗi thì thử lại history_logs (phòng hờ)
                                supabase.table("history_logs").insert(data).execute()
                                
                            success_count += 1
                            
                        except Exception as e:
                            error_count += 1
                            errors_log.append(f"Dòng {idx}: {str(e)}")
                        
                        progress_bar.progress((idx + 1) / len(df_old))
                    
                    st.success(f"✅ Đã chuyển thành công: {success_count} dòng.")
                    
                    if error_count > 0:
                        st.error(f"⚠️ Có {error_count} dòng bị lỗi.")
                        with st.expander("Xem chi tiết lỗi"):
                            for err in errors_log:
                                st.write(err)
                    else:
                        st.balloons()
                        time.sleep(1)
                        st.rerun()
