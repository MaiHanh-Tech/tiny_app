import streamlit as st
import json
import re

# 1. CẤU HÌNH TRANG (Bắt buộc dòng đầu tiên)
st.set_page_config(page_title="Super AI System", layout="wide", page_icon="🏢")

# 2. KHỐI BẢO MẬT (Import Auth Block)
try:
    from auth_block import AuthBlock
    auth = AuthBlock()
except ImportError:
    st.error("❌ Thiếu file 'auth_block.py'. Hãy tạo file này trước!")
    st.stop()

# Trong phần sidebar (sau dòng 15), thêm:

with st.sidebar:
    st.title("🗂️ DANH MỤC ỨNG DỤNG")
    st.info(f"👤 Xin chào: **{st.session_state.current_user}**")
    
    app_choice = st.radio("Chọn công việc:", [
        "💰 1. Cognitive Weaver (Sách & Graph)", 
        "🌏 2. AI Translator (Dịch thuật)",
        "🧠 3. CFO Controller (Tài chính)",
        "🔐 4. Hash Generator (Admin)"  # ← THÊM DÒNG NÀY
    ])
    
    st.divider()
    if st.button("Đăng Xuất"):
        st.session_state.user_logged_in = False
        st.rerun()

# Trong phần điều hướng (sau dòng 38), thêm:

try:
    if app_choice == "💰 1. Cognitive Weaver (Sách & Graph)":
        import module_weaver
        module_weaver.run()
         
    elif app_choice == "🌏 2. AI Translator (Dịch thuật)":
        import module_translator
        module_translator.run()
        
    elif app_choice == "🧠 3. CFO Controller (Tài chính)":
        import module_cfo
        module_cfo.run()
    
    # ← THÊM ĐOẠN NÀY
    elif app_choice == "🔐 4. Hash Generator (Admin)":
        import hash_generator
        hash_generator.run()
        
# 3. MÀN HÌNH ĐĂNG NHẬP
if 'user_logged_in' not in st.session_state:
    st.session_state.user_logged_in = False

if not st.session_state.user_logged_in:
    st.title("🔐 Đăng Nhập Hệ Thống")
    
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        # ✅ KHÔNG HIỂN THỊ PASSWORD!
        pwd = st.text_input(
            "Nhập mật khẩu:", 
            type="password",
            placeholder="Nhập mật khẩu của bạn",
            help="Liên hệ admin nếu quên mật khẩu"
        )
        
        if st.button("Truy cập", use_container_width=True):
            if auth.login(pwd):
                st.success("✅ Đăng nhập thành công!")
                st.rerun()
            else:
                st.error("❌ Sai mật khẩu!")
                
                # Hiển thị số lần thử còn lại
                attempts = st.session_state.get('login_attempts', {}).get('global', [])
                remaining = 5 - len(attempts)
                if remaining > 0:
                    st.warning(f"⚠️ Còn {remaining} lần thử")
    
    st.stop()  # Dừng lại, không chạy phần dưới nếu chưa login


# 4. GIAO DIỆN CHÍNH (SAU KHI LOGIN)
with st.sidebar:
    st.title("🗂️ DANH MỤC ỨNG DỤNG")
    st.info(f"👤 Xin chào: **{st.session_state.current_user}**")
    
    # Menu chọn App
    app_choice = st.radio("Chọn công việc:", [
        "💰 1. Cognitive Weaver (Sách & Graph)", 
        "🌏 2. AI Translator (Dịch thuật)",
        "🧠 3. CFO Controller (Tài chính)"
    ])
    
    st.divider()
    if st.button("Đăng Xuất"):
        st.session_state.user_logged_in = False
        st.rerun()

# 5. ĐIỀU HƯỚNG (GỌI CÁC FILE CON)
try:
    if app_choice == "💰 1. Cognitive Weaver (Sách & Graph)":
        import module_weaver
        module_weaver.run()
         
        
    elif app_choice == "🌏 2. AI Translator (Dịch thuật)":
        import module_translator
        module_translator.run()
        
    elif app_choice == "🧠 3. CFO Controller (Tài chính)":
        import module_cfo
        module_cfo.run()
        
except ImportError as e:
    st.error(f"⚠️ Lỗi: Không tìm thấy file module tương ứng!\nChi tiết: {e}")
    st.info("👉 Hãy đảm bảo chị đã đổi tên các file cũ thành: module_cfo.py, module_translator.py, module_weaver.py")
