import streamlit as st

# 1. CẤU HÌNH TRANG (Bắt buộc dòng đầu tiên)
st.set_page_config(page_title="Super AI System", layout="wide", page_icon="🏢")

# 2. KHỐI BẢO MẬT (Import Auth Block)
try:
    from auth_block import AuthBlock
    auth = AuthBlock()
except ImportError:
    st.error("❌ Thiếu file 'auth_block.py'. Hãy tạo file này trước!")
    st.stop()

# 3. MÀN HÌNH ĐĂNG NHẬP
if 'user_logged_in' not in st.session_state:
    st.session_state.user_logged_in = False

if not st.session_state.user_logged_in:
    st.title("🔐 Đăng Nhập Hệ Thống")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        pwd = st.text_input("Nhập mật khẩu:", type="password")
        if st.button("Truy cập", use_container_width=True):
            if auth.login(pwd): # Gọi hàm login từ Auth Block
                st.success("Thành công!")
                st.rerun()
            else:
                st.error("Sai mật khẩu!")
    st.stop() # Dừng lại, không chạy phần dưới nếu chưa login

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
