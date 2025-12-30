import streamlit as st
import pandas as pd 
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
except Exception as e:
    st.error(f"❌ Lỗi khởi tạo Auth: {e}")
    st.stop()

# 3. MÀN HÌNH ĐĂNG NHẬP
if 'user_logged_in' not in st.session_state:
    st.session_state.user_logged_in = False

if not st.session_state.user_logged_in:
    st.title("🔐 Đăng Nhập Hệ Thống")
    
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
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
    
    st.stop()

# 4. GIAO DIỆN CHÍNH (SAU KHI LOGIN)
with st.sidebar:
    st.title("🗂️ DANH MỤC ỨNG DỤNG")
    st.info(f"👤 Xin chào: **{st.session_state.current_user}**")
    
    app_choice = st.radio("Chọn công việc:", [
        "💰 1. Cognitive Weaver (Sách & Graph)", 
        "🌏 2. AI Translator (Dịch thuật)",
        "🧠 3. CFO Controller (Tài chính)"
    ])
    
    st.divider()
    if st.button("Đăng Xuất"):
        st.session_state.user_logged_in = False
        st.rerun()

    # ✅ GIAO DIỆN QUẢN TRỊ (CHỈ HIỆN VỚI ADMIN)
    # Code này chỉ chạy OK nếu chị đã đổi file auth_block.py sang bản Supabase
    if st.session_state.get("is_admin"):
        st.divider()
        st.write("👑 **Admin Panel**")
        
        with st.expander("Quản lý Người dùng"):
            # 1. Danh sách User
            all_users = auth.get_all_users()
            if all_users:
                df_users = pd.DataFrame(all_users)
                # Ẩn cột mật khẩu đi cho bảo mật, chỉ hiện các cột cần thiết
                # Lưu ý: Cần đảm bảo các cột này có trong DB Supabase
                display_cols = [col for col in ['username', 'role', 'is_active', 'created_at'] if col in df_users.columns]
                st.dataframe(df_users[display_cols], hide_index=True)
            
            # 2. Tạo User Mới
            st.write("➕ **Thêm User mới**")
            new_u = st.text_input("Username:")
            new_p = st.text_input("Password:", type="password")
            new_role = st.selectbox("Role:", ["user", "admin"])
            
            if st.button("Tạo User"):
                if new_u and new_p:
                    ok, msg = auth.create_user(new_u, new_p, new_role)
                    if ok: st.success(msg); st.rerun()
                    else: st.error(msg)
            
            # 3. Xóa User
            st.write("❌ **Xóa User**")
            # Lấy danh sách username để chọn xóa
            user_list = [u['username'] for u in all_users] if all_users else []
            del_u = st.selectbox("Chọn User xóa:", user_list)
            
            if st.button("Xóa"):
                if del_u:
                    # Không cho phép tự xóa chính mình (nếu đang là admin)
                    if del_u == st.session_state.current_user:
                        st.error("Không thể tự xóa tài khoản đang đăng nhập!")
                    else:
                        ok, msg = auth.delete_user(del_u)
                        if ok: st.success(msg); st.rerun()
                        else: st.error(msg)

# --- HÀM AN TOÀN (ERROR BOUNDARY) ---
def safe_run_module(module_func, module_name):
    """Wrapper an toàn cho module"""
    try:
        module_func()
    except Exception as e:
        st.error(f"❌ Module {module_name} gặp lỗi:")
        st.exception(e)
        st.info("💡 Hãy reload trang hoặc chọn module khác")

# 5. ĐIỀU HƯỚNG (GỌI CÁC FILE CON)
try:
    if app_choice == "💰 1. Cognitive Weaver (Sách & Graph)":
        import module_weaver
        # ✅ Dùng wrapper an toàn
        safe_run_module(module_weaver.run, "Cognitive Weaver")
         
    elif app_choice == "🌏 2. AI Translator (Dịch thuật)":
        import module_translator
        # ✅ Dùng wrapper an toàn
        safe_run_module(module_translator.run, "AI Translator")
        
    elif app_choice == "🧠 3. CFO Controller (Tài chính)":
        import module_cfo
        # ✅ Dùng wrapper an toàn
        safe_run_module(module_cfo.run, "CFO Controller")
        
except ImportError as e:
    st.error(f"⚠️ Lỗi: Không tìm thấy file module tương ứng!\nChi tiết: {e}")
    st.info("👉 Hãy đảm bảo đã có các file: module_cfo.py, module_translator.py, module_weaver.py")
except Exception as e:
    st.error(f"❌ Lỗi nghiêm trọng: {e}")
    st.exception(e)
