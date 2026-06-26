import streamlit as st
import base64
import os

# Import modules
import login
import lsb_arn
import dwt_svd
import dct_dwt
import attack_streamlit

# Page configuration
st.set_page_config(page_title="Watermarking App", layout="centered")

# --- Helpers ---
def set_background(image_file):
    if os.path.exists(image_file):
        with open(image_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        page_bg = f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """
        st.markdown(page_bg, unsafe_allow_html=True)

def card(title, subtitle, icon, color_start, color_end, key_name, value):
    selected = st.session_state.get("algorithm") == value
    
    if selected:
        bg_style = f"background: linear-gradient(135deg, {color_start}, {color_end}); box-shadow: 0 0 15px {color_start}; transform: scale(1.02);"
    else:
        bg_style = "background: rgba(40, 40, 40, 0.7); border: 1px solid #444;"

    st.markdown(
        f"""
        <div style='
            text-align: center;
            padding: 30px 20px;
            border-radius: 15px;
            color: white;
            transition: all 0.3s ease;
            cursor: pointer;
            margin-bottom: 20px;
            {bg_style}
        '>
            <div style='font-size: 40px; margin-bottom: 10px;'>{icon}</div>
            <h3 style='margin: 0; font-size: 20px;'>{title}</h3>
            <p style='margin: 5px 0 0 0; font-size: 12px; opacity: 0.8;'>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if st.button(f"Select {title}", key=f"btn_{value}", use_container_width=True):
        st.session_state[key_name] = value
        st.rerun()

# --- Main App Flow ---

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "algorithm" not in st.session_state:
    st.session_state["algorithm"] = None

if os.path.exists("background.png"):
    set_background("background.png")

# 1. Check Authentication
if not st.session_state["authenticated"]:
    login.show_login_page()

# 2. Show Main Menu
else:
    # Retrieve current user from session state
    current_user = st.session_state.get("user", "User")
    
    col1, col2 = st.columns([1, 5])
    with col1:
        st.markdown("<h1>👤</h1>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<h2 style='color: white; margin-top: 10px;'>Hello, {current_user}</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color: #bbb;'>Secure your images with our watermarking tools.</p>", unsafe_allow_html=True)
    
    st.divider()

    if st.session_state["algorithm"] is None:
        st.subheader("🤖 Select Algorithm")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            card("LSB + Arnold", "High Capacity", "🔏", "#00C853", "#1B5E20", "algorithm", "lsb_arn")
        with c2:
            card("Hybrid DWT-SVD", "Copyright Protection", "🛡️", "#2962FF", "#002171", "algorithm", "dwt_svd")
        with c3:
            card("DCT-DWT + Chaos", "Social Sharing", "📉", "#FF6D00", "#E65100", "algorithm", "dct_dwt")
        with c4:
            card("Attack & Measure", "Test Robustness", "🔒", "#9C27B0", "#4A148C", "algorithm", "attack")

    else:
        # Navigation Header
        col_back, col_title = st.columns([1, 5])
        with col_back:
            if st.button("⬅ Menu"):
                st.session_state["algorithm"] = None
                st.rerun()
        with col_title:
            algo_map = {
                "lsb_arn": "Adaptive LSB + Arnold Transform",
                "dwt_svd": "Hybrid DWT-SVD",
                "dct_dwt": "DCT-DWT + Chaotic Map",
                "attack": "Attack & Quality Measurement"
            }
            title = algo_map.get(st.session_state["algorithm"], "Unknown Algorithm")
            st.markdown(f"## {title}")

        st.divider()

        # Route to the specific python file
        # We pass current_user to run() to fix your specific error, 
        # but we also allow run() to handle it if it's None.
        selected = st.session_state["algorithm"]
        
        if selected == "lsb_arn":
            lsb_arn.run(current_user)
        elif selected == "dwt_svd":
            dwt_svd.run(current_user)
        elif selected == "dct_dwt":
            dct_dwt.run(current_user)
        elif selected == "attack":
            attack_streamlit.run()

    st.markdown("---")
    if st.button("🚪 Logout", key="logout"):
        st.session_state.clear()
        st.session_state["authenticated"] = False
        st.rerun()