import streamlit as st
import json
import os
import hashlib

USER_DB = "users.json"

def load_users():
    if not os.path.exists(USER_DB):
        with open(USER_DB, "w") as f:
            json.dump({}, f)
        return {}
    try:
        with open(USER_DB, "r") as f:
            data = f.read().strip()
            return json.loads(data) if data else {}
    except (json.JSONDecodeError, ValueError):
        return {}

def save_users(users):
    with open(USER_DB, "w") as f:
        json.dump(users, f, indent=4)

def hash_data(data):
    """Hashes passwords to create a secure key."""
    return hashlib.sha256(data.encode()).hexdigest()

def get_user_key(username):
    """
    Retrieves the password hash of a specific user. 
    This acts as the 'Public Key' for encryption.
    """
    users = load_users()
    if username in users:
        return users[username]["password"]
    return None

def show_login_page():
    st.markdown("## 🔐 Account Access")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    users = load_users()

    # --- LOGIN TAB ---
    with tab1:
        st.subheader("Welcome Back")
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        
        if st.button("Log In", use_container_width=True):
            if username in users:
                if users[username]["password"] == hash_data(password):
                    st.session_state["authenticated"] = True
                    st.session_state["user"] = username
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error("Incorrect password.")
            else:
                st.error("User not found.")

    # --- REGISTER TAB ---
    with tab2:
        st.subheader("Create New Account")
        new_user = st.text_input("Choose Username", key="reg_user")
        new_pass = st.text_input("Choose Password", type="password", key="reg_pass")
        confirm_pass = st.text_input("Confirm Password", type="password", key="reg_confirm")

        if st.button("Register", use_container_width=True):
            if not new_user or not new_pass:
                st.warning("Please fill in all fields.")
            elif new_user in users:
                st.error("Username already exists.")
            elif new_pass != confirm_pass:
                st.error("Passwords do not match.")
            else:
                # Removed PIN storage
                users[new_user] = {
                    "password": hash_data(new_pass)
                }
                save_users(users)
                st.success("Account created! You can now log in.")