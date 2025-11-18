import streamlit as st

# Hardcoded user credentials (can be replaced with DB later)
USERS = {
    "admin": "admin123",
    "manager": "manager123",
    "analyst": "analyst123"
}

def login_page():
    st.title("🔐 Secure Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in USERS and USERS[username] == password:
            st.session_state["authenticated"] = True
            st.session_state["user_role"] = username
            st.session_state["username"] = username
            st.success("Login Successful!")
        else:
            st.error("Invalid Username or Password")
