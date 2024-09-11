import streamlit as st
import pandas as pd

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
# Function to authenticate user
def authenticate_user(email):
    # Load the Excel file
    df = pd.read_excel('user.xlsx')
    # Convert the input email to lowercase
    email = email.lower()
    # Convert the emails in the dataframe to lowercase
    df['Email'] = df['Email'].str.lower()
    # Check if the email matches any entry in the file
    user = df[df['Email'] == email]
    if not user.empty:
        return True
    return False

# Login page
def create_ui():
    if not st.session_state.authenticated:
        st.markdown("<h3 style='color: #4682B4;'>Login</h3>", unsafe_allow_html=True)
        with st.form(key='login_form'):
            email = st.text_input("Email")
            login_button = st.form_submit_button(label='Login')
            if login_button:
                if authenticate_user(email):
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid email or password. Please try again.")
        return

# Main app
def main_app():
    # Importing main.py functionality here
    import app


if st.session_state.authenticated:
    main_app()
else:
    create_ui()
