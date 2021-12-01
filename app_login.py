import streamlit as st
from about import intro_page
from Automated_Tuning_of_Money_Loundering_System import Automated_Tuning_page

# Security
#passlib,hashlib,bcrypt,scrypt
import hashlib
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False

# DB Management
import sqlite3 
conn = sqlite3.connect('data.db', check_same_thread=False)
c = conn.cursor()

# DB  Functions
def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')

def add_userdata(username,password):
    c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
    conn.commit()

def login_user(username,password):
    c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
    data = c.fetchall()
    return data

def view_all_users():
    c.execute('SELECT * FROM userstable')
    data = c.fetchall()
    return data

# Main login page where other functions are called
def login_page():
    """ Login page """

    menu = ["Home","Login","SignUp"]
    choice = st.sidebar.selectbox("Menu",menu)

    # Home tab
    if choice == "Home":
        st.sidebar.subheader("Home")
    
    # Login tab
    elif choice == "Login":
        st.sidebar.subheader("Login Section")

        username = st.sidebar.text_input("User Name")
        password = st.sidebar.text_input("Password",type='password')
        if st.sidebar.checkbox("Login"):
            # if password == '12345':
            create_usertable()
            hashed_pswd = make_hashes(password)

            result = login_user(username,check_hashes(password,hashed_pswd))
            if result:
                st.sidebar.success("Logged In as {}".format(username))
                task = st.sidebar.selectbox("DashBoard",["About","Automated Tuning","Contact"])
                if task == "About":
                    st.info("About")
                    intro_page()
                elif task == "Automated Tuning":
                    st.info("Automated Tuning")
                    Automated_Tuning_page()
                elif task == "Contact":
                    st.info('Contact')
                    st.markdown("Let us know, if you need any help")
    
            else:
                st.warning("Incorrect Username/Password")

    # Sign-Up tab
    elif choice == "SignUp":
        st.subheader("Create New Account")
        new_user = st.sidebar.text_input("Username")
        new_password = st.sidebar.text_input("Password",type='password')

        if st.sidebar.button("Signup"):
            create_usertable()
            add_userdata(new_user,make_hashes(new_password))
            st.sidebar.success("You have successfully created a valid Account")
            st.info("Go to Login Menu to login")
