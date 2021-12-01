import streamlit as st
import streamlit.components.v1 as components
import streamlit as st
from about import intro_page
from Automated_Tuning_of_Money_Loundering_System import Automated_Tuning_page

def main():
    # Set the configuration of the page
    st.set_page_config(page_title='Atom UNO',
                       page_icon="🧊", 
                       layout='wide',
                       initial_sidebar_state="expanded"
                      )
    
    
    st.sidebar.image('images/tt.png', width=300)
    col1, col2, col3 = st.columns([6,1,1])

    with col1:
        st.title("**Atom UNO**")

    with col2:
        st.write("")
    
    with col3:
        st.image('images/inf.png', width=75)

    st.markdown("*Automated Tuning of Money Loundering System*", unsafe_allow_html=True)
    st.sidebar.image('images/bul.png', width=300)
    
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
if __name__ == '__main__':
    main()
