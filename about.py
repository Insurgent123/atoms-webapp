import streamlit as st

def intro_page():
    """ Introduction Page """
    #st.markdown("""---""")
    st.subheader('*Legal Notices*')
    st.markdown("""---""")
    st.subheader('Trademark Notice')
    st.markdown('Solyitcs Partners and MMM – Model Monitoring Module are '+\
    'registered trademarks of Solytics Partners LLC and/or its affiliates. '+\
    'Other names may be trademarks of their respective owners.')
    st.subheader('Warranty Disclaimer')
    st.markdown('The information contained herein is subject to change without notice '+\
    'and is not warranted to be error-free. If you find any errors, please report '+\
    'them to us in writing.')
    st.subheader('Consequential Damages Disclaimer')
    st.markdown('This software and related documentation are provided under a '+\
    'license agreement containing restrictions on use and disclosure and are '+\
    'protected by intellectual property laws. Except as expressly permitted in '+\
    'your license agreement or allowed by law, you may not use, copy, reproduce, '+\
    'translate, broadcast, modify, license, transmit, distribute, exhibit, perform, '+\
    'publish or display any part, in any form, or by any means. Reverse engineering, '+\
    'disassembly, or decompilation of this software, unless required by law for '+\
    'interoperability, is prohibited.')
    st.subheader('Third Party Content, Products Services and Disclaimer')
    st.markdown('This software or hardware and documentation may provide access to '+\
    'or information on content, products and services from third parties. Oracle '+\
    'Corporation and its affiliates are not responsible for and expressly disclaim '+\
    'all warranties of any kind with respect to third-party content, products, and services. '+\
    'Solytics Partners and its affiliates will not be responsible for any loss, costs, or '+\
    'damages incurred due to your access to or use of third-party content, products, or services.')
    st.markdown("""---""")
    st.text('© 2019 MMM. All Rights Reserved')
