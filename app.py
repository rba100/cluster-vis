import streamlit as st
from ui_extract import main as extract
from ui_classify import main as classify

st.set_page_config(layout="wide")

if not hasattr(st.session_state, 'mode'):
    st.session_state.mode = 'Extract'

with st.sidebar:
    # Create a sidebar for navigation
    st.sidebar.header('Thematic extraction and classification', divider='rainbow')
    sub_apps = ['Extract', 'Classify']
    st.session_state.mode = st.radio('Choose a workflow:', sub_apps)

    with st.expander('Examples'):
        st.caption("Public domain sample data from the FY18 Tenant Satisfaction Survey conducted by the US government.")
        if st.button('Load example'):
            with open('sample.txt', 'r', encoding='utf-8') as f:
                st.session_state.data_strings_raw = f.read()
        st.caption("You will find that clustering focuses too much on 'office' or 'facility' concepts. Try removing the concept of 'office building' and setting clusters to 25.")

# Display the selected sub app
if st.session_state.mode == 'Extract':
    extract()
elif st.session_state.mode == 'Classify':
    classify()