import streamlit as st
from ui_extract import main as extract
from ui_classify import main as classify

st.set_page_config(layout="wide")

if not hasattr(st.session_state, 'mode'):
    st.session_state.mode = 'Extract'


with st.sidebar:
    # Create a sidebar for navigation
    st.sidebar.title('Thematic extraction and classification')
    sub_apps = ['Extract', 'Classify']
    st.session_state.mode = st.radio('Choose a workflow:', sub_apps)

    with st.expander('Examples'):
        st.caption("Public domain sample data from the FY18 Tenant Satisfaction Survey conducted by the US government.")
        if st.button('Load example'):
            with open('sample.txt', 'r', encoding='utf-8') as f:
                st.session_state.data_strings_raw = f.read()
        st.caption("If you find that clustering focusses to much on 'office' concecepts, try the 'remove concept' feature with a prompt like 'office building'.")

# Display the selected sub app
if st.session_state.mode == 'Extract':
    extract()
elif st.session_state.mode == 'Classify':
    classify()