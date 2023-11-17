import streamlit as st
from ui_extract import main as extract
from ui_classify import main as classify

st.set_page_config(layout="wide")

if not hasattr(st.session_state, 'mode'):
    st.session_state.mode = 'Extract'

# Create a sidebar for navigation
st.sidebar.title('Thematic extraction and classification')
sub_apps = ['Extract', 'Classify']
st.session_state.mode = st.sidebar.radio('Choose a workflow:', sub_apps)

# Display the selected sub app
if st.session_state.mode == 'Extract':
    extract()
elif st.session_state.mode == 'Classify':
    classify()
