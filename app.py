import streamlit as st
from ui_extract import main as extract
from ui_classify import main as classify
import requests

st.set_page_config(layout="wide")

def get_public_ip():
    response = requests.get('https://httpbin.org/ip')
    return response.json()['origin']

if(st.secrets.has_key("maintenance") and st.secrets["maintenance"] == "true"):
    message = st.secrets.has_key("maintenance_message") and st.secrets["maintenance_message"] \
              or "This app is currently under maintenance. Please contact Robin Anderson for assistance or try again later."
    st.write(message)
    public_ip = get_public_ip()
    st.write(f'{public_ip}')
    st.stop()

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
                st.session_state._n_clusters = 25
                st.session_state._removeConceptText = 'office building'

# Display the selected sub app
if st.session_state.mode == 'Extract':
    extract()
elif st.session_state.mode == 'Classify':
    classify()