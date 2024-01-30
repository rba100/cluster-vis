import streamlit as st

def init_session_state(empty=None, strings=None, dicts=None, false=None, true=None):
    if empty is None:
        empty = []
    if strings is None:
        strings = []
    if dicts is None:
        dicts = []
    if false is None:
        false = []
    if true is None:
        true = []
    
    for item in empty:
        if item not in st.session_state:
            st.session_state[item] = None
    
    for item in strings:
        if item not in st.session_state:
            st.session_state[item] = ""
    
    for item in dicts:
        if item not in st.session_state:
            st.session_state[item] = {}

    for item in false:
        if item not in st.session_state:
            st.session_state[item] = False

    for item in true:
        if item not in st.session_state:
            st.session_state[item] = True

def value_persister(valueKey, storeKey=None):
    """
    Works around a bug in streamlit where you can't bind a varaible to a widget
    if that widget is transient.
    The return value is (key, update) which are args to pass to the widget.
    """

    if storeKey is None:
        return value_persister_single(valueKey)
    
    internalKey = f"_{storeKey}-{valueKey}"
    if storeKey not in st.session_state:
        st.session_state[storeKey] = {}
    if valueKey not in st.session_state[storeKey]:
        st.session_state[storeKey][valueKey] = None
    if internalKey not in st.session_state:
        st.session_state[internalKey] = st.session_state[storeKey][valueKey]
    
    def update_value():
        st.session_state[storeKey][valueKey] = st.session_state[internalKey]
    
    return internalKey, update_value

def value_persister_single(valueKey):
    internalKey = f"_{valueKey}"
    if valueKey not in st.session_state:
        st.session_state[valueKey] = None
    if internalKey not in st.session_state:
        st.session_state[internalKey] = st.session_state[valueKey]
    
    def update_value():
        st.session_state[valueKey] = st.session_state[internalKey]
    
    return internalKey, update_value