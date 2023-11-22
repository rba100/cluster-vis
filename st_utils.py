import streamlit as st

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
        #print(f"Updating {storeKey} {valueKey} to {st.session_state[internalKey]}")
        st.session_state[storeKey][valueKey] = st.session_state[internalKey]
    
    return internalKey, update_value

def value_persister_single(valueKey):
    internalKey = f"_{valueKey}"
    if valueKey not in st.session_state:
        st.session_state[valueKey] = None
    if internalKey not in st.session_state:
        st.session_state[internalKey] = st.session_state[valueKey]
    
    def update_value():
        #print(f"Updating {valueKey} to {st.session_state[internalKey]}")
        st.session_state[valueKey] = st.session_state[internalKey]
    
    return internalKey, update_value