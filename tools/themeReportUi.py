import streamlit as st
from joblib import Memory
import json
import pandas as pd
from metadata.columns import getColumnMetadata
from stats.coincidenceAnalysis import getCoincidenceStats
from generativeai.summarise import summariseStats

#st.set_page_config(layout="wide")

memory = Memory(location='.', verbose=0, bytes_limit=100*1024*1024)

globalStore = {}
def save(key, dataframe, metadata):
    globalStore[key] = dataframe
    globalStore[key + "_metadata"] = metadata
def remove(key):
    globalStore.pop(key, None)
    globalStore.pop(key + "_metadata", None)

@memory.cache
def getStats(key):
    df = globalStore[key]
    metadata = globalStore[key + "_metadata"]
    stats, ignoredStats = getCoincidenceStats(df, metadata)
    return stats, ignoredStats

@memory.cache
def getSummary(report):
    return summariseStats(report, model="gpt-4-1106-preview")

sheet = None
df = None
columnMetadata = None
columnMetadataOverrides = None
stats = None
ignoredStats = None
summary = None

st.header("Theme Report")

file = st.file_uploader("Upload an excel file", type="xlsx", accept_multiple_files=False)

if file is not None:
    excelFile = pd.ExcelFile(file)
    sheet = st.selectbox("Select sheet", excelFile.sheet_names)

if sheet is not None:
    df = pd.read_excel(excelFile, na_filter=None, keep_default_na=False, dtype=str, sheet_name=sheet)
    columnMetadata = getColumnMetadata(df)
    with st.expander("Column metadata", expanded=False):
        st.text(json.dumps(columnMetadata, indent=4))

if columnMetadata is not None:
    save(file.name, df, columnMetadata)
    stats, ignoredStats = getStats(file.name)
    remove(file.name)
    with st.expander("Stats", expanded=False):
        st.text(stats)
    with st.expander("Ignored stats", expanded=False):
        st.text(ignoredStats)

if stats is not None and stats != "":
    generateReport = st.button("Generate summary")
    if generateReport:
        summary = getSummary(stats)

if summary is not None:
    st.markdown(summary, unsafe_allow_html=False)