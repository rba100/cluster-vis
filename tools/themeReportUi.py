import streamlit as st
from joblib import Memory
import json
import os
import subprocess
import pandas as pd
from metadata.columns import getColumnMetadata
from stats.coincidenceAnalysis import getCoincidenceStats
from generativeai.summarise import summariseStats
from generativeai.charts import getPythonForCharts

memory = Memory(location='.', verbose=0, bytes_limit=100*1024*1024)

globalStore = {}
def save(key, dataframe, metadata):
    globalStore[key] = dataframe
    globalStore[key + "_metadata"] = metadata
def remove(key):
    globalStore.pop(key, None)
    globalStore.pop(key + "_metadata", None)

@memory.cache
def getStats(key, metadata):
    df = globalStore[key]
    #metadata = globalStore[key + "_metadata"]
    stats, ignoredStats = getCoincidenceStats(df, metadata)
    return stats, ignoredStats

@memory.cache
def getSummary(report):
    return summariseStats(report, model="gpt-4-1106-preview")

@memory.cache
def getPython(summary):
    return getPythonForCharts(summary)

sheet = None
df = None
columnMetadata = None
columnMetadataOverrides = None
stats = None
ignoredStats = None
summary = None
generateImages = False
classificationSources = ["hardcoded", "openai", "user"]
classificationTypes = ["ignored", "classification", "boolean"]

if "summary" not in st.session_state:
    st.session_state["summary"] = None
if "images" not in st.session_state:
    st.session_state["images"] = []

st.header("Theme Report")

file = st.file_uploader("Upload an excel file", type="xlsx", accept_multiple_files=False)

if file is not None:
    excelFile = pd.ExcelFile(file)
    sheet = st.selectbox("Select sheet", excelFile.sheet_names)

if sheet is not None:
    df = pd.read_excel(excelFile, na_filter=None, keep_default_na=False, dtype=str, sheet_name=sheet)
    columnMetadata = getColumnMetadata(df)
    columnMetadataOverrides = {}
    with st.expander("Column metadata", expanded=False):
        for key, value in columnMetadata.items():
            columnMetadataOverrides[key] = {}
            columnMetadataOverrides[key]["type"] = st.selectbox(key, classificationTypes, index=classificationTypes.index(value["type"]))
            isChanged = columnMetadataOverrides[key]["type"] != columnMetadata[key]["type"]
            columnMetadataOverrides[key]["classifiedBy"] = "user" if isChanged else columnMetadata[key]["classifiedBy"]

if columnMetadataOverrides is not None:
    save(file.name, df, columnMetadataOverrides)
    stats, ignoredStats = getStats(file.name, columnMetadataOverrides)
    remove(file.name)
    with st.expander("Stats", expanded=False):
        st.text(stats)
    with st.expander("Ignored stats", expanded=False):
        st.text(ignoredStats)

if stats is not None and stats != "":
    generateReport = st.button("Generate summary")
    if generateReport:
        st.session_state["summary"] = getSummary(stats)

if st.session_state["summary"] is not None:
    generateImages = st.button("Generate images")
    st.markdown(st.session_state["summary"], unsafe_allow_html=False)

if generateImages:
    pythonCode = getPython(st.session_state["summary"])
    folderPath = ".temp/" + file.name
    imagesFolderPath = folderPath + "/images"
    if not os.path.exists(imagesFolderPath):
        os.makedirs(imagesFolderPath)
    with open(folderPath + "/charts.py", "w") as f:
        f.write(pythonCode)
    subprocess.run(["python", "charts.py"], cwd=folderPath)
    images = os.listdir(imagesFolderPath)
    st.session_state["images"] = [imagesFolderPath + "/" + image for image in images]

for image in st.session_state["images"]:
    st.image(image, use_column_width=True)
    