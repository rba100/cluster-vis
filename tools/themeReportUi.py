import streamlit as st
from joblib import Memory
import json
import os
import subprocess
import pandas as pd
from metadata.columns import getColumnMetadata
from stats.coincidenceAnalysis import getCoincidenceStats
from generativeai.summarise import summariseStats
from generativeai.charts import getPythonForCharts, insertChartsIntoSummary

memory = Memory(location='.', verbose=0, bytes_limit=100*1024*1024)

globalStore = {}
def save(key, dataframe, metadata):
    globalStore[key] = dataframe
def remove(key):
    globalStore.pop(key, None)

@memory.cache
def getStats(key, metadata):
    df = globalStore[key]
    stats, ignoredStats = getCoincidenceStats(df, metadata)
    return stats, ignoredStats

@memory.cache
def getSummary(report):
    return summariseStats(report, model="gpt-4-1106-preview")

@memory.cache
def getPython(summary, stats):
    return getPythonForCharts(summary, stats)

@memory.cache
def getSummaryWithCharts(summary, images):
    return insertChartsIntoSummary(summary, images)

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
if "combinedReport" not in st.session_state:
    st.session_state["combinedReport"] = None
combinedReport = st.session_state["combinedReport"]

st.header("Theme Report")

file = st.file_uploader("Upload an excel file", type="xlsx", accept_multiple_files=False)

if file is not None:
    excelFile = pd.ExcelFile(file)
    sheet = st.selectbox("Select sheet", excelFile.sheet_names)

if sheet is not None:
    df = pd.read_excel(excelFile, na_filter=None, keep_default_na=False, dtype=str, sheet_name=sheet)
    columnMetadata = getColumnMetadata(df)
    columnMetadataOverrides = {}
    with st.expander("Columns", expanded=False):
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

if generateImages:
    pythonCode = getPython(st.session_state["summary"], stats)
    folderPath = ".temp/" + file.name
    imagesFolderPath = folderPath + "/images"
    if not os.path.exists(imagesFolderPath):
        os.makedirs(imagesFolderPath)
    with open(folderPath + "/charts.py", "w", encoding="utf-8") as f:
        f.write(pythonCode)
    subprocess.run(["python", "charts.py"], cwd=folderPath)
    images = os.listdir(imagesFolderPath)
    st.session_state["images"] = [imagesFolderPath + "/" + image for image in images]
    combinedMd = getSummaryWithCharts(st.session_state["summary"], st.session_state["images"])
    with open(folderPath + "/combined.md", "w", encoding="utf-8") as f:
        f.write(combinedMd)
    subprocess.run(["mdpdf", "combined.md", "-o", "report.pdf"], cwd=folderPath)
    with open(folderPath + "/report.pdf", "rb") as f:
        combinedReport = f.read()
        st.session_state["combinedReport"] = combinedReport

if st.session_state["combinedReport"] is not None or combinedReport is not None:
    report = combinedReport if combinedReport is not None else st.session_state["combinedReport"]
    st.download_button('Download Report', report, file.name + ".pdf", mime="application/pdf")  

if st.session_state["summary"] is not None:
    st.markdown(st.session_state["summary"], unsafe_allow_html=False)

for image in st.session_state["images"]:
    st.image(image, use_column_width=True, caption=image)
